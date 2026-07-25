"""
backend.py — 抽出的可复用 chat session (Step 5: Web 后端共享)
=============================================================

职责:
    把 CLI (main.py) 和 Web (server.py) 共用的 chat 逻辑集中到这,
    通过事件回调 (on_event) 让上层 (CLI print / Web SSE) 自由消费.

事件类型 (除 user 外都带 round_id; raw_request/raw_response/tool_call/tool_result/assistant/error 还带 call_id):
    "user"        - 用户输入 {text}
    "tool_call"   - 工具被调用 {name, arguments, round_id, call_id}
    "tool_result" - 工具返回 {name, output, round_id, call_id}
    "assistant"   - assistant 最终回复 {text, round_id, call_id}
    "error"       - 异常 {message, round_id, call_id?}
    "done"        - 一个 turn 结束 {round_id, messages_count, tokens_est}
    "raw_request" - L0+ 调试: 调 LLM 前的完整请求 {url, request, round_id, call_id}
    "raw_response"- L0+ 调试: 调 LLM 后的完整响应 {finish_reason, response, round_id, call_id}
    "round_start" - 一轮开始 {user_text, messages_before, round_id}
    "round_end"   - 一轮结束 {messages_after, tokens_est, round_id}

学到的概念:
    1. 事件流 vs 直接 print: 同一份业务逻辑多个前端都能用
    2. 回调函数: 把"副作用"交给调用方, 函数本身保持纯
    3. 解耦: backend 不知道也不关心是 CLI 还是 Web 在用它
"""
from __future__ import annotations

from typing import Callable

from context import estimate_tokens, trim_messages
from llm import chat_step
from tools import all_tool_schemas, dispatch_tool


# 事件类型常量, 防止拼错
EVT_USER = "user"
EVT_TOOL_CALL = "tool_call"
EVT_TOOL_RESULT = "tool_result"
EVT_ASSISTANT = "assistant"
EVT_ERROR = "error"
EVT_DONE = "done"
EVT_RAW_REQUEST = "raw_request"
EVT_RAW_RESPONSE = "raw_response"
EVT_ROUND_START = "round_start"  # 新一轮 user input 开始
EVT_ROUND_END = "round_end"      # 新一轮结束 (final assistant 已发出)


EventCallback = Callable[[str, dict], None]


def run_agent_turn(
    messages: list[dict],
    on_event: EventCallback | None = None,
    debug_level: int = 0,
) -> None:
    """
    跑一次 "agent turn": 一次 user input -> 一次 assistant 回复.
    内部可能往返 LLM 多次 (调工具 -> 拿结果 -> 再调 -> ...).

    参数:
        messages:    完整历史 (含本轮 user), 会原地追加 assistant/tool 消息
        on_event:    事件回调 (type, payload), None 表示不发事件
        debug_level: 0=只发业务事件, 1=加 raw_request/raw_response

    事件里的 round_id / call_id:
        round_id 在一轮 turn 开始时生成, 整轮不变
        call_id 每次 LLM 调用递增, 用于前端 chat 消息 ↔ call-card 互跳
    """

    def emit(evt_type: str, payload: dict) -> None:
        if on_event:
            try:
                on_event(evt_type, payload)
            except Exception:
                # 回调出错不影响主流程
                pass

    # 这一轮一个唯一 id (一个 turn 一个 round)
    round_id = f"r-{id(messages)}"
    call_counter = 0

    def emit_round(evt_type: str, payload: dict) -> None:
        """对带 round_id 的事件统一加 id 前缀"""
        payload = {**payload, "round_id": round_id}
        emit(evt_type, payload)

    def emit_call(evt_type: str, payload: dict) -> None:
        """对带 call_id 的事件统一加 round+call id 前缀"""
        payload = {**payload, "round_id": round_id, "call_id": call_id}
        emit(evt_type, payload)

    # 一轮 user input 开始
    emit_round(EVT_ROUND_START, {
        "user_text": messages[-1].get("content", "") if messages and messages[-1].get("role") == "user" else "",
        "messages_before": len(messages),
    })

    while True:
        try:
            # 每次 LLM 调用前分配 call_id
            call_counter += 1
            call_id = f"{round_id}.c{call_counter}"

            result = chat_step(messages, tools=all_tool_schemas())

            # 调试: 发完整 wire-level 数据 (请求 + 响应)
            # debug_level 1: 发所有调用的完整 payload
            # debug_level 2: 同 1, 但前端能区分 (后续可以加更多细节)
            if debug_level >= 1:
                emit_call(EVT_RAW_REQUEST, {
                    "url": "/v1/chat/completions",  # 模拟 wire-level URL
                    "request": result.request,       # 完整 kwargs (model / messages / tools / ...)
                })
                emit_call(EVT_RAW_RESPONSE, {
                    "finish_reason": result.finish_reason,
                    "response": result.response,     # 完整 response (choices / usage / id / ...)
                })

            if result.finish_reason == "stop":
                emit_call(EVT_ASSISTANT, {"text": result.content or ""})
                messages.append({"role": "assistant", "content": result.content})
                break

            elif result.finish_reason == "tool_calls":
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": result.tool_calls,
                })
                for tc in result.tool_calls:
                    name = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    # tool_call/tool_result 跟本轮 LLM 调用同 call_id
                    # (前端能跟 call-card 对上, 也能看到调用顺序)
                    emit_call(EVT_TOOL_CALL, {"name": name, "arguments": args})
                    try:
                        output = dispatch_tool(name, args)
                    except Exception as e:
                        output = f"工具执行失败: {e}"
                        emit_call(EVT_ERROR, {"message": str(e), "tool": name})
                    emit_call(EVT_TOOL_RESULT, {"name": name, "output": output})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": output,
                    })
                continue

            elif result.finish_reason == "length":
                emit_call(EVT_ERROR, {"message": "LLM 输出被截断 (finish_reason=length)"})
                break

            else:
                emit_call(EVT_ERROR, {"message": f"未知 finish_reason: {result.finish_reason}"})
                break

        except Exception as e:
            emit_call(EVT_ERROR, {"message": str(e)})
            break

    emit_round(EVT_DONE, {"messages_count": len(messages), "tokens_est": estimate_tokens(messages)})

    # 一轮结束
    emit_round(EVT_ROUND_END, {
        "messages_after": len(messages),
        "tokens_est": estimate_tokens(messages),
    })
