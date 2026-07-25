"""
llm.py — LLM 客户端
====================

职责:
    把"和 LLM 说一句话"封装成一个函数, main.py 不用关心 SDK 细节。

API 演进:
    - chat()      Step 1 引入, 简单调用, 只返 string
    - chat_step() Step 3 引入, 支持 tools, 返结构化结果 (含 finish_reason / tool_calls)

学到的概念:
    1. OpenAI 风格消息结构: [{"role": ..., "content": ...}, ...]
       role 有: system / user / assistant / tool
    2. client.chat.completions.create(...) 是 agent 心脏的一次跳动
    3. finish_reason: "stop"=普通回复, "tool_calls"=我要调工具, "length"=截断
"""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from config import load_config

# 进程启动时建一次 client, 复用连接池
_cfg = load_config()
_client = OpenAI(
    api_key=_cfg.api_key,
    base_url=_cfg.base_url,
)

# 系统提示词: agent 的"人设"和"行为规则"
# 教学版先用最简单的, Step 5 会展开讲怎么设计 system prompt
DEFAULT_SYSTEM_PROMPT = "你是一个友好、简洁的助手, 用中文回答用户的问题。"


def chat(messages: list[dict], model: str | None = None) -> str:
    """
    简单调用, 只返回 assistant 的 content 字符串.
    保留给不需要 tool calling 的场景. Step 3+ 建议用 chat_step().
    """
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages
    response = _client.chat.completions.create(
        model=model or _cfg.model,
        messages=messages,
    )
    return response.choices[0].message.content


@dataclass
class ChatStepResult:
    """chat_step() 的返回结构.

    字段:
        content:       assistant 文字内容, 普通回复时有值, 工具调用时通常为 None
        tool_calls:    工具调用列表, 形如 [{"id": ..., "function": {"name": ..., "arguments": ...}}, ...]
                       普通回复时为 []
        finish_reason: "stop" / "tool_calls" / "length" / ...
        request:       调 LLM 时的完整 kwargs (含 messages 全文本 + tools schemas), 调试用
        response:      LLM 返回的完整 response (model_dump), 调试用 (含 usage / id / created / ...)
    """
    content: str | None
    tool_calls: list[dict]
    finish_reason: str
    request: dict | None = None
    response: dict | None = None


def chat_step(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
) -> ChatStepResult:
    """
    单步 LLM 调用, 返回结构化结果.

    一次调用可能:
    - finish_reason == "stop":       普通回复, 看 content
    - finish_reason == "tool_calls": LLM 想调工具, 看 tool_calls, 调用方要 dispatch
    - finish_reason == "length":     截断, 一般要 break

    参数:
        messages: 历史消息 (内部会加 system)
        tools:    工具 schema 列表, None 表示不让 LLM 调任何工具
    """
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages
    kwargs = {
        "model": model or _cfg.model,
        "messages": messages,
    }
    if tools is not None:
        kwargs["tools"] = tools

    print(f"→ LLM call, {len(messages)} msgs, {len(tools or [])} tools")
    response = _client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message
    content = message.content
    tool_calls = [tc.model_dump() for tc in (message.tool_calls or [])]
    finish_reason = choice.finish_reason

    return ChatStepResult(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        request=kwargs,                     # 完整请求 (messages + tools + model + ...)
        response=response.model_dump(),     # 完整响应 (choices / usage / id / created / ...)
    )


if __name__ == "__main__":
    # 冒烟测试: 不传 tools, 应该跟 chat() 一样, 返普通回复
    result = chat_step([{"role": "user", "content": "用一句话介绍你自己"}])
    print(f"finish_reason={result.finish_reason!r}")
    print(f"content={result.content!r}")
    print(f"tool_calls={result.tool_calls!r}")
