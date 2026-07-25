"""
main.py — Agent 入口
====================

这是 agent 的"主线程", 其它能力都挂在这个 while 循环上:
  - Step 1: 只有 user -> LLM -> print
  - Step 2: 加 trim + /clear / /history
  - Step 3: 加 inner loop 处理 tool_calls
  - Step 4+: 工具注册表化 / 多工具 / 更复杂能力

命令约定 (统一 / 前缀, 和 Claude Code / Cursor 风格一致):
  /quit    - 退出
  /clear   - 清空对话历史
  /history - 打印消息数和 token 估算
  /tools   - 列出可用工具
  /help    - 打印帮助

退出:
  - /quit
  - Ctrl-C / Ctrl-D

Agent Loop 流程 (Step 3):
  1. 读 user input
  2. messages.append(user)
  3. 调 trim_messages(messages)
  4. === INNER LOOP ===
     a. result = chat_step(messages, tools=ALL_TOOL_SCHEMAS)
     b. if finish_reason == "stop":       普通回复, 打印, 退出 inner loop
     c. if finish_reason == "tool_calls": 执行每个 tool_call,
                                          把结果以 role=tool 加进 messages,
                                          回到 a
     d. if finish_reason == "length":     截断警告, 退出 inner loop
  === INNER LOOP END ===
"""
from __future__ import annotations

import sys

from backend import (
    EVT_ASSISTANT,
    EVT_ERROR,
    EVT_TOOL_CALL,
    EVT_TOOL_RESULT,
    run_agent_turn as _run_agent_turn,
)
from context import estimate_tokens, trim_messages
from tools import all_tool_schemas


PROMPT = "👤 "
COMMAND_PREFIX = "/"


def run_agent_turn(messages: list[dict]) -> None:
    """CLI 版的 run_agent_turn: 把事件翻译成 print."""

    def on_event(evt_type: str, payload: dict) -> None:
        if evt_type == EVT_ASSISTANT:
            print(payload["text"])
        elif evt_type == EVT_TOOL_CALL:
            print(f"  🔧 {payload['name']}({payload['arguments']})")
        elif evt_type == EVT_TOOL_RESULT:
            print(f"  ← {payload['output']}")
        elif evt_type == EVT_ERROR:
            print(f"⚠️  {payload['message']}")

    _run_agent_turn(messages, on_event=on_event)


def main() -> int:
    """REPL 主循环."""
    messages: list[dict] = []

    while True:
        try:
            user_input = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye 👋")
            return 0

        if not user_input:
            continue

        # --- 命令分发 ---
        if user_input.startswith(COMMAND_PREFIX):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit"):
                print("bye 👋")
                return 0
            elif cmd == "/clear":
                messages.clear()
                print("🧹 history cleared")
                continue
            elif cmd == "/history":
                print_history(messages)
                continue
            elif cmd == "/tools":
                print_tools()
                continue
            elif cmd == "/help":
                print_help()
                continue
            else:
                print(f"未知命令: {cmd} (试试 /help)")
                continue

        # --- 正常对话 (Step 3: 走 inner loop) ---
        messages.append({"role": "user", "content": user_input})
        messages = trim_messages(messages)
        run_agent_turn(messages)


def print_help() -> None:
    """打印可用命令."""
    print("可用命令:")
    print("  /quit     退出")
    print("  /clear    清空对话历史")
    print("  /history  查看当前消息数和 token 估算")
    print("  /tools    列出可用工具")
    print("  /help     打印本帮助")


def print_history(messages: list[dict]) -> None:
    """打印当前消息数和 token 估算."""
    print(f"📊 {len(messages)} msgs, ~{estimate_tokens(messages)} tokens (估)")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"  {i+1}. [{role}] {content[:50]}{'...' if len(content) > 50 else ''}")


def print_tools() -> None:
    """打印当前可用工具清单."""
    schemas = all_tool_schemas()
    print(f"🛠  共 {len(schemas)} 个工具:")
    for schema in schemas:
        fn = schema["function"]
        print(f"  - {fn['name']}: {fn['description']}")


if __name__ == "__main__":
    sys.exit(main())
