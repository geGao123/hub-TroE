"""
server.py — FastAPI Web 后端
=============================

提供:
    GET  /          - 单页 Web UI
    POST /api/chat  - 发送一条用户消息, 返回 SSE 流 (Server-Sent Events)
    POST /api/clear - 清空会话

设计要点:
    1. 用 SSE (text/event-stream) 推送事件, 浏览器原生支持, 无需 WebSocket
    2. Session 存内存 dict (session_id -> messages), 单进程够用, 多用户要换 Redis
    3. debug_level 通过 POST body 传, 前端每个请求可独立控制
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import (
    EVT_ASSISTANT,
    EVT_DONE,
    EVT_ERROR,
    EVT_RAW_REQUEST,
    EVT_RAW_RESPONSE,
    EVT_TOOL_CALL,
    EVT_TOOL_RESULT,
    EVT_USER,
    run_agent_turn,
)
from context import trim_messages
from tools import all_tool_schemas


app = FastAPI(title="timiAgent Web")

# 静态文件目录
STATIC_DIR = Path(__file__).parent / "static"

# 内存 session store: {session_id: messages}
# 教学用, 真实场景要换持久化
SESSIONS: dict[str, list[dict]] = {}


# === Pydantic models ===

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    debug_level: int = 0


# === SSE 辅助 ===

def _sse(event: str, data: dict) -> str:
    """格式化成 SSE 协议: event: <name>\\ndata: <json>\\n\\n"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# === 路由 ===

@app.get("/")
async def index():
    """返回单页 Web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/tools")
async def list_tools():
    """返回所有可用工具, 含完整 JSON Schema (前端调试面板展示用)."""
    schemas = all_tool_schemas()
    return {
        "tools": [
            {"name": s["function"]["name"], "description": s["function"]["description"]}
            for s in schemas
        ],
        "schemas": schemas,
    }


@app.get("/api/system")
async def get_system_prompt():
    """返回当前 session 用的 system prompt (前端调试面板展示用)."""
    from llm import DEFAULT_SYSTEM_PROMPT
    return {"prompt": DEFAULT_SYSTEM_PROMPT}


@app.post("/api/clear")
async def clear_session(req: ChatRequest):
    """清空指定 session 的历史."""
    sid = req.session_id or "default"
    SESSIONS[sid] = []
    return {"ok": True, "session_id": sid}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    接收用户消息, 流式返回事件.

    流程:
        1. 拿/建 session
        2. 把 user 消息塞进去
        3. trim_messages
        4. 跑 run_agent_turn, 把每个事件转成 SSE 推送出去
        5. 流结束
    """
    sid = req.session_id or str(uuid.uuid4())
    messages = SESSIONS.setdefault(sid, [])
    messages.append({"role": "user", "content": req.message})
    messages = trim_messages(messages)
    SESSIONS[sid] = messages

    async def event_gen():
        # 同步函数跑在 thread 里, 不阻塞 event loop
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def on_event(evt_type: str, payload: dict):
            loop.call_soon_threadsafe(queue.put_nowait, (evt_type, payload))

        def run_in_thread():
            try:
                run_agent_turn(
                    SESSIONS[sid],
                    on_event=on_event,
                    debug_level=req.debug_level,
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        task = loop.run_in_executor(None, run_in_thread)

        # 先推 session_id
        yield _sse("session", {"session_id": sid})

        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            evt_type, payload = item
            yield _sse(evt_type, payload)

        await task

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx buffering
        },
    )


# 挂载静态资源 (CSS/JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
