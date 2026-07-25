# timiAgent

个人学习项目：从零实现一个 Agent Harness, 一步步理解 Mavis 这类 agent 框架是怎么工作的。

## 学习路径

| Step | 主题 | 状态 |
|---|---|---|
| 1 | 最小 REPL: 单轮 LLM 调用 | ✅ done |
| 2 | 多轮上下文 + 长度控制 | ✅ done |
| 3 | Tool calling 基础 | ✅ done |
| 4 | 工具注册表 (装饰器) | ✅ done |
| 5 | System prompt 设计 | 🚧 TODO |
| 6 | 流式输出 | ⏳ |
| 7 | Memory 持久化 | ⏳ |
| 8 | Sub-agent | ⏳ |
| 9 | Hooks / 中间件 | ⏳ |
| 10 | 进阶: cron / async / context 压缩 | ⏳ |
| - | Web UI (辅助查看, 非主学习线) | ✅ done |

## 快速开始

### CLI 模式
```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Web 模式
```bash
source .venv/bin/activate
pip install -r requirements.txt
python server.py
# 浏览器打开 http://127.0.0.1:8000
```

Web UI 特性:
- 实时聊天, SSE 流式推送
- 右侧调试面板: 选 L0/L1/L2 调试级别
- 工具列表实时展示
- 事件流单独面板, 可勾选要看哪些事件类型
- `/clear` 一键清空

## Step 1 笔记

### 涉及的概念
- 12-factor 配置分离
- OpenAI 风格 messages 结构
- REPL (read-eval-print loop) 是 agent 的主线程
- fail-fast: 配置错误启动时就要暴露

### 文件分工
- `config.py` — 加载和校验环境变量
- `llm.py` — 封装一次 LLM 调用
- `main.py` — 主循环: 读用户输入 -> 调 LLM -> 打印回复

### 验收标准
- [x] `python config.py` 打印 Config(...) (说明 .env 加载对了)
- [x] `python llm.py` 打印一句中文回复 (说明 LLM 通了)
- [x] `python main.py` 进入 REPL, 能聊, 输入 /quit 退出
- [x] 多轮对话时, LLM 知道上文

## Step 2 笔记

### 涉及的概念
- token 估算 (字符级粗估)
- 截断策略: 保留 system + 最近 N 条
- "上下文管理"是 agent 工程里独立的模块, 不该塞进 main.py

### 新增 / 改动
- `context.py` (新) — `estimate_tokens()` + `trim_messages()`
- `main.py` (改) — 命令前缀统一 `/` (跟 Claude Code / Cursor 一致), 加 `/clear` `/history` `/help`

### 验收标准
- [x] `python context.py` 冒烟测试: 25 条 trim 到 ≤ 20, system 保留
- [x] `python main.py` 聊 25 轮, `/history` 显示 ~20 条
- [x] `/clear` 后问"我刚才说了什么", LLM 应该失忆
- [x] 截断后最近几轮对话依然连贯

## Step 3 笔记

### 涉及的概念
- OpenAI function calling schema (JSON Schema 描述工具)
- finish_reason: "stop" 普通回复 / "tool_calls" 调工具 / "length" 截断
- Inner loop: 一次 user input 可能要往返 LLM 多次
- `role=tool` 消息格式 (tool_call_id 必须匹配)

### 新增 / 改动
- `tools.py` (新) — 工具 schema + `get_current_time()` + `dispatch_tool()`
- `llm.py` (改) — 新增 `chat_step()`, 返 `ChatStepResult(content, tool_calls, finish_reason)`
- `main.py` (改) — `run_agent_turn()` 实现 inner loop, 加 `/tools` 命令

### Agent Loop 流程
```
user input
  ↓
messages.append(user)
  ↓
trim_messages
  ↓
[INNER LOOP]
  ↓
chat_step(messages, tools=...)
  ↓
stop? ──→ print, 退出
  ↓ tool_calls
执行工具
  ↓
messages.append(tool result)
  ↓
回到 chat_step
```

### 验收标准
- [x] `python tools.py` 冒烟: `dispatch_tool("get_current_time", "{}")` 返回时间字符串
- [x] `python llm.py` 冒烟: `chat_step()` 返 `finish_reason="stop"`, `tool_calls=[]`
- [x] `python main.py`: 问"现在几点了?" 看到 `🔧 get_current_time({})` 被打印, LLM 最终回复含时间
- [x] `/tools` 列出当前工具

## Step 4 笔记

### 涉及的概念
- 装饰器: 接函数返函数, 不改本体只"包"一层
- 注册表模式: 全局 dict 收集, 遍历统一暴露
- 同样的套路 Flask 用 @app.route, Click 用 @cli.command, pytest 用 @pytest.fixture

### 新增 / 改动
- `tools.py` (重写) — `@tool(...)` 装饰器 + `_REGISTRY` + `all_tool_schemas()`
- `main.py` (小改) — 用 `all_tool_schemas()` 替代 `ALL_TOOL_SCHEMAS` 常量

### 前后对比

之前 (Step 3) 加新工具要改 4 处:
```python
GET_X_SCHEMA = {...}   # 1. 手写 schema
def get_x(): ...        # 2. 函数
ALL_TOOL_SCHEMAS.append(GET_X_SCHEMA)  # 3. 加到 schema 列表
_TOOL_FUNCS["get_x"] = get_x           # 4. 加到 dispatch 表
```

现在 (Step 4) 只改 1 处:
```python
@tool(description="...", parameters={...})
def get_x(): ...
```

### 验收标准
- [ ] `python tools.py` 冒烟: "已注册 2 个工具" + dispatch 都对
- [ ] `python main.py` `/tools` 仍列出 2 个工具
- [ ] 问 "北京天气如何" 仍走通 inner loop
- [ ] 试着加第 3 个工具 (比如 calculator), 验证只改一处就能用

## 命令约定

所有命令以 `/` 开头 (user 偏好, 跟主流 agent 工具一致):

| 命令 | 作用 |
|---|---|
| `/quit` | 退出 |
| `/clear` | 清空对话历史 |
| `/history` | 查看消息数和 token 估算 |
| `/tools` | 列出可用工具 |
| `/help` | 帮助 |
