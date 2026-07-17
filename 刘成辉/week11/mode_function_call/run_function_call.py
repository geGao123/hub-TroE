"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 循环闭环：模型输出 tool_call → 宿主执行 → 结果以 role=tool 回填 → 再次请求模型
  重复这一过程直到模型不再返回 tool_calls（视为已收集足够信息，生成最终回答），
  或达到 max_iterations（防止 LLM 死循环）。这正是 ReAct / Agent 的核心机制。
  3. 并行工具调用：模型在同一次响应里输出多个 tool_call（如同时查北京+上海天气），
  宿主逐个执行后一并回填；与"循环多次"是不同维度，循环里每轮都可以并行。
  4. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题（可能触发多轮工具调用）
  python mode_function_call/run_function_call.py --question "查北京、上海、广州三座城市的天气，挑出最热的"

  # 内置示例问题（演示并行 / 循环 / 幻觉控制）
  python mode_function_call/run_function_call.py --demo

  # 自定义最大循环轮次（防 LLM 死循环）
  python mode_function_call/run_function_call.py -q "..." --max-iterations 8

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

与其它方式的关系：
  本文件的 LLM 循环代码，和 mode_mcp/run_mcp.py、mode_cli/run_cli.py 几乎一样，
  差异只在"工具从哪来"和"调用怎么执行"——这正是三者对比的教学点。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_weather  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",  # 即 deepseek-v4-flash
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_report",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录 5 家公司：贵州茅台(600519)/五粮液(000858)/"
                "宁德时代(300750)/海康威视(002415)/中国平安(601318)，"
                "年份仅 2021/2022/2023。不在库内的公司请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题，自然语言。重要：不要包含公司名和年份"
                            "（已由 stock_code/year 参数过滤），只用简短财务术语，"
                            "例如 '营收和净利润'、'研发投入'、'主营业务'。"
                            "把公司名写进 query 会稀释检索精度。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤，如 '300750'。不传则跨公司检索",
                    },
                    "year": {
                        "type": "string",
                        "description": "可选，按年份过滤：'2021' / '2022' / '2023'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数，默认5，建议不超过10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "列出年报知识库中收录的所有公司、股票代码与可查年份。用于确认目标公司在库内。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "get_weather": get_weather,
}


# ── 循环闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时调用 get_weather。"
    "你可以多轮调用工具：例如问多个城市天气时，可分别多次调用 get_weather；"
    "问对比+天气时，可先调 search_annual_report 拿到数据，再调 get_weather 补充天气信息。"
    "信息收集充分后再生成最终回答。同一次响应里也可以并行输出多个 tool_call。"
    "如果用户问题在工具能力之外，直接说明，不要硬调工具。"
)


def run(client, model: str, question: str, verbose: bool = True, max_iterations: int = 5) -> dict:
    """
    循环闭环：模型可以多轮调工具，直到信息充分后生成最终回答。
    每轮：模型输出 tool_call → 宿主逐个执行 → 以 role=tool 回填 → 再次请求模型。
    退出条件：模型不再返回 tool_calls（视为已收集足够信息）或达到 max_iterations。
    返回 {answer, tool_calls, elapsed, iterations, truncated} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    iterations = 0
    truncated = False

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        if verbose:
            print(f"  [iter {iteration}] 请求模型...")

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 【退出条件 1】模型没有 tool_calls → 视为已收集足够信息,生成最终回答
        if not msg.tool_calls:
            answer = msg.content or ""
            elapsed = time.time() - t0
            if verbose:
                print(f"  → [llm] 最终回答（{elapsed:.1f}s, 共 {iteration} 轮迭代）")
            return {
                "answer": answer,
                "tool_calls": tool_call_log,
                "elapsed": elapsed,
                "iterations": iterations,
                "truncated": truncated,
            }

        # 【教学时刻 3】：模型输出了 tool_calls → 逐个执行后端函数
        # 把 assistant 这条带 tool_calls 的消息原样回填,保持上下文
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "iteration": iteration})
            if verbose:
                print(f"  → [tool] {name}({args})")
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    # 工具执行！！
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            # 以 role=tool 把每个工具的结果回填,tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # 本轮 tool 执行完,回到循环顶部再次请求模型
        # 模型看到刚回填的工具结果,可能: 1) 又输出 tool_calls → 继续  2) 直接给最终答案 → 退出

    # 【退出条件 2】达到 max_iterations 仍收敛不了
    # 取最后一次响应作为"部分答案"返回,并标记 truncated
    truncated = True
    answer = (msg.content or "") + f"\n\n[警告] 已达到 max_iterations={max_iterations} 仍未收敛"
    elapsed = time.time() - t0
    if verbose:
        print(f"  ⚠️  达到 max_iterations={max_iterations},强制结束（{elapsed:.1f}s, {iterations} 轮）")
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
        "iterations": iterations,
        "truncated": truncated,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",  # 单 tool,1 轮收敛
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",  # 2 tool 并行(RAG + 天气),1 轮收敛
    "对比贵州茅台和五粮液2023年的营收。",  # 2 tool 并行(都是 RAG),1 轮收敛
    "比亚迪2023年营收是多少？",  # 幻觉控制:不在知识库,直接拒绝不调 tool
    "查北京、上海、广州三座城市的天气,挑出最热的那个城市,并给出去那里旅游的穿衣建议。",  # 天气循环:3 tool 串行,演示多轮
    "宁德时代 2023 研发投入是多少?查出来后再告诉我宁德现在下不下雨。",  # 跨 tool 链式调用:RAG → 天气
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call（循环版）")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="最大循环轮次,防止 LLM 死循环,默认 5")
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model} max_iterations={args.max_iterations}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, verbose=not (args.quiet or args.json),
                     max_iterations=args.max_iterations)
        result["question"] = q
        results.append(result)
        if not args.json:
            print(f"\n[统计] 迭代 {result['iterations']} 轮,共 {len(result['tool_calls'])} 次工具调用"
                  f"{', 强制截断' if result['truncated'] else ''}")
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        # 单问题输出单对象；demo 输出数组
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
