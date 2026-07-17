#!/usr/bin/env python3
"""
民法典 RAG 问答（端到端）：
  1) BM25 + Dense + RRF 混合检索 top-k
  2) 把 chunks 拼进 prompt 喂给 LLM
  3) 输出：问题、检索引用、LLM 回答

LLM 通过 Anthropic SDK 调用（你环境里 ANTHROPIC_BASE_URL 指向 DeepSeek 兼容端点）。
DeepSeek 这条兼容通道默认会返回 ThinkingBlock，脚本只取 TextBlock。

用法：
    python scripts/rag_answer.py "自然人下落不明多久才能宣告失踪？"
    python scripts/rag_answer.py "违约金约定多少合理？" --k 5
    python scripts/rag_answer.py --quiet-check   # 用预设问题跑一遍自检
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 复用 search_mindian 的混合检索函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_mindian import (  # noqa: E402
    load_bm25, get_qdrant_client, search_bm25, search_dense, rrf_merge,
)


# ──────────────────────────────────────────────────────────────
# 1. Prompt 模板
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是"民法典智能问答助手"。你的回答必须完全基于"参考资料"里提供的法条原文。

【行为规则 — 严格遵守，违反任何一条都不可接受】
1. **严禁引用任何参考资料之外的内容**。包括但不限于："司法实践""通常认为""一般情况下""实践中""通常""一般""大致""大概""约""30% 左右""通常情况下认定"等表达。
2. **严禁补充任何参考资料里没有的事实、数字、比例、案例**。
3. 资料不足以回答时，**直接回复一句**："未在提供的法条中找到直接依据"。不要再尝试猜测、补全或加任何额外的解释。
4. 引用法条时使用「第x条」格式，每条只引一次。
5. 答案结构：先给一句≤30字的基于条文原文的结论，再引用条文原文展开（直接照抄条文里的字，不要改写）。总篇幅≤200字。
6. 你是法律助手，不替人下判断，也不替人创作。"""

USER_PROMPT_TEMPLATE = """【参考资料】（按相关性排序，最相关的法条在前）
{context}

【用户问题】
{question}

【你的回答】"""


def build_context(chunks: list[dict]) -> str:
    """把 chunks 编号化塞进 prompt。多法源时显式标注法源名。"""
    parts = []
    for i, c in enumerate(chunks, 1):
        article = c.get("article_index", "?")
        section = c.get("section", "?")
        page = c.get("start_page", "?")
        law_short = c.get("law_short", "")
        text = c.get("text", "").replace("\n", " ").strip()
        prefix = f"[{i}] 《{law_short}》" if law_short else f"[{i}] "
        parts.append(f"{prefix}{article}（{section}，第{page}页）\n{text}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────
# 2. Query 改写（多轮对话 → 自包含 query）
# ──────────────────────────────────────────────────────────────
REWRITE_SYSTEM = """你是查询改写助手。你的任务是把用户最新一轮的问题改写成一个**自包含、清晰的法律问题**，消除代词、省略和对话上下文依赖，让它脱离对话历史也能被完全理解。

要求：
1. 改写后必须**自包含**——脱离对话历史也能被完全理解。
2. **保留**原问的法律关键词（如"违约金"、"借款合同"、"诉讼时效"、"遗嘱继承"等）。
3. **不要**解释、**不要**加"请问"等客套话、**不要**加引号。
4. 如果原问已经自包含（如首轮问题或本身完整），**原样返回**。
5. 只输出改写后的问题本身，一行。"""

REWRITE_USER_TEMPLATE = """[对话历史]
{history}

[用户的最新问题]
{question}

[改写后的问题]"""


def rewrite_query(question: str, chat_history: list[dict] | None = None) -> str:
    """用 LLM 把省略/代词的 query 改写成自包含 query。无历史时原样返回。

    改写用独立 LLM 调用（不走 chat_history messages，避免循环嵌套）；
    用 top_p=0.3 让改写结果稳定。
    """
    if not chat_history:
        return question

    history_lines = []
    for h in chat_history:
        history_lines.append(f"{h['role']}: {h['content']}")
    history_text = "\n".join(history_lines) or "(无历史)"

    user_prompt = REWRITE_USER_TEMPLATE.format(history=history_text, question=question)

    from anthropic import Anthropic
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-flash")
    if not base_url or not api_key:
        # 没配 LLM 就直接用原 query（不改写）
        return question

    client = Anthropic(base_url=base_url, api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            top_p=0.3,                       # 改写要确定性
            system=REWRITE_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        print(f"[rewrite_query] LLM 失败: {e}", file=sys.stderr)
        return question

    # 取文本（DeepSeek 可能带 ThinkingBlock，自动跳过）
    text = "".join(b.text for b in resp.content if hasattr(b, "text") and b.text).strip()
    # 兜底
    return text or question


# ──────────────────────────────────────────────────────────────
# 3. LLM 调用（Anthropic 协议，覆盖 DeepSeek/Claude 兼容端点）
# ──────────────────────────────────────────────────────────────
def call_llm(
    system: str,
    messages: list[dict],
    max_tokens: int = 1500,
    top_p: float = 0.9,
) -> tuple[str, str | None]:
    """
    返回 (answer_text, thinking_text_or_None)
    兼容 DeepSeek 这类带思考模式的 provider：跳过 ThinkingBlock，
    仅把 TextBlock 拼成 answer。Debug 时可拿到 thinking 块。

    `messages` 是 Anthropic 多轮格式：
        [{"role": "user"|"assistant", "content": str|list}, ...]
    第一个必须是 user。允许空列表（调用方负责保证）。

    `top_p` 是 nucleus sampling 阈值，控制 token 选择的概率质量：
      - 0.05：近似贪心，每次回答几乎一致（server 拒绝 0）
      - 0.9：常用推荐值（默认）
      - 1.0：完全不限制，token 选择最随机
    """
    from anthropic import Anthropic

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-flash")
    if not base_url or not api_key:
        sys.exit("❌ 需要环境变量 ANTHROPIC_BASE_URL 和 ANTHROPIC_API_KEY")

    if not messages:
        sys.exit("❌ messages 不能为空")
    if messages[0]["role"] != "user":
        sys.exit("❌ messages 第一个必须是 user 角色")

    client = Anthropic(base_url=base_url, api_key=api_key)
    create_kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if top_p is not None and 0.05 <= top_p <= 1.0:
        create_kwargs["top_p"] = top_p
    resp = client.messages.create(**create_kwargs)

    answer_parts: list[str] = []
    thinking_parts: list[str] = []
    for blk in resp.content:
        if hasattr(blk, "text") and blk.text:
            # TextBlock 类型
            answer_parts.append(blk.text)
        if hasattr(blk, "thinking") and blk.thinking:
            thinking_parts.append(blk.thinking)
    answer = "\n".join(answer_parts).strip()
    thinking = "\n".join(thinking_parts).strip() or None
    return answer, thinking


# ──────────────────────────────────────────────────────────────
# 3. 输出美化
# ──────────────────────────────────────────────────────────────
def print_results(question: str, context_chunks: list[dict], answer: str, thinking: str | None):
    print("=" * 78)
    print(f"❓ 问题: {question}\n")

    print(f"📚 检索 top-{len(context_chunks)}:")
    for i, c in enumerate(context_chunks, 1):
        article = c.get("article_index", "?")
        section = c.get("section", "?")
        page = c.get("start_page", "?")
        print(f"  [{i}] {article}（{section}，p.{page}）")
    print()

    print("🤖 回答:\n")
    print(answer)
    print()

    if thinking:
        print("─" * 78)
        print(f"🧠 Thinking (debug, 可选展示):\n{thinking}\n")

    print("=" * 78)
    print("📖 引用条文（供交叉验证）:")
    for i, c in enumerate(context_chunks, 1):
        article = c.get("article_index", "?")
        section = c.get("section", "?")
        page = c.get("start_page", "?")
        text = c.get("text", "").replace("\n", " ").strip()
        snippet = text[:90] + ("…" if len(text) > 90 else "")
        print(f"  [{i}] 《{article}》{section} · p.{page}")
        print(f"      {snippet}")


# ──────────────────────────────────────────────────────────────
# 4. main
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None,
                    help="用户问题（中文自然语言）")
    ap.add_argument("--server", default="http://192.168.31.101:6333")
    ap.add_argument("--collection", default="mfd_law_small")
    ap.add_argument("--bm25", default="data/pdf/all_laws.articles.bm25.pkl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-print-context", action="store_true",
                    help="不打印引用条文细节，只显示回答")
    ap.add_argument("--quiet-check", action="store_true",
                    help="跑几个预设问题做自检")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="nucleus sampling 阈值，0=贪心, 1=最大随机 (默认 0.9)")
    args = ap.parse_args()

    # 跑预设问题自检
    preset = [
        "自然人下落不明多久才能宣告失踪？",
        "约定的违约金过高是否可以请求法院减少？",
        "借款合同没有约定利息，债权人能否主张利息？",
        "小区电梯广告收益归谁所有？",
    ]
    if args.quiet_check:
        for q in preset:
            print(f"\n{'=' * 78}\n预设问题: {q}\n{'=' * 78}")
            run_one(q, args)
        return

    if not args.query:
        ap.error("需要问题，或用 --quiet-check 跑自检")
    run_one(args.query, args)


def run_one(question: str, args):
    # 1) 检索
    bm25, chunks = load_bm25(Path(args.bm25).resolve())
    client = get_qdrant_client(args)
    N = max(args.k * 2, 20)
    bm25_hits = search_bm25(bm25, chunks, question, N)
    dense_hits = search_dense(client, args.collection, question,
                              "BAAI/bge-small-zh-v1.5", N)
    fused = rrf_merge(bm25_hits, dense_hits, k_rrf=60, top_k=args.k)
    context_chunks = [f["chunk"] for f in fused]

    # 2) 拼 prompt
    ctx_text = build_context(context_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=ctx_text, question=question)

    # 3) 调 LLM（单轮：只发一条 user message）
    messages = [{"role": "user", "content": user_prompt}]
    top_p = getattr(args, "top_p", 0.9)
    answer, thinking = call_llm(SYSTEM_PROMPT, messages, top_p=top_p)

    # 4) 输出
    print_results(question, context_chunks, answer, thinking)


if __name__ == "__main__":
    main()