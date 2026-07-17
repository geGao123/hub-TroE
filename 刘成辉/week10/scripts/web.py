#!/usr/bin/env python3
"""
民法典 RAG 问答 - Streamlit 前端
  - 调试模式开关（侧边栏）
  - chat 风格对话
  - 调试模式下可看：召回条文 + Prompt + LLM 推理 + 三路对比 + 原始 payload

启动：
    streamlit run scripts/web.py --server.port 8501
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from search_mindian import (  # noqa: E402
    load_bm25, get_qdrant_client, search_bm25, search_dense, rrf_merge,
)
from rag_answer import (  # noqa: E402
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, build_context, call_llm, rewrite_query,
)

st.set_page_config(
    page_title="民法典 RAG 问答",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────
# 资源加载（只在第一次 / cache miss 时跑一次）
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔌 加载 BM25 + Qdrant …")
def load_resources(bm25_path: str, server: str):
    bm25, chunks = load_bm25(Path(bm25_path))
    ns = argparse.Namespace(server=server, local=None)
    client = get_qdrant_client(ns)
    return bm25, chunks, client


@st.cache_resource(show_spinner="🧠 加载 Embedding 模型 …")
def load_embed_model(model_name: str):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name)


# ──────────────────────────────────────────────────────────────
# 调一次 RAG（带全量调试信息）
# ──────────────────────────────────────────────────────────────
def ask(
    question: str,
    k: int,
    top_p: float,
    bm25, chunks, qclient, embed_model,
    chat_history: list[dict] | None = None,
    use_rewrite: bool = True,
    law_filter: list[str] | None = None,
) -> dict:
    """
    chat_history: 之前对话的 [{role, content}, ...]，仅喂给 LLM 用于多轮指代消解；
    检索只用最新 question（保持相关）。

    use_rewrite: 启用 query rewriting（用 LLM 把省略式 query 改写成自包含 query 再检索）。
                 关闭则直接用原 question 检索（首轮 / 简单 follow-up 时可关）。

    law_filter: 法源过滤。None 或 ["全部"] = 不限；其他 = 只保留这些 law_short 的 chunk。
    """
    # 1) Query 改写（多轮对话 → 自包含 query）
    if use_rewrite and chat_history:
        search_query = rewrite_query(question, chat_history)
    else:
        search_query = question
    was_rewritten = (search_query != question)

    # 2) 混合检索（用改写后 query 召回，提升 follow-up 检索质量）
    N = max(k * 2, 20)
    bm25_hits = search_bm25(bm25, chunks, search_query, N)
    dense_hits = search_dense(qclient, "mfd_law_small", search_query, "BAAI/bge-small-zh-v1.5", N)
    fused = rrf_merge(bm25_hits, dense_hits, k_rrf=60, top_k=max(k * 3, k))

    # ── 法源过滤（在 RRF 后截断）
    if law_filter and "全部" not in law_filter:
        filter_set = set(law_filter)
        fused = [f for f in fused if f["chunk"].get("law_short") in filter_set]
        fused = fused[:k]

    context_chunks = [f["chunk"] for f in fused]

    # 3) 拼当前轮 prompt（参考资料 + 当前 question — 用**原 question**给 LLM 看，保留用户语义）
    ctx_text = build_context(context_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=ctx_text, question=question)

    # 4) 拼 messages 数组：[history..., 当前 user]
    messages = []
    if chat_history:
        for h in chat_history:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})

    # 5) 调 LLM
    answer, thinking = call_llm(SYSTEM_PROMPT, messages, top_p=top_p)

    # 6) embedding（debug 用，用改写后 query 算 embedding，与检索一致）
    q_vec = next(embed_model.embed([search_query])).tolist()

    return {
        "question": question,
        "search_query": search_query,             # 实际用于检索的 query（可能 = question）
        "was_rewritten": was_rewritten,
        "law_filter": law_filter or ["全部"],
        "answer": answer,
        "thinking": thinking,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "messages_sent": messages,
        "chat_history_used": chat_history or [],
        "bm25_hits": bm25_hits[:k],
        "dense_hits": dense_hits[:k],
        "fused": fused,
        "context_chunks": context_chunks,
        "question_embedding": q_vec,
        "embedding_dim": len(q_vec),
        "top_p": top_p,
    }


# ──────────────────────────────────────────────────────────────
# 调试面板 UI
# ──────────────────────────────────────────────────────────────
def _fmt_score(v):
    if v is None:
        return "—"
    return f"{float(v):.4f}"


def render_debug(result: dict):
    st.markdown("---")
    st.markdown("#### 🛠  调试面板")
    tab_summary, tab_chunks, tab_compare, tab_prompt, tab_thinking, tab_payload, tab_history = st.tabs(
        ["📊 召回综述", "📚 召回条文", "⚖️  三路对比", "📝 Prompt", "🧠 LLM 推理",
         "📦 原始 Payload", "💬 上下文"]
    )

    # ───── 召回综述（默认展开） ─────
    with tab_summary:
        st.markdown(f"**Query**: {result['question']}")
        # Query 改写对比
        if result.get("was_rewritten"):
            st.info(
                f"🔄 **Query 已改写**（用于检索）\n\n"
                f"- 原 query：`{result['question']}`\n"
                f"- 改写 query：`{result['search_query']}`"
            )
        else:
            st.caption("🔍 检索 query：`{}`（未改写）".format(result.get("search_query", result["question"])))
        st.caption(
            f"Embedding dim: `{result['embedding_dim']}` · 召回 top-{len(result['fused'])} · "
            f"LLM top_p: `{result.get('top_p', 0.9)}`"
        )
        rows = []
        for h in result["fused"]:
            c = h["chunk"]
            rows.append({
                "混合 #": h["rrf_rank"],
                "法源": c.get("law_short", "?"),
                "条文": c.get("article_index", "?"),
                "节/章": c.get("section", "?"),
                "页": c.get("start_page"),
                "RRF 分": round(h["rrf"], 4),
                "BM25 #": h["bm25_rank"] if h["bm25_rank"] is not None else "—",
                "BM25 分": _fmt_score(h["bm25_score"]),
                "Dense #": h["dense_rank"] if h["dense_rank"] is not None else "—",
                "Dense 余弦": _fmt_score(h["dense_score"]),
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

    # ───── 召回条文原文 ─────
    with tab_chunks:
        for h in result["fused"]:
            c = h["chunk"]
            law_short = c.get("law_short", "?")
            label = f"#{h['rrf_rank']} 【{law_short}】《{c.get('article_index','?')}》{c.get('section','?')} · p.{c.get('start_page','?')}"
            with st.expander(label, expanded=(h["rrf_rank"] <= 2)):
                st.markdown(c.get("text", "").replace("\n", "  \n"))
                st.json({
                    "law": c.get("law"),
                    "law_short": c.get("law_short"),
                    "law_slug": c.get("law_slug"),
                    "article_id": c.get("article_id"),
                    "article_index": c.get("article_index"),
                    "section": c.get("section"),
                    "section_index": c.get("section_index"),
                    "section_kind": c.get("section_kind"),
                    "start_page": c.get("start_page"),
                    "end_page": c.get("end_page"),
                    "pages": c.get("pages"),
                    "rrf_score": round(h["rrf"], 4),
                    "bm25_rank": h["bm25_rank"],
                    "bm25_score": round(h["bm25_score"], 4) if h["bm25_score"] is not None else None,
                    "dense_rank": h["dense_rank"],
                    "dense_score": round(h["dense_score"], 4) if h["dense_score"] is not None else None,
                })

    # ───── 三路对比 ─────
    with tab_compare:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📕 BM25-only**")
            for h in result["bm25_hits"]:
                c = h["chunk"]
                st.caption(f"#{h['rank']} {c.get('article_index','?')}")
                st.caption(f"`{_fmt_score(h['score'])}` {c.get('section','?')}")
                st.markdown("---")
        with col2:
            st.markdown("**🔍 Dense-only**")
            for h in result["dense_hits"]:
                c = h["chunk"]
                st.caption(f"#{h['rank']} {c.get('article_index','?')}")
                st.caption(f"`{_fmt_score(h['score'])}` {c.get('section','?')}")
                st.markdown("---")
        with col3:
            st.markdown("**🔀 RRF 混合**")
            for h in result["fused"]:
                c = h["chunk"]
                st.caption(f"#{h['rrf_rank']} {c.get('article_index','?')}")
                st.caption(f"`RRF={_fmt_score(h['rrf'])}` {c.get('section','?')}")
                st.markdown("---")

    # ───── Prompt ─────
    with tab_prompt:
        st.markdown("**🧑‍⚖️ System Prompt**")
        st.code(result["system_prompt"], language="markdown")
        st.markdown("**👤 User Prompt**")
        st.code(result["user_prompt"], language="markdown")
        with st.expander("🔢 Query Embedding（前 20 维）"):
            st.code(
                "[" + ", ".join(f"{x:.4f}" for x in result["question_embedding"][:20]) + ", …]",
                language="text",
            )

    # ───── LLM 推理 ─────
    with tab_thinking:
        if result["thinking"]:
            st.markdown(result["thinking"])
        else:
            st.info("该 provider 未返回 thinking 块。")

    # ───── 原始 payload ─────
    with tab_payload:
        st.caption("第一条 chunk 的完整 payload（Qdrant → RRF → LLM context）")
        if result["context_chunks"]:
            st.json(result["context_chunks"][0])

    # ───── 多轮上下文 ─────
    with tab_history:
        history = result.get("chat_history_used", [])
        messages = result.get("messages_sent", [])
        st.caption(f"本次发给 LLM 的 messages 共 {len(messages)} 条 · 其中历史 {len(history)} 条")
        if history:
            st.markdown("##### ① 历史对话（用于指代消解）")
            for i, h in enumerate(history, 1):
                role = h["role"]
                content = h["content"]
                emoji = "👤" if role == "user" else "🤖"
                with st.expander(f"{emoji} [{i}] {role}: {content[:60]}{'…' if len(content) > 60 else ''}",
                                 expanded=(i == len(history))):
                    st.markdown(content)
        else:
            st.info("本次无历史上下文（多轮上下文设为 0 或首轮）")
        st.markdown("##### ② 本次问题 + 参考资料（最后一条 user）")
        if messages:
            last_user = messages[-1]
            st.code(last_user["content"], language="markdown")
        st.caption(
            f"检索只用最新问题（与历史无关），保证召回相关性。\n"
            f"💡 把对话切多轮时，'那过低呢？' 这类指代会由 LLM 基于历史自动理解。"
        )


# ──────────────────────────────────────────────────────────────
# 主区
# ──────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──
    with st.sidebar:
        st.title("⚙️  设置")
        debug = st.checkbox("🛠  调试模式", value=False, help="显示召回条文 + Prompt + LLM 推理")
        k = st.slider("检索 top-k", min_value=3, max_value=15, value=5, step=1)
        top_p = st.slider(
            "top_p (核采样)",
            min_value=0.05, max_value=1.0, value=0.9, step=0.05,
            help="nucleus sampling 阈值\n"
                 "• 0.05 ≈ 完全贪心（server 端拒绝 0）\n"
                 "• 0.9 = 推荐值（默认）\n"
                 "• 1.0 = 最大随机性",
        )
        history_turns = st.slider(
            "💬 多轮上下文",
            min_value=0, max_value=6, value=2, step=1,
            help="把最近 N 轮对话喂给 LLM（仅指代消解，不影响检索）\n"
                 "• 0 = 单轮（不参考历史）\n"
                 "• 2 = 推荐值\n"
                 "• 6 = 走最远 6 轮",
        )
        use_rewrite = st.checkbox(
            "🔄 Query 改写",
            value=True,
            help="用 LLM 把省略式 query 改写成自包含 query 再检索\n"
                 "例:「那约定过低呢？」 → 「约定的违约金约定过低，能否请求法院增加？」\n"
                 "• 开启 = 多轮对话检索质量大幅提升（推荐）\n"
                 "• 关闭 = 用原 query 直接检索（首轮/简单 follow-up 时可省一次 LLM 调用）",
        )

        st.divider()
        st.markdown("### 📊 数据")
        st.markdown("""
- **民法典** (1260 条) + **刑法** (452 条) + **宪法** (143 条)
- **合计 1855 条**
- 模型：`bge-small-zh-v1.5` (512d)
- 检索：BM25 (jieba) + Dense Cosine → **RRF**
        """)

        # ── 法源过滤（默认全部）
        law_filter = st.multiselect(
            "📚 限定法源",
            options=["全部", "民法典", "刑法", "宪法"],
            default=["全部"],
            help="限定检索的法律来源；多选即跨法检索",
        )
        # 存入 session_state 供 ask() 使用
        st.session_state["law_filter"] = law_filter

        st.divider()
        if st.button("🗑  清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("🔌  Qdrant: `192.168.31.101:6333` / `mfd_law_small`")
        st.caption("🧠  LLM: Anthropic 兼容 · DeepSeek v4-flash")

    # ── Header ──
    st.title("📖 中国法律智能问答")
    st.caption("基于混合检索 + LLM 的法律问答 · 严格基于法条原文 · 覆盖民法典 / 刑法 / 宪法")

    # ── Load resources ──
    bm25, chunks, qclient = load_resources(
        bm25_path="data/pdf/all_laws.articles.bm25.pkl",
        server="http://192.168.31.101:6333",
    )
    embed_model = load_embed_model("BAAI/bge-small-zh-v1.5")

    # ── Chat history ──
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 默认给个示例提示
        with st.chat_message("assistant"):
            st.markdown("你好，我是法律智能问答助手。可查 **民法典 / 刑法 / 宪法** 1855 条法条。试试问我：\n\n"
                        "- **「自然人下落不明多久才能宣告失踪？」（民法典 §40）**\n"
                        "- **「约定的违约金过高是否可以请求法院减少？」（民法典 §585）**\n"
                        "- **「盗窃罪怎么判？」（刑法 §264）**\n"
                        "- **「公民的人身自由受宪法保护吗？」（宪法 §37）**\n"
                        "- **「正当防卫造成损害要赔偿吗？」（民法典 §181 + 刑法 §20）**\n\n"
                        "在左侧开启 **🛠 调试模式** 可以看到召回的法条 + LLM 推理过程。\n\n"
                        "💡 跨法问题会被自动召回多部法律相关条文（如\"正当防卫\"同时召回刑法+民法典）。")

    # ── Render history ──
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and debug and "result" in m:
                render_debug(m["result"])

    # ── Input ──
    if prompt := st.chat_input("请输入你的法律问题 …"):
        # user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # assistant
        with st.chat_message("assistant"):
            with st.spinner("🔍 混合检索 + 🧠 LLM 推理中 …"):
                # 多轮上下文：取最近 history_turns 轮 user/assistant 对
                # 注意 session_state 当前末尾是刚 append 的 user，需要先排除
                prior_msgs = st.session_state.messages[:-1]
                chat_history = prior_msgs[-(history_turns * 2):] if history_turns else []
                result = ask(
                    prompt, k, top_p,
                    bm25, chunks, qclient, embed_model,
                    chat_history=chat_history,
                    use_rewrite=use_rewrite,
                    law_filter=st.session_state.get("law_filter", ["全部"]),
                )
            st.markdown(result["answer"])
            if debug:
                render_debug(result)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "result": result,
        })
        # No st.rerun() - allow current message to render inline naturally


if __name__ == "__main__":
    main()