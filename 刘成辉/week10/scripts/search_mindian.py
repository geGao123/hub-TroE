#!/usr/bin/env python3
"""
民法典混合检索（方案 A）：BM25 + dense 向量 → RRF 融合。

支持多种模式：
  * 默认: --server <url> 或 --local <dir>  → Qdrant client
  * --bm25 <path>                            → BM25 pickle (默认与 json 同名)

用法：
    python scripts/search_mindian.py "自然人下落不明"
    python scripts/search_mindian.py "违约金" --k 5
    python scripts/search_mindian.py --server http://192.168.31.101:6333 "失踪宣告"

输出：
    BM25-only top-k
    Dense-only top-k
    RRF 混合 top-k（含原始两个分数）
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# 复用 ingest 的分词（同一进程内）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_mindian import tokenize  # noqa: E402

import jieba  # noqa: E402


# ──────────────────────────────────────────────────────────────
# 1. 加载 BM25 / Qdrant
# ──────────────────────────────────────────────────────────────
def load_bm25(path: Path):
    if not path.exists():
        sys.exit(f"❌ 找不到 BM25 pickle: {path}")
    payload = pickle.loads(path.read_bytes())
    return payload["bm25"], payload["chunks"]


def get_qdrant_client(args):
    from qdrant_client import QdrantClient
    if args.server:
        return QdrantClient(url=args.server, timeout=60)
    if args.local:
        return QdrantClient(path=str(Path(args.local).resolve()))
    sys.exit("❌ 需要 --server 或 --local")


# ──────────────────────────────────────────────────────────────
# 2. 各路召回
# ──────────────────────────────────────────────────────────────
def search_bm25(bm25, chunks, query: str, k: int) -> list[dict]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    # 取 top-k by score
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"chunk": chunks[i], "score": float(scores[i]), "rank": r + 1}
        for r, i in enumerate(idxs)
    ]


def search_dense(client, collection: str, query: str, model: str, k: int) -> list[dict]:
    """多库适配：fastembed / FlagEmbedding / BGEM3FlagModel"""
    from ingest_mindian import dense_embed_batch
    q_vec = dense_embed_batch([query], model)[0]
    res = client.query_points(
        collection_name=collection,
        query=q_vec,
        limit=k,
        with_payload=True,
    )
    out = []
    for r, pt in enumerate(res.points, start=1):
        out.append({"chunk": dict(pt.payload), "score": float(pt.score), "rank": r})
    return out


# ──────────────────────────────────────────────────────────────
# 3. RRF 融合
# ──────────────────────────────────────────────────────────────
def rrf_merge(bm25_hits, dense_hits, k_rrf: int = 60, top_k: int = 10) -> list[dict]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank), rank 从 1 开始。
    key 用 article_id 字符串。"""
    fused: dict[str, dict] = {}

    def add(hits):
        for h in hits:
            aid = str(h["chunk"].get("article_id"))
            if aid not in fused:
                fused[aid] = {
                    "chunk": h["chunk"],
                    "rrf": 0.0,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "dense_rank": None,
                    "dense_score": None,
                }
            fused[aid]["rrf"] += 1.0 / (k_rrf + h["rank"])
            if "bm25" in str(hits[0].get("_src", "")) if hits else False:
                fused[aid]["bm25_rank"] = h["rank"]
                fused[aid]["bm25_score"] = h["score"]
            else:
                fused[aid]["dense_rank"] = h["rank"]
                fused[aid]["dense_score"] = h["score"]

    # 上面 add 函数写得不清楚，重写清晰的版本：
    fused.clear()
    for h in bm25_hits:
        aid = str(h["chunk"].get("article_id"))
        if aid not in fused:
            fused[aid] = {
                "chunk": h["chunk"], "rrf": 0.0,
                "bm25_rank": None, "bm25_score": None,
                "dense_rank": None, "dense_score": None,
            }
        fused[aid]["rrf"] += 1.0 / (k_rrf + h["rank"])
        fused[aid]["bm25_rank"] = h["rank"]
        fused[aid]["bm25_score"] = h["score"]
    for h in dense_hits:
        aid = str(h["chunk"].get("article_id"))
        if aid not in fused:
            fused[aid] = {
                "chunk": h["chunk"], "rrf": 0.0,
                "bm25_rank": None, "bm25_score": None,
                "dense_rank": None, "dense_score": None,
            }
        fused[aid]["rrf"] += 1.0 / (k_rrf + h["rank"])
        fused[aid]["dense_rank"] = h["rank"]
        fused[aid]["dense_score"] = h["score"]

    items = list(fused.values())
    items.sort(key=lambda x: x["rrf"], reverse=True)
    for i, it in enumerate(items, start=1):
        it["rrf_rank"] = i
    return items[:top_k]


# ──────────────────────────────────────────────────────────────
# 4. 打印
# ──────────────────────────────────────────────────────────────
def _fmt_hit(hit, include_text=True, max_text=80) -> str:
    c = hit["chunk"]
    law = c.get("law_short", "")
    law_tag = f"[{law}] " if law else ""
    head = f"{law_tag}[{c.get('article_index','?'):>8}] {c.get('section','?')}"
    if include_text:
        text = c.get("text", "").replace("\n", " ")
        snippet = text[:max_text] + ("…" if len(text) > max_text else "")
        head += f"\n    {snippet}"
    return head


def print_results(query: str, bm25_hits, dense_hits, fused, k: int):
    print("=" * 78)
    print(f"❓ Query: {query}\n")

    print(f"📕 BM25-only top-{k}:")
    if not bm25_hits:
        print("    (无结果)")
    for h in bm25_hits:
        print(f"  rank {h['rank']:<2}  score={h['score']:.4f}  {_fmt_hit(h, include_text=False)}")

    print(f"\n🔍 Dense-only top-{k}:")
    if not dense_hits:
        print("    (无结果)")
    for h in dense_hits:
        print(f"  rank {h['rank']:<2}  cosine={h['score']:.4f}  {_fmt_hit(h, include_text=False)}")

    print(f"\n🔀 RRF 混合 top-{k}:")
    for h in fused:
        bm_str = f"BM25#{h['bm25_rank']}" if h['bm25_rank'] else "BM25✗"
        dv_str = f"D#{h['dense_rank']}" if h['dense_rank'] else "D✗"
        c = h["chunk"]
        print(f"  rrf#{h['rrf_rank']:<2} ({bm_str} {dv_str}) score={h['rrf']:.4f}")
        print(f"      [{c.get('article_index','?')}] {c.get('section','?')}")
        text = c.get("text", "").replace("\n", " ")
        print(f"      {text[:120]}{'…' if len(text) > 120 else ''}")
    print("=" * 78)


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="查询文本（中文自然语言）")
    ap.add_argument("--server", default="http://192.168.31.101:6333")
    ap.add_argument("--local", default=None)
    ap.add_argument("--collection", default="mfd_law_small")
    ap.add_argument("--bm25", default="data/pdf/all_laws.articles.bm25.pkl")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # 加载 BM25 + Qdrant client
    bm25_path = Path(args.bm25).resolve()
    if not bm25_path.exists():
        # 尝试与默认 json 同目录找
        cand = Path("data/pdf/all_laws.articles.bm25.pkl").resolve()
        if cand.exists():
            bm25_path = cand
        else:
            sys.exit(f"❌ BM25 文件不存在: {args.bm25}")
    bm25, chunks = load_bm25(bm25_path)
    print(f"📖 BM25: {len(chunks)} 条 (from {bm25_path})\n")

    client = get_qdrant_client(args)
    print(f"📖 Qdrant collection: {args.collection}\n")

    # 各路召回（多取一些做 RRF）
    N = max(args.k * 2, 20)
    bm25_hits = search_bm25(bm25, chunks, args.query, N)
    dense_hits = search_dense(client, args.collection, args.query, args.model, N)

    fused = rrf_merge(bm25_hits, dense_hits, k_rrf=60, top_k=args.k)
    print_results(args.query, bm25_hits[:args.k], dense_hits[:args.k], fused, args.k)


if __name__ == "__main__":
    main()