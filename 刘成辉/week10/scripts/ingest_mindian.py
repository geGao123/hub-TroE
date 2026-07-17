#!/usr/bin/env python3
"""
多法来源 RAG 入库（合并 chunks → Qdrant + BM25）

支持：
  * 单法（民法典 / 宪法 / 刑法）：
        python scripts/ingest_mindian.py --json data/pdf/中华人民共和国民法.articles.json
  * 多法合并：
        python scripts/ingest_mindian.py --merged-json data/pdf/all_laws.articles.json
  * 默认（向后兼容）：
        python scripts/ingest_mindian.py
        # → 读 data/pdf/all_laws.articles.json（如不存在，回退单法路径）

输入 JSON 格式（兼容单法和合并）：
  单法: {chunks: [{article_id, article_index, text, section, ...}], ...}
  合并: {chunks: [{law, law_short, law_slug, article_id, ...}], laws: [...]}

输出：
  * Qdrant collection（每个点带 law / law_short / law_slug payload 字段）
  * 本地 BM25 索引（pickle）

Point ID 方案：
  使用全局顺序 ID（1, 2, 3, …）。article_id 仅在 law 内有效（同一部法律的"第1条"可能多个），
  所以不能用 article_id 做主键。point_id = 0-indexed seq。

  Payload 同时保存 (law_slug, article_id)，需要时可通过 filter 限定某部法律。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

# 在 import qdrant/fastembed 之前设环境变量，避免进度条污染 stdout
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# ──────────────────────────────────────────────────────────────
# 1. 加载 chunks（单法 or 合并）
# ──────────────────────────────────────────────────────────────
def load_chunks_merged(json_path: Path) -> tuple[list[dict], dict]:
    """加载合并的 articles JSON。

    返回 (chunks, src_meta)。
    每个 chunk 标准化字段：
      law, law_short, law_slug, article_id, article_index,
      text, section, section_index, section_kind,
      start_page, end_page, pages, point_id (assigned later)
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # 合并格式：顶层 chunks 数组
    if "chunks" in data and "laws" in data:
        chunks = data["chunks"]
        meta = {
            "format": "merged",
            "laws": data["laws"],
            "total_articles": data.get("total_articles", len(chunks)),
            "total_sections": data.get("total_sections", 0),
        }
    # 单法格式：顶层 sections（嵌套）+ chunks 平铺
    elif "chunks" in data:
        chunks = data["chunks"]
        meta = {
            "format": "single",
            "law": data.get("law"),
            "law_short": data.get("law_short"),
            "law_slug": data.get("law_slug"),
            "total_articles": data.get("total_articles", len(chunks)),
            "total_sections": data.get("total_sections", 0),
        }
    else:
        sys.exit(f"❌ 不识别的 JSON 格式: {json_path}")

    # 标准化 + 分配 point_id
    norm = []
    for i, c in enumerate(chunks):
        norm.append({
            "point_id": i,
            "law": c.get("law") or meta.get("law") or "未知",
            "law_short": c.get("law_short") or meta.get("law_short") or "未知",
            "law_slug": c.get("law_slug") or meta.get("law_slug") or "unk",
            "article_id": c["article_id"],
            "article_index": c["article_index"],
            "text": c["text"],
            "section": c.get("section", ""),
            "section_index": c.get("section_index", ""),
            "section_kind": c.get("section_kind", "节"),
            "start_page": c.get("start_page"),
            "end_page": c.get("end_page"),
            "pages": c.get("pages", []),
        })
    return norm, meta


# 向后兼容：给单法 ingest_mindian.py 老调用用
def load_chunks(json_path: Path) -> list[dict]:
    chunks, _ = load_chunks_merged(json_path)
    return chunks, _


# ──────────────────────────────────────────────────────────────
# 2. BM25 构建
# ──────────────────────────────────────────────────────────────
_CN_PUNC = re.compile(r"[\s,。、;；:：！？\"\“\”()（）\[\]【】<>《》\\.\\+\\-]+")
_TOKEN_CACHE: dict[int, list[str]] = {}


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 去停用词/单字符。"""
    h = hash(text)
    if h in _TOKEN_CACHE:
        return _TOKEN_CACHE[h]
    toks = [t for t in jieba.cut(text) if t.strip()]
    out = [t for t in toks if any('\u4e00' <= ch <= '\u9fff' for ch in t) or t.isalnum()]
    _TOKEN_CACHE[h] = out
    return out


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus_tokens, k1=1.5, b=0.75)


# ──────────────────────────────────────────────────────────────
# 3. Qdrant 入库
# ──────────────────────────────────────────────────────────────
def get_qdrant_client(args) -> "qdrant_client.QdrantClient":
    from qdrant_client import QdrantClient
    if args.server:
        print(f"→ 使用远程 Qdrant: {args.server}")
        return QdrantClient(url=args.server, timeout=60)
    path = Path(args.local).resolve()
    path.mkdir(parents=True, exist_ok=True)
    print(f"→ 使用本地嵌入式 Qdrant: {path}")
    return QdrantClient(path=str(path))


def make_collection(client, name: str, dim: int, recreate: bool):
    from qdrant_client.http import models as qm

    existing = {c.name for c in client.get_collections().collections}
    if recreate and name in existing:
        print(f"  删除旧 collection: {name}")
        client.delete_collection(name)
    if name not in existing or recreate:
        print(f"  创建 collection: {name} (dim={dim})")
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        # ── payload 索引
        client.create_payload_index(name, "law_slug", qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "law_short", qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "article_id", qm.PayloadSchemaType.INTEGER)
        client.create_payload_index(name, "section_kind", qm.PayloadSchemaType.KEYWORD)


def dense_embed(texts: list[str], model_name: str) -> list[list[float]]:
    """fastembed 一次性把全部文本向量化。"""
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=model_name)
    return [v.tolist() for v in model.embed(texts, batch_size=32, parallel=0)]


def dense_embed_batch(texts: list[str], model_name: str, lib: str = "auto") -> list[list[float]]:
    """支持 fastembed / FlagEmbedding 的统一入口。"""
    if lib == "fastembed" or model_name.endswith("bge-small-zh-v1.5"):
        return dense_embed(texts, model_name)
    # 其他走 FlagEmbedding
    from FlagEmbedding import FlagModel
    if "bge-m3" in model_name:
        from FlagEmbedding import BGEM3FlagModel
        m = BGEM3FlagModel(model_name, use_fp16=True)
        out = m.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return out["dense_vecs"].tolist()
    m = FlagModel(model_name, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in m.encode(texts)]


def ingest(client, collection: str, chunks: list[dict], model_name: str, batch: int):
    from qdrant_client.http import models as qm

    total = len(chunks)
    print(f"  生成 dense 向量: {total} 条")
    texts = [c["text"] for c in chunks]
    vectors = dense_embed_batch(texts, model_name)
    print(f"  ✓ {len(vectors)} 个向量已生成，dim={len(vectors[0])}")

    # 分批 upsert，避免单次 HTTP 超过 32MB
    # 每条约 4*dim + payload bytes，1024d ~ 4KB，加上 text/payload ~6-8KB
    # 保守按 300 条/批（HTTP 单次 ~2-3MB）
    SAFE_BATCH = 300
    n_batches = (len(chunks) + SAFE_BATCH - 1) // SAFE_BATCH
    print(f"  upsert {len(chunks)} points → {collection}（分 {n_batches} 批，每批 ≤{SAFE_BATCH}）")

    for i in range(0, len(chunks), SAFE_BATCH):
        batch_chunks = chunks[i:i + SAFE_BATCH]
        batch_vecs = vectors[i:i + SAFE_BATCH]
        points: list[qm.PointStruct] = []
        for vec, c in zip(batch_vecs, batch_chunks):
            pid = int(c["point_id"])
            payload = {
                "law": c["law"],
                "law_short": c["law_short"],
                "law_slug": c["law_slug"],
                "article_id": c["article_id"],
                "article_index": c["article_index"],
                "section": c["section"],
                "section_index": c["section_index"],
                "section_kind": c["section_kind"],
                "text": c["text"],
                "pages": c.get("pages", []),
                "start_page": c.get("start_page"),
                "end_page": c.get("end_page"),
            }
            points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))
        client.upsert(collection_name=collection, points=points, wait=True)
        bi = i // SAFE_BATCH + 1
        print(f"    批 {bi}/{n_batches}: {len(points)} points 已 upsert")


# ──────────────────────────────────────────────────────────────
# 4. 保存 BM25
# ──────────────────────────────────────────────────────────────
def save_bm25(bm25, chunks: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": bm25,
        "chunks": chunks,
    }
    out_path.write_bytes(pickle.dumps(payload))
    print(f"  BM25 索引已保存: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


# ──────────────────────────────────────────────────────────────
# 5. main
# ──────────────────────────────────────────────────────────────
def resolve_input_json(args) -> Path:
    """优先级：--merged-json > --json > 默认 merged > 默认 单法。"""
    if args.merged_json:
        return Path(args.merged_json).resolve()
    if args.json:
        return Path(args.json).resolve()
    # 默认 merged
    default_merged = Path("data/pdf/all_laws.articles.json").resolve()
    if default_merged.exists():
        return default_merged
    # 回退到单法
    return Path("data/pdf/中华人民共和国民法.articles.json").resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-json", default=None, help="合并的 articles JSON 路径")
    ap.add_argument("--json", default=None, help="单法 articles JSON 路径（向后兼容）")
    ap.add_argument("--collection", default="mfd_law_small")
    ap.add_argument("--local", default="./qdrant_local")
    ap.add_argument("--server", default=None, help="Qdrant HTTP URL，覆盖 --local")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--recreate", action="store_true",
                    help="删除已存在的 collection 重建")
    args = ap.parse_args()

    json_path = resolve_input_json(args)
    if not json_path.exists():
        sys.exit(f"❌ 找不到输入: {json_path}")

    print(f"📖 读取条款 JSON: {json_path}")
    chunks, src = load_chunks_merged(json_path)
    print(f"   共 {len(chunks)} 条 chunk, format={src['format']}")
    if src["format"] == "merged":
        print(f"   法律:")
        for law in src["laws"]:
            print(f"     • {law['law_short']:6s} ({law['law_slug']}): "
                  f"{law['total_articles']:4d} 条  {law['total_sections']:3d} 节  "
                  f"{law['article_index_range'][0]} ~ {law['article_index_range'][1]}")
    else:
        print(f"   单法: {src.get('law_short', '?')} ({src.get('law_slug', '?')})")

    # Qdrant
    print("\n📥 Qdrant 入库")
    client = get_qdrant_client(args)
    # 探维度（fastembed 仅支持部分模型，其他靠查表 / 硬编码）
    KNOWN_DIMS = {
        "BAAI/bge-small-zh-v1.5": 512,
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-m3": 1024,
    }
    if args.model in KNOWN_DIMS:
        dim = KNOWN_DIMS[args.model]
    else:
        try:
            from fastembed import TextEmbedding
            dim = TextEmbedding(model_name=args.model).embedding_size
        except Exception as e:
            sys.exit(f"❌ 无法探测模型 {args.model} 的维度（fastembed 不支持）。请加到 KNOWN_DIMS 或换库。\n{e}")
    print(f"   embedding dim = {dim}")
    make_collection(client, args.collection, dim, args.recreate)
    ingest(client, args.collection, chunks, args.model, args.batch)

    # BM25
    print("\n📥 BM25 索引")
    bm25 = build_bm25(chunks)
    bm25_path = json_path.with_suffix(".bm25.pkl")
    save_bm25(bm25, chunks, bm25_path)

    # sanity
    print("\n🔎 入库后核校:")
    info = client.get_collection(args.collection)
    print(f"   Qdrant points: {info.points_count}")
    from qdrant_client.http import models as qm
    sample = client.scroll(args.collection, limit=3, with_payload=True, with_vectors=False)[0]
    for p in sample:
        pl = p.payload
        print(f"   - id={p.id} | {pl.get('law_short','?'):6s} {pl.get('article_index','?'):>8} | {pl.get('section','?')}")

    # 统计每个 law 的入库数
    print("\n📊 按 law 统计入库:")
    by_law: dict[str, int] = {}
    for c in chunks:
        by_law[c["law_short"]] = by_law.get(c["law_short"], 0) + 1
    for k, v in sorted(by_law.items(), key=lambda kv: -kv[1]):
        print(f"   • {k:6s}: {v:4d} 条")

    print("\n✅ 完成")


if __name__ == "__main__":
    main()