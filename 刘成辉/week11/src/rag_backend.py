"""
rag_backend.py — A股年报 RAG 检索后端（三种方式共享的业务逻辑）

教学重点：
  1. 这一层是"纯业务逻辑"，不感知被哪种方式调用（Function Call / MCP / CLI 都复用它）
  2. FAISS 索引在 import 时加载一次（轻量、必须），Embedding 模型懒加载
     —— 查天气时根本不会加载 bge-m3（省 ~18 秒），第一次 RAG 检索时才加载
  3. L2 归一化 + IndexFlatIP 内积 = 余弦相似度
  4. 元数据过滤（stock_code / year）在检索后做，过滤条件越多越大搜回数
  5. 本地 Embedding：使用 sentence-transformers 加载 BAAI/bge-m3，无需联网/API key

使用方式（作为模块）：
  from src.rag_backend import search_annual_report, list_companies
  print(search_annual_report("宁德时代2023年营收", stock_code="300750", year="2023", top_k=3))

依赖：
  pip install faiss-cpu numpy sentence-transformers
  向量数据位于 vectorstore/（运行 scripts/copy_data.py 复制）
  Embedding 模型位于 /Volumes/timi/embedding/bge-m3（首次 RAG 检索时才加载）

知识库说明：
  公司（stock_code）：贵州茅台(600519) / 五粮液(000858) / 宁德时代(300750)
                      海康威视(002415) / 中国平安(601318)
  年份：2021 / 2022 / 2023
  规模：15 份年报，共 10353 个语义分块

Embedding 切换说明：
  原：DashScope text-embedding-v3 (1024 维) + OpenAI 兼容接口
  现：本地 BAAI/bge-m3 (1024 维) + sentence-transformers
  维度一致，FAISS 索引可直接复用，无需重建。

懒加载说明：
  之前：模块级 import 时立即加载 bge-m3，导致 fincli weather / fincli list-companies
        都会白白等 18 秒模型加载
  现在：只在第一次调用 search_annual_report 时才加载 bge-m3，list_companies / 纯
        weather 场景不付出模型加载成本
"""

import json
import os
import sys
from pathlib import Path

# Windows 上 torch 与 numpy 各自链接 OpenMP 会冲突，必须打开此开关
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# ── 常量 ──────────────────────────────────────────────────────────────────

# 本地 Embedding 模型路径（BAAI/bge-m3，1024 维，与原 text-embedding-v3 维度一致）
EMBED_MODEL_DIR = os.environ.get("EMBED_MODEL_DIR", "/Volumes/timi/embedding/bge-m3")
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
# bge-m3 推荐 query 前缀，但 sentence-transformers 加载 bge-m3 时默认不加，
# 我们也不加，跟原索引保持一致（索引是用同样的方式构建的）

# 用 __file__ 定位项目根目录，无论从哪个工作目录启动都能找到 vectorstore/
BASE_DIR = Path(__file__).parent.parent
FAISS_INDEX_PATH = BASE_DIR / "vectorstore" / "faiss_index.bin"
FAISS_META_PATH = BASE_DIR / "vectorstore" / "faiss_meta.json"

# 公司信息表（用于 list_companies 和参数说明）
COMPANIES = [
    {"name": "贵州茅台", "stock_code": "600519", "years": ["2021", "2022", "2023"]},
    {"name": "五粮液",   "stock_code": "000858", "years": ["2021", "2022", "2023"]},
    {"name": "宁德时代", "stock_code": "300750", "years": ["2021", "2022", "2023"]},
    {"name": "海康威视", "stock_code": "002415", "years": ["2021", "2022", "2023"]},
    {"name": "中国平安", "stock_code": "601318", "years": ["2021", "2022", "2023"]},
]

# ── 初始化（模块导入时执行一次）────────────────────────────────────────────

if not Path(EMBED_MODEL_DIR).exists():
    print(f"错误：本地 Embedding 模型不存在：{EMBED_MODEL_DIR}", file=sys.stderr)
    print(f"  请将模型放在该路径，或通过环境变量 EMBED_MODEL_DIR 覆盖", file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("错误：未安装 sentence-transformers，请运行 pip install sentence-transformers", file=sys.stderr)
    sys.exit(1)

# bge-m3 模型懒加载：模块级只占位，第一次调 get_embedding() 时才实例化
# 这样 fincli weather / fincli list-companies 等不需要 RAG 的场景
# 不会白白等 ~18 秒模型加载
_embed_model = None
_embed_device: str | None = None

try:
    import faiss
except ImportError:
    print("错误：未安装 faiss-cpu，请运行 pip install faiss-cpu", file=sys.stderr)
    sys.exit(1)

if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
    print(f"错误：向量索引文件不存在，请先运行 scripts/copy_data.py", file=sys.stderr)
    print(f"  期望路径：{FAISS_INDEX_PATH}", file=sys.stderr)
    sys.exit(1)

_index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
    _meta_list: list[dict] = json.load(f)

# 维度一致性校验：如果索引维度和模型维度不一致，要立刻报错，不能静默错下去
if _index.d != EMBED_DIM:
    print(
        f"错误：FAISS 索引维度 ({_index.d}) 与 Embedding 模型维度 ({EMBED_DIM}) 不一致，"
        f"请重建索引（运行 scripts/build_index.py）",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"[rag_backend] 就绪：{_index.ntotal} 个向量，{len(_meta_list)} 条元数据",
    file=sys.stderr,
)


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def _ensure_embed_model():
    """
    懒加载 bge-m3 模型：第一次调用时实例化并缓存到模块全局变量。
    进程内后续调用直接复用，单次加载约 18 秒（MPS）/ 更久（CPU）。

    教学要点：Python 模块级副作用是双刃剑。本模块改成"重资源懒加载"
    是因为同时被 fincli weather / fincli list-companies 复用，
    后两者根本不需要 bge-m3，提前加载是浪费。
    MCP 模式天然没这问题（子进程隔离），Function Call / CLI 受益于此改动。
    """
    global _embed_model, _embed_device
    if _embed_model is not None:
        return _embed_model

    # device 选择也在懒加载里：MPS/CUDA 检测只在真用 RAG 时才触发
    import torch
    if torch.backends.mps.is_available():
        _embed_device = "mps"
    elif torch.cuda.is_available():
        _embed_device = "cuda"
    else:
        _embed_device = "cpu"

    _embed_model = SentenceTransformer(EMBED_MODEL_DIR, device=_embed_device)
    print(
        f"[rag_backend] Embedding 模型已加载：{EMBED_MODEL_NAME} "
        f"(device={_embed_device}, dim={_embed_model.get_sentence_embedding_dimension()})",
        file=sys.stderr,
    )
    return _embed_model


def get_embedding(text: str) -> np.ndarray:
    """
    调用本地 bge-m3 模型做 Embedding，返回 L2 归一化后的 float32 向量。
    FAISS 使用 IndexFlatIP（内积），预先 L2 归一化后内积等价于余弦相似度。
    bge-m3 默认输出已归一化，这里再保险 normalize_embeddings=True 一次。
    第一次调用会触发模型懒加载（~18 秒 MPS），后续直接复用。
    """
    model = _ensure_embed_model()
    vec = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float32)


# ── 对外接口 ──────────────────────────────────────────────────────────────

def search_annual_report(
    query: str,
    stock_code: str | None = None,
    year: str | None = None,
    top_k: int = 5,
) -> str:
    """
    在A股年报语料库中检索与问题最相关的段落。

    Args:
        query:      检索问题，自然语言，例如 "宁德时代2023年营收和净利润"
        stock_code: 可选，按公司过滤。600519(茅台)/000858(五粮液)/
                    300750(宁德时代)/002415(海康威视)/601318(中国平安)
        year:       可选，按年份过滤。"2021" / "2022" / "2023"
        top_k:      返回段落数，默认5，建议不超过10

    Returns:
        按相关度排序的段落列表，每段含来源（公司、年份、章节、页码）
    """
    try:
        query_vec = get_embedding(query)
    except Exception as e:
        return f"Embedding 调用失败：{e}"

    # 有过滤条件时多搜几倍，再过滤；无过滤时搜略多一点
    search_k = min(top_k * 10 if (stock_code or year) else top_k * 3, _index.ntotal)
    distances, indices = _index.search(query_vec.reshape(1, -1), search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_meta_list):
            continue
        meta = _meta_list[idx]
        if stock_code and meta.get("stock_code") != stock_code:
            continue
        if year and str(meta.get("year")) != str(year):
            continue
        results.append({
            "score": float(dist),
            "content": meta.get("content", ""),
            "stock_code": meta.get("stock_code", ""),
            "year": str(meta.get("year", "")),
            "section": meta.get("section", ""),
            "page_num": meta.get("page_num", ""),
        })
        if len(results) >= top_k:
            break

    if not results:
        filter_parts = []
        if stock_code:
            filter_parts.append(f"股票代码={stock_code}")
        if year:
            filter_parts.append(f"年份={year}")
        filter_str = f"（过滤条件：{', '.join(filter_parts)}）" if filter_parts else ""
        return f"未找到相关内容{filter_str}，请尝试换一种问法或去掉过滤条件"

    lines = [f"检索到 {len(results)} 条相关段落：\n"]
    for i, r in enumerate(results, 1):
        company_name = next(
            (c["name"] for c in COMPANIES if c["stock_code"] == r["stock_code"]),
            r["stock_code"],
        )
        lines.append(
            f"【{i}】{company_name}（{r['stock_code']}）{r['year']}年报"
            f" | 第{r['page_num']}页 | 相关度：{r['score']:.3f}"
        )
        lines.append(f"章节：{r['section']}")
        lines.append(r["content"])
        lines.append("")

    return "\n".join(lines)


def list_companies() -> str:
    """
    列出年报知识库中包含的所有公司及可查询的年份范围。

    Returns:
        公司列表，含名称、股票代码、可查年份
    """
    lines = ["年报知识库收录公司列表：\n"]
    for c in COMPANIES:
        years_str = " / ".join(c["years"])
        lines.append(f"  {c['name']}  股票代码：{c['stock_code']}  年份：{years_str}")
    lines.append("\n共 5 家公司，每家 3 年，合计 15 份年报")
    return "\n".join(lines)


if __name__ == "__main__":
    # 自检：直接运行看检索结果
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("search")
    p1.add_argument("--query", required=True)
    p1.add_argument("--stock-code", default=None)
    p1.add_argument("--year", default=None)
    p1.add_argument("--top-k", type=int, default=5)
    sub.add_parser("list-companies")
    args = parser.parse_args()

    if args.cmd == "search":
        print(search_annual_report(args.query, args.stock_code, args.year, args.top_k))
    else:
        print(list_companies())
