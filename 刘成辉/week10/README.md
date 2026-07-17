# ai-ask · 中国法律 RAG 问答系统

> 民法典 / 刑法 / 宪法 条文语义检索 + 问答。基于 BM25 + Dense 双路召回 + RRF 融合 + LLM 答案生成。

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![Qdrant](https://img.shields.io/badge/qdrant-1.x-red)]()
[![License](https://img.shields.io/badge/license-internal-lightgrey)]()

## 这是什么

一个**离线本地可跑**的中文法律 RAG 系统：输入自然语言问题，召回相关法条原文，喂给 LLM 生成带条款引用的回答。

- **数据**：1855 条结构化法律条文（民法典 1260 + 刑法 552 + 宪法 143）
- **检索**：BM25（关键词）+ BGE-M3（语义）+ RRF 倒数排名融合
- **生成**：Claude / DeepSeek 等兼容 Anthropic 协议的 LLM
- **前端**：Streamlit 单页应用，含调试模式（看召回 / Prompt / 三路对比）

## 快速开始

```bash
# 1. 启动 Qdrant
docker compose up -d qdrant

# 2. 装依赖
uv pip install -r requirements.txt
uv pip install FlagEmbedding streamlit anthropic

# 3. 配置 LLM 环境变量
export ANTHROPIC_BASE_URL="https://your-llm-gateway/v1"
export ANTHROPIC_API_KEY="sk-..."
export ANTHROPIC_MODEL="deepseek-v4-flash"   # 可选，默认 deepseek-v4-flash

# 4. （首次）灌库 — 见下方「数据准备」

# 5. 启动 Web UI
streamlit run scripts/web.py --server.port 8501
```

打开 `http://localhost:8501` 即可。

## 数据准备

首次跑需要把原始 PDF 灌成向量库，4 步流水线：

```bash
# 1. PDF → 结构化 JSON
python scripts/pdf_to_json.py data/raw/民法典.pdf data/json/mfd.json

# 2. JSON → 条文级 JSONL（按章切条）
python scripts/pdf_to_articles.py data/json/mfd.json data/articles/mfd.jsonl

# 3. 入 Qdrant + 建 BM25 索引
python scripts/ingest_mindian.py --collection mfd_law_small

# 4. 验证检索
python scripts/search_mindian.py "约定的违约金过高，能否请求法院减少？"
```

完整说明见 **[ARCHITECTURE.md](./ARCHITECTURE.md)**，包括每个脚本的入参出参、关键设计、常见操作。

## 常用命令

| 任务 | 命令 |
|---|---|
| 启动 UI | `streamlit run scripts/web.py --server.port 8501` |
| 跑 benchmark | `python scripts/benchmark_models.py` |
| 重灌某部法律 | `python scripts/ingest_mindian.py --collection <name> --reset` |
| 单独试检索 | `python scripts/search_mindian.py "你的问题"` |
| 看 Qdrant 状态 | `curl http://localhost:6333/collections` |

## 技术栈

| 层 | 选型 |
|---|---|
| **向量库** | Qdrant 1.x（Docker 本地部署） |
| **Embedding** | BAAI/bge-m3（默认，1024d，跨法场景 100% hit@5） |
| **稀疏检索** | rank_bm25（纯 Python，零依赖） |
| **LLM** | 任意 Anthropic 兼容协议（Claude / DeepSeek / 自部署） |
| **Web UI** | Streamlit |
| **PDF 解析** | pdfplumber |
| **包管理** | uv |

为什么是 bge-m3？看 [embedding benchmark 报告](./docs/benchmark_embedding_models_multi_law.md) — 在跨法场景下，small / large 都有召回偏差，m3 dense 自身就 100% 命中。

## 项目结构

```
ai-ask/
├── scripts/                # 8 个核心脚本（pdf_to_json → ingest → search → web）
├── data/                   # raw / json / articles / bm25 四级数据
├── qdrant_local/           # 本地 Qdrant 存储（已 gitignore）
├── docs/                   # benchmark 报告
├── docker-compose.yml      # Qdrant 服务编排
├── requirements.txt        # Python 依赖
├── pyproject.toml          # uv 项目元数据
├── ARCHITECTURE.md         # 详细架构 + 常见操作（必读）
└── README.md               # 本文件
```

详细目录树、每个脚本的作用、数据流水线的图示、6 个关键设计决策的"为什么" — 全在 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 状态

| 项 | 数值 |
|---|---|
| 已灌条文 | 1855（民法典 1260 + 刑法 552 + 宪法 143）|
| Collection | `mfd_law_small`（HNSW + 4 个 payload index） |
| Benchmark | 12 个跨法查询用例（见 `docs/`） |
| 已选 embedding | bge-m3（替换 small / large 前后对比已留档） |

## 路线图

- [ ] **判决案例接入**：`data/判决案例/` 已下载，待结构化入 RAG
- [ ] **诉讼文书样式**：`data/诉讼文书样式/` 同样待接入
- [ ] **query rewrite**：当前 query 改写效果一般，待优化
- [ ] **RAGAS 评估**：人工标注集 + RAGAS 三维度自动评估

## 免责声明

本系统是**检索 + 生成辅助工具**，输出内容**不构成法律意见**。真实场景请以官方发布法律文本为准，并咨询专业律师。
