# ai-ask 项目架构

> **项目定位**：基于 RAG（Retrieval-Augmented Generation）的中国法律智能问答系统
> **数据范围**：民法典（1260 条）+ 刑法（452 条）+ 宪法（143 条）= **1855 条法条**
> **核心技术**：BM25 + 向量检索混合 + LLM 生成 + Streamlit 交互前端

---

## 一、项目目录结构

```
ai-ask/
├── 📁 data/                          # 原始数据 + 中间产物
│   ├── 📁 pdf/                       # 法律 PDF + 解析缓存
│   │   ├── 中华人民共和国民法.pdf      # 民法典原文 PDF
│   │   ├── 中华人民共和国宪法.pdf      # 宪法原文 PDF
│   │   ├── 中华人民共和国刑法.pdf      # 刑法原文 PDF
│   │   ├── *.json                    # 各 PDF 解析后的 page-level JSON
│   │   ├── *.articles.json           # 各法律切条后的 articles JSON
│   │   ├── all_laws.articles.json    # 合并的 3 部法律 articles（总 1855 条）
│   │   └── all_laws.articles.bm25.pkl # 合并的 BM25 索引
│   ├── 判决案例/                      # DOCX 判决案例（预留，未启用）
│   └── 诉讼文书样式/                   # 诉讼文书样式（爬虫产物，未启用）
│
├── 📁 scripts/                       # 所有可执行脚本（核心代码）
│   ├── pdf_to_json.py                # ① PDF → page-level JSON
│   ├── pdf_to_articles.py            # ② page JSON → 节→条 结构
│   ├── ingest_mindian.py             # ③ articles → Qdrant + BM25 入库
│   ├── search_mindian.py             # ④ 单次混合检索 CLI
│   ├── rag_answer.py                 # ⑤ 端到端问答（CLI 版）
│   ├── web.py                        # ⑥ Streamlit 前端
│   ├── benchmark_models.py           # 多 embedding 模型对比 benchmark
│   └── crawl_susongyangshi.py        # 爬虫：抓诉讼文书样式（备用）
│
├── 📁 docs/                          # 评估报告 / 文档
│   ├── benchmark_embedding_models.md          # 单法（1260 条）三路 benchmark
│   └── benchmark_embedding_models_multi_law.md # 多法（1855 条）三路 benchmark
│
├── 📁 qdrant_local/                  # 本地嵌入式 Qdrant 存储（备用）
│
├── docker-compose.yml                # Qdrant 服务编排
├── requirements.txt                   # Python 依赖（实际用 uv 管理）
├── pyproject.toml                    # 项目元数据（半空）
├── .python-version                   # Python 版本（3.12）
├── .gitignore
├── (无 main.py)                      # 入口是 scripts/web.py
├── README.md                         # 空
└── 合同法律RAG问答系统_实施方案.md    # 早期方案设计文档
```

---

## 二、数据流水线（4 步）

```
PDF 文件
   │  pdf_to_json.py
   ↓
page-level JSON
   │  pdf_to_articles.py
   ↓
articles JSON（按"第X条"切块，带 law 标签）
   │  ingest_mindian.py
   ↓
┌─────────────┐    ┌──────────────────┐
│ Qdrant 向量库 │    │ 本地 BM25 pickle │
└─────────────┘    └──────────────────┘
   │                      │
   └──── search_dense() ─┴── search_bm25() ──→ RRF 融合
                                          │
                                          ↓
                                   rag_answer.py
                                          │  拼 prompt
                                          ↓
                                  Anthropic 兼容 LLM
                                  （DeepSeek v4-flash）
                                          ↓
                                     Streamlit UI
```

---

## 三、核心文件详解

### 3.1 `scripts/pdf_to_json.py` — PDF → page-level JSON

**作用**：把 PDF 解析为每页的 JSON（文本、页码、表格、图片数等元数据）。

**用法**：
```bash
# 单文件
python scripts/pdf_to_json.py data/pdf/中华人民共和国宪法.pdf

# 批量（目录下所有 *.pdf）
python scripts/pdf_to_json.py --all --dir data/pdf
```

**输出**：`<pdf_basename>.json`（如 `中华人民共和国宪法.json`），结构：
```json
{
  "filename": "中华人民共和国宪法.pdf",
  "total_pages": 40,
  "pages": [
    {
      "page_number": 6,
      "char_count": 18537,
      "tables": [...],
      "text": "..."
    }
  ]
}
```

**依赖**：`pdfplumber`

---

### 3.2 `scripts/pdf_to_articles.py` — page JSON → 节→条 结构

**作用**：从 page-level JSON 提取"节/章 → 条"结构，每条法条一个 chunk。

**用法**：
```bash
# 单文件
python scripts/pdf_to_articles.py data/pdf/中华人民共和国宪法.json

# 批量（处理目录下所有 page JSON + 合并输出）
python scripts/pdf_to_articles.py --all --dir data/pdf
```

**输出**：
- 每部法律一个：`<law>.articles.json`（含 `law/law_short/law_slug` 字段）
- 合并版：`all_laws.articles.json`（3 部法律 chunks 数组拼接）

**关键特性**：
- 自动识别法律（文件名 → law 元数据）
- **dedup 去重**：修正刑法 PDF 双份重复（修正案合集形态）
- 中文数字 → int（第X条 → article_id）

**输出 schema**（每个 chunk）：
```json
{
  "law": "中华人民共和国民法典",
  "law_short": "民法典",
  "law_slug": "mfd",
  "article_id": 585,
  "article_index": "第五百八十五条",
  "text": "当事人可以约定...",
  "section": "第八章 违约责任",
  "section_index": "第八章",
  "section_kind": "章",
  "start_page": 76,
  "end_page": 76,
  "pages": [76]
}
```

**法律识别规则**：

| 文件名前缀 | law_full | law_short | law_slug |
|---|---|---|---|
| 中华人民共和国民法 | 中华人民共和国民法典 | 民法典 | mfd |
| 中华人民共和国宪法 | 中华人民共和国宪法 | 宪法 | xfa |
| 中华人民共和国刑法 | 中华人民共和国刑法 | 刑法 | xf |

---

### 3.3 `scripts/ingest_mindian.py` — articles → Qdrant + BM25

**作用**：把 articles JSON 灌进 Qdrant（dense 向量）+ 生成 BM25 pickle（本地）。

**用法**：
```bash
# 默认：读 all_laws.articles.json → mfd_law_small（bge-small-zh-v1.5）
python scripts/ingest_mindian.py \
  --server http://192.168.31.101:6333

# 切换模型 / collection
python scripts/ingest_mindian.py \
  --collection mfd_law_large \
  --model BAAI/bge-large-zh-v1.5 \
  --server http://192.168.31.101:6333

# bge-m3
python scripts/ingest_mindian.py \
  --collection mfd_law_m3 \
  --model BAAI/bge-m3 \
  --server http://192.168.31.101:6333
```

**支持的模型 + 库**：

| 模型 | dim | 库 |
|---|---|---|
| BAAI/bge-small-zh-v1.5 | 512 | fastembed |
| BAAI/bge-large-zh-v1.5 | 1024 | FlagEmbedding |
| BAAI/bge-m3 | 1024 | FlagEmbedding (BGEM3FlagModel) |

**关键技术**：
- **batch upsert**（每批 ≤300 points，避开 Qdrant HTTP 32MB 限制）
- **point_id 用全局 seq**（不用 article_id，因为多法下会冲突）
- payload 索引：`law_slug` / `law_short` / `article_id` / `section_kind`

---

### 3.4 `scripts/search_mindian.py` — 混合检索 CLI

**作用**：单次查询的混合检索（BM25 + Dense → RRF）。

**用法**：
```bash
python scripts/search_mindian.py "盗窃罪" --k 5
python scripts/search_mindian.py "违约金 减少" --collection mfd_law_large --k 5
python scripts/search_mindian.py "正当防卫" --server http://192.168.31.101:6333
```

**输出**：BM25-only / Dense-only / RRF 混合 三路 top-k。

---

### 3.5 `scripts/rag_answer.py` — 端到端问答（CLI 版）

**作用**：检索 + 拼 prompt + 调 LLM（命令行版）。

**用法**：
```bash
python scripts/rag_answer.py "约定的违约金过高，能否请求法院减少？"
python scripts/rag_answer.py "盗窃罪怎么判？" --collection mfd_law_large --k 5
```

**LLM 调用**：Anthropic SDK，base_url 走环境变量 `ANTHROPIC_BASE_URL`（默认指向 DeepSeek 兼容端点）。

---

### 3.6 `scripts/web.py` — Streamlit 前端（主入口）

**作用**：浏览器问答界面 + 调试面板。

**启动**：
```bash
streamlit run scripts/web.py --server.port 8501
# 浏览器访问 http://localhost:8501
```

**核心功能**：
- Chat 风格对话
- 侧边栏：调试模式 / top-k / top_p / 多轮上下文 / **法源过滤** / Query 改写
- 调试面板（7 个 tab）：
  - 📊 召回综述（含改写对比）
  - 📚 召回条文（按法源标签）
  - ⚖️ 三路对比（BM25 / Dense / RRF）
  - 📝 Prompt
  - 🧠 LLM 推理
  - 📦 原始 Payload
  - 💬 上下文

---

### 3.7 `scripts/benchmark_models.py` — 多 embedding 模型对比

**作用**：对比 bge-small-zh / bge-large-zh / bge-m3 在 12 个测试用例上的 hit@K / MRR / 延迟。

**用法**：
```bash
# 全跑（首次：入库 + benchmark）
python scripts/benchmark_models.py

# 跳过入库（已入库时）
python scripts/benchmark_models.py --skip-ingest

# 自定义输出
python scripts/benchmark_models.py --skip-ingest --out docs/benchmark_v3.md
```

**输出**：`docs/benchmark_*.md` + `docs/benchmark_*.json`

**当前测试用例（12 条）**：
- 民法典 5 条（基础 / 复合 / 否定 / 长句 / 口语）
- 刑法 3 条（盗窃 / 故意杀人 / 正当防卫）
- 宪法 2 条（公民权利 / 人身自由）
- 跨法 2 条（正当防卫 / 人身自由）

---

### 3.8 `scripts/crawl_susongyangshi.py` — 备用爬虫

**作用**：从最高法官网抓诉讼文书样式 → Markdown。

**状态**：✅ 可用，但当前未集成进 RAG pipeline（数据未入库）。

---

## 四、辅助文件

### `data/数据处理.md` — 数据流图（早期手绘版本）
简化的 pipeline 流程图，新数据版本（多法）请看本文件。

### `合同法律RAG问答系统_实施方案.md` — 早期方案设计
项目初期的整体方案、技术选型、数据源计划（覆盖范围比当前实际更广）。

### `docker-compose.yml` — Qdrant 服务
```bash
# 启动 Qdrant 服务（如未运行）
docker compose up -d qdrant
```

### `requirements.txt` / `pyproject.toml`
依赖声明。实际安装用 `uv`：
```bash
uv pip install -r requirements.txt
uv pip install FlagEmbedding  # bge-large / bge-m3 需要
```

### `qdrant_local/`
本地嵌入式 Qdrant 存储目录（备用，平时用远程 Qdrant）。

---

## 五、常见操作

### 5.1 第一次跑通（从零开始）

```bash
# 1. 启动 Qdrant
docker compose up -d qdrant

# 2. 装依赖
uv pip install -r requirements.txt
uv pip install FlagEmbedding

# 3. 解析 PDF → JSON
python scripts/pdf_to_json.py --all --dir data/pdf

# 4. 切条 → articles
python scripts/pdf_to_articles.py --all --dir data/pdf

# 5. 入库（默认走 bge-small）
python scripts/ingest_mindian.py --server http://192.168.31.101:6333

# 6. 启动 web
streamlit run scripts/web.py
```

### 5.2 添加新法律 PDF

1. 把 PDF 放进 `data/pdf/`（文件名按 `中华人民共和国<法律名>.pdf` 格式）
2. 如果法律不在识别表里，在 `scripts/pdf_to_articles.py` 的 `LAW_REGISTRY` 加一行
3. 重跑步骤 3-5

### 5.3 切换 embedding 模型

```bash
# 重新入库到不同 collection（保留旧 collection 可回滚）
python scripts/ingest_mindian.py \
  --collection mfd_law_m3 \
  --model BAAI/bge-m3 \
  --server http://192.168.31.101:6333

# 改 scripts/web.py 的 collection 名 + embed_model
```

### 5.4 添加新法源到 benchmark

编辑 `scripts/benchmark_models.py` 的 `TEST_CASES` 列表，加：
```python
{
    "id": "Q13",
    "query": "...",
    "expected": [("law_slug", article_id)],
    "category": "...",
    "note": "...",
}
```

### 5.5 调试召回

```bash
# 单条查询看三路分数
python scripts/search_mindian.py "你的问题" --k 10
```

或开 `streamlit run scripts/web.py` 看完整调试面板（含改写 prompt + LLM 推理）。

---

## 六、关键设计决策

### 6.1 多法律融合：合并 chunk 到单 collection

- **不分开 collection**（如 `mfd_civil_law` / `mfd_criminal_law`）
- **用 payload 字段 `law_slug` 区分**
- 优势：跨法查询天然支持，单次 RRF 召回混合法条

### 6.2 point_id 用全局 seq，不用 article_id

- 民法典 §1 和刑法 §1 都存在，article_id 冲突
- 用 0-indexed 全局序号作为 Qdrant point_id
- 通过 payload 字段 `(law_slug, article_id)` 联合定位

### 6.3 三路 embedding 模型并列

- `mfd_law_small` (512d, 0.09 GB) ← 默认
- `mfd_law_large` (1024d, 1.3 GB)
- `mfd_law_m3` (1024d, 2.7 GB)
- 3 个同时存在，可切换、可对比

### 6.4 BM25 + Dense + RRF 三步混合

- 单 BM25：精确术语召回强，但语义召回弱
- 单 Dense：语义召回强，但精确术语差
- RRF 融合：两个互补

### 6.5 多轮对话增强

- **Query 改写**：用 LLM 把省略式 query 改写成自包含 query 再检索
- **多轮上下文**：把最近 N 轮 user/assistant 对喂给 LLM
- 检索只用最新 query（不被历史污染），指代消解靠 LLM

---

## 七、依赖清单

```
# requirements.txt
qdrant-client
fastembed       # bge-small-zh
rank_bm25
pdfplumber
python-docx
pandas
openpyxl

# 额外需要
FlagEmbedding   # bge-large-zh + bge-m3
streamlit       # web UI
anthropic       # LLM 调用（兼容 DeepSeek）
jieba           # 中文分词
```

---

## 八、版本与规模快照

| 项 | 当前 |
|---|---|
| 数据 | 3 部法律，**1855 条**（民法典 1260 + 刑法 452 + 宪法 143） |
| 节/章 | **157 节** |
| BM25 索引 | `all_laws.articles.bm25.pkl` (~1.6 MB) |
| Qdrant collections | 3 个（small/large/m3），各 1855 points |
| 测试用例 | 12 条（覆盖 3 部法律 + 跨法） |
| Python | 3.12 |
| Qdrant | 远程 `192.168.31.101:6333`（也可本地嵌入式） |