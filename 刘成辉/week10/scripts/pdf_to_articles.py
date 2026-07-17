#!/usr/bin/env python3
"""
把 PDF 解析后的 page-level JSON 切成"节→条"结构。

支持多部法律：
  * 单文件: pdf_to_articles.py <page_json>
  * 批量:   pdf_to_articles.py --all [--dir data/pdf]
            → 每个 PDF 输出 <pdf>.articles.json
            → 同时输出合并的 all_laws.articles.json（含 law 字段）

法律识别（文件名 → law 元数据）：
  中华人民共和国民法*  → "中华人民共和国民法典" / "民法典" / "mfd"
  中华人民共和国宪法*  → "中华人民共和国宪法" / "宪法" / "xfa"
  中华人民共和国刑法*  → "中华人民共和国刑法" / "刑法" / "xf"

每个 chunk 新增字段：
  law        — 法律全称（如 "中华人民共和国民法典"）
  law_short  — 简称（如 "民法典"）
  law_slug   — ASCII 缩写（用于 ID、collection 命名等）

规则：
- 节/章标题: "第X节  标题..." / "第X章  标题..."  作为分组节点
- 条文:   "第X条..."  切一条，跨行正文合并
- 页码 metadata: start_page / end_page / pages[]
- 空数据（只标题/节标题无条文）丢弃
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CN = "零一二三四五六七八九十百千〇○两"
SECTION_RE = re.compile(rf"^\s*(第[{CN}]+(?:章|节))(?:[\s　]+(.+?))?\s*$")
ARTICLE_RE = re.compile(rf"^\s*(第[{CN}]+条)(?:[\s　]+(.*))?$")

NOISE_PATTERNS = [
    re.compile(r"^\s*8/23/2020\b"),
    re.compile(r"^\s*www\.npc\.gov\.cn"),
    re.compile(r"^\s*\d+/\d+\s*$"),
    re.compile(r"^\s*当前位置[:：]"),
    re.compile(r"^\s*来源[:：]"),
    re.compile(r"^\s*浏览字号[:：]"),
]


def is_noise(line: str) -> bool:
    if not line.strip():
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


DIGITS = {
    "零": 0, "〇": 0, "○": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}


def cn_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    total = 0
    cur = 0
    for ch in s:
        if ch in DIGITS:
            cur = DIGITS[ch]
        elif ch == "千":
            total += (cur or 1) * 1000
            cur = 0
        elif ch == "百":
            total += (cur or 1) * 100
            cur = 0
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        else:
            return -1
    return total + cur


# ──────────────────────────────────────────────────────────────
# 法律识别
# ──────────────────────────────────────────────────────────────
LAW_REGISTRY: list[tuple[str, tuple[str, str, str]]] = [
    # (filename 前缀, (law_full, law_short, law_slug))
    ("中华人民共和国民法", ("中华人民共和国民法典", "民法典", "mfd")),
    ("中华人民共和国宪法", ("中华人民共和国宪法", "宪法", "xfa")),
    ("中华人民共和国刑法", ("中华人民共和国刑法", "刑法", "xf")),
]


def detect_law(filename: str) -> tuple[str, str, str]:
    """从 page-level JSON 的 filename 字段识别法律。

    Returns:
        (law_full, law_short, law_slug)
    """
    stem = Path(filename).stem  # e.g. "中华人民共和国民法"
    for prefix, (full, short, slug) in LAW_REGISTRY:
        if stem == prefix or stem.startswith(prefix):
            return full, short, slug
    # fallback
    return stem, stem, stem[:3].lower()


# ──────────────────────────────────────────────────────────────
# 核心 parser（接受 page dict list + law 元数据）
# ──────────────────────────────────────────────────────────────
def parse_articles(
    pages: list[dict],
    law_full: str,
    law_short: str,
    law_slug: str,
    source_pdf: str = "",
    source_json: str = "",
    extracted_at: str = "",
    filename: str = "",
    dedup: bool = True,
) -> dict:
    """从 page-level JSON 切出 sections + chunks。

    Returns dict with keys: filename, law, law_short, law_slug,
                            source_pdf, source_json, extracted_at,
                            total_sections, total_articles,
                            article_index_range, sections, chunks

    dedup=True 时，同一 (law, article_id) 重复出现只保留第一份
    （用于"修正案合集"型 PDF：第一份是合并后的全文，后面是历次修正案决定），
    同时 sections 也会按这个原则去重。
    """
    raw_lines: list[tuple[str, int]] = []
    for p in pages:
        pn = p["page_number"]
        for line in (p["text"] or "").splitlines():
            if not is_noise(line):
                raw_lines.append((line, pn))

    chunks: list[dict] = []
    sections_index: dict[str, dict] = {}
    section_order: list[str] = []

    started = False
    cur_section: dict | None = None
    cur_article: dict | None = None

    def flush_article():
        nonlocal cur_article
        if not cur_article:
            return
        text = re.sub(r"\s+", "", cur_article["text_buf"])
        cur_article["text"] = text.strip()
        pages_set = sorted(set(cur_article["pages"]))
        cur_article["pages"] = pages_set
        cur_article["start_page"] = pages_set[0]
        cur_article["end_page"] = pages_set[-1]
        del cur_article["text_buf"]
        if not cur_article["text"]:
            cur_article = None
            return
        if not cur_section:
            cur_article = None
            return
        # ── 加 law 元数据
        cur_article["law"] = law_full
        cur_article["law_short"] = law_short
        cur_article["law_slug"] = law_slug
        cur_article["section"] = cur_section["section_full"]
        cur_article["section_index"] = cur_section["section_index"]
        cur_article["section_kind"] = cur_section["section_kind"]
        cur_section["articles"].append(cur_article)
        cur_article = None

    def new_article(line: str, page_no: int):
        nonlocal cur_article
        flush_article()
        m = ARTICLE_RE.match(line)
        title = m.group(1)
        rest = (m.group(2) or "").strip()
        cur_article = {
            "article_index": title,
            "article_id": cn_to_int(title.replace("第", "").replace("条", "")),
            "text_buf": rest,
            "pages": [page_no],
        }

    def new_section(line: str, page_no: int):
        nonlocal cur_section
        flush_article()
        m = SECTION_RE.match(line)
        title = m.group(1)
        kind = title[-1]
        name = (m.group(2) or "").strip() or "(无标题)"
        full = f"{title} {name}"
        if full not in sections_index:
            sections_index[full] = {
                "section_index": title,
                "section_kind": kind,
                "section_title": name,
                "section_full": full,
                "first_page": page_no,
                "articles": [],
            }
            section_order.append(full)
        cur_section = sections_index[full]

    for line, page_no in raw_lines:
        if SECTION_RE.match(line):
            new_section(line, page_no)
            continue
        if ARTICLE_RE.match(line):
            if not started:
                started = True  # 跳过目录区
            new_article(line, page_no)
            continue
        if not started:
            continue
        if cur_article is not None:
            cur_article["text_buf"] += "\n" + line.strip()
            cur_article["pages"].append(page_no)

    flush_article()

    sections_out = []
    for full in section_order:
        sec = sections_index[full]
        if not sec["articles"]:
            continue
        spages = sorted({p for a in sec["articles"] for p in a["pages"]})
        sections_out.append({
            "section_index": sec["section_index"],
            "section_kind": sec["section_kind"],
            "section_title": sec["section_title"],
            "section_full": sec["section_full"],
            "first_page": spages[0],
            "last_page": spages[-1],
            "pages": spages,
            "articles": sec["articles"],
        })

    # ── 去重：修正案合集型 PDF 会出现同一 article_id 重复（前半 + 后半）
    if dedup:
        seen_articles: dict[int, dict] = {}  # article_id → first occurrence
        for sec in sections_out:
            kept = []
            for a in sec["articles"]:
                aid = a["article_id"]
                if aid in seen_articles:
                    # 重复，丢弃
                    continue
                seen_articles[aid] = a
                kept.append(a)
            sec["articles"] = kept
        # 过滤掉空 section
        sections_out = [s for s in sections_out if s["articles"]]

    flat = []
    for sec in sections_out:
        for a in sec["articles"]:
            flat.append({
                "law": law_full,
                "law_short": law_short,
                "law_slug": law_slug,
                "article_index": a["article_index"],
                "article_id": a["article_id"],
                "section": sec["section_full"],
                "section_index": sec["section_index"],
                "section_kind": sec["section_kind"],
                "text": a["text"],
                "start_page": a["start_page"],
                "end_page": a["end_page"],
                "pages": a["pages"],
            })

    return {
        "filename": filename,
        "law": law_full,
        "law_short": law_short,
        "law_slug": law_slug,
        "source_pdf": source_pdf,
        "source_json": source_json,
        "extracted_at": extracted_at,
        "total_sections": len(sections_out),
        "total_articles": sum(len(s["articles"]) for s in sections_out),
        "article_index_range": [
            flat[0]["article_index"] if flat else None,
            flat[-1]["article_index"] if flat else None,
        ],
        "sections": sections_out,
        "chunks": flat,
    }


# ──────────────────────────────────────────────────────────────
# 单文件 / 批量入口
# ──────────────────────────────────────────────────────────────
def process_one(json_path: Path, out_dir: Path | None = None) -> dict:
    """处理一个 page-level JSON，返回 articles dict。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    law_full, law_short, law_slug = detect_law(data.get("filename", json_path.stem))
    print(f"📑 {json_path.name} → {law_short}（{law_slug}）")

    out = parse_articles(
        pages=data["pages"],
        law_full=law_full,
        law_short=law_short,
        law_slug=law_slug,
        source_pdf=data.get("path", ""),
        source_json=str(json_path),
        extracted_at=data.get("parsed_at", ""),
        filename=data.get("filename", json_path.stem),
    )

    out_dir = out_dir or json_path.parent
    out_path = out_dir / f"{json_path.stem}.articles.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"   ✅ → {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"      节={out['total_sections']}  条={out['total_articles']}  范围={out['article_index_range'][0]} ~ {out['article_index_range'][1]}")
    if out["sections"]:
        first_sec = out["sections"][0]
        print(f"      首节: {first_sec['section_full']}")
        for a in first_sec["articles"][:2]:
            print(f"        [{a['article_index']}] (p{a['start_page']}) {a['text'][:50]}…")
    return out


def process_all(json_dir: Path, merged_out: Path | None = None):
    """批量：每个 page JSON 输出 .articles.json，再合成 all_laws.articles.json。"""
    json_dir = json_dir.resolve()
    page_jsons = sorted(json_dir.glob("*.json"))
    # 排除已生成的 .articles.json（避免重复处理）
    page_jsons = [p for p in page_jsons if ".articles." not in p.name]
    if not page_jsons:
        sys.exit(f"❌ 目录无 page-level JSON: {json_dir}")

    print(f"📂 扫描: {json_dir}  → 找到 {len(page_jsons)} 个 JSON\n")

    all_results: list[dict] = []
    for jp in page_jsons:
        all_results.append(process_one(jp, out_dir=json_dir))
        print()

    # 合并
    merged = {
        "merged_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "total_laws": len(all_results),
        "total_articles": sum(r["total_articles"] for r in all_results),
        "total_sections": sum(r["total_sections"] for r in all_results),
        "laws": [
            {
                "law": r["law"],
                "law_short": r["law_short"],
                "law_slug": r["law_slug"],
                "filename": r["filename"],
                "total_articles": r["total_articles"],
                "total_sections": r["total_sections"],
                "article_index_range": r["article_index_range"],
            }
            for r in all_results
        ],
        "chunks": [a for r in all_results for a in r["chunks"]],
    }

    merged_out = merged_out or (json_dir / "all_laws.articles.json")
    merged_out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 合并: {merged_out.name} ({merged_out.stat().st_size / 1024:.1f} KB)")
    print(f"   法律={merged['total_laws']}  条={merged['total_articles']}  节={merged['total_sections']}")
    for r in all_results:
        print(f"     • {r['law_short']:6s} ({r['law_slug']}): {r['total_articles']:4d} 条  {r['total_sections']:3d} 节")
    return merged


def main():
    ap = argparse.ArgumentParser(description="切分 page-level JSON 为 节→条 结构")
    ap.add_argument("input", type=Path, nargs="?", help="page-level JSON 路径")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="输出 JSON（默认: <input>.articles.json）")
    ap.add_argument("--all", action="store_true", help="批量模式")
    ap.add_argument("--dir", type=Path, default=Path("data/pdf"), help="批量模式目录")
    ap.add_argument("--merged-out", type=Path, default=None,
                    help="批量模式下合并 JSON 输出路径")
    args = ap.parse_args()

    if args.all:
        process_all(args.dir, args.merged_out)
    else:
        if not args.input:
            ap.error("需要 input 路径，或 --all 批量模式")
        out_dir = args.output.parent if args.output else None
        process_one(args.input, out_dir=out_dir)


if __name__ == "__main__":
    main()