#!/usr/bin/env python3
"""
把 PDF 解析为 JSON，保留每页的文本、页码、表格、图片统计等元数据。

用法：
    python scripts/pdf_to_json.py <pdf> [-o <output.json>]          # 单文件
    python scripts/pdf_to_json.py --all [--dir data/pdf]            # 批量（目录下所有 *.pdf）

输出：
    默认 <pdf>.json (同目录)
    批量时输出 <pdf_basename>.json 与每个 PDF 同名
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

warnings.filterwarnings("ignore", message=".*FontBBox.*")


def _safe(v):
    """把非 JSON-serializable 的值转成字符串。"""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
    return str(v)


def parse_pdf(pdf_path: Path) -> dict:
    pdf_path = pdf_path.resolve()
    result: dict = {
        "filename": pdf_path.name,
        "path": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "tool": f"pdfplumber {pdfplumber.__version__}",
        "total_pages": 0,
        "metadata": {},
        "pages": [],
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        # 顶层 PDF metadata
        raw_md = pdf.metadata or {}
        result["metadata"] = _safe(raw_md)
        result["total_pages"] = len(pdf.pages)

        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            images = page.images or []
            chars = page.chars or []

            result["pages"].append({
                "page_number": i,
                "width": round(float(page.width), 2),
                "height": round(float(page.height), 2),
                "rotation": int(getattr(page, "rotation", 0) or 0),
                "char_count": len(text),
                "char_obj_count": len(chars),
                "image_count": len(images),
                "table_count": len(tables),
                "tables": [
                    [[cell if cell is not None else "" for cell in row] for row in tbl]
                    for tbl in tables
                ],
                "text": text,
            })

    return result


def process_one(pdf_path: Path, out_dir: Path | None = None) -> Path:
    """单文件处理，返回输出 JSON 路径。"""
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        sys.exit(f"❌ 文件不存在: {pdf_path}")

    print(f"📄 解析: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)", flush=True)
    data = parse_pdf(pdf_path)

    out_dir = out_dir or pdf_path.parent
    out = out_dir / f"{pdf_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_chars = sum(p["char_count"] for p in data["pages"])
    total_tables = sum(p["table_count"] for p in data["pages"])
    total_images = sum(p["image_count"] for p in data["pages"])
    print(f"   ✅ → {out.name} ({out.stat().st_size / 1024:.1f} KB)")
    print(f"      页面={data['total_pages']}  字符={total_chars:,}  表格={total_tables}  图片={total_images}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Parse PDF to JSON with per-page metadata.")
    ap.add_argument("pdf", type=Path, nargs="?", help="input PDF path")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output JSON path (default: <pdf>.json next to input)")
    ap.add_argument("--all", action="store_true",
                    help="批量处理目录下所有 *.pdf")
    ap.add_argument("--dir", type=Path, default=Path("data/pdf"),
                    help="批量模式下的扫描目录")
    args = ap.parse_args()

    if args.all:
        pdfs = sorted(args.dir.resolve().glob("*.pdf"))
        if not pdfs:
            sys.exit(f"❌ 目录下没有 PDF: {args.dir}")
        print(f"📂 扫描: {args.dir}  → 找到 {len(pdfs)} 个 PDF\n")
        for pdf in pdfs:
            process_one(pdf, out_dir=args.dir.resolve())
            print()
        print(f"✅ 全部完成: {len(pdfs)} 个 PDF 已解析")
    else:
        if not args.pdf:
            ap.error("需要指定 PDF 路径，或用 --all 批量模式")
        process_one(args.pdf, out_dir=args.output.parent if args.output else None)


if __name__ == "__main__":
    main()