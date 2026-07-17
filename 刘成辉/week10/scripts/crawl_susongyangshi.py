#!/usr/bin/env python3
"""
从最高人民法院网站 https://www.court.gov.cn/susongyangshi/ 抓取所有诉讼文书样式，
按分类树保存到 data/诉讼文书样式/ 目录下，每个文书一份 Markdown。

用法：
    python scripts/crawl_susongyangshi.py [--workers 8] [--out data/诉讼文书样式]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://www.court.gov.cn"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
INDEX_URL = f"{BASE}/susong.html"
PER_PAGE = 20

# ---------- HTTP ----------
_session_local = threading_local() if False else None  # placeholder


def _opener():
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(),
        urllib.request.HTTPHandler(debuglevel=0),
    )


def fetch(url: str, retries: int = 3, sleep: float = 0.3) -> str:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
            with _opener().open(req, timeout=30) as r:
                data = r.read()
            # 强制 utf-8
            return data.decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            time.sleep(sleep * (i + 1))
    raise RuntimeError(f"fetch {url} failed: {last_err}")


# ---------- 分类树 ----------
NODE_RE = re.compile(
    r"\{'id':'(\d+)','pid':'(\d+)','name':'([^']+)','open':(true|false),"
    r"'url':(false|'([^']*)'),'target':'([^']+)'\}"
)


def parse_tree(html_text: str):
    raw = NODE_RE.findall(html_text)
    nodes = [
        {"id": m[0], "pid": m[1], "name": m[2], "url": m[5] or False}
        for m in raw
    ]
    by_pid: dict[str, list[dict]] = {}
    for n in nodes:
        by_pid.setdefault(n["pid"], []).append(n)

    # 每个叶子节点计算完整路径（顶层 → 叶子）
    leaves: list[dict] = []

    def walk(node, path):
        for n in by_pid.get(node["id"], []):
            cur = path + [n["name"]]
            if n["url"]:
                leaves.append({**n, "path": cur})
            else:
                walk(n, cur)

    for root in by_pid.get("1", []):
        cur = [root["name"]]
        if root["url"]:
            leaves.append({**root, "path": cur})
        walk(root, cur)
    return nodes, leaves


# ---------- 分类列表页 ----------
LIST_ITEM_RE = re.compile(
    r'<a title="([^"]+)"[^>]*href="(/susongyangshi/xiangqing/(\d+)\.html)"[^>]*>[^<]+</a>\s*'
    r'<i class="date">(\d{4}-\d{2}-\d{2})</i>'
)
TOTAL_RE = re.compile(r'共<span class="num">(\d+)</span>篇')
NEXT_RE = re.compile(r'<li class="next">\s*<a href="([^"]+)">')


def parse_list_page(html_text: str) -> tuple[list[dict], int | None, str | None]:
    items = [
        {"title": m[0], "url": m[1], "id": int(m[2]), "date": m[3]}
        for m in LIST_ITEM_RE.findall(html_text)
    ]
    total_m = TOTAL_RE.search(html_text)
    total = int(total_m.group(1)) if total_m else None
    next_m = NEXT_RE.search(html_text)
    next_url = next_m.group(1) if next_m else None
    return items, total, next_url


def list_all_in_category(cat_url: str) -> list[dict]:
    """翻页抓取某个分类下的所有文书。"""
    full = f"{BASE}{cat_url}"
    html_text = fetch(full)
    items, total, next_url = parse_list_page(html_text)
    pages_fetched = 1
    # 计算总页数
    if total is not None:
        total_pages = (total + PER_PAGE - 1) // PER_PAGE
    else:
        total_pages = 1
    # 用 next_url 翻页
    while next_url and pages_fetched < total_pages:
        full = f"{BASE}{next_url}" if next_url.startswith("/") else next_url
        ht = fetch(full)
        more_items, _, next_url = parse_list_page(ht)
        items.extend(more_items)
        pages_fetched += 1
        time.sleep(0.05)
    return items


# ---------- 详情页 → Markdown ----------
ZOOM_RE = re.compile(r'<div class="txt_txt"[^>]*id="zoom">(.*?)</div>\s*</div>\s*</div>', re.DOTALL)
TITLE_RE = re.compile(r'<div class="title">([^<]+)</div>')
DATE_RE = re.compile(r'发布时间：(\d{4}-\d{2}-\d{2})')


class StripHTML(HTMLParser):
    """非常简单的 HTML → Markdown，去掉样式 span，保留段落/标题/换行。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_skip = 0
        self.tag_stack: list[str] = []
        self.skip_tags = {"script", "style", "noscript", "iframe"}

    def handle_starttag(self, tag, attrs):
        attr_d = dict(attrs)
        if tag in self.skip_tags:
            self.in_skip += 1
            return
        self.tag_stack.append(tag)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self.parts.append("\n\n" + "#" * level + " ")
        elif tag == "p":
            self.parts.append("\n\n")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "strong" or tag == "b":
            self.parts.append("**")
        elif tag in {"em", "i"}:
            self.parts.append("*")
        elif tag == "li":
            self.parts.append("\n- ")

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            if self.in_skip > 0:
                self.in_skip -= 1
            return
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n\n")
        elif tag == "p":
            self.parts.append("\n")
        elif tag in {"strong", "b", "em", "i"}:
            self.parts.append("**" if tag in {"strong", "b"} else "*")
        elif tag == "div":
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_skip:
            return
        self.parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self.parts)
        # 合并多余空行
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        # 行首尾空白
        lines = [ln.rstrip() for ln in raw.split("\n")]
        return "\n".join(lines).strip()


def html_to_markdown(html_fragment: str) -> str:
    # 先去掉所有样式属性 / 字体声明，保留文本结构
    cleaned = re.sub(r"\s+style=\"[^\"]*\"", "", html_fragment)
    cleaned = re.sub(r"\s+(align|width|height)=\"[^\"]*\"", "", cleaned)
    p = StripHTML()
    p.feed(cleaned)
    return p.get_text()


def parse_detail(html_text: str) -> dict:
    title_m = TITLE_RE.search(html_text)
    date_m = DATE_RE.search(html_text)
    zoom_m = ZOOM_RE.search(html_text)
    body = html_to_markdown(zoom_m.group(1)) if zoom_m else ""
    return {
        "title": title_m.group(1).strip() if title_m else "",
        "date": date_m.group(1) if date_m else "",
        "body": body,
    }


# ---------- 文件名安全化 ----------
def safe_filename(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str, default="data/诉讼文书样式")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个分类（调试用）")
    ap.add_argument("--skip-fetch-list", action="store_true", help="复用 /tmp 缓存的列表页（调试用）")
    ap.add_argument("--only-id", type=str, default="", help="只跑指定 id 的叶子分类（多个用逗号分隔）")
    args = ap.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    out_root = workspace / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{out_root}")

    print("正在拉取分类树...")
    index_html = fetch(INDEX_URL)
    nodes, leaves = parse_tree(index_html)
    print(f"分类树节点 {len(nodes)} 个，叶子分类 {len(leaves)} 个")

    if args.limit:
        leaves = leaves[: args.limit]
        print(f"DEBUG: 只处理前 {args.limit} 个分类")

    if args.only_id:
        wanted = set(args.only_id.split(","))
        before = len(leaves)
        leaves = [l for l in leaves if l["id"] in wanted]
        print(f"DEBUG: 按 id 过滤 {before} → {len(leaves)} 个分类")

    # 第一步：抓取所有分类页的文书清单（轻量，单线程即可）
    all_items: list[tuple[dict, list[dict]]] = []  # (leaf, items)
    for i, leaf in enumerate(leaves, 1):
        try:
            items = list_all_in_category(leaf["url"])
            all_items.append((leaf, items))
            total_count = sum(len(it) for _, it in all_items)
            print(f"[{i:3d}/{len(leaves)}] {leaf['path']} → {len(items)} 篇 (累计 {total_count})")
        except Exception as e:
            print(f"[{i:3d}/{len(leaves)}] {leaf['path']} ❌ {e}", file=sys.stderr)
            all_items.append((leaf, []))

    grand_total = sum(len(it) for _, it in all_items)
    print(f"\n=== 总共需要下载 {grand_total} 篇文书 ===\n")

    # 第二步：并发抓取详情页
    detail_jobs: list[tuple[dict, dict]] = []  # (leaf, item)
    for leaf, items in all_items:
        for it in items:
            detail_jobs.append((leaf, it))
    print(f"准备并发抓取 {len(detail_jobs)} 篇...")

    def job(leaf, item):
        url = f"{BASE}{item['url']}"
        try:
            html_text = fetch(url)
            data = parse_detail(html_text)
            return (leaf, item, data, None)
        except Exception as e:
            return (leaf, item, None, str(e))

    success = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, leaf, it): (leaf, it) for leaf, it in detail_jobs}
        for fut in as_completed(futs):
            leaf, item, data, err = fut.result()
            if err:
                fail += 1
                print(f"  ❌ {item['url']} ({err})", file=sys.stderr)
                continue
            # 保存 Markdown
            target_dir = out_root.joinpath(*leaf["path"])
            target_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{item['id']:04d}_{safe_filename(item['title'])}.md"
            target_file = target_dir / fname
            title = data["title"] or item["title"]
            frontmatter = (
                f"---\n"
                f"id: {item['id']}\n"
                f"title: {title}\n"
                f"date: {item['date']}\n"
                f"source: {BASE}{item['url']}\n"
                f"category: {' / '.join(leaf['path'])}\n"
                f"---\n\n"
            )
            body_md = data["body"] if data["body"] else "（无正文）"
            target_file.write_text(frontmatter + f"# {title}\n\n" + body_md + "\n", encoding="utf-8")
            success += 1
            if success % 50 == 0:
                print(f"  ...已写入 {success} 篇")

    print(f"\n=== 完成：成功 {success} 篇，失败 {fail} 篇 ===")
    print(f"输出：{out_root}")


if __name__ == "__main__":
    main()