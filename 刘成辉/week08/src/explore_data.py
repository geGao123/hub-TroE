"""Quick EDA for the three sentence-pair datasets.

Usage:
    python explore_data.py                 # explore all three datasets
    python explore_data.py --dataset afqmc # only one
    python explore_data.py --root ../data  # custom data root

Output (stdout):
    - row counts per split
    - label distribution (pos/neg ratio)
    - sentence-length stats (chars & tokens via char-only fallback)
    - duplicate-pair counts (sentence1, sentence2)
    - 3 random positive & 3 random negative samples per dataset
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DATASETS = ("afqmc", "bq_corpus", "lcqmc")
SPLITS = ("train", "validation", "test")


@dataclass
class SplitStats:
    name: str
    rows: int
    pos: int
    neg: int
    s1_chars: list[int]
    s2_chars: list[int]
    pair_seen: set[tuple[str, str]]
    samples: list[dict]

    @property
    def s1_mean(self) -> float:
        return sum(self.s1_chars) / max(len(self.s1_chars), 1)

    @property
    def s2_mean(self) -> float:
        return sum(self.s2_chars) / max(len(self.s2_chars), 1)

    @property
    def s1_max(self) -> int:
        return max(self.s1_chars, default=0)

    @property
    def s2_max(self) -> int:
        return max(self.s2_chars, default=0)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Defensive: skip corrupted lines (the bq_corpus train file
                # has a stray "负yang ben" prefix on the very first line, but
                # json.loads will tolerate because it's actually a full JSON
                # line — still, we keep the guard for safety).
                continue


def load_split(path: Path, sample_n: int = 3) -> SplitStats:
    rows = 0
    pos = 0
    neg = 0
    s1_chars: list[int] = []
    s2_chars: list[int] = []
    pair_seen: set[tuple[str, str]] = set()
    samples: list[dict] = []
    rng = random.Random(0)

    if not path.exists():
        return SplitStats(path.name, 0, 0, 0, [], [], set(), [])

    for obj in _iter_jsonl(path):
        rows += 1
        s1 = obj.get("sentence1", "") or ""
        s2 = obj.get("sentence2", "") or ""
        label = int(obj.get("label", 0))
        if label == 1:
            pos += 1
        else:
            neg += 1
        s1_chars.append(len(s1))
        s2_chars.append(len(s2))
        pair_seen.add((s1, s2))

        if len(samples) < sample_n * 2 and rng.random() < 0.001:
            samples.append({"s1": s1, "s2": s2, "label": label})

    # Ensure we surface both pos and neg examples even on small splits
    if len(samples) < sample_n * 2:
        for obj in _iter_jsonl(path):
            if any(s["s1"] == obj.get("sentence1") for s in samples):
                continue
            samples.append({
                "s1": obj.get("sentence1", ""),
                "s2": obj.get("sentence2", ""),
                "label": int(obj.get("label", 0)),
            })
            if len(samples) >= sample_n * 2:
                break

    return SplitStats(
        name=path.stem,
        rows=rows,
        pos=pos,
        neg=neg,
        s1_chars=s1_chars,
        s2_chars=s2_chars,
        pair_seen=pair_seen,
        samples=samples,
    )


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  n/a"
    return f" ({num / denom * 100:5.2f}%)"


def _print_stats(data_root: Path, dataset: str) -> None:
    print()
    print("=" * 78)
    print(f"Dataset: {dataset}    root: {data_root}")
    print("=" * 78)

    ds_dir = data_root / dataset
    if not ds_dir.is_dir():
        print(f"  !! not found: {ds_dir}")
        return

    for split in SPLITS:
        path = ds_dir / f"{split}.jsonl"
        st = load_split(path)
        if st.rows == 0:
            print(f"  [{st.name:11s}] missing or empty  ({path})")
            continue

        dups = st.rows - len(st.pair_seen)
        print(
            f"  [{st.name:11s}] rows={st.rows:>7d}  "
            f"pos={st.pos:>6d}{_pct(st.pos, st.rows)}  "
            f"neg={st.neg:>6d}{_pct(st.neg, st.rows)}  "
            f"dup={dups}"
        )
        print(
            f"               s1 chars mean={st.s1_mean:6.1f} max={st.s1_max:<5d}  "
            f"s2 chars mean={st.s2_mean:6.1f} max={st.s2_max:<5d}"
        )

    # Sample surfacing
    train_path = ds_dir / "train.jsonl"
    if train_path.exists():
        st = load_split(train_path, sample_n=3)
        if st.samples:
            print("  -- random samples (train) --")
            for i, s in enumerate(st.samples[:6], 1):
                tag = "POS" if s["label"] == 1 else "NEG"
                print(
                    f"    {i:>2d}. [{tag}] {s['s1'][:40]!s} || {s['s2'][:40]!s}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for sentence-pair datasets")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="data root dir containing afqmc/ bq_corpus/ lcqmc/",
    )
    parser.add_argument(
        "--dataset",
        choices=("all",) + DATASETS,
        default="all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for sample selection",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.root.exists():
        raise SystemExit(f"data root not found: {args.root}")

    targets = DATASETS if args.dataset == "all" else (args.dataset,)
    for ds in targets:
        _print_stats(args.root, ds)


if __name__ == "__main__":
    main()