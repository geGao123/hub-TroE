"""Badcase analyzer.

Loads a trained model, runs inference on the validation (or test) split, dumps
misclassified examples to a JSONL file, and prints simple error-pattern
statistics:

* error rate by sentence-length bucket (short / medium / long)
* error rate by character-level Jaccard similarity bucket
* top-N most-confident-wrong examples (and least-confident-right, for sanity)

Usage:
    python analyze_badcases.py --mode bi --dataset afqmc \
        --model_name runs/afqmc/bi-pair/ckpt-3000 \
        --output runs/afqmc/bi-pair/badcases.jsonl

    # YAML config for optional knobs
    python analyze_badcases.py --mode bi --dataset afqmc \
        --model_name runs/afqmc/bi-pair/ckpt-3000 \
        --output runs/afqmc/bi-pair/badcases.jsonl \
        --config configs/badcase.yaml
"""

from __future__ import annotations

import argparse
import json
from argparse import SUPPRESS
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dataset import build_tokenizer, load_jsonl
from evaluate import compute_metrics, predict_bi, predict_cross
from model import build_model
import config
from utils_plot import plot_badcase_distribution


ANALYZE_DEFAULTS = {
    "data_root": Path(__file__).resolve().parent.parent / "data",
    "cache_dir": None,
    "pretrain_dir": None,  # None = use config.PRETRAIN_DIR
    "split": "validation",
    "max_length": 64,
    "max_length_pair": 128,
    "batch_size": 64,
    "no_cuda": False,
    "pooling": "mean",
    "top_n": 10,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze badcases from a trained sentence-pair model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="YAML file with init params (CLI > YAML > default). See config.example.yaml.",
    )
    # required-on-CLI
    p.add_argument("--mode", choices=("bi", "cross"), required=True)
    p.add_argument("--dataset", choices=("afqmc", "bq_corpus", "lcqmc"), required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="where to dump the badcase jsonl")
    # YAML-fillable
    p.add_argument("--data_root", type=Path, default=SUPPRESS)
    p.add_argument("--cache_dir", default=SUPPRESS)
    p.add_argument(
        "--pretrain_dir", type=Path, default=SUPPRESS,
        help="Override the pretrained model root; combined with --model_name "
             "to locate the checkpoint. Default uses config.PRETRAIN_DIR "
             "(env: TROE_PRETRAIN_DIR).",
    )
    p.add_argument("--split", choices=("validation", "test"), default=SUPPRESS)
    p.add_argument("--max_length", type=int, default=SUPPRESS)
    p.add_argument("--max_length_pair", type=int, default=SUPPRESS)
    p.add_argument("--batch_size", type=int, default=SUPPRESS)
    p.add_argument("--no_cuda", action="store_true", default=SUPPRESS)
    p.add_argument("--pooling", choices=("mean", "cls"), default=SUPPRESS)
    p.add_argument("--top_n", type=int, default=SUPPRESS)
    return p


def parse_args() -> tuple[argparse.Namespace, dict]:
    return config.parse_with_yaml(
        build_parser(), fallback_defaults=ANALYZE_DEFAULTS,
    )


def _jaccard_chars(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _length_bucket(n: int) -> str:
    if n < 16:
        return "short(<16)"
    if n < 40:
        return "medium(16-39)"
    return "long(>=40)"


def main() -> None:
    args, _yaml_cfg = parse_args()
    roots = [args.pretrain_dir] if args.pretrain_dir else None
    args.model_name = config.resolve_model_path(args.model_name, roots=roots)
    if args.pretrain_dir:
        print(f"using pretrained model root: {args.pretrain_dir}")
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = config.get_device()
    print(f"device: {config.device_name(device)}")

    tokenizer = build_tokenizer(args.model_name, cache_dir=args.cache_dir)
    model = build_model(
        args.mode, args.model_name,
        pooling=args.pooling, cache_dir=args.cache_dir,
    ).to(device)
    model.eval()

    rows = load_jsonl(args.data_root / args.dataset / f"{args.split}.jsonl")
    if args.mode == "bi":
        probs, labels = predict_bi(
            model, rows, tokenizer, device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
    else:
        probs, labels = predict_cross(
            model, rows, tokenizer, device,
            batch_size=args.batch_size, max_length_pair=args.max_length_pair,
        )

    preds = (probs > 0.5).astype(np.int64)
    metrics = compute_metrics(probs, labels)
    print("metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # ---- per-row annotations ----
    annotated = []
    for row, prob, pred, label in zip(rows, probs, preds, labels):
        s1, s2 = row["sentence1"], row["sentence2"]
        annotated.append({
            "sentence1": s1,
            "sentence2": s2,
            "label": int(label),
            "pred": int(pred),
            "prob": float(prob),
            "conf": float(abs(prob - 0.5) * 2),  # 0 = unsure, 1 = very sure
            "wrong": bool(int(pred) != int(label)),
            "len_s1": len(s1),
            "len_s2": len(s2),
            "jaccard": _jaccard_chars(s1, s2),
        })

    bad = [a for a in annotated if a["wrong"]]
    bad_sorted = sorted(bad, key=lambda a: -a["conf"])
    good_sorted = sorted(
        (a for a in annotated if not a["wrong"]),
        key=lambda a: a["conf"],
    )

    # ---- write badcase jsonl ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for a in bad_sorted:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"wrote {len(bad_sorted)} badcases -> {args.output}")

    # ---- error patterns ----
    def _err_rate_by(keyfn) -> list[tuple[str, int, int, float]]:
        bucket: dict[str, list[int]] = {}
        for a in annotated:
            k = keyfn(a)
            bucket.setdefault(k, [0, 0])
            bucket[k][0] += int(a["wrong"])
            bucket[k][1] += 1
        out = []
        for k, (w, n) in bucket.items():
            out.append((k, w, n, w / max(n, 1)))
        return sorted(out, key=lambda x: -x[3])

    by_len_s1 = _err_rate_by(lambda a: _length_bucket(a["len_s1"]))
    by_len_s2 = _err_rate_by(lambda a: _length_bucket(a["len_s2"]))
    by_jac = _err_rate_by(lambda a: f"{int(a['jaccard'] * 10) / 10:.1f}")

    print("\nerror rate by sentence1 length bucket:")
    for k, w, n, r in by_len_s1:
        print(f"  {k:<14s} {w:>5d}/{n:<6d}  {r*100:6.2f}%")

    print("\nerror rate by sentence2 length bucket:")
    for k, w, n, r in by_len_s2:
        print(f"  {k:<14s} {w:>5d}/{n:<6d}  {r*100:6.2f}%")

    print("\nerror rate by char-level Jaccard bucket:")
    for k, w, n, r in by_jac:
        print(f"  jaccard={k}    {w:>5d}/{n:<6d}  {r*100:6.2f}%")

    print(f"\ntop-{args.top_n} most-confident WRONG predictions:")
    for a in bad_sorted[: args.top_n]:
        tag = "FN" if a["label"] == 1 else "FP"
        print(
            f"  [{tag} p={a['prob']:.3f}] "
            f"{a['sentence1'][:30]!s} || {a['sentence2'][:30]!s} "
            f"(label={a['label']})"
        )

    print(f"\ntop-{args.top_n} least-confident RIGHT predictions (uncertain zone):")
    for a in good_sorted[: args.top_n]:
        print(
            f"  [p={a['prob']:.3f}] "
            f"{a['sentence1'][:30]!s} || {a['sentence2'][:30]!s} "
            f"(label={a['label']})"
        )

    # ---- PNG: error distribution ----
    png_path = (
        args.output.with_suffix(".png") if str(args.output).endswith(".jsonl")
        else config.PLOTS_DIR / f"{args.dataset}_{args.mode}_errdist.png"
    )
    try:
        plot_badcase_distribution(
            by_len_s1,
            by_jac,
            png_path,
            title=f"{args.dataset} {args.mode} — error distribution",
        )
        print(f"saved png -> {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}")


if __name__ == "__main__":
    main()