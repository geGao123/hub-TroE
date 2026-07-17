"""Standalone evaluation script — loads a trained checkpoint and reports metrics.

Usage:
    python evaluate.py --mode bi --dataset afqmc \
        --model_name runs/afqmc/bi-pair/ckpt-3000 \
        --data_root ../data

    python evaluate.py --mode cross --dataset bq_corpus \
        --model_name runs/bq_corpus/cross-pair/ckpt-1500 \
        --split test

    # YAML for optional knobs (CLI > YAML > default)
    python evaluate.py --mode bi --dataset afqmc \
        --model_name runs/afqmc/bi-pair/ckpt-3000 \
        --config configs/eval.yaml

Metrics
-------
* Accuracy
* Precision / Recall / F1  (positive class = 1)
* AUC   (only if ``--probs`` can be computed, always for sigmoid heads)
* Confusion matrix counts
"""

from __future__ import annotations

import argparse
import json
from argparse import SUPPRESS
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset import PairDataset, build_tokenizer, collate_pair, load_jsonl
from model import build_model
import config


# Hardcoded fallback defaults. CLI > YAML > these.
EVAL_DEFAULTS = {
    "data_root": Path(__file__).resolve().parent.parent / "data",
    "cache_dir": None,
    "pretrain_dir": None,  # None = use config.PRETRAIN_DIR
    "split": "validation",
    "max_length": 64,
    "max_length_pair": 128,
    "batch_size": 64,
    "no_cuda": False,
    "pooling": "mean",
    "output_json": None,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained sentence-pair model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # YAML bootstrap
    p.add_argument(
        "--config", type=Path, default=None,
        help="YAML file with init params (CLI > YAML > default). See config.example.yaml.",
    )
    # Required-when-no-default fields stay required; YAML cannot satisfy argparse
    # `required=True` so user must pass these on CLI even when using --config.
    p.add_argument("--mode", choices=("bi", "cross"), required=True)
    p.add_argument("--dataset", choices=("afqmc", "bq_corpus", "lcqmc"), required=True)
    p.add_argument("--model_name", required=True,
                   help="HF model id OR path to a saved checkpoint dir")
    # YAML-fillable optionals
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
    p.add_argument("--output_json", type=Path, default=SUPPRESS,
                   help="if set, dump metrics to this JSON file")
    return p


def parse_args() -> tuple[argparse.Namespace, dict]:
    return config.parse_with_yaml(
        build_parser(), fallback_defaults=EVAL_DEFAULTS,
    )


@torch.no_grad()
def predict_bi(
    model, rows: list[dict], tokenizer, device, batch_size: int, max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    s1_list = [r["sentence1"] for r in rows]
    s2_list = [r["sentence2"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.int64)

    logits_all: list[float] = []
    for i in range(0, len(rows), batch_size):
        e1 = tokenizer(
            s1_list[i:i + batch_size], max_length=max_length,
            padding=True, truncation=True, return_tensors="pt",
        )
        e2 = tokenizer(
            s2_list[i:i + batch_size], max_length=max_length,
            padding=True, truncation=True, return_tensors="pt",
        )
        e1 = {k: v.to(device) for k, v in e1.items()}
        e2 = {k: v.to(device) for k, v in e2.items()}
        logits = model.forward_pair(e1, e2)
        logits_all.extend(logits.float().cpu().tolist())
    probs = 1.0 / (1.0 + np.exp(-np.array(logits_all)))
    return probs, labels


@torch.no_grad()
def predict_cross(
    model, rows: list[dict], tokenizer, device, batch_size: int, max_length_pair: int,
) -> tuple[np.ndarray, np.ndarray]:
    pad_id = tokenizer.pad_token_id or 0
    ds = PairDataset(
        rows, tokenizer, mode="cross",
        max_length_pair=max_length_pair,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_pair(b, pad_id),
    )
    logits_all: list[float] = []
    labels_all: list[int] = []
    for batch in loader:
        labels = batch.pop("labels")
        labels_all.extend(labels.long().tolist())
        inputs = {k: v.to(device) for k, v in batch.items()}
        logits = model(**inputs)
        logits_all.extend(logits.float().cpu().tolist())
    probs = 1.0 / (1.0 + np.exp(-np.array(logits_all)))
    return probs, np.array(labels_all, dtype=np.int64)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = (probs > 0.5).astype(np.int64)
    acc = accuracy_score(labels, preds)
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0,
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(labels, preds).tolist()
    return {
        "n": int(len(labels)),
        "acc": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "auc": float(auc),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }


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

    metrics = compute_metrics(probs, labels)
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "model_name": args.model_name,
        **metrics,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    print()
    print(classification_report(
        labels, (probs > 0.5).astype(int),
        target_names=["neg", "pos"], digits=4, zero_division=0,
    ))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"saved metrics -> {args.output_json}")


if __name__ == "__main__":
    main()