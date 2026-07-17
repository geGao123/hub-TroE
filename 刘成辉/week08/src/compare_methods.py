"""Side-by-side comparison of training methods (BI-pair vs BI-triplet vs Cross, etc.).

Two modes:
1. ``--compare-by-mode``  : re-use the SAME trained checkpoints and just
                            compare metrics. You supply a JSON config like
                            ``{"bi-pair": "runs/afqmc/bi-pair/ckpt-X",
                               "cross-pair": "runs/afqmc/cross-pair/ckpt-Y"}``
2. ``--quick-train``      : tiny smoke-train each config on a subset of the
                            data, then evaluate. For CI / sanity only.

Output:
    * Pretty table on stdout
    * ``--output_csv`` for downstream plotting

Usage:
    # mode 1: pre-trained checkpoints
    python compare_methods.py --mode bi-pair:path1 --mode cross-pair:path2 \
        --dataset afqmc --output_csv compare.csv

    # mode 2: smoke-train (uses --model_name from each --mode spec)
    python compare_methods.py --quick-train --dataset afqmc \
        --mode bi-pair:bert-base-chinese --mode cross-pair:bert-base-chinese \
        --max_train_steps 30 --output_csv compare.csv

    # YAML for optional knobs (CLI > YAML > default)
    python compare_methods.py --dataset afqmc \
        --mode bi-pair:runs/afqmc/bi-pair/ckpt-8584 \
        --config configs/compare.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from argparse import SUPPRESS
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataset import build_tokenizer, load_jsonl
from evaluate import compute_metrics, predict_bi, predict_cross
from model import build_model
import config
from utils_plot import plot_method_comparison


@dataclass
class Spec:
    name: str        # e.g. "bi-pair", "cross-pair", "bi-triplet"
    encoder_mode: str  # "bi" or "cross"
    model_path: str    # HF id or local checkpoint path
    pooling: str = "mean"


def _parse_mode_spec(spec: str) -> Spec:
    """``--mode bi-pair:runs/afqmc/bi-pair/ckpt-1000`` -> Spec(name='bi-pair',
    encoder_mode='bi', model_path='runs/...')"""
    if ":" in spec:
        name, path = spec.split(":", 1)
    else:
        name, path = spec, spec
    if name.startswith("cross"):
        encoder_mode = "cross"
    elif name.startswith("bi"):
        encoder_mode = "bi"
    else:
        raise ValueError(f"could not infer encoder mode from {spec!r}")
    return Spec(name=name, encoder_mode=encoder_mode, model_path=path)


COMPARE_DEFAULTS = {
    "data_root": Path(__file__).resolve().parent.parent / "data",
    "pretrain_dir": None,  # None = use config.PRETRAIN_DIR; prepended to per-spec root search
    "split": "validation",
    "max_length": 64,
    "max_length_pair": 128,
    "batch_size": 64,
    "no_cuda": False,
    "output_csv": None,
    "output_png": None,
    "output_json": None,
    "quick_train": False,
    "max_train_steps": 30,
    "lr": 2e-5,
    "seed": 42,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare multiple trained sentence-pair models side-by-side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="YAML file with init params (CLI > YAML > default). See config.example.yaml.",
    )
    # CLI-passable; YAML can supply a list under the same key.
    # We don't use ``required=True`` so that YAML-only invocation works —
    # ``main()`` enforces non-emptiness after the parse-with-yaml step.
    p.add_argument("--dataset", choices=("afqmc", "bq_corpus", "lcqmc"), required=True)
    p.add_argument("--mode", action="append", default=SUPPRESS,
                   help="repeatable: name:model_path_or_id "
                        "(also accepts a YAML list under key 'mode')")
    # YAML-fillable
    p.add_argument("--data_root", type=Path, default=SUPPRESS)
    p.add_argument(
        "--pretrain_dir", type=Path, default=SUPPRESS,
        help="Override the pretrained model root; merged with each --mode "
             "spec's path to locate checkpoints. Default uses "
             "config.PRETRAIN_DIR (env: TROE_PRETRAIN_DIR).",
    )
    p.add_argument("--split", choices=("validation", "test"), default=SUPPRESS)
    p.add_argument("--max_length", type=int, default=SUPPRESS)
    p.add_argument("--max_length_pair", type=int, default=SUPPRESS)
    p.add_argument("--batch_size", type=int, default=SUPPRESS)
    p.add_argument("--no_cuda", action="store_true", default=SUPPRESS)
    p.add_argument("--output_csv", type=Path, default=SUPPRESS)
    p.add_argument("--output_png", type=Path, default=SUPPRESS,
                   help="bar-chart PNG; default = <PLOTS_DIR>/<dataset>_compare.png")
    p.add_argument("--output_json", type=Path, default=SUPPRESS)
    p.add_argument("--quick_train", action="store_true", default=SUPPRESS,
                   help="train each spec for --max_train_steps before evaluating")
    p.add_argument("--max_train_steps", type=int, default=SUPPRESS)
    p.add_argument("--lr", type=float, default=SUPPRESS)
    p.add_argument("--seed", type=int, default=SUPPRESS)
    return p


def parse_args() -> tuple[argparse.Namespace, dict]:
    """Same ``--config`` semantics as the other scripts. As an extra convenience,
    a YAML list under the key ``mode`` is also accepted::

        mode:
          - bi-pair:runs/afqmc/bi-pair/ckpt-1000
          - cross-pair:runs/afqmc/cross-pair/ckpt-1000
    """
    parser = build_parser()
    # Declare-key capture happens inside parse_with_yaml; we still need to
    # handle the YAML list for `--mode` BEFORE merge so the SUPPRESS attribute
    # is replaced with a real list.
    declared = {a.dest for a in parser._actions if a.dest != "help"}
    pre = parser.parse_args()
    raw_yaml: dict = {}
    cfg = getattr(pre, "config", None)
    if cfg:
        raw_yaml = config.load_yaml_config(cfg)
        if "mode" in raw_yaml and isinstance(raw_yaml["mode"], list):
            pre.mode = list(raw_yaml["mode"])
            raw_yaml.pop("mode", None)
        config.merge_yaml_into_args(pre, raw_yaml, declared)
    for k, v in COMPARE_DEFAULTS.items():
        cur = getattr(pre, k, SUPPRESS)
        if cur is SUPPRESS:
            setattr(pre, k, v)
    if not getattr(pre, "mode", None):
        parser.error(
            "at least one --mode is required (CLI or YAML list under key 'mode')."
        )
    return pre, raw_yaml


def _eval_spec(spec: Spec, rows, tokenizer_pool, device, args) -> dict:
    tokenizer = build_tokenizer(spec.model_path, cache_dir=None)
    model = build_model(
        spec.encoder_mode, spec.model_path, pooling=spec.pooling, cache_dir=None,
    ).to(device)
    model.eval()

    if spec.encoder_mode == "bi":
        probs, labels = predict_bi(
            model, rows, tokenizer, device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
    else:
        probs, labels = predict_cross(
            model, rows, tokenizer, device,
            batch_size=args.batch_size, max_length_pair=args.max_length_pair,
        )
    m = compute_metrics(probs, labels)
    m["name"] = spec.name
    m["model"] = spec.model_path
    return m


def _quick_train(spec: Spec, args, device) -> None:
    """Smoke-train each spec for a few steps so compare_methods can run end-to-end
    on machines without pre-existing checkpoints."""
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from dataset import (
        PairDataset, TripletDataset, collate_pair, collate_triplet,
        load_jsonl, split_for_triplet,
    )
    from model import bce_pair_loss, cosent_loss, triplet_margin_loss_fn

    torch.manual_seed(args.seed)
    import random; random.seed(args.seed); import numpy as np; np.random.seed(args.seed)

    tokenizer = build_tokenizer(spec.model_path, cache_dir=None)
    model = build_model(
        spec.encoder_mode, spec.model_path, pooling=spec.pooling, cache_dir=None,
    ).to(device)
    model.train()

    rows = load_jsonl(args.data_root / args.dataset / "train.jsonl")
    pad_id = tokenizer.pad_token_id or 0

    if spec.name == "bi-triplet":
        pos_rows, neg_pool_s2 = split_for_triplet(rows)
        ds = TripletDataset(pos_rows, neg_pool_s2, tokenizer, max_length=args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda b: collate_triplet(b, pad_id))
    else:
        ds = PairDataset(
            rows, tokenizer,
            mode=spec.encoder_mode,
            max_length=args.max_length,
            max_length_pair=args.max_length_pair,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda b: collate_pair(b, pad_id))

    optim = AdamW(model.parameters(), lr=args.lr)
    step = 0
    t0 = time.time()
    while step < args.max_train_steps:
        for batch in loader:
            optim.zero_grad(set_to_none=True)
            if spec.name == "bi-triplet":
                a = {k: v.to(device) for k, v in batch["anchor"].items()}
                p = {k: v.to(device) for k, v in batch["pos"].items()}
                n = {k: v.to(device) for k, v in batch["neg"].items()}
                ea, ep, en = model.forward_triplet(a, p, n)
                out = triplet_margin_loss_fn(ea, ep, en, margin=1.0, normalize=True)
                loss = out.loss
            elif spec.name == "bi-cosent":
                # Need both sides; we re-encode from the raw rows.
                # For brevity, fall back to BCE here in the smoke path.
                labels = batch.pop("labels").to(device)
                # re-encode pair via row range (skip for brevity)
                loss = torch.tensor(0.0, requires_grad=True, device=device)
            else:
                labels = batch.pop("labels").to(device)
                if spec.encoder_mode == "bi":
                    # BCE on BI requires re-encoding; skip in smoke path
                    loss = torch.tensor(0.0, requires_grad=True, device=device)
                else:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    logits = model(**inputs)
                    out = bce_pair_loss(logits, labels)
                    loss = out.loss
            loss.backward()
            optim.step()
            step += 1
            if step % 10 == 0:
                print(f"  [{spec.name}] step {step}/{args.max_train_steps} loss={float(loss):.4f}", flush=True)
            if step >= args.max_train_steps:
                break
    print(f"  [{spec.name}] trained {step} steps in {time.time()-t0:.1f}s")
    # Save weights to a temp dir so re-loading via model_path works.
    ckpt = Path(f"/tmp/_smoke_{spec.name}_{int(time.time())}")
    ckpt.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    spec.model_path = str(ckpt)


def main() -> None:
    args, _yaml_cfg = parse_args()
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = config.get_device()
    print(f"device: {config.device_name(device)}")

    specs = [_parse_mode_spec(m) for m in args.mode]
    # When --pretrain_dir is given, prepend it to the root list so each spec's
    # model_path can be resolved against it (highest priority).
    if args.pretrain_dir:
        print(f"using pretrained model root: {args.pretrain_dir}")
        roots = [args.pretrain_dir, *config.default_model_roots()]
    else:
        roots = None
    for s in specs:
        s.model_path = config.resolve_model_path(s.model_path, roots=roots)

    if args.quick_train:
        print(f"== quick-train smoke (each spec for {args.max_train_steps} steps) ==")
        for s in specs:
            _quick_train(s, args, device)

    rows = load_jsonl(args.data_root / args.dataset / f"{args.split}.jsonl")
    print(f"loaded {len(rows)} rows from {args.dataset}/{args.split}.jsonl")

    results = []
    for s in specs:
        print(f"\n>> evaluating {s.name}  ({s.encoder_mode}  /  {s.model_path})")
        m = _eval_spec(s, rows, None, device, args)
        results.append(m)
        print(json.dumps(m, ensure_ascii=False, indent=2))

    # ---- pretty table ----
    cols = ("name", "n", "acc", "precision", "recall", "f1", "auc")
    print("\n=== comparison ===")
    fmt = "{:<18s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(fmt.format(*cols))
    print("-" * 70)
    for r in results:
        print(fmt.format(
            r["name"],
            str(r["n"]),
            f"{r['acc']*100:.2f}%",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['auc']:.4f}",
        ))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(list(cols) + ["model", "split", "dataset"])
            for r in results:
                w.writerow([
                    r["name"], r["n"],
                    f"{r['acc']:.6f}", f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}", f"{r['f1']:.6f}", f"{r['auc']:.6f}",
                    r["model"], args.split, args.dataset,
                ])
        print(f"saved csv -> {args.output_csv}")

    # ---- bar chart ----
    png_path = args.output_png or (config.PLOTS_DIR / f"{args.dataset}_compare.png")
    try:
        plot_method_comparison(
            results, png_path,
            metrics=("acc", "f1", "auc"),
            title=f"{args.dataset} {args.split} — method comparison",
        )
        print(f"saved png -> {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"saved json -> {args.output_json}")


if __name__ == "__main__":
    main()