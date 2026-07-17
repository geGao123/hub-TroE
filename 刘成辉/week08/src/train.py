"""Trainer for the three training modes.

Usage examples
--------------

# 1. BI-pair (BCE) on AFQMC
python train.py --mode bi --loss pair --dataset afqmc \
    --model_name bert-base-chinese --epochs 3 --batch_size 32

# 2. BI-triplet on BQ Corpus
python train.py --mode bi --loss triplet --dataset bq_corpus \
    --model_name bert-base-chinese --epochs 2 --batch_size 32 --margin 1.0

# 3. BI hybrid (pair + triplet) on LCQMC
python train.py --mode bi --loss hybrid --dataset lcqmc \
    --model_name bert-base-chinese --epochs 1 --batch_size 32

# 4. Cross on AFQMC
python train.py --mode cross --loss pair --dataset afqmc \
    --model_name bert-base-chinese --epochs 3 --batch_size 32

# Optional: smoke-test on a tiny model
python train.py --mode bi --loss pair --dataset afqmc \
    --model_name google/electra-small-discriminator --max_train_steps 20

# Optional: read most params from a YAML file. Priority is
#     CLI flag  >  YAML  >  hardcoded default
# python train.py --config configs/afqmc_bce.yaml --epochs 5
#         # YAML fills everything else; CLI bumps epochs to 5

Outputs
-------
* ``--output_dir/<dataset>/<mode>-<loss>/ckpt-*/`` — model checkpoints
* ``--output_dir/<dataset>/<mode>-<loss>/eval.json`` — validation metrics
* ``--output_dir/<dataset>/<mode>-<loss>/trainer_state.json`` — step-wise metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from argparse import SUPPRESS
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import (
    PairDataset,
    TripletDataset,
    build_tokenizer,
    collate_pair,
    collate_triplet,
    load_jsonl,
    split_for_triplet,
)
from model import (
    BiEncoder,
    CrossEncoder,
    bce_pair_loss,
    build_model,
    cosent_loss,
    triplet_margin_loss_fn,
)
import config
from progress import ProgressBar, should_show_progress

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


# `set_seed` / `device` moved to config.py for reuse across all entry points.
set_seed = config.seed_everything
device = config.get_device

# Hardcoded fallback defaults. Priority: CLI > YAML > these.
# Keep this in sync with the docstring's CLI examples.
TRAIN_DEFAULTS = {
    "mode": "bi",
    "loss": "pair",
    "dataset": "afqmc",
    "data_root": Path(__file__).resolve().parent.parent / "data",
    "model_name": "bert-base-chinese",
    "cache_dir": None,
    "pretrain_dir": None,  # None = use config.PRETRAIN_DIR
    "epochs": 3,
    "batch_size": 32,
    "eval_batch_size": 64,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "max_train_steps": -1,
    "eval_steps": 200,
    "save_steps": 0,
    "logging_steps": 50,
    "seed": 42,
    "max_length": 64,
    "max_length_pair": 128,
    "num_workers": 0,
    "pooling": "mean",
    "proj_dim": 0,
    "margin": 1.0,
    "no_normalize": False,
    "output_dir": Path(__file__).resolve().parent / "runs",
    "no_amp": False,
    "no_progress_bar": False,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the parser. All tunable args use ``SUPPRESS`` as default so
    the YAML/CLI overlay in ``config.parse_with_yaml`` can detect "unset"."""
    p = argparse.ArgumentParser(
        description="Train a sentence-pair model (BI or Cross encoder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # YAML bootstrap
    p.add_argument(
        "--config", type=Path, default=None,
        help="YAML file with init params (CLI > YAML > default). See config.example.yaml.",
    )
    # core
    p.add_argument("--mode", choices=("bi", "cross"), default=SUPPRESS)
    p.add_argument(
        "--loss",
        choices=("pair", "cosent", "triplet", "hybrid"),
        default=SUPPRESS,
        help="pair=BCE / cosent=CosineEntropy / triplet=TripletMargin / "
             "hybrid=alternate pair+triplet (bi only)",
    )
    p.add_argument("--dataset", choices=("afqmc", "bq_corpus", "lcqmc"), default=SUPPRESS)
    p.add_argument("--data_root", type=Path, default=SUPPRESS)
    p.add_argument(
        "--model_name",
        default=SUPPRESS,
        help="HF model id or local path",
    )
    p.add_argument("--cache_dir", default=SUPPRESS)
    p.add_argument(
        "--pretrain_dir", type=Path, default=SUPPRESS,
        help="Override the pretrained model root; combined with --model_name "
             "to locate the checkpoint. Default uses config.PRETRAIN_DIR "
             "(env: TROE_PRETRAIN_DIR).",
    )
    # optimization
    p.add_argument("--epochs", type=int, default=SUPPRESS)
    p.add_argument("--batch_size", type=int, default=SUPPRESS)
    p.add_argument("--eval_batch_size", type=int, default=SUPPRESS)
    p.add_argument("--lr", type=float, default=SUPPRESS)
    p.add_argument("--weight_decay", type=float, default=SUPPRESS)
    p.add_argument("--warmup_ratio", type=float, default=SUPPRESS)
    p.add_argument("--max_grad_norm", type=float, default=SUPPRESS)
    p.add_argument("--max_train_steps", type=int, default=SUPPRESS,
                   help="if >0, override total steps (for smoke tests)")
    p.add_argument("--eval_steps", type=int, default=SUPPRESS)
    p.add_argument("--save_steps", type=int, default=SUPPRESS,
                   help="0 = only save at end")
    p.add_argument("--logging_steps", type=int, default=SUPPRESS)
    p.add_argument("--seed", type=int, default=SUPPRESS)
    # data
    p.add_argument("--max_length", type=int, default=SUPPRESS,
                   help="per-sentence max length (BI mode)")
    p.add_argument("--max_length_pair", type=int, default=SUPPRESS,
                   help="pair max length (Cross mode)")
    p.add_argument("--num_workers", type=int, default=SUPPRESS)
    # model & loss knobs
    p.add_argument("--pooling", choices=("mean", "cls"), default=SUPPRESS)
    p.add_argument("--proj_dim", type=int, default=SUPPRESS,
                   help="if >0, project encoder hidden to this dim")
    p.add_argument("--margin", type=float, default=SUPPRESS,
                   help="triplet margin")
    p.add_argument("--no_normalize", action="store_true", default=SUPPRESS,
                   help="disable L2 normalize before triplet distance")
    # outputs
    p.add_argument("--output_dir", type=Path, default=SUPPRESS)
    p.add_argument("--no_amp", action="store_true", default=SUPPRESS,
                   help="disable mixed precision even on cuda")
    p.add_argument("--no_progress_bar", action="store_true", default=SUPPRESS,
                   help="disable the single-line CLI progress bar "
                        "(auto-disabled when stdout is not a TTY)")
    return p


def parse_args() -> tuple[argparse.Namespace, dict]:
    return config.parse_with_yaml(
        build_parser(), fallback_defaults=TRAIN_DEFAULTS,
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model, eval_loader: DataLoader, mode: str, device_: torch.device) -> dict:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []

    for batch in eval_loader:
        labels = batch.pop("labels").to(device_)
        if mode == "bi":
            # For evaluation, encode s1 and s2 separately using a single
            # PairDataset in mode="bi" produces separate tokenizations.
            s1 = {k: v.to(device_) for k, v in batch.items() if k != "token_type_ids"}
            # We need s2 separately for BI eval. Build a sidecar from the same
            # dataset and pull by index. To keep eval simple, we assume the
            # dataset stores (s1, s2, label) raw rows and we re-tokenize on the fly.
            raise RuntimeError(
                "BI evaluation should use evaluate_bi(); see evaluate.py"
            )
        else:  # cross
            inputs = {k: v.to(device_) for k, v in batch.items()}
            logits = model(**inputs)
            all_logits.extend(logits.float().cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())

    if not all_labels:
        return {"acc": 0.0, "n": 0}

    preds = (torch.sigmoid(torch.tensor(all_logits)) > 0.5).long().tolist()
    acc = sum(int(p == y) for p, y in zip(preds, all_labels)) / len(all_labels)
    return {"acc": acc, "n": len(all_labels)}


# --------------------------------------------------------------------------- #
# BI eval that re-tokenizes s2 separately
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate_bi(
        model: BiEncoder,
        eval_rows: list[dict],
        tokenizer,
        device_: torch.device,
        batch_size: int = 64,
        max_length: int = 64,
) -> dict:
    """Evaluation for BI encoder.

    Tokenizes s1 and s2 separately per the BI contract.
    """
    model.eval()
    s1_list = [r["sentence1"] for r in eval_rows]
    s2_list = [r["sentence2"] for r in eval_rows]
    labels = torch.tensor([r["label"] for r in eval_rows], dtype=torch.float)

    pad_id = tokenizer.pad_token_id or 0
    all_logits: list[float] = []
    for i in range(0, len(eval_rows), batch_size):
        s1_b = s1_list[i:i + batch_size]
        s2_b = s2_list[i:i + batch_size]
        e1 = tokenizer(s1_b, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        e2 = tokenizer(s2_b, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        e1 = {k: v.to(device_) for k, v in e1.items()}
        e2 = {k: v.to(device_) for k, v in e2.items()}
        logits = model.forward_pair(e1, e2)
        all_logits.extend(logits.float().cpu().tolist())

    preds = (torch.sigmoid(torch.tensor(all_logits)) > 0.5).long()
    acc = (preds == labels.long()).float().mean().item()
    return {"acc": acc, "n": len(labels)}


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def make_optimizer(model, lr: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return AdamW(params, lr=lr)


def make_scheduler(optimizer, num_warmup: int, num_total: int):
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=num_total,
    )


def save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> Path:
    ckpt_dir = output_dir / f"ckpt-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    return ckpt_dir


def train(args) -> Path:
    set_seed(args.seed)
    dev = device()
    print(f"device: {dev} ({config.device_name(dev)})")

    # ---- tokenizer ----
    tokenizer = build_tokenizer(args.model_name, cache_dir=args.cache_dir)
    pad_id = tokenizer.pad_token_id or 0

    # ---- model ----
    model = build_model(
        args.mode,
        args.model_name,
        pooling=args.pooling,
        proj_dim=args.proj_dim if args.proj_dim > 0 else None,
        cache_dir=args.cache_dir,
    ).to(dev)

    # ---- data ----

    train_rows = load_jsonl(os.path.join(args.data_root, args.dataset, "train.jsonl"))
    valid_rows = load_jsonl(os.path.join(args.data_root, args.dataset, "validation.jsonl"))

    if args.mode == "bi":
        train_pair_ds = PairDataset(
            train_rows, tokenizer, mode="bi",
            max_length=args.max_length,
        )
        train_pair_loader = DataLoader(
            train_pair_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=False,
            collate_fn=lambda b: collate_pair(b, pad_id),
        )
    else:  # cross
        train_pair_ds = PairDataset(
            train_rows, tokenizer, mode="cross",
            max_length_pair=args.max_length_pair,
        )
        train_pair_loader = DataLoader(
            train_pair_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=False,
            collate_fn=lambda b: collate_pair(b, pad_id),
        )

    train_triplet_loader: Optional[DataLoader] = None
    if args.loss in ("triplet", "hybrid"):
        pos_rows, neg_pool_s2 = split_for_triplet(train_rows)
        train_triplet_ds = TripletDataset(
            pos_rows, neg_pool_s2, tokenizer,
            max_length=args.max_length, seed=args.seed,
        )
        train_triplet_loader = DataLoader(
            train_triplet_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=False,
            collate_fn=lambda b: collate_triplet(b, pad_id),
        )

    # ---- loss routing ----
    if args.mode == "cross" and args.loss not in ("pair",):
        raise ValueError(f"--loss {args.loss} only supported in --mode bi")

    def compute_pair_loss(model, batch) -> tuple[torch.Tensor, dict]:
        labels = batch.pop("labels").to(dev)
        if args.mode == "bi":
            # PairDataset(mode='bi') produces a SINGLE encoding per row
            # (sentence1 only). For BI-pair loss we still want to encode s1
            # and s2 separately. So we re-fetch via the underlying rows.
            raise RuntimeError(
                "BI-pair loss should call compute_bi_pair_loss; see below"
            )
        else:  # cross
            inputs = {k: v.to(dev) for k, v in batch.items()}
            logits = model(**inputs)
            out = bce_pair_loss(logits, labels)
            return out.loss, {"acc": out.extra["acc"]}

    def compute_bi_pair_loss(model, batch, rows_offset: int) -> tuple[torch.Tensor, dict]:
        """BI-pair loss over a (s1, s2, label) batch.

        ``batch`` here is what ``PairDataset(mode='bi')`` returned — i.e.
        encoded sentence1 only. To get sentence2, we walk back through the
        underlying ``rows`` list using the DataLoader's shuffle order.

        Simpler approach: we re-tokenize sentence2 on the fly via the dataset's
        row pointer, using a parallel iteration. But the cleanest way is to
        switch PairDataset to return BOTH s1 and s2 encodings. That's what we
        do below in the loop by indexing into the dataset directly.
        """
        raise NotImplementedError  # unused; handled in the loop body below

    # ---- optimizer / scheduler ----
    epochs = args.epochs
    steps_per_epoch = len(train_pair_loader)
    if args.loss in ("triplet", "hybrid") and train_triplet_loader is not None:
        # Hybrid: 1 pair batch + 1 triplet batch per "step"
        if args.loss == "hybrid":
            steps_per_epoch = len(train_pair_loader) + len(train_triplet_loader)
        else:
            steps_per_epoch = len(train_triplet_loader)
    total_steps = (
        args.max_train_steps if args.max_train_steps > 0
        else steps_per_epoch * epochs
    )
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer,
        num_warmup=int(total_steps * args.warmup_ratio),
        num_total=total_steps,
    )
    use_amp = (not args.no_amp) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Wire up file logging
    log_file = os.path.join(args.dataset, f"{args.mode}-{args.loss}.log")
    logger = config.setup_logging(
        name="troe.train",
        log_file=log_file,
        level=logging.INFO,
    )
    logger.info("device=%s | use_amp=%s", config.device_name(dev), use_amp)

    # ---- output dir ----
    run_name = f"{args.mode}-{args.loss}"
    out_dir = Path(os.path.join(args.output_dir, args.dataset, run_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    state_log: list[dict] = []
    print(f"output_dir: {out_dir}")
    print(f"total_steps: {total_steps}   steps/epoch: {steps_per_epoch}")
    logger.info("output_dir=%s", out_dir)
    logger.info("total_steps=%d steps_per_epoch=%d", total_steps, steps_per_epoch)

    # ---- progress bar ----
    bar = ProgressBar(
        total=total_steps,
        enabled=should_show_progress(args),
    )
    if bar.enabled:
        print("progress bar: ON (TTY detected, pass --no_progress_bar to disable)")

    # ---- training loop ----
    model.train()
    global_step = 0
    t0 = time.time()
    triplet_iter = iter(train_triplet_loader) if train_triplet_loader is not None else None

    def bi_pair_step(batch_idx: int) -> tuple[torch.Tensor, dict]:
        """Run a BI-pair step. Re-tokenizes sentence2 from the underlying rows."""
        # Determine which rows this batch corresponds to. The DataLoader uses
        # shuffle; we read the dataset in shuffled order via a shared RNG.
        rows = _rows_in_batch_order(train_pair_ds, batch_idx, args.batch_size)
        s1 = tokenizer(
            [r["sentence1"] for r in rows], max_length=args.max_length,
            padding=True, truncation=True, return_tensors="pt",
        )
        s2 = tokenizer(
            [r["sentence2"] for r in rows], max_length=args.max_length,
            padding=True, truncation=True, return_tensors="pt",
        )
        s1 = {k: v.to(dev) for k, v in s1.items()}
        s2 = {k: v.to(dev) for k, v in s2.items()}
        labels = torch.tensor([r["label"] for r in rows], dtype=torch.float, device=dev)

        if args.loss == "cosent":
            u = model.encode(s1["input_ids"], s1["attention_mask"])
            v = model.encode(s2["input_ids"], s2["attention_mask"])
            out = cosent_loss(u, v, labels)
            return out.loss, {"acc": out.extra.get("acc", 0.0)}
        # default: BCE
        logits = model.forward_pair(s1, s2)
        out = bce_pair_loss(logits, labels)
        return out.loss, {"acc": out.extra["acc"]}

    while global_step < total_steps:
        for batch in train_pair_loader:
            if global_step >= total_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            # ---- BI-pair / BI-cosent ----
            if args.mode == "bi" and args.loss in ("pair", "cosent"):
                # batch is the (s1-only) encoded PairDataset batch; not used directly.
                # We re-encode both sides from the underlying rows for correctness.
                # Compute the row range covered by this batch using the
                # DataLoader's current state.
                batch_size_actual = batch["input_ids"].size(0)
                start = _last_batch_start[0]
                rows = train_pair_ds.rows[start:start + batch_size_actual]
                _last_batch_start[0] = start + batch_size_actual
                if len(rows) == 0:
                    continue

                s1 = tokenizer(
                    [r["sentence1"] for r in rows], max_length=args.max_length,
                    padding=True, truncation=True, return_tensors="pt",
                )
                s2 = tokenizer(
                    [r["sentence2"] for r in rows], max_length=args.max_length,
                    padding=True, truncation=True, return_tensors="pt",
                )
                s1 = {k: v.to(dev) for k, v in s1.items()}
                s2 = {k: v.to(dev) for k, v in s2.items()}
                labels = torch.tensor(
                    [r["label"] for r in rows], dtype=torch.float, device=dev,
                )

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        if args.loss == "cosent":
                            u = model.encode(s1["input_ids"], s1["attention_mask"])
                            v = model.encode(s2["input_ids"], s2["attention_mask"])
                            out = cosent_loss(u, v, labels)
                            loss = out.loss
                            extra = {"acc": out.extra.get("acc", 0.0)}
                        else:
                            logits = model.forward_pair(s1, s2)
                            out = bce_pair_loss(logits, labels)
                            loss = out.loss
                            extra = {"acc": out.extra["acc"]}
                else:
                    if args.loss == "cosent":
                        u = model.encode(s1["input_ids"], s1["attention_mask"])
                        v = model.encode(s2["input_ids"], s2["attention_mask"])
                        out = cosent_loss(u, v, labels)
                        loss = out.loss
                        extra = {"acc": out.extra.get("acc", 0.0)}
                    else:
                        logits = model.forward_pair(s1, s2)
                        out = bce_pair_loss(logits, labels)
                        loss = out.loss
                        extra = {"acc": out.extra["acc"]}

            # ---- BI-triplet ----
            elif args.mode == "bi" and args.loss == "triplet":
                try:
                    tri_batch = next(triplet_iter)
                except StopIteration:
                    triplet_iter = iter(train_triplet_loader)
                    tri_batch = next(triplet_iter)
                a = {k: v.to(dev) for k, v in tri_batch["anchor"].items()}
                p = {k: v.to(dev) for k, v in tri_batch["pos"].items()}
                n = {k: v.to(dev) for k, v in tri_batch["neg"].items()}
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        emb_a, emb_p, emb_n = model.forward_triplet(a, p, n)
                        out = triplet_margin_loss_fn(
                            emb_a, emb_p, emb_n,
                            margin=args.margin, normalize=not args.no_normalize,
                        )
                else:
                    emb_a, emb_p, emb_n = model.forward_triplet(a, p, n)
                    out = triplet_margin_loss_fn(
                        emb_a, emb_p, emb_n,
                        margin=args.margin, normalize=not args.no_normalize,
                    )
                loss, extra = out.loss, out.extra

            # ---- BI-hybrid: alternate pair & triplet ----
            elif args.mode == "bi" and args.loss == "hybrid":
                # Step type alternates every global step.
                if global_step % 2 == 0:
                    try:
                        tri_batch = next(triplet_iter)
                    except StopIteration:
                        triplet_iter = iter(train_triplet_loader)
                        tri_batch = next(triplet_iter)
                    a = {k: v.to(dev) for k, v in tri_batch["anchor"].items()}
                    p = {k: v.to(dev) for k, v in tri_batch["pos"].items()}
                    n = {k: v.to(dev) for k, v in tri_batch["neg"].items()}
                    emb_a, emb_p, emb_n = model.forward_triplet(a, p, n)
                    out = triplet_margin_loss_fn(
                        emb_a, emb_p, emb_n,
                        margin=args.margin, normalize=not args.no_normalize,
                    )
                    loss, extra = out.loss, {**out.extra, "kind": "triplet"}
                else:
                    batch_size_actual = batch["input_ids"].size(0)
                    start = _last_batch_start[0]
                    rows = train_pair_ds.rows[start:start + batch_size_actual]
                    _last_batch_start[0] = start + batch_size_actual
                    if len(rows) == 0:
                        continue
                    s1 = tokenizer(
                        [r["sentence1"] for r in rows], max_length=args.max_length,
                        padding=True, truncation=True, return_tensors="pt",
                    )
                    s2 = tokenizer(
                        [r["sentence2"] for r in rows], max_length=args.max_length,
                        padding=True, truncation=True, return_tensors="pt",
                    )
                    s1 = {k: v.to(dev) for k, v in s1.items()}
                    s2 = {k: v.to(dev) for k, v in s2.items()}
                    labels = torch.tensor(
                        [r["label"] for r in rows], dtype=torch.float, device=dev,
                    )
                    logits = model.forward_pair(s1, s2)
                    out = bce_pair_loss(logits, labels)
                    loss, extra = out.loss, {"acc": out.extra["acc"], "kind": "pair"}

            # ---- Cross-pair ----
            else:
                labels = batch.pop("labels").to(dev)
                inputs = {k: v.to(dev) for k, v in batch.items()}
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(**inputs)
                        out = bce_pair_loss(logits, labels)
                else:
                    logits = model(**inputs)
                    out = bce_pair_loss(logits, labels)
                loss, extra = out.loss, {"acc": out.extra["acc"]}

            # ---- backward ----
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()

            global_step += 1

            # ---- progress bar update (silent, every step) ----
            bar.update(
                global_step,
                loss=loss.item(),
                lr=scheduler.get_last_lr()[0],
                **extra,
            )

            # ---- logging ----
            if global_step % args.logging_steps == 0 or global_step == 1:
                elapsed = time.time() - t0
                msg = (
                    f"step {global_step}/{total_steps}  loss={loss.item():.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"elapsed={elapsed:.1f}s"
                )
                if "acc" in extra:
                    msg += f"  acc={extra['acc']:.4f}"
                if "d_pos" in extra:
                    msg += (
                        f"  d_pos={extra['d_pos']:.3f}  d_neg={extra['d_neg']:.3f}  "
                        f"gap={extra['gap']:.3f}"
                    )
                # write_log_line breaks above the bar so the per-step
                # message doesn't garble the running progress line.
                bar.write_log_line(msg)
                logger.info(msg)
                state_log.append({
                    "step": global_step,
                    "loss": float(loss.item()),
                    **{k: float(v) for k, v in extra.items() if isinstance(v, (int, float))},
                })

            # ---- eval ----
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                m = _eval_now(model, valid_rows, tokenizer, args, dev)
                m["step"] = global_step
                state_log.append({"step": global_step, "eval": m})
                bar.write_log_line(f"  >> eval @ step {global_step}: {m}")
                logger.info("eval @ step %d: %s", global_step, m)
                model.train()

            # ---- save ----
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, out_dir, global_step)
                bar.write_log_line(f"  >> saved checkpoint @ step {global_step}")
                logger.info("saved checkpoint @ step %d", global_step)

        # End of epoch — reset the row pointer for BI pair loader so we
        # iterate from the start next epoch (DataLoader shuffle already
        # redraws order; we just need to walk through rows in the new order).
        if args.mode == "bi":
            _last_batch_start[0] = 0

    # ---- final eval ----
    final_eval = _eval_now(model, valid_rows, tokenizer, args, dev)
    state_log.append({"step": global_step, "eval": final_eval})

    # ---- save final model ----
    ckpt = save_checkpoint(model, tokenizer, out_dir, global_step)
    (out_dir / "eval.json").write_text(json.dumps(final_eval, ensure_ascii=False, indent=2))
    (out_dir / "trainer_state.json").write_text(json.dumps(state_log, ensure_ascii=False, indent=2))
    # Drop the progress bar cleanly before the post-loop printout.
    bar.finish()
    print(f"FINAL eval: {final_eval}")
    logger.info("FINAL eval: %s", final_eval)
    print(f"checkpoint: {ckpt}")
    logger.info("checkpoint=%s", ckpt)

    # ---- plot loss curve ----
    try:
        from utils_plot import plot_loss_curve  # noqa: WPS433 (lazy import)
        plot_path = config.PLOTS_DIR / f"{args.dataset}_{args.mode}-{args.loss}_loss.png"
        plot_loss_curve(state_log, plot_path, title=f"{args.dataset} {args.mode}-{args.loss}")
        logger.info("loss-curve plot=%s", plot_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_loss_curve skipped: %s", exc)

    return ckpt


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

# Module-level so train() can reset between epochs without `global` trickery.
_last_batch_start = [0]


def _rows_in_batch_order(ds: PairDataset, batch_idx: int, batch_size: int) -> list[dict]:
    start = batch_idx * batch_size
    return ds.rows[start:start + batch_size]


def _eval_now(model, valid_rows, tokenizer, args, dev) -> dict:
    if args.mode == "bi":
        return evaluate_bi(
            model, valid_rows, tokenizer, dev,
            batch_size=args.eval_batch_size, max_length=args.max_length,
        )
    # cross
    pad_id = tokenizer.pad_token_id or 0
    ds = PairDataset(
        valid_rows, tokenizer, mode="cross",
        max_length_pair=args.max_length_pair,
    )
    loader = DataLoader(
        ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=lambda b: collate_pair(b, pad_id),
    )
    return evaluate(model, loader, args.mode, dev)


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #


def main() -> None:
    args, yaml_cfg = parse_args()
    if yaml_cfg:
        print(f"loaded YAML config: {len(yaml_cfg)} keys overridden")
    roots = [args.pretrain_dir] if args.pretrain_dir else None
    args.model_name = config.resolve_model_path(args.model_name, roots=roots)
    if args.model_name != config.PRETRAIN_DIR and args.pretrain_dir:
        # Surface the resolved path so it's obvious which root won.
        print(f"using pretrained model root: {args.pretrain_dir}")
    train(args)


if __name__ == "__main__":
    main()
