"""PyTorch Dataset classes for sentence-pair training.

Three dataset variants cover the training modes required by the spec:

* ``PairDataset``            — (s1, s2, label)            used by BI-pair & Cross
* ``TripletDataset``         — (anchor=s1, pos=s2, neg)   used by BI-triplet
* ``BiTripletDataset``       — pair + triplet mixed       used by hybrid BI

Tokenization is shared via ``build_tokenizer`` so a single AutoTokenizer is
created from the model name passed in ``--model_name`` (e.g. ``bert-base-chinese``).

The triplet sampler follows the spec exactly:
    * anchor  = sentence1 of a ``label==1`` row
    * pos     = sentence2 of the same ``label==1`` row
    * neg     = sentence2 of a *random other* ``label==0`` row

Negatives are drawn fresh per ``__getitem__`` call (with a per-worker RNG) so
each epoch effectively sees a fresh batch of negatives even without a custom
``Sampler``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# File loading
# --------------------------------------------------------------------------- #


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a jsonl file into a list of {sentence1, sentence2, label} dicts."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "sentence1": obj.get("sentence1", "") or "",
                    "sentence2": obj.get("sentence2", "") or "",
                    "label": int(obj.get("label", 0)),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Tokenization helpers
# --------------------------------------------------------------------------- #


def build_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    """Build an AutoTokenizer. Imported lazily to keep this module light."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


def tokenize_pair(
    tokenizer,
    s1: str,
    s2: str,
    max_length: int = 128,
):
    """Tokenize a single sentence pair. Returns dict suitable for **BatchEncoding**."""
    return tokenizer(
        s1,
        s2,
        max_length=max_length,
        padding=False,
        truncation=True,
        return_tensors=None,
    )


def tokenize_single(tokenizer, text: str, max_length: int = 64):
    return tokenizer(
        text,
        max_length=max_length,
        padding=False,
        truncation=True,
        return_tensors=None,
    )


def collate_pair(features: list[dict], pad_token_id: int):
    """Collate for BI-pair (s1, s2, label) or Cross (single sequence, label).

    Each ``feature`` is a dict from ``tokenize_pair`` (which may be either a
    pair of strings for BI or a single concatenated string for Cross); we
    collate the keys present.
    """
    keys = [k for k in features[0].keys() if k != "label"]
    batch: dict[str, torch.Tensor] = {}

    max_len = max(len(f["input_ids"]) for f in features)
    for k in keys:
        if k == "input_ids":
            pad_value = pad_token_id
        elif k.endswith("ids"):
            pad_value = 0
        else:
            pad_value = 0
        padded = []
        for f in features:
            seq = f[k]
            padded.append(seq + [pad_value] * (max_len - len(seq)))
        batch[k] = torch.tensor(padded, dtype=torch.long)

    if "label" in features[0]:
        batch["labels"] = torch.tensor(
            [float(f["label"]) for f in features], dtype=torch.float
        )
    return batch


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class PairDataset(Dataset):
    """Sentence-pair dataset for BI-pair and Cross modes.

    Tokenization strategy depends on ``mode``:
        * ``mode="bi"``    — tokenize s1 and s2 *separately* (length controlled
                             by ``max_length`` each)
        * ``mode="cross"`` — tokenize the concatenated pair as a single sequence
                             so the model can use cross-attention
    """

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        mode: str = "bi",
        max_length: int = 64,
        max_length_pair: int = 128,
    ) -> None:
        if mode not in ("bi", "cross"):
            raise ValueError(f"mode must be 'bi' or 'cross', got {mode!r}")
        self.rows = rows
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.max_length_pair = max_length_pair

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        s1, s2, label = row["sentence1"], row["sentence2"], row["label"]
        if self.mode == "bi":
            enc = self.tokenizer(
                s1,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors=None,
            )
        else:  # cross
            enc = self.tokenizer(
                s1,
                s2,
                max_length=self.max_length_pair,
                padding=False,
                truncation=True,
                return_tensors=None,
            )
        enc["label"] = label
        return enc


class TripletDataset(Dataset):
    """Triplet dataset for BI-triplet mode.

    Anchors/positives come from rows where ``label==1``. Negatives are drawn
    fresh from the pool of rows where ``label==0`` at every ``__getitem__``
    call so that each epoch effectively sees a fresh batch of negatives.

    Parameters
    ----------
    pos_rows: rows with label == 1 (anchor & pos)
    neg_pool: rows with label == 0 (negatives drawn from sentence2 of these)
    neg_pool_s2: pre-extracted list of sentence2 strings from neg_pool,
                 so we don't reach into row dicts in the hot path
    """

    def __init__(
        self,
        pos_rows: list[dict],
        neg_pool_s2: list[str],
        tokenizer,
        max_length: int = 64,
        seed: int = 0,
    ) -> None:
        if not pos_rows:
            raise ValueError("TripletDataset requires at least one label==1 row")
        if not neg_pool_s2:
            raise ValueError("TripletDataset requires a non-empty negative pool")
        self.pos_rows = pos_rows
        self.neg_pool_s2 = neg_pool_s2
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Per-worker RNG so DataLoader workers don't all draw the same negative.
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pos_rows)

    def _encode(self, text: str) -> dict:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None,
        )

    def __getitem__(self, idx: int) -> dict:
        row = self.pos_rows[idx]
        neg = self._rng.choice(self.neg_pool_s2)
        anchor = self._encode(row["sentence1"])
        pos = self._encode(row["sentence2"])
        neg_enc = self._encode(neg)
        return {
            "anchor": anchor,
            "pos": pos,
            "neg": neg_enc,
            # Expose label=1 so a hybrid sampler can mix pair & triplet batches.
            "label": 1,
        }


def collate_triplet(batch: list[dict], pad_token_id: int):
    """Collate a list of TripletDataset items into a stacked batch."""

    def _stack(side: str) -> dict[str, torch.Tensor]:
        feats = [b[side] for b in batch]
        keys = [k for k in feats[0].keys() if k != "label"]
        max_len = max(len(f["input_ids"]) for f in feats)
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            pad = pad_token_id if k == "input_ids" else 0
            out[k] = torch.tensor(
                [f[k] + [pad] * (max_len - len(f[k])) for f in feats],
                dtype=torch.long,
            )
        return out

    return {
        "anchor": _stack("anchor"),
        "pos": _stack("pos"),
        "neg": _stack("neg"),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.float),
    }


# --------------------------------------------------------------------------- #
# Convenience builders
# --------------------------------------------------------------------------- #


def split_for_triplet(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Split a row list into (pos_rows, neg_pool_s2).

    neg_pool_s2 is the list of sentence2 strings from label==0 rows, as required
    by the spec ("neg 从其他 label=0 的数据中随机抽选").
    """
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_pool_s2 = [r["sentence2"] for r in rows if r["label"] == 0]
    return pos_rows, neg_pool_s2


@dataclass
class DataBundle:
    """Bundle returned by ``build_dataloaders`` for the trainer."""

    train_pair: Optional[torch.utils.data.DataLoader] = None
    train_triplet: Optional[torch.utils.data.DataLoader] = None
    eval_pair: Optional[torch.utils.data.DataLoader] = None


def build_dataloaders(
    data_root: str | Path,
    dataset_name: str,
    tokenizer,
    mode: str = "bi",
    loss_type: str = "pair",
    batch_size: int = 32,
    eval_batch_size: int = 64,
    max_length: int = 64,
    max_length_pair: int = 128,
    num_workers: int = 0,
    seed: int = 0,
) -> DataBundle:
    """Build train + eval DataLoaders for the given dataset & training config.

    Parameters
    ----------
    mode: ``"bi"`` or ``"cross"``
    loss_type: ``"pair"``, ``"triplet"``, or ``"hybrid"``
        * pair    -> PairDataset, BCE loss
        * triplet -> TripletDataset, TripletMargin loss
        * hybrid  -> returns BOTH pair and triplet loaders (trainer alternates)
    """
    from torch.utils.data import DataLoader

    data_root = Path(data_root)
    train_rows = load_jsonl(data_root / dataset_name / "train.jsonl")
    valid_rows = load_jsonl(data_root / dataset_name / "validation.jsonl")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    bundle = DataBundle()

    if loss_type in ("pair", "hybrid"):
        train_ds = PairDataset(
            train_rows, tokenizer, mode=mode,
            max_length=max_length, max_length_pair=max_length_pair,
        )
        bundle.train_pair = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=lambda b: collate_pair(b, pad_id),
            drop_last=False,
        )
    if loss_type in ("triplet", "hybrid"):
        pos_rows, neg_pool_s2 = split_for_triplet(train_rows)
        train_tri = TripletDataset(
            pos_rows, neg_pool_s2, tokenizer,
            max_length=max_length, seed=seed,
        )
        bundle.train_triplet = DataLoader(
            train_tri, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=lambda b: collate_triplet(b, pad_id),
            drop_last=False,
        )

    valid_ds = PairDataset(
        valid_rows, tokenizer, mode=mode,
        max_length=max_length, max_length_pair=max_length_pair,
    )
    bundle.eval_pair = DataLoader(
        valid_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=lambda b: collate_pair(b, pad_id),
    )

    return bundle