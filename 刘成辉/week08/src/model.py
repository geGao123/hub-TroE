"""Model definitions + loss functions for the three training modes.

Two architectures:

* ``BiEncoder``     — encodes s1 and s2 separately, then classifies via
                       ``[u, v, |u-v|]`` dot-product head. Supports both pair
                       (BCE) and triplet (TripletMargin) losses.
* ``CrossEncoder``  — encodes ``[CLS] s1 [SEP] s2 [SEP]`` jointly and outputs a
                       single logit. Supports pair (BCE) loss only.

Both backbones are loaded via ``AutoModel`` so any HuggingFace encoder works
(``bert-base-chinese``, ``hfl/chinese-roberta-wwm-ext``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool with masking. Works for any BERT-like encoder."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    return last_hidden[:, 0]


# --------------------------------------------------------------------------- #
# Bi-Encoder
# --------------------------------------------------------------------------- #


class BiEncoder(nn.Module):
    """Twin-tower encoder. Shared weights across the two towers."""

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        proj_dim: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.pooling = pooling
        hidden = self.config.hidden_size

        if proj_dim and proj_dim != hidden:
            self.proj = nn.Linear(hidden, proj_dim)
            out_dim = proj_dim
        else:
            self.proj = nn.Identity()
            out_dim = hidden

        # Classification head: [u; v; |u-v|] -> 1 logit (binary).
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, 1),
        )

    # ----- encoding helpers -----
    def _pool(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return cls_pool(last_hidden)
        return mean_pool(last_hidden, mask)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kw) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kw)
        emb = self._pool(out.last_hidden_state, attention_mask)
        return self.proj(emb)

    def encode_pair(
        self,
        s1: dict[str, torch.Tensor],
        s2: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.encode(s1["input_ids"], s1["attention_mask"])
        v = self.encode(s2["input_ids"], s2["attention_mask"])
        return u, v

    # ----- forward heads -----
    def forward_pair(
        self,
        s1: dict[str, torch.Tensor],
        s2: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return logits for binary classification. Shape: (B,)."""
        u, v = self.encode_pair(s1, s2)
        feat = torch.cat([u, v, torch.abs(u - v)], dim=-1)
        return self.classifier(feat).squeeze(-1)

    def forward_triplet(
        self,
        anchor: dict[str, torch.Tensor],
        pos: dict[str, torch.Tensor],
        neg: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (a, p, n) embeddings, each (B, D)."""
        a = self.encode(anchor["input_ids"], anchor["attention_mask"])
        p = self.encode(pos["input_ids"], pos["attention_mask"])
        n = self.encode(neg["input_ids"], neg["attention_mask"])
        return a, p, n


# --------------------------------------------------------------------------- #
# Cross-Encoder
# --------------------------------------------------------------------------- #


class CrossEncoder(nn.Module):
    """Single-tower cross-encoder. Used for the ``--mode cross`` setting."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = 1,
        cache_dir: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kw,
    ) -> torch.Tensor:
        kw_in = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw_in["token_type_ids"] = token_type_ids
        out = self.backbone(**kw_in)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #


@dataclass
class LossOutput:
    loss: torch.Tensor
    extra: dict[str, float]


def bce_pair_loss(logits: torch.Tensor, labels: torch.Tensor) -> LossOutput:
    """Binary cross-entropy for sentence-pair matching."""
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == labels.long()).float().mean().item()
    return LossOutput(loss, {"acc": acc})


def cosent_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> LossOutput:
    """Cosine-entropy / circle-loss style soft objective (text-matching friendly).

    Uses cosine similarity + label-aware cross-entropy:

        sim = cos(u, v)                                  # (B,)
        sim_diff = sim.unsqueeze(1) - sim.unsqueeze(0)    # (B, B) pairwise
        labels_diff = (labels.unsqueeze(1) - labels.unsqueeze(0) > 0).float()
        logits = sim_diff / tau - (1 - labels_diff) * margin
        target = labels_diff / labels_diff.sum(-1, keepdim=True).clamp(min=1)
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    A clean, scale-stable ranking loss that pairs well with BI encoders.
    """
    tau = 0.05
    sim = F.cosine_similarity(u, v, dim=-1)
    sim_diff = sim.unsqueeze(1) - sim.unsqueeze(0)
    labels_diff = (labels.unsqueeze(1) > labels.unsqueeze(0)).float()
    logits = sim_diff / tau - (1 - labels_diff) * margin
    log_probs = F.log_softmax(logits, dim=-1)
    # Avoid division by zero on rows with no positives.
    denom = labels_diff.sum(-1).clamp(min=1.0)
    target = labels_diff / denom.unsqueeze(-1)
    loss = -(target * log_probs).sum(-1).mean()
    with torch.no_grad():
        preds = (sim > 0.5).long()
        acc = (preds == labels.long()).float().mean().item()
    return LossOutput(loss, {"acc": acc, "sim_mean": sim.mean().item()})


def triplet_margin_loss_fn(
    anchor: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    margin: float = 1.0,
    normalize: bool = True,
) -> LossOutput:
    """Triplet margin loss. Optionally L2-normalizes embeddings first."""
    if normalize:
        anchor = F.normalize(anchor, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)
    d_pos = F.pairwise_distance(anchor, pos)
    d_neg = F.pairwise_distance(anchor, neg)
    loss = F.relu(d_pos - d_neg + margin).mean()
    return LossOutput(
        loss,
        {
            "d_pos": d_pos.mean().item(),
            "d_neg": d_neg.mean().item(),
            "gap": (d_neg - d_pos).mean().item(),
        },
    )


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build_model(
    mode: str,
    model_name: str,
    pooling: str = "mean",
    proj_dim: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> nn.Module:
    if mode == "bi":
        return BiEncoder(
            model_name=model_name,
            pooling=pooling,
            proj_dim=proj_dim,
            cache_dir=cache_dir,
        )
    if mode == "cross":
        return CrossEncoder(model_name=model_name, cache_dir=cache_dir)
    raise ValueError(f"mode must be 'bi' or 'cross', got {mode!r}")