"""Lightweight plotting helpers used by train.py / compare_methods.py.

All output goes to ``config.PLOTS_DIR`` so the directory layout stays
predictable.

This module is intentionally lazy-imported (matplotlib only loaded when
actually needed) to keep CLI startup fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import config


def _ensure_parent(path: Path) -> None:
    config.ensure_dirs("plots")
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_loss_curve(
    state_log: Sequence[dict],
    output_path: str | Path,
    title: str = "training loss",
) -> Path:
    """Plot loss (and acc, if present) over training steps.

    ``state_log`` is the list produced by ``train.py``; each entry is either
    ``{"step", "loss", ...}`` for a training step or ``{"step", "eval": {...}}``
    for an evaluation snapshot.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    _ensure_parent(output_path)

    steps_loss: list[int] = []
    losses: list[float] = []
    accs: list[float] = []
    steps_acc: list[int] = []
    steps_eval: list[int] = []
    eval_accs: list[float] = []

    for entry in state_log:
        if "loss" in entry:
            steps_loss.append(int(entry["step"]))
            losses.append(float(entry["loss"]))
            if "acc" in entry:
                accs.append(float(entry["acc"]))
                steps_acc.append(int(entry["step"]))
        if "eval" in entry:
            steps_eval.append(int(entry["step"]))
            ev = entry["eval"]
            eval_accs.append(float(ev.get("acc", 0.0)))

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps_loss, losses, color="#1f77b4", linewidth=1.4, label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    if steps_acc:
        ax2 = ax1.twinx()
        ax2.plot(steps_acc, accs, color="#2ca02c", linewidth=1.2, alpha=0.8, label="acc")
        ax2.set_ylabel("acc", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")

    if steps_eval:
        ax1.scatter(steps_eval, [losses[0] if losses else 0] * len(steps_eval),
                    marker="|", s=120, color="#d62728", label="eval")

    plt.title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_method_comparison(
    rows: Iterable[dict],
    output_path: str | Path,
    metrics: Sequence[str] = ("acc", "f1", "auc"),
    title: str = "method comparison",
) -> Path:
    """Bar chart comparing N methods on K metrics. Saves PNG to ``output_path``.

    Each ``row`` is the dict returned by ``compare_methods._eval_spec``; we
    expect ``name``, and one float per metric.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(rows)
    output_path = Path(output_path)
    _ensure_parent(output_path)

    names = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4.5))
    width = 0.8 / max(len(metrics), 1)
    x = np.arange(len(names))

    for i, m in enumerate(metrics):
        vals = [float(r.get(m, 0.0) or 0.0) for r in rows]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    # Tiny self-test
    fake_state = [
        {"step": 1, "loss": 0.9, "acc": 0.3},
        {"step": 2, "loss": 0.7, "acc": 0.5},
        {"step": 3, "loss": 0.5, "acc": 0.6},
        {"step": 3, "eval": {"acc": 0.55}},
        {"step": 4, "loss": 0.4, "acc": 0.7},
    ]
    p1 = plot_loss_curve(fake_state, config.PLOTS_DIR / "_selftest_loss.png",
                         title="self-test loss")
    p2 = plot_method_comparison(
        [{"name": "a", "acc": 0.7, "f1": 0.65, "auc": 0.8},
         {"name": "b", "acc": 0.75, "f1": 0.7, "auc": 0.82}],
        config.PLOTS_DIR / "_selftest_compare.png",
        title="self-test compare",
    )
    print(f"wrote {p1}\nwrote {p2}")


def plot_badcase_distribution(
    by_len: list[tuple[str, int, int, float]],
    by_jaccard: list[tuple[str, int, int, float]],
    output_path: str | Path,
    title: str = "error distribution",
) -> Path:
    """Two-bar-group PNG: error rate by sentence length and by char-Jaccard.

    Each tuple in ``by_len`` / ``by_jaccard`` is ``(label, wrong, total, rate)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    _ensure_parent(output_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- length buckets ----
    ax = axes[0]
    if by_len:
        labels = [r[0] for r in by_len]
        rates = [r[3] * 100 for r in by_len]
        ax.bar(labels, rates, color="#1f77b4")
        ax.set_ylabel("error rate (%)")
        ax.set_title("by sentence length")
        ax.tick_params(axis="x", rotation=20)
        for i, r in enumerate(rates):
            ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", fontsize=8)

    # ---- jaccard buckets ----
    ax = axes[1]
    if by_jaccard:
        labels = [r[0] for r in by_jaccard]
        rates = [r[3] * 100 for r in by_jaccard]
        ax.bar(labels, rates, color="#d62728")
        ax.set_ylabel("error rate (%)")
        ax.set_title("by char-level Jaccard")
        ax.tick_params(axis="x", rotation=20)
        for i, r in enumerate(rates):
            ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path