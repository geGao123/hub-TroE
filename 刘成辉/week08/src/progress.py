"""Lightweight single-line training progress bar.

A pure-stdlib replacement for ``tqdm`` aimed at the train.py CLI. The bar
overwrites its previous line with ``\\r`` and redraws at a throttled rate
(20 Hz max) so it stays out of the way of the per-step hot loop.

Layout
------

    [████████░░░░░░░░░░░░░░] 41% step 1234/3000  loss=0.234  acc=0.85  lr=1.2e-5  3.1it/s  eta=4:32

Why not just ``tqdm``? This codebase keeps its dep surface tiny and the
``train.py`` loop already knows what it wants to log. A small class keeps
the formatting honest and avoids pulling tqdm's multiprocessing / locking
machinery into a single-process CLI tool.

Coexistence with the file logger
--------------------------------

``ProgressBar.write_log_line(msg)`` clears the bar, emits ``msg`` on its
own line, then forces the bar to redraw on the next ``update``. This is the
correct hook for train.py's logging boundary (every ``logging_steps``) —
calling ``print()`` directly would garble the bar.

Output suppression
------------------

* ``enabled=False``  →  no bar at all (used by ``--no_progress_bar``).
* Non-TTY stdout     →  bar auto-disabled (so ``train.py | tee run.log``
                        doesn't dump ANSI codes into the captured file).
"""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Optional


class ProgressBar:
    """Single-line progress bar with smoothed loss / acc + ETA."""

    BAR_WIDTH = 30

    # Default fields that get special formatting (rendered in this order).
    KNOWN_FIELDS = ("loss", "acc", "lr")

    def __init__(
        self,
        total: int,
        *,
        smoothing: int = 50,
        enabled: bool = True,
        stream=None,
    ) -> None:
        self.total = max(int(total), 1)
        self.smoothing = max(int(smoothing), 1)
        self.stream = stream or sys.stdout
        # Auto-disable on non-TTY (captured / piped output) so we don't
        # emit ANSI escapes into log files. Explicit ``enabled=True`` still
        # wins so unit tests can drive it via StringIO.
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = bool(enabled) and is_tty
        self._loss_buf: deque[float] = deque(maxlen=self.smoothing)
        self._acc_buf: deque[float] = deque(maxlen=self.smoothing)
        self._t0 = time.time()
        self._last_print = 0.0
        self._min_interval = 0.05  # throttle to 20 Hz

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, step: int, **metrics) -> None:
        """Advance the bar to ``step`` and (maybe) redraw.

        Recognized keyword metrics:

        * ``loss``  — appended to the smoothed-loss window.
        * ``acc``   — appended to the smoothed-acc window.
        * ``lr``    — shown verbatim on this update only (no smoothing).

        Any other kwargs (``d_pos``, ``gap``, ``kind``, …) are appended at
        the end of the line so the bar surfaces everything ``train.py``
        already computes.
        """
        if not self.enabled:
            return
        loss = metrics.get("loss")
        if loss is not None:
            self._loss_buf.append(float(loss))
        acc = metrics.get("acc")
        if acc is not None:
            self._acc_buf.append(float(acc))

        now = time.time()
        if now - self._last_print < self._min_interval and step != self.total:
            return
        self._last_print = now

        line = self._format(step, metrics)
        self.stream.write("\r" + line)
        self.stream.flush()

    def write_log_line(self, line: str) -> None:
        """Break the bar to emit ``line`` on its own row, then redraw next time.

        Use this for per-step log messages that need to coexist with the
        bar. On a non-TTY / disabled bar this is just a plain newline write.
        """
        if not self.enabled:
            self.stream.write(line + "\n")
            self.stream.flush()
            return
        # Clear the current bar row (pad with spaces), then newline + msg.
        # 120 cols is a safe over-estimate — terminals that are narrower will
        # just wrap, which is still readable.
        self.stream.write("\r" + " " * 120 + "\r")
        self.stream.write(line + "\n")
        self.stream.flush()
        # Force redraw on next update so the bar doesn't sit blank until
        # the throttle window re-opens.
        self._last_print = 0.0

    def finish(self, final_msg: Optional[str] = None) -> None:
        """Jump to 100 %, drop the bar, optionally print a final message."""
        if not self.enabled:
            if final_msg:
                self.stream.write(final_msg + "\n")
                self.stream.flush()
            return
        # Force the last redraw at 100 % before the newline.
        self._last_print = 0.0
        self.update(self.total)
        self.stream.write("\n")
        self.stream.flush()
        if final_msg:
            self.stream.write(final_msg + "\n")
            self.stream.flush()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _format(self, step: int, metrics: dict) -> str:
        pct = min(100.0, step / self.total * 100.0)
        filled = int(round(pct / 100.0 * self.BAR_WIDTH))
        bar = "█" * filled + "░" * (self.BAR_WIDTH - filled)

        elapsed = max(time.time() - self._t0, 1e-6)
        rate = step / elapsed
        eta_s = (self.total - step) / rate if rate > 0 else 0.0

        parts: list[str] = [
            f"[{bar}] {pct:5.1f}%",
            f"step {step}/{self.total}",
        ]
        if self._loss_buf:
            mean_loss = sum(self._loss_buf) / len(self._loss_buf)
            parts.append(f"loss={mean_loss:.4f}")
        if self._acc_buf:
            mean_acc = sum(self._acc_buf) / len(self._acc_buf)
            parts.append(f"acc={mean_acc:.4f}")
        if "lr" in metrics:
            parts.append(f"lr={float(metrics['lr']):.2e}")
        parts.append(f"{rate:.2f}it/s")
        parts.append(f"eta={_fmt_eta(eta_s)}")

        # Surface any extra fields last (d_pos / gap / kind / …).
        for k, v in metrics.items():
            if k in self.KNOWN_FIELDS or k == "loss":
                continue
            try:
                parts.append(f"{k}={float(v):.4f}" if isinstance(v, float) else f"{k}={v}")
            except (TypeError, ValueError):
                parts.append(f"{k}={v}")
        return "  ".join(parts)


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds > 86400 * 30:
        return "--:--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def should_show_progress(args, stream=None) -> bool:
    """Resolve whether to render the bar for this run.

    Off when:

    * ``--no_progress_bar`` was passed (or ``no_progress_bar: true`` in YAML).
    * stdout isn't a TTY (captured output / piped log).
    """
    if getattr(args, "no_progress_bar", False):
        return False
    s = stream if stream is not None else sys.stdout
    if hasattr(s, "isatty") and not s.isatty():
        return False
    return True


# ---------------------------------------------------------------------- #
# Self-test
# ---------------------------------------------------------------------- #


if __name__ == "__main__":
    """Tiny visual self-test — drives a fake 100-step run.

    Tries to force-enable the bar even on non-TTY so you can see the layout
    by piping output. Run ``python progress.py`` for the live demo.
    """
    import math
    import random

    print("=== progress bar self-test ===")
    # Force-enable on non-TTY for the demo by wrapping stdout.
    bar = ProgressBar(total=100, enabled=True, stream=sys.stdout)
    try:
        for step in range(1, 101):
            loss = 1.0 / (1 + step * 0.1) + random.uniform(-0.02, 0.02)
            acc = min(0.99, step / 100 + random.uniform(-0.02, 0.02))
            bar.update(step, loss=loss, acc=acc, lr=2e-5)
            if step % 20 == 0:
                bar.write_log_line(
                    f"step {step}/100  loss={loss:.4f}  acc={acc:.4f}  "
                    f"(simulated logging boundary)",
                )
            time.sleep(0.03)
        bar.finish("done — all 100 fake steps finished")
    except KeyboardInterrupt:
        bar.finish("interrupted")