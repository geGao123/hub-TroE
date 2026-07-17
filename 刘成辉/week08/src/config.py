"""Centralized configuration for the week08 project.

Responsibilities
----------------
1. Resolve all paths (data, runs, logs, plots, baseline-model cache,
   pretrained-model cache) once at import time so the rest of the codebase
   can use plain constants.
2. Expose ``get_device()`` that picks the best available accelerator:
       CUDA  >  MPS (Apple Silicon GPU)  >  CPU
3. Provide ``setup_logging()`` so train/eval all write to ``LOGS_DIR`` with a
   consistent format.
4. Allow every constant to be overridden by an env var, which is useful for
   CI / container runs.
5. Provide ``resolve_model_path(name, *, roots=...)`` that combines a model
   **root directory** with a **model name** to locate the actual checkpoint.
6. Load YAML config files via ``load_yaml_config()`` and provide
   ``parse_with_yaml()`` so each entry-point can absorb ``--config <yaml>``
   with a consistent CLI > YAML > default priority order.

Override env vars
-----------------
    TROE_PROJECT_ROOT       default = parent of this file's parent (week08/)
    TROE_DATA_ROOT          default = <project_root>/data
    TROE_RUNS_DIR           default = <project_root>/runs
    TROE_LOGS_DIR           default = <project_root>/logs
    TROE_PLOTS_DIR          default = <project_root>/plots
    TROE_BASELINE_MODEL_DIR default = <project_root>/baseline_models
    TROE_PRETRAIN_DIR       default = <project_root>/pretrain_models
                           (legacy: PRETRAIN_MODELS_ROOT — still honored)
    TROE_DEVICE             one of {auto, cuda, mps, cpu}   default = auto

Model resolution
----------------
``resolve_model_path(name, *, roots=...)`` combines a model name with one or
more root directories. Lookup order:

    1. If ``name`` is an existing filesystem path  → return as-is.
    2. For each root in ``roots`` (default ``[PRETRAIN_DIR, BASELINE_MODEL_DIR]``):
       ``<root>/<name>`` if it exists  → return that path.
    3. Otherwise return ``name`` unchanged so HF Hub can try.

The two defaults cover the common split: ``PRETRAIN_DIR`` is for raw
HuggingFace-style pretrained checkpoints (``bert-base-chinese``,
``electra-small``, ...), ``BASELINE_MODEL_DIR`` is for downstream baselines /
locally fine-tuned snapshots. CLI / env can override either individually.

YAML config
-----------
Train/eval/analyze/compare scripts accept ``--config <path>``. Layered
priority: **CLI flag  >  YAML  >  hardcoded fallback default**.

    python train.py --config configs/afqmc_bce.yaml --epochs 5
        # CLI 5 epochs wins; YAML everything else wins; rest from defaults
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from argparse import SUPPRESS
from logging import Logger
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


# src/ -> week08/ (project root for this exercise)
_THIS_FILE = Path(__file__).resolve()
SRC_DIR = _THIS_FILE.parent
PROJECT_ROOT = Path(os.environ.get("TROE_PROJECT_ROOT", SRC_DIR.parent))
OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_ROOT", PROJECT_ROOT / "outputs"))

DATA_ROOT = Path(os.environ.get("TROE_DATA_ROOT", PROJECT_ROOT / "data"))
RUNS_DIR = Path(os.environ.get("TROE_RUNS_DIR", OUTPUTS_ROOT / "runs"))
LOGS_DIR = Path(os.environ.get("TROE_LOGS_DIR", OUTPUTS_ROOT / "logs"))
PLOTS_DIR = Path(os.environ.get("TROE_PLOTS_DIR", OUTPUTS_ROOT / "plots"))
BASELINE_MODEL_DIR = Path(
    os.environ.get("TROE_BASELINE_MODEL_DIR", PROJECT_ROOT / "baseline_models")
)
# Pretrained-model cache (bert-base-chinese, electra-small, ...). Two env
# names accepted for back-compat: TROE_PRETRAIN_DIR (new) wins over the
# legacy PRETRAIN_MODELS_ROOT.
PRETRAIN_DIR = Path(
    os.environ.get("TROE_PRETRAIN_DIR")
    or os.environ.get("PRETRAIN_MODELS_ROOT")
    or PROJECT_ROOT / "pretrain_models"
)
# Legacy alias kept so any old import of ``PRETRAIN_MODELS`` still resolves.
PRETRAIN_MODELS = PRETRAIN_DIR

# Convenience aliases used elsewhere
PATHS = {
    "project_root": PROJECT_ROOT,
    "src": SRC_DIR,
    "data": DATA_ROOT,
    "runs": RUNS_DIR,
    "logs": LOGS_DIR,
    "plots": PLOTS_DIR,
    "baseline_models": BASELINE_MODEL_DIR,
    "pretrain_dir": PRETRAIN_DIR,
}


def ensure_dirs(*keys: str) -> None:
    """Create the requested path directories if missing."""
    for k in keys:
        if k not in PATHS:
            raise KeyError(f"unknown path key {k!r}; valid: {list(PATHS)}")
        PATHS[k].mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #


def _mps_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    except AttributeError:
        return False


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best device, with optional user override.

    Priority: explicit ``prefer`` arg  >  ``TROE_DEVICE`` env  >  auto-detect
    (cuda > mps > cpu).
    """
    choice = (prefer or os.environ.get("TROE_DEVICE", "auto")).lower()

    if choice in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("TROE_DEVICE=cuda requested but CUDA is unavailable")
    if choice == "mps":
        if _mps_available():
            return torch.device("mps")
        raise RuntimeError("TROE_DEVICE=mps requested but MPS is unavailable")
    if choice == "cpu":
        return torch.device("cpu")
    if choice != "auto":
        raise ValueError(f"unknown TROE_DEVICE={choice!r}; expected auto/cuda/mps/cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return f"cuda:{dev.index or 0}"
    return dev.type


def default_model_roots() -> list[Path]:
    """Default root list used by ``resolve_model_path`` when no ``roots``
    override is passed. Order: PRETRAIN_DIR first, then BASELINE_MODEL_DIR."""
    return [PRETRAIN_DIR, BASELINE_MODEL_DIR]


def resolve_model_path(
    name: str | Path,
    *,
    roots: Optional[Iterable[str | Path]] = None,
) -> str:
    """Combine a model name with one or more root directories to find the
    actual on-disk checkpoint.

    The two parameters that determine which model gets used are:

    * ``name``         — model id (``"bert-base-chinese"``) or absolute path
    * ``roots``        — an iterable of root directories to look in; the
                         first root containing ``<root>/<name>`` wins.
                         Default: ``[PRETRAIN_DIR, BASELINE_MODEL_DIR]``.

    Lookup order:

        1. If ``name`` already points to an existing filesystem path,
           return it as-is (passes through untouched so HF from_pretrained
           and absolute-path workflows both work).
        2. For each root in ``roots``, check ``<root>/<name>``. The first hit
           is returned as a string (HF will treat it as a local repo path).
        3. If no root has it, return ``name`` unchanged so HF can fall back
           to the Hub.

    Both root directories can be overridden via env vars:

        TROE_PRETRAIN_DIR            (or legacy PRETRAIN_MODELS_ROOT)
        TROE_BASELINE_MODEL_DIR

    Examples:

        >>> resolve_model_path("bert-base-chinese")
        '/abs/pretrain_models/bert-base-chinese'   # if cached
        'bert-base-chinese'                        # otherwise

        >>> resolve_model_path("/tmp/my_run/ckpt-1000", roots=[PRETRAIN_DIR])
        '/tmp/my_run/ckpt-1000'    # absolute paths always win
    """
    p = Path(name)
    if p.exists():
        return str(p)

    if roots is None:
        roots_iter = default_model_roots()
    else:
        roots_iter = list(roots)

    for root in roots_iter:
        local = Path(root) / name
        if local.exists():
            return str(local)
    return str(name)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(
    name: str = "troe",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    also_console: bool = True,
) -> Logger:
    """Configure a logger that writes to ``LOGS_DIR/<log_file>`` (if given) and
    stdout. Safe to call multiple times — handlers are reset per call.
    """
    ensure_dirs("logs")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Reset previous handlers to avoid duplicate output when called repeatedly.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.propagate = False

    fmt = logging.Formatter(_DEFAULT_FORMAT)

    if log_file is not None:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = LOGS_DIR / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if also_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# YAML config loading
# --------------------------------------------------------------------------- #


try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


class YamlConfigError(RuntimeError):
    """Raised when a YAML config file is missing, unparsable, or malformed."""


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML file and return its root mapping as a plain dict.

    Empty files yield ``{}``. A non-mapping root raises ``YamlConfigError``.
    """
    if not _HAS_YAML:
        raise YamlConfigError(
            "PyYAML not installed; `pip install pyyaml` to use YAML configs."
        )
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise YamlConfigError(f"YAML config not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise YamlConfigError(f"failed to parse YAML at {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise YamlConfigError(
            f"YAML root must be a mapping, got {type(data).__name__}"
        )
    return data


def merge_yaml_into_args(
    ns: argparse.Namespace,
    yaml_cfg: dict,
    declared_keys: Optional[Iterable[str]] = None,
    *,
    warn_unknown: bool = True,
) -> argparse.Namespace:
    """Overlay ``yaml_cfg`` onto ``ns`` for attrs left at ``argparse.SUPPRESS``.

    Priority after this call:
        **CLI flag (anything the user typed)  >  YAML  >  fallback default**.

    Unknown YAML keys are ignored (with one warning) so a typo won't silently
    land a parameter the script doesn't know about.

    Note: ``argparse.SUPPRESS`` default means the attribute is ABSENT from
    the namespace until the user provides the flag. We therefore can't rely on
    ``vars(ns).keys()`` to know what the script accepts — pass ``declared_keys``
    (e.g. from ``{a.dest for a in parser._actions}``) when you have access to
    the parser. ``parse_with_yaml`` handles this automatically.
    """
    if declared_keys is None:
        declared_keys = set(vars(ns).keys())
    declared = set(declared_keys)
    unknown = [k for k in yaml_cfg if k not in declared]
    if unknown and warn_unknown:
        logging.getLogger("troe.config").warning(
            "YAML config has unknown keys (ignored): %s", sorted(unknown),
        )
    for k, v in yaml_cfg.items():
        if k not in declared:
            continue
        cur = getattr(ns, k, SUPPRESS)
        if cur is SUPPRESS:
            setattr(ns, k, v)
    return ns


def parse_with_yaml(
    parser: argparse.ArgumentParser,
    *,
    fallback_defaults: Optional[dict] = None,
    require_config: bool = False,
) -> tuple[argparse.Namespace, dict]:
    """One-shot CLI parse + YAML overlay + fallback default fill.

    Layered priority: **CLI > YAML > ``fallback_defaults`` > parser default**.

    Args:
        parser:           an ``ArgumentParser`` whose ``--config`` (Path-like,
                          default ``None``) is used to trigger YAML loading.
                          Every other arg in ``parser`` that the user did NOT
                          pass on CLI must have ``default=argparse.SUPPRESS``
                          so we can tell "set on CLI" apart from "left blank".
        fallback_defaults: hardcoded defaults used when neither CLI nor YAML
                          supplied a value. Keys not present in the namespace
                          are ignored (so callers can pass a partial dict).
        require_config:    when ``True``, raise ``YamlConfigError`` if the
                          user did not pass ``--config``. Useful for CI jobs
                          that want YAML-only config (no surprises from env).

    Returns:
        ``(args, raw_yaml_dict)`` — the merged namespace plus the loaded
        YAML dict (empty dict when ``--config`` is absent).
    """
    # Capture the full set of declared keys BEFORE parse_args, because
    # argparse.SUPPRESS defaults leave the attribute absent from the
    # namespace when the user didn't pass the flag.
    declared = {a.dest for a in parser._actions if a.dest != "help"}
    args = parser.parse_args()
    raw_yaml: dict = {}
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        raw_yaml = load_yaml_config(cfg_path)
        merge_yaml_into_args(args, raw_yaml, declared)
    elif require_config:
        raise YamlConfigError("--config is required (require_config=True)")

    if fallback_defaults:
        for k, v in fallback_defaults.items():
            cur = getattr(args, k, SUPPRESS)
            if cur is SUPPRESS:
                setattr(args, k, v)

    return args, raw_yaml


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    print("=== config summary ===")
    for k, v in PATHS.items():
        print(f"  {k:<18s} {v}")
    print(f"  device             {device_name(get_device())}")
    ensure_dirs("runs", "logs", "plots", "baseline_models", "pretrain_dir")
    print("  (runs / logs / plots / baseline_models / pretrain_dir created if missing)")
    print(f"  PyYAML available   {_HAS_YAML}")