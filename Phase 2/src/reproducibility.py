"""Seeding and run-provenance helpers (audit M3).

``seed_everything`` makes Python, NumPy and (if installed) PyTorch deterministic for a given seed.
``collect_run_info`` records what is needed to reproduce a saved result: seed, library versions
and the git commit the code was run from.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _torch_or_none() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU and all CUDA devices) from one integer.

    With ``deterministic=True`` PyTorch is asked for deterministic kernels (``warn_only`` so an op
    without a deterministic implementation warns instead of crashing).
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be >= 0.")
    seed = int(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    torch = _torch_or_none()
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_state() -> dict[str, Any]:
    """Current commit SHA and whether tracked files have uncommitted changes (None if not a git repo)."""
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha or None, "git_dirty": None if status is None else bool(status)}


def collect_run_info(seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """JSON-safe provenance record to store beside saved results."""
    torch = _torch_or_none()
    info: dict[str, Any] = {
        "seed": int(seed),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": None if torch is None else torch.__version__,
        "cuda_available": False if torch is None else bool(torch.cuda.is_available()),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = None
    if extra:
        info.update(extra)
    return info
