"""Deterministic, namespaced random-seed helpers."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np


UINT32_MODULUS = 2**32


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable uint32 seed without correlating subsystem streams."""
    payload = f"ibvs-interceptor-v1:{int(base_seed)}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % UINT32_MODULUS


def seed_everything(seed: int, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed process-level RNGs and return the applied reproducibility state.

    Gym environments and wrappers are still seeded through ``reset(seed=...)``;
    this function covers the training process, NumPy's legacy global RNG, and
    PyTorch/SB3 initialization.
    """
    resolved = int(seed)
    random.seed(resolved)
    np.random.seed(resolved)

    state: dict[str, Any] = {
        "python_random": resolved,
        "numpy_global": resolved,
        "torch": None,
        "torch_deterministic_algorithms": False,
    }
    try:
        import torch

        torch.manual_seed(resolved)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        state["torch"] = resolved
        state["torch_deterministic_algorithms"] = deterministic_torch
    except ImportError:
        pass

    return state
