"""Resolve MORPH ``--device_idx`` from PyTorch CUDA visibility (e.g. after Slurm sets ``CUDA_VISIBLE_DEVICES``)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


def resolve_training_device_index(explicit: Optional[int]) -> Tuple[int, Dict[str, Any]]:
    """
    Return ``(device_idx, info)`` for ``finetune_MORPH.py --device_idx``.

    * ``explicit is not None``: use that index (validate against ``device_count()`` when possible).
    * Auto mode (``explicit is None``): ``MORPH_DEVICE_IDX`` if set, else
      ``torch.cuda.current_device()`` when CUDA is available and ``device_count() > 0``, else ``0``.

    After Slurm narrows visible GPUs, ``device_count()`` is typically 1 and
    ``current_device()`` is ``0`` — that is the correct index for MORPH.
    """
    info: Dict[str, Any] = {}

    if explicit is not None:
        idx = int(explicit)
        info["source"] = "cli"
        _validate_and_fill_torch_stats(idx, info)
        return idx, info

    env_raw = os.environ.get("MORPH_DEVICE_IDX")
    if env_raw is not None and str(env_raw).strip() != "":
        idx = int(env_raw)
        info["source"] = "MORPH_DEVICE_IDX"
        _validate_and_fill_torch_stats(idx, info)
        return idx, info

    try:
        import torch
    except ImportError:
        info["source"] = "no_torch"
        return 0, info

    if not torch.cuda.is_available():
        info["source"] = "cpu"
        return 0, info

    n = torch.cuda.device_count()
    info["torch_device_count"] = n
    if n <= 0:
        info["source"] = "cuda_no_devices"
        return 0, info

    cur = torch.cuda.current_device()
    info["torch_current_device"] = cur
    info["source"] = "torch.cuda.current_device"

    if cur < 0 or cur >= n:
        idx = 0
        info["note"] = "current_device out of range; using 0"
    else:
        idx = cur

    _validate_and_fill_torch_stats(idx, info)
    return idx, info


def _validate_and_fill_torch_stats(idx: int, info: Dict[str, Any]) -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    n = torch.cuda.device_count()
    info.setdefault("torch_device_count", n)
    if n > 0 and (idx < 0 or idx >= n):
        info["warning"] = f"device_idx {idx} not in [0, {n - 1}]"


def format_device_resolution_log(idx: int, info: Dict[str, Any], *, explicit_cli: bool) -> str:
    parts = [f"morph_wrap: MORPH --device_idx={idx}"]
    src = info.get("source")
    if src:
        parts.append(f"source={src}")
    if "torch_device_count" in info:
        parts.append(f"torch.cuda.device_count()={info['torch_device_count']}")
    if "torch_current_device" in info:
        parts.append(f"torch.cuda.current_device()={info['torch_current_device']}")
    if "note" in info:
        parts.append(f"({info['note']})")
    if "warning" in info:
        parts.append(f"WARNING:{info['warning']}")
    if explicit_cli:
        parts.append("(explicit --device-index)")
    return " ".join(parts)
