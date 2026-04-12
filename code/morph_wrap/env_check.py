#!/usr/bin/env python3
"""
Lightweight checks before launching finetune on a compute node.

Run inside the **same** conda/env you use for MORPH (needs PyTorch).

Example (Slurm prolog)::

    PYTHONPATH=code python -m morph_wrap.env_check --require-cuda || exit 1
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="HPC / env sanity checks for MORPH sweeps")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit 1 if torch.cuda.is_available() is False",
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=0.0,
        help="If >0 and CUDA visible, warn if free memory on device 0 is below this (GB)",
    )
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("env_check: PyTorch not importable (activate MORPH conda env).", file=sys.stderr)
        return 1

    print(f"env_check: torch {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"env_check: cuda.is_available = {cuda_ok}")
    if cuda_ok:
        n = torch.cuda.device_count()
        print(f"env_check: torch.cuda.device_count() = {n}")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            print(f"env_check: logical device {i}: {name}")
        if n > 0:
            cur = torch.cuda.current_device()
            print(f"env_check: torch.cuda.current_device() = {cur}")
        if args.min_free_gb > 0 and n > 0:
            try:
                dev = torch.cuda.current_device()
                free, _total = torch.cuda.mem_get_info(dev)
                free_gb = free / (1024**3)
                print(f"env_check: cuda:{dev} free_mem_gb ≈ {free_gb:.2f}")
                if free_gb < args.min_free_gb:
                    print(
                        f"env_check: WARNING free_mem_gb {free_gb:.2f} < {args.min_free_gb}",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"env_check: could not read mem_get_info: {e}", file=sys.stderr)

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis:
        print(f"env_check: CUDA_VISIBLE_DEVICES={vis!r}")

    if args.require_cuda and not cuda_ok:
        print("env_check: FAIL --require-cuda", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
