#!/usr/bin/env python3
"""
Preprocess shallow-water trajectories for MORPH fine-tuning.

Two input modes:
  1. **folder** (default) — pack per-trajectory ``????__data.npy`` files from
     ``MORPH/datasets/SW2d/`` into a single array.
  2. **h5** — read groups directly from a raw HDF5 file.

Outputs (written to ``MORPH/datasets/``):
  - ``shallow_water_train.npy``  (N_train, T, H, W, 1)
  - ``shallow_water_val.npy``    (N_val,   T, H, W, 1)
  - ``shallow_water_test.npy``   (N_test,  T, H, W, 1)
  - ``shallow_water_all.npy``    (N,       T, H, W, 1)  — for compatibility
     with ``finetune_MORPH_general.py``
  - ``shallow_water_manifest.json`` — records N, T, H, W, split info,
     and the ``dataset_specs`` to pass to the training script.

No normalization is applied; RevIN stats are computed at training time.

Packing uses **one pass** over trajectories (each file read once). Train/val/test
arrays are filled directly, so RAM does not spike with a temporary “list + stack”
of the full dataset. The optional ``*_all.npy`` is written via ``numpy`` memmap
row-by-row so peak resident memory stays close to one full dataset worth of
floats (the three split buffers), not two copies plus splits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_SEED,
    PROCESSED_DIR,
    SW2D_DIR,
    SW_DATASET_NAME,
    SW_DATASET_SPECS,
    discover_trajectory_files,
    get_logger,
    group_split_indices,
    pack_splits_streaming_from_files,
    pack_splits_streaming_from_h5,
    save_manifest,
)

log = get_logger("preprocess")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pack & split shallow-water data for MORPH fine-tuning.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--sw2d-dir",
        type=Path,
        default=SW2D_DIR,
        help="Directory of per-trajectory ????__data.npy files (default: %(default)s)",
    )
    src.add_argument(
        "--h5",
        type=Path,
        default=None,
        help="Path to raw HDF5 shallow-water file (alternative to --sw2d-dir)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory to write output .npy / manifest (default: %(default)s)",
    )
    p.add_argument(
        "--name",
        type=str,
        default=SW_DATASET_NAME,
        help="Base filename prefix for outputs (default: %(default)s)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="RNG seed for train/val/test split (default: %(default)s)",
    )
    p.add_argument(
        "--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0],
    )
    p.add_argument(
        "--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1],
    )
    p.add_argument(
        "--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2],
    )
    p.add_argument(
        "--skip-all",
        action="store_true",
        help="Do not write the combined *_all.npy (saves disk space).",
    )
    return p


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def preprocess(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover size + split indices (no full (N,...) tensor yet) ───────
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        log.error("Split ratios sum to %.4f, must equal 1.0", total)
        sys.exit(1)

    if args.h5 is not None:
        log.info("Packing from HDF5 (streaming): %s", args.h5)
        import h5py

        with h5py.File(args.h5, "r") as _f:
            n_traj = len(_f.keys())
        if n_traj == 0:
            log.error("HDF5 file has no groups.")
            sys.exit(1)
    else:
        log.info("Packing from folder (streaming): %s", args.sw2d_dir)
        files = discover_trajectory_files(args.sw2d_dir)
        n_traj = len(files)

    log.info("Trajectories: N=%d", n_traj)
    indices = group_split_indices(n_traj, ratios=ratios, seed=args.seed)

    all_npy_path: Path | None = None
    if not args.skip_all:
        all_npy_path = out_dir / f"{args.name}_all.npy"

    if args.h5 is not None:
        splits = pack_splits_streaming_from_h5(
            args.h5,
            indices,
            all_npy_path=all_npy_path,
            logger=log,
        )
    else:
        splits = pack_splits_streaming_from_files(
            files,
            indices,
            all_npy_path=all_npy_path,
            logger=log,
        )

    probe = splits["train"]
    T, H, W, C = probe.shape[1], probe.shape[2], probe.shape[3], probe.shape[4]
    N = n_traj
    log.info("Dataset: N=%d, T=%d, H=%d, W=%d, C=%d", N, T, H, W, C)
    log.info("Validation passed (per-trajectory checks in packer; float32)")

    for name, arr in splits.items():
        log.info("  %-6s %d trajectories  shape %s", name, arr.shape[0], arr.shape)

    split_sizes = {k: int(v.shape[0]) for k, v in splits.items()}

    # ── Save splits ─────────────────────────────────────────────────────
    files_written: dict[str, str] = {}

    for split_name, arr in splits.items():
        fname = f"{args.name}_{split_name}.npy"
        fpath = out_dir / fname
        np.save(fpath, arr)
        files_written[split_name] = fname
        log.info("Saved %s  (%s)", fpath, arr.shape)

    if all_npy_path is not None:
        files_written["all"] = f"{args.name}_all.npy"
        log.info("Saved %s  (written via memmap during pack)", all_npy_path)

    del splits

    # ── Manifest ────────────────────────────────────────────────────────
    manifest_path = out_dir / f"{args.name}_manifest.json"
    save_manifest(
        manifest_path,
        n=N,
        t=T,
        h=H,
        w=W,
        split_seed=args.seed,
        split_ratios=ratios,
        split_sizes=split_sizes,
        dataset_specs=SW_DATASET_SPECS,
        files=files_written,
    )
    log.info("Manifest: %s", manifest_path)

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("── Preprocessing complete (%.1fs) ──", elapsed)
    log.info("  dataset_specs for training: --dataset_specs %d %d %d %d %d",
             SW_DATASET_SPECS["F"], SW_DATASET_SPECS["C"],
             SW_DATASET_SPECS["D"], SW_DATASET_SPECS["H"], SW_DATASET_SPECS["W"])
    log.info("  Splits: train=%d  val=%d  test=%d",
             split_sizes["train"], split_sizes["val"], split_sizes["test"])
    log.info("  Output dir: %s", out_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
