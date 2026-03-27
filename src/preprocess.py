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
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATASETS_DIR,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_SEED,
    OUT_DIR,
    PROCESSED_DIR,
    SW2D_DIR,
    SW_DATASET_NAME,
    SW_DATASET_SPECS,
    SW_EXPECTED_SHAPE_PER_TRAJ,
    apply_split,
    discover_trajectory_files,
    get_logger,
    group_split_indices,
    load_trajectories,
    load_trajectories_from_h5,
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

    # ── Load ────────────────────────────────────────────────────────────
    if args.h5 is not None:
        log.info("Loading from HDF5: %s", args.h5)
        dataset = load_trajectories_from_h5(args.h5, logger=log)
    else:
        log.info("Loading from folder: %s", args.sw2d_dir)
        files = discover_trajectory_files(args.sw2d_dir)
        dataset = load_trajectories(files, logger=log)

    N, T, H, W, C = dataset.shape
    log.info("Dataset: N=%d, T=%d, H=%d, W=%d, C=%d", N, T, H, W, C)

    # ── Validate ────────────────────────────────────────────────────────
    assert dataset.dtype == np.float32, f"Expected float32, got {dataset.dtype}"
    assert np.isfinite(dataset).all(), "Dataset contains non-finite values"
    log.info("Validation passed (float32, all finite)")

    # ── Group-wise split ────────────────────────────────────────────────
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        log.error("Split ratios sum to %.4f, must equal 1.0", total)
        sys.exit(1)

    indices = group_split_indices(N, ratios=ratios, seed=args.seed)
    splits = apply_split(dataset, indices)

    for name, arr in splits.items():
        log.info("  %-6s %d trajectories  shape %s", name, arr.shape[0], arr.shape)

    # ── Save splits ─────────────────────────────────────────────────────
    files_written: dict[str, str] = {}

    for split_name, arr in splits.items():
        fname = f"{args.name}_{split_name}.npy"
        fpath = out_dir / fname
        np.save(fpath, arr)
        files_written[split_name] = fname
        log.info("Saved %s  (%s)", fpath, arr.shape)

    if not args.skip_all:
        all_fname = f"{args.name}_all.npy"
        all_path = out_dir / all_fname
        np.save(all_path, dataset)
        files_written["all"] = all_fname
        log.info("Saved %s  (%s)", all_path, dataset.shape)

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
        split_sizes={k: int(v.shape[0]) for k, v in splits.items()},
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
             splits["train"].shape[0], splits["val"].shape[0], splits["test"].shape[0])
    log.info("  Output dir: %s", out_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
