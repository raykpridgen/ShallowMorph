"""
Shared utilities for the shallow-water MORPH fine-tuning pipeline.

Provides path resolution, data loading/validation, group-wise splitting,
and constants shared across preprocessing, training, evaluation, and
visualization modules.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # .../morph
MORPH_ROOT = PROJECT_ROOT / "MORPH"
DATASETS_DIR = MORPH_ROOT / "datasets"
SW2D_DIR = DATASETS_DIR / "SW2d"
PROCESSED_DIR = DATASETS_DIR                                 # final .npy files land here
MODELS_DIR = MORPH_ROOT / "models"
DATA_DIR = PROJECT_ROOT.parent / "data"                      # ../ml/data (raw HDF5 etc.)
OUT_DIR = PROJECT_ROOT / "out"

# ---------------------------------------------------------------------------
# Dataset constants (shallow-water 2D)
# ---------------------------------------------------------------------------

SW_DATASET_NAME = "shallow_water"
SW_DATASET_SPECS = {"F": 1, "C": 1, "D": 1, "H": 128, "W": 128}
SW_EXPECTED_SHAPE_PER_TRAJ = (101, 128, 128, 1)  # (T, H, W, 1)

DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(name)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_trajectory_files(sw2d_dir: Path = SW2D_DIR) -> list[Path]:
    """Return sorted list of ``????__data.npy`` files in *sw2d_dir*."""
    files = sorted(sw2d_dir.glob("????__data.npy"))
    if not files:
        raise FileNotFoundError(
            f"No trajectory files (????__data.npy) found in {sw2d_dir}"
        )
    return files


def load_trajectories(
    files: Sequence[Path],
    *,
    expected_shape: tuple[int, ...] | None = SW_EXPECTED_SHAPE_PER_TRAJ,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Load and stack per-trajectory ``.npy`` files into ``(N, T, H, W, 1)``.

    Performs shape-consistency and finite-value checks as required by the
    preprocessing spec.  Casts to ``float32``.
    """
    log = logger or get_logger("load_trajectories")
    arrays: list[np.ndarray] = []

    for i, fp in enumerate(files):
        arr = np.load(fp).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., np.newaxis]
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(
                f"Trajectory {fp.name} has shape {arr.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"Trajectory {fp.name} contains non-finite values")
        arrays.append(arr)

    dataset = np.stack(arrays, axis=0)  # (N, T, H, W, 1)
    log.info(
        "Loaded %d trajectories -> shape %s, dtype %s",
        len(files), dataset.shape, dataset.dtype,
    )
    return dataset


def load_trajectories_from_h5(
    h5_path: Path,
    *,
    expected_spatial: tuple[int, ...] | None = None,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Load all groups from a raw HDF5 shallow-water file.

    Each HDF5 group key (e.g. ``'0000'``) is expected to contain a ``data``
    dataset shaped ``(T, X, Y, 1)`` or ``(T, X, Y)``.
    Returns ``(N, T, H, W, 1)`` as ``float32``.
    """
    import h5py

    log = logger or get_logger("load_h5")
    arrays: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
        log.info("HDF5 groups: %d (first=%s, last=%s)", len(keys), keys[0], keys[-1])

        for key in keys:
            grp = f[key]
            if "data" not in grp:
                raise KeyError(f"Group '{key}' missing 'data' dataset")
            arr = np.asarray(grp["data"], dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., np.newaxis]
            if expected_spatial is not None:
                if arr.shape[1:3] != expected_spatial:
                    raise ValueError(
                        f"Group '{key}' spatial dims {arr.shape[1:3]} != {expected_spatial}"
                    )
            if not np.isfinite(arr).all():
                raise ValueError(f"Group '{key}' contains non-finite values")
            arrays.append(arr)

    ref_shape = arrays[0].shape
    min_T = min(a.shape[0] for a in arrays)
    if any(a.shape[0] != ref_shape[0] for a in arrays):
        log.warning(
            "Inconsistent T across groups (min=%d, max=%d). Trimming to min.",
            min_T, max(a.shape[0] for a in arrays),
        )
        arrays = [a[:min_T] for a in arrays]

    dataset = np.stack(arrays, axis=0)
    log.info(
        "Loaded %d groups from HDF5 -> shape %s, dtype %s",
        len(keys), dataset.shape, dataset.dtype,
    )
    return dataset

# ---------------------------------------------------------------------------
# Group-wise splitting
# ---------------------------------------------------------------------------

def group_split_indices(
    n: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, np.ndarray]:
    """Return disjoint train/val/test index arrays for *n* groups.

    Uses a fixed RNG seed for reproducibility.  The ratios are
    ``(train, val, test)`` and must sum to 1.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1, got {sum(ratios)}"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))

    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val :])

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def apply_split(
    dataset: np.ndarray,
    indices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Index *dataset* along axis-0 with the split *indices*."""
    return {name: dataset[idx] for name, idx in indices.items()}

# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def save_manifest(
    path: Path,
    *,
    n: int,
    t: int,
    h: int,
    w: int,
    split_seed: int,
    split_ratios: tuple[float, ...],
    split_sizes: dict[str, int],
    dataset_specs: dict[str, int],
    files: dict[str, str],
) -> None:
    """Write a small JSON manifest recording preprocessing metadata."""
    doc = {
        "N": n,
        "T": t,
        "H": h,
        "W": w,
        "split_seed": split_seed,
        "split_ratios": list(split_ratios),
        "split_sizes": split_sizes,
        "dataset_specs": dataset_specs,
        "files": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())

# ---------------------------------------------------------------------------
# MORPH model constants
# ---------------------------------------------------------------------------

MORPH_MODELS: dict[str, list[int]] = {
    #             filters, dim,  heads, depth, mlp_dim
    "Ti": [8, 256,  4,  4, 1024],
    "S":  [8, 512,  8,  4, 2048],
    "M":  [8, 768, 12,  8, 3072],
    "L":  [8, 1024, 16, 16, 4096],
}

MORPH_FM_FILENAMES: dict[str, str] = {
    "Ti": "morph-Ti-FM-max_ar1_ep225.pth",
    "S":  "morph-S-FM-max_ar1_ep225.pth",
    "M":  "morph-M-FM-max_ar1_ep290_latestbatch.pth",
    "L":  "morph-L-FM-max_ar16_ep189_latestbatch.pth",
}

RESULTS_DIR = MORPH_ROOT / "experiments" / "results" / "test"
NORM_STATS_DIR = MORPH_ROOT / "data"

# ---------------------------------------------------------------------------
# MORPH import helpers
# ---------------------------------------------------------------------------

def import_morph_modules() -> dict:
    """Import MORPH internal modules with sys.path isolation.

    Our ``src/utils.py`` shadows MORPH's ``src/utils/`` package when both
    ``src/`` directories live on ``sys.path`` as namespace packages.  This
    helper temporarily reconfigures ``sys.path`` so that ``from src.utils.xxx``
    resolves to MORPH's tree, imports everything the pipeline needs, then
    restores the original path.

    Returns a dict mapping short names to imported classes/modules.
    """
    import sys

    morph_str = str(MORPH_ROOT)
    script_dir = str(Path(__file__).resolve().parent)
    project_root = str(PROJECT_ROOT)

    saved_path = sys.path[:]
    stale_keys = [k for k in sys.modules if k == "src" or k.startswith("src.")]
    saved_modules = {k: sys.modules.pop(k) for k in stale_keys}

    conflict_dirs = {
        script_dir,
        project_root,
        os.path.abspath(""),
        os.path.abspath("."),
    }
    sys.path = [morph_str] + [
        p for p in saved_path
        if os.path.abspath(p) not in conflict_dirs and p not in ("", ".")
    ]

    try:
        from src.utils.data_preparation_fast import FastARDataPreparer
        from src.utils.device_manager import DeviceManager
        from src.utils.metrics_3d import Metrics3DCalculator
        from src.utils.normalization import RevIN
        from src.utils.select_fine_tuning_parameters import SelectFineTuningParameters
        from src.utils.trainers import Trainer
        from src.utils.uptf7 import UPTF7
        from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
    finally:
        sys.path = saved_path
        for k in list(sys.modules):
            if k == "src" or k.startswith("src."):
                del sys.modules[k]
        sys.modules.update(saved_modules)

    return {
        "FastARDataPreparer": FastARDataPreparer,
        "DeviceManager": DeviceManager,
        "Metrics3DCalculator": Metrics3DCalculator,
        "RevIN": RevIN,
        "SelectFineTuningParameters": SelectFineTuningParameters,
        "Trainer": Trainer,
        "UPTF7": UPTF7,
        "ViT3DRegression": ViT3DRegression,
    }


def load_split_npy(
    name: str = SW_DATASET_NAME,
    datasets_dir: Path = PROCESSED_DIR,
) -> dict[str, np.ndarray]:
    """Load the train/val/test ``.npy`` splits produced by ``preprocess.py``.

    Returns ``{"train": ..., "val": ..., "test": ...}`` each shaped
    ``(N_split, T, H, W, 1)`` as ``float32``.
    """
    splits: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        p = datasets_dir / f"{name}_{split}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}. Run preprocess.py first."
            )
        splits[split] = np.load(p).astype(np.float32)
    return splits
