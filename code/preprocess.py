#!/usr/bin/env python3
"""
Prepare raw DARUS-style HDF5 files for MORPH finetuning (this repo's sweep).

Mirrors the BE1D, shallow-water 2D, and DR2D sections of
``MORPH/scripts/data_normalization_revin.py`` by calling the same
``split_and_save_h5``, dataloader, and ``RevIN`` code paths.

Expected raw layouts match ``specs/plan.md``:

* **BE1D** — dataset ``tensor`` with shape ``(N, T, W)`` e.g. ``(10000, 201, 1024)``.
* **SW** — groups ``0000`` … each with dataset ``data`` shaped ``(T, H, W, C)`` e.g.
  ``(101, 128, 128, 1)``.
* **DR2D** (diffusion–reaction 2D) — same HDF5 layout as SW; ``data`` shaped
  ``(101, 128, 128, 2)``.

Usage (from repo root, MORPH conda env active)::

    python code/preprocess.py --morph-root MORPH --be1d raw/1D.hdf5 --sw raw/2D_SW.hdf5 \\
        --dr2d raw/2D_DR.hdf5

Outputs under ``<morph-root>/datasets/normalized_revin/{1dbe_pdebench,2dSW_pdebench,DR2d_data_pdebench}/``
with ``train/``, ``val/``, ``test/``, and RevIN statistics under ``<morph-root>/data/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Expected shapes from specs/plan.md (DARUS); N may vary per download.
# ---------------------------------------------------------------------------
_EXPECTED = {
    "be1d": {
        "tensor_tail": (201, 1024),  # (T, W)
        "describe": "dataset 'tensor' (N, 201, 1024) float32",
    },
    "sw": {
        "data_tail": (101, 128, 128, 1),  # (T, H, W, C)
        "describe": "group '####/data' (101, 128, 128, 1) float32",
    },
    "dr2d": {
        "data_tail": (101, 128, 128, 2),  # (T, H, W, C)
        "describe": "group '####/data' (101, 128, 128, 2) float32",
    },
}


def _repo_default_morph_root() -> Path:
    return Path(__file__).resolve().parents[1] / "MORPH"


def _setup_morph_imports(morph_root: Path) -> Path:
    root = morph_root.resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def _validate_be1d(path: Path, strict: bool) -> None:
    exp = _EXPECTED["be1d"]["tensor_tail"]
    with h5py.File(path, "r") as f:
        if "tensor" not in f:
            msg = f"{path}: missing dataset 'tensor'"
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}", file=sys.stderr)
            return
        sh = f["tensor"].shape
        if len(sh) != 3 or sh[1:] != exp:
            msg = f"{path}: tensor shape {sh}, expected (N, {exp[0]}, {exp[1]})"
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}", file=sys.stderr)


def _validate_sw(path: Path, strict: bool) -> None:
    exp = _EXPECTED["sw"]["data_tail"]
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if not k.startswith("."))
        if not keys:
            msg = f"{path}: no top-level groups"
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}", file=sys.stderr)
            return
        first = keys[0]
        if "data" not in f[first]:
            msg = f"{path}: group {first!r} missing 'data'"
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}", file=sys.stderr)
            return
        sh = f[first]["data"].shape
        if sh != exp:
            msg = f"{path}: {first}/data shape {sh}, expected {exp}"
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}", file=sys.stderr)


def _validate_dr2d(path: Path, strict: bool) -> None:
    exp = _EXPECTED["dr2d"]["data_tail"]
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if not k.startswith("."))
        if not keys:
            msg = f"{path}: no top-level groups"
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}", file=sys.stderr)
            return
        first = keys[0]
        if "data" not in f[first]:
            msg = f"{path}: group {first!r} missing 'data'"
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}", file=sys.stderr)
            return
        sh = f[first]["data"].shape
        if sh != exp:
            msg = f"{path}: {first}/data shape {sh}, expected {exp}"
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}", file=sys.stderr)


def _prepare_out_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"{path} already exists; pass --force to replace normalized outputs for this dataset."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_be1d(
    morph_root: Path,
    raw_file: Path,
    *,
    force: bool,
    skip_roundtrip: bool,
) -> None:
    from src.utils.dataloaders.dataloader_be1d import BE1DDataLoader, split_and_save_h5
    from src.utils.normalization import RevIN

    norm_dir = morph_root / "datasets" / "normalized_revin" / "1dbe_pdebench"
    stats_dir = morph_root / "data"
    os.makedirs(stats_dir, exist_ok=True)
    _prepare_out_dir(norm_dir, force)

    rev = RevIN(stats_dir)
    tol = 7e-5

    with tempfile.TemporaryDirectory(prefix="morph_be1d_", dir=morph_root / "datasets") as td:
        work = Path(td)
        staged = work / raw_file.name
        shutil.copy2(raw_file, staged)

        split_and_save_h5(
            raw_h5_loadpath=os.fspath(work),
            savepath=os.fspath(work),
            selected_idx=0,
            dataset_name="be1d",
            train_frac=0.8,
            rand=True,
        )

        loader = BE1DDataLoader(data_path=os.fspath(work), dataset_name="BE1d")
        train, val = loader.split_train(selected_idx=0)
        test = loader.split_test(selected_idx=0)
        dataset = np.concatenate((train, val, test), axis=0)
        del train, val, test

        dataset = dataset.transpose(0, 1, 6, 5, 2, 3, 4)
        rev.compute_stats(dataset, prefix="stats_be1d")
        dataset_n = rev.normalize(dataset, prefix="stats_be1d")

        if not skip_roundtrip:
            recovered = rev.denormalize(dataset_n, prefix="stats_be1d")
            max_err = float(np.max(np.abs(recovered - dataset)))
            del recovered
            assert max_err < tol, f"BE1D RevIN round-trip max error {max_err} >= {tol}"

        dataset_sq = dataset_n.transpose(0, 1, 4, 5, 6, 3, 2)
        dataset_sq = np.squeeze(dataset_sq, axis=(2, 3, 5, 6))

        raw_top = [f for f in os.listdir(work) if f.endswith(".h5") or f.endswith(".hdf5")]
        filename = sorted(raw_top)[0]
        out_path = norm_dir / filename
        with h5py.File(out_path, "w") as f_out:
            f_out.create_dataset("tensor", data=dataset_sq, compression="lzf")

        split_and_save_h5(
            raw_h5_loadpath=os.fspath(norm_dir),
            savepath=os.fspath(norm_dir),
            dataset_name="be1d",
            selected_idx=0,
            train_frac=0.8,
            rand=False,
        )
        if out_path.is_file():
            out_path.unlink()


def _run_sw(
    morph_root: Path,
    raw_file: Path,
    *,
    force: bool,
    skip_roundtrip: bool,
) -> None:
    from src.utils.dataloaders.dataloader_sw2d import SW2dDataLoader, split_and_save_h5
    from src.utils.normalization import RevIN

    norm_dir = morph_root / "datasets" / "normalized_revin" / "2dSW_pdebench"
    stats_dir = morph_root / "data"
    os.makedirs(stats_dir, exist_ok=True)
    _prepare_out_dir(norm_dir, force)

    rev = RevIN(stats_dir)
    tol = 1e-5

    with tempfile.TemporaryDirectory(prefix="morph_sw_", dir=morph_root / "datasets") as td:
        work = Path(td)
        shutil.copy2(raw_file, work / raw_file.name)

        split_and_save_h5(
            raw_h5_loadpath=os.fspath(work),
            savepath=os.fspath(work),
            dataset_name="SW2d",
            train_frac=0.8,
            rand=True,
        )

        loader = SW2dDataLoader(os.fspath(work))
        train, val = loader.split_train()
        test = loader.split_test()
        dataset = np.concatenate((train, val, test), axis=0)
        del train, val, test

        dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)
        rev.compute_stats(dataset_tr, prefix="stats_sw")
        dataset_n = rev.normalize(dataset_tr, prefix="stats_sw")

        if not skip_roundtrip:
            recovered = rev.denormalize(dataset_n, prefix="stats_sw")
            max_err = max(
                float(np.max(np.abs(recovered[i] - dataset_tr[i]))) for i in range(recovered.shape[0])
            )
            del recovered
            assert max_err < tol, f"SW RevIN round-trip max error {max_err} >= {tol}"

        dataset_sq = dataset_n.transpose(0, 1, 4, 5, 6, 3, 2)
        dataset_sq = np.squeeze(dataset_sq, axis=2)[:, :, :, :, 0, :]

        raw_top = [f for f in os.listdir(work) if f.endswith(".h5") or f.endswith(".hdf5")]
        filename = sorted(raw_top)[0]
        out_path = norm_dir / filename
        with h5py.File(out_path, "w") as f_out:
            for i in range(dataset_sq.shape[0]):
                grp = f_out.create_group(f"{i:04d}")
                grp.create_dataset("data", data=dataset_sq[i], compression="lzf")

        split_and_save_h5(
            raw_h5_loadpath=os.fspath(norm_dir),
            savepath=os.fspath(norm_dir),
            dataset_name="SW2d",
            train_frac=0.8,
            rand=False,
        )
        if out_path.is_file():
            out_path.unlink()


def _run_dr2d(
    morph_root: Path,
    raw_file: Path,
    *,
    force: bool,
    skip_roundtrip: bool,
) -> None:
    from src.utils.dataloaders.dataloader_dr import DR2DDataLoader, split_and_save_h5
    from src.utils.normalization import RevIN

    norm_dir = morph_root / "datasets" / "normalized_revin" / "DR2d_data_pdebench"
    stats_dir = morph_root / "data"
    os.makedirs(stats_dir, exist_ok=True)
    _prepare_out_dir(norm_dir, force)

    rev = RevIN(stats_dir)
    tol = 1e-5

    with tempfile.TemporaryDirectory(prefix="morph_dr2d_", dir=morph_root / "datasets") as td:
        work = Path(td)
        shutil.copy2(raw_file, work / raw_file.name)

        split_and_save_h5(
            raw_h5_loadpath=os.fspath(work),
            savepath=os.fspath(work),
            dataset_name="DR",
            train_frac=0.8,
            rand=True,
        )

        loader = DR2DDataLoader(os.fspath(work), dataset_name="DR")
        train, val = loader.split_train()
        test = loader.split_test()
        dataset = np.concatenate((train, val, test), axis=0)
        del train, val, test

        dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)
        rev.compute_stats(dataset_tr, prefix="stats_dr2d")
        dataset_n = rev.normalize(dataset_tr, prefix="stats_dr2d")

        if not skip_roundtrip:
            recovered = rev.denormalize(dataset_n, prefix="stats_dr2d")
            max_err = max(
                float(np.max(np.abs(recovered[i] - dataset_tr[i]))) for i in range(recovered.shape[0])
            )
            del recovered
            assert max_err < tol, f"DR2D RevIN round-trip max error {max_err} >= {tol}"

        dataset_sq = dataset_n.transpose(0, 1, 4, 5, 6, 3, 2)
        dataset_sq = np.squeeze(dataset_sq, axis=2)[:, :, :, :, 0, :]

        raw_top = [f for f in os.listdir(work) if f.endswith(".h5") or f.endswith(".hdf5")]
        filename = sorted(raw_top)[0]
        out_path = norm_dir / filename
        with h5py.File(out_path, "w") as f_out:
            for i in range(dataset_sq.shape[0]):
                grp = f_out.create_group(f"{i:04d}")
                grp.create_dataset("data", data=dataset_sq[i], compression="lzf")

        split_and_save_h5(
            raw_h5_loadpath=os.fspath(norm_dir),
            savepath=os.fspath(norm_dir),
            dataset_name="DR",
            train_frac=0.8,
            rand=False,
        )
        if out_path.is_file():
            out_path.unlink()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Raw DARUS HDF5 → MORPH normalized_revin + RevIN stats (BE1D, SW, DR2D)."
    )
    p.add_argument(
        "--morph-root",
        type=Path,
        default=_repo_default_morph_root(),
        help="MORPH clone root (default: ./MORPH next to this repo)",
    )
    p.add_argument("--be1d", type=Path, metavar="FILE", help="Raw BE1D (Burgers) .h5 / .hdf5")
    p.add_argument("--sw", type=Path, metavar="FILE", help="Raw shallow-water 2D .h5 / .hdf5")
    p.add_argument(
        "--dr2d",
        type=Path,
        metavar="FILE",
        help="Raw diffusion–reaction 2D PDEBench .h5 / .hdf5 (see specs/plan.md)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Remove existing normalized_revin output folder for each selected dataset before writing",
    )
    p.add_argument(
        "--strict-shapes",
        action="store_true",
        help="Fail if HDF5 layout does not match specs/plan.md expected shapes",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Only check shapes; do not write outputs",
    )
    p.add_argument(
        "--skip-roundtrip",
        action="store_true",
        help="Skip RevIN denormalize sanity check (faster on huge arrays)",
    )

    args = p.parse_args(argv)
    morph_root = _setup_morph_imports(args.morph_root)

    if not morph_root.is_dir():
        print(f"error: --morph-root not a directory: {morph_root}", file=sys.stderr)
        return 1

    jobs: list[tuple[str, Path, Callable[..., None]]] = []
    if args.be1d is not None:
        jobs.append(("BE1D", args.be1d, _run_be1d))
    if args.sw is not None:
        jobs.append(("SW", args.sw, _run_sw))
    if args.dr2d is not None:
        jobs.append(("DR2D", args.dr2d, _run_dr2d))

    if not jobs:
        p.error("pass at least one of --be1d, --sw, --dr2d")

    (morph_root / "datasets").mkdir(parents=True, exist_ok=True)

    for name, path, runner in jobs:
        path = path.resolve()
        if not path.is_file():
            print(f"error: {name} input not a file: {path}", file=sys.stderr)
            return 1
        suf = path.suffix.lower()
        if suf not in (".h5", ".hdf5"):
            print(f"error: {name} expected .h5 or .hdf5, got {path}", file=sys.stderr)
            return 1

        if name == "BE1D":
            _validate_be1d(path, args.strict_shapes)
        elif name == "SW":
            _validate_sw(path, args.strict_shapes)
        else:
            _validate_dr2d(path, args.strict_shapes)

        if args.validate_only:
            print(f"ok: {name} shape check passed for {path}")
            continue

        print(f"→ preprocessing {name} from {path}", flush=True)
        runner(
            morph_root,
            path,
            force=args.force,
            skip_roundtrip=args.skip_roundtrip,
        )
        print(f"→ done {name}", flush=True)

    if args.validate_only:
        return 0

    print(f"→ RevIN stats under {morph_root / 'data'}", flush=True)
    print(
        "→ normalized splits under "
        f"{morph_root / 'datasets' / 'normalized_revin'}/{{1dbe_pdebench,2dSW_pdebench,DR2d_data_pdebench}}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
