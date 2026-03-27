#!/usr/bin/env python3
"""
Visualization utilities for the shallow-water MORPH pipeline.

Supports three sources of 2-D water-height sequences:

  1. **HDF5 groups** (original raw data, CLI ``--h5 --group``)
  2. **NumPy arrays** loaded from ``.npy`` files (CLI ``--npy``)
  3. **In-process numpy arrays** passed to the library functions directly

Library functions (importable from other modules):
  - ``render_sequence_gif``  — animate a ``(T, H, W)`` sequence to GIF
  - ``plot_diff_frame``      — actual / predicted / diff triptych
  - ``plot_sequence_grid``   — multi-column evolution grid (actual vs predicted)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    """Ensure a frame is 2-D ``(H, W)``."""
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D frame after squeeze, got shape {arr.shape}")
    return arr


def _global_range(seq: np.ndarray) -> tuple[float, float]:
    return float(np.nanmin(seq)), float(np.nanmax(seq))

# ---------------------------------------------------------------------------
# 1) GIF of a (T, H, W) sequence
# ---------------------------------------------------------------------------

def render_sequence_gif(
    seq: np.ndarray,
    out_path: Path | str,
    *,
    title_prefix: str = "water height",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    fps: int = 10,
    dpi: int = 150,
) -> Path:
    """Render every frame of *seq* ``(T, H, W)`` into a GIF at *out_path*."""
    from PIL import Image
    import tempfile, shutil

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 4 and seq.shape[-1] == 1:
        seq = seq[..., 0]
    assert seq.ndim == 3, f"Expected (T, H, W), got {seq.shape}"

    if vmin is None or vmax is None:
        lo, hi = _global_range(seq)
        vmin = vmin if vmin is not None else lo
        vmax = vmax if vmax is not None else hi

    tmp_dir = Path(tempfile.mkdtemp())
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(seq[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="water height")
    title = ax.set_title(f"{title_prefix}, t=0")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()

    frame_paths: list[Path] = []
    for t in range(seq.shape[0]):
        im.set_data(seq[t])
        title.set_text(f"{title_prefix}, t={t}")
        fp = tmp_dir / f"f_{t:04d}.png"
        fig.savefig(fp, dpi=dpi)
        frame_paths.append(fp)
    plt.close(fig)

    images = []
    for p in frame_paths:
        with Image.open(p) as img:
            images.append(img.convert("P", palette=Image.Palette.ADAPTIVE))

    duration_ms = int(1000 / max(1, fps))
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0)
    shutil.rmtree(tmp_dir)
    return out_path

# ---------------------------------------------------------------------------
# 2) Single-frame diff triptych
# ---------------------------------------------------------------------------

def plot_diff_frame(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    t_idx: int = 0,
    label: str = "",
    cmap: str = "viridis",
    diff_cmap: str = "coolwarm",
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> None:
    """Plot actual | predicted | signed-diff for one ``(H, W)`` frame."""
    actual = _squeeze_hw(actual)
    predicted = _squeeze_hw(predicted)
    diff = predicted - actual

    vmin = min(float(actual.min()), float(predicted.min()))
    vmax = max(float(actual.max()), float(predicted.max()))
    dlim = max(abs(float(diff.min())), abs(float(diff.max()))) or 1.0

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(actual, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title("Actual")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(predicted, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title("Predicted")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, origin="lower", cmap=diff_cmap, vmin=-dlim, vmax=dlim, aspect="equal")
    axes[2].set_title("Diff (pred − actual)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    suptitle = f"t={t_idx}  MAE={mae:.4f}  RMSE={rmse:.4f}"
    if label:
        suptitle = f"{label}  |  {suptitle}"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 3) Sequence evolution grid (actual vs predicted, optional diff row)
# ---------------------------------------------------------------------------

def plot_sequence_grid(
    actual_seq: np.ndarray,
    predicted_seq: np.ndarray,
    *,
    n_cols: int = 8,
    show_diff: bool = True,
    cmap: str = "viridis",
    diff_cmap: str = "coolwarm",
    label: str = "",
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> None:
    """Plot a multi-column grid comparing actual vs predicted over time.

    *actual_seq* and *predicted_seq* are ``(T, H, W)`` (or ``(T, H, W, 1)``).
    A stride is chosen automatically so that *n_cols* frames span the full
    sequence.
    """
    def _prep(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float32)
        if s.ndim == 4 and s.shape[-1] == 1:
            s = s[..., 0]
        return s

    actual_seq = _prep(actual_seq)
    predicted_seq = _prep(predicted_seq)
    T = min(actual_seq.shape[0], predicted_seq.shape[0])
    stride = max(1, T // n_cols)
    indices = list(range(0, T, stride))[:n_cols]
    n = len(indices)

    vmin, vmax = _global_range(np.concatenate([actual_seq, predicted_seq], axis=0))

    n_rows = 3 if show_diff else 2
    fig, axes = plt.subplots(n_rows, n, figsize=(3.2 * n, 3.2 * n_rows))
    if n == 1:
        axes = axes.reshape(n_rows, 1)

    for col, t in enumerate(indices):
        a = actual_seq[t]
        p = predicted_seq[t]

        axes[0, col].imshow(a, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        axes[0, col].set_title(f"t={t}", fontsize=10)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        axes[1, col].imshow(p, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

        if show_diff:
            diff = p - a
            dlim = max(abs(float(diff.min())), abs(float(diff.max()))) or 1.0
            axes[2, col].imshow(diff, origin="lower", cmap=diff_cmap, vmin=-dlim, vmax=dlim, aspect="equal")
            axes[2, col].set_xticks([]); axes[2, col].set_yticks([])

    row_labels = ["Actual", "Predicted"]
    if show_diff:
        row_labels.append("Diff")
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=12, rotation=90, labelpad=10)

    title = f"Sequence grid (stride={stride})"
    if label:
        title = f"{label}  |  {title}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# CLI (backward-compatible with the original HDF5 script + new --npy mode)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    morph_root = Path(__file__).resolve().parents[1]
    repo_root = morph_root.parent
    default_h5 = repo_root / "data" / "shallow_data.h5"
    default_out_dir = morph_root / "out"

    p = argparse.ArgumentParser(
        description="Visualize water-height sequences (HDF5 groups or .npy arrays).",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--h5", type=Path, default=default_h5,
                     help="Path to shallow_data.h5")
    src.add_argument("--npy", type=Path, default=None,
                     help="Path to a .npy array of shape (T,H,W) or (T,H,W,1)")

    p.add_argument("--group", type=str, default="0000",
                   help="HDF5 group key (ignored when --npy is given)")
    p.add_argument("--t", type=int, default=-1,
                   help="Timestep index (-1 = all → GIF)")
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--global-scale", action="store_true")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def _load_npy_sequence(path: Path) -> np.ndarray:
    """Load a .npy and return (T, H, W)."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,H,W) or (T,H,W,1), got {arr.shape}")
    return arr


def main() -> None:
    args = _parse_args()
    morph_root = Path(__file__).resolve().parents[1]
    default_out_dir = morph_root / "out"

    # ── .npy path ───────────────────────────────────────────────────────
    if args.npy is not None:
        seq = _load_npy_sequence(args.npy)
        label = args.npy.stem

        if args.t >= 0:
            frame = seq[args.t]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(frame, origin="lower", cmap=args.cmap,
                           vmin=args.vmin, vmax=args.vmax, aspect="equal")
            fig.colorbar(im, ax=ax, label="water height")
            ax.set_title(f"{label}, t={args.t}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            out_path = args.out or default_out_dir / f"{label}_t{args.t}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=args.dpi)
            plt.close(fig)
            print(f"Saved: {out_path}")
        else:
            out_path = args.out or default_out_dir / f"{label}.gif"
            render_sequence_gif(
                seq, out_path,
                title_prefix=label, cmap=args.cmap,
                vmin=args.vmin, vmax=args.vmax, fps=args.fps, dpi=args.dpi,
            )
            print(f"Saved GIF: {out_path}")
        return

    # ── HDF5 path (original behaviour) ─────────────────────────────────
    try:
        import h5py
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `h5py`. Install with: pip install h5py"
        ) from e

    if not args.h5.exists():
        raise FileNotFoundError(f"HDF5 file not found: {args.h5}")

    with h5py.File(args.h5, "r") as f:
        if args.group not in f:
            raise KeyError(f"Group '{args.group}' not found. Example keys: {list(f.keys())[:5]}")
        dset = f[args.group]["data"]
        n_t = int(dset.shape[0])

        if args.t >= 0:
            frame = _squeeze_hw(np.asarray(dset[args.t]))
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(frame, origin="lower", cmap=args.cmap,
                           vmin=args.vmin, vmax=args.vmax, aspect="equal")
            fig.colorbar(im, ax=ax, label="water height")
            ax.set_title(f"group={args.group}, t={args.t}")
            fig.tight_layout()
            out_path = args.out or default_out_dir / f"frame_{args.group}_t{args.t}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=args.dpi)
            if not args.no_show:
                plt.show()
            plt.close(fig)
        else:
            seq = np.stack([_squeeze_hw(np.asarray(dset[t])) for t in range(n_t)])
            out_path = args.out or default_out_dir / f"water_height_{args.h5.stem}_group_{args.group}.gif"
            render_sequence_gif(
                seq, out_path,
                title_prefix=f"group={args.group}", cmap=args.cmap,
                vmin=args.vmin, vmax=args.vmax, fps=args.fps, dpi=args.dpi,
            )
            print(f"Saved GIF: {out_path}")


if __name__ == "__main__":
    main()
