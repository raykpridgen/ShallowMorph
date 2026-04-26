#!/usr/bin/env python3
"""
Standalone rollout evaluator for MORPH checkpoints (single-file pipeline).

Design goals from specs/eval.md:
- n-step rollout on BE1D / SW / DR2D (and other MORPH datasets)
- per-step [actual, predicted, diff] frames + GIF
- overview grid(s): row 1 actual subset, row 2 predicted subset
- per-timestep MSE and SSIM
- metric plot showing progression/accumulation over time

Notes:
- Uses MORPH model + dataloader paradigms (same core tensor shapes).
- Does NOT use MORPH visualization utilities.
- Works on CPU and GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

matplotlib.use("Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # optional dependency
    skimage_ssim = None

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parent
MORPH_ROOT = REPO_ROOT / "MORPH"
sys.path.insert(0, str(MORPH_ROOT))

from config.data_config import DataConfig  # noqa: E402
from src.utils.dataloaders.dataloaderchaos import DataloaderChaos  # noqa: E402
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression  # noqa: E402

MORPH_MODELS: Dict[str, List[int]] = {
    "Ti": [8, 256, 4, 4, 1024],
    "S": [8, 512, 8, 4, 2048],
    "M": [8, 768, 12, 8, 3072],
    "L": [8, 1024, 16, 16, 4096],
}

_DATASETS = [
    "DR2d_data_pdebench",
    "MHD3d_data_thewell",
    "1dcfd_pdebench",
    "2dSW_pdebench",
    "2dcfd_ic_pdebench",
    "3dcfd_pdebench",
    "1ddr_pdebench",
    "2dcfd_pdebench",
    "3dcfd_turb_pdebench",
    "1dbe_pdebench",
    "2dgrayscottdr_thewell",
    "3dturbgravitycool_thewell",
    "2dFNS_KF_pdegym",
]


@dataclass
class EvalConfig:
    checkpoint_path: Path
    dataset: str
    timesteps: int
    sample_idx: int
    model_size: str
    model_choice: str
    ar_context: int
    device: torch.device
    output_root: Path
    gif_fps: int
    subset_frames: int
    heads_xa: int
    dropout: float
    emb_dropout: float
    dataset_path: Path


def default_dataset_paths() -> Dict[str, Path]:
    base = MORPH_ROOT / "datasets" / "normalized_revin"
    return {
        "MHD": base / _DATASETS[1],
        "DR": base / _DATASETS[0],
        "CFD1D": base / _DATASETS[2],
        "CFD2D-IC": base / _DATASETS[4],
        "CFD3D": base / _DATASETS[5],
        "SW": base / _DATASETS[3],
        "DR1D": base / _DATASETS[6],
        "CFD2D": base / _DATASETS[7],
        "CFD3D-TURB": base / _DATASETS[8],
        "BE1D": base / _DATASETS[9],
        "GSDR2D": base / _DATASETS[10],
        "TGC3D": base / _DATASETS[11],
        "FNS_KF_2D": base / _DATASETS[12],
        "DR2D": base / _DATASETS[0],
    }


def parse_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(spec)
    raise ValueError(f"Unsupported --device value: {spec}")


def parse_checkpoint(model_path: str, model_file: str | None) -> Path:
    p = Path(model_path)
    if model_file:
        ckpt = p / model_file
    else:
        ckpt = p
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt.resolve()


def load_model(cfg: EvalConfig) -> torch.nn.Module:
    DataConfig(str(MORPH_ROOT), 8)
    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[cfg.model_size]
    if cfg.model_choice != "FM":
        dc = DataConfig(str(MORPH_ROOT), 8)
        patch_size = dc[cfg.dataset]["patch_size"]
        max_patches = dc[cfg.dataset]["max_patches"]
        max_fields = dc[cfg.dataset]["fields"]
        max_components = dc[cfg.dataset]["components"]
    else:
        patch_size = 8
        max_patches = 4096
        max_fields = 3
        max_components = 3

    model = ViT3DRegression(
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        heads_xa=cfg.heads_xa,
        mlp_dim=mlp_dim,
        max_components=max_components,
        conv_filter=filters,
        max_ar=cfg.ar_context,
        max_patches=max_patches,
        max_fields=max_fields,
        dropout=cfg.dropout,
        emb_dropout=cfg.emb_dropout,
        lora_r_attn=0,
        lora_r_mlp=0,
        lora_alpha=None,
        lora_p=0.0,
    ).to(cfg.device)

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing = model.load_state_dict(state, strict=False)
    if missing.unexpected_keys:
        print(f"Warning: unexpected keys (showing up to 10): {missing.unexpected_keys[:10]}")
    model.eval()
    return model


def extract_pred(raw):
    if isinstance(raw, tuple):
        if len(raw) == 0:
            raise TypeError("Model returned empty tuple.")
        return raw[-1]
    return raw


def tensor_to_plot_array(field_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert model field tensor to a plottable array (1D or 2D).
    Handles shapes like (C,D,H,W), (C,H,W), (H,W), (W).
    """
    x = field_tensor.detach().cpu()
    while x.ndim > 2 and 1 in x.shape:
        x = x.squeeze()
    while x.ndim > 2:
        idx = 0 if x.shape[0] <= 8 else x.shape[0] // 2
        x = x[idx]
    if x.ndim == 0:
        x = x.unsqueeze(0)
    return x.numpy()


def safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if skimage_ssim is None:
        return float("nan")
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range <= 0:
        data_range = 1.0
    kwargs = {"data_range": data_range}
    # For small spatial dimensions, skimage default win_size may be too large.
    min_dim = min(a.shape) if a.ndim > 1 else a.shape[0]
    if min_dim < 7:
        win = min_dim if min_dim % 2 == 1 else min_dim - 1
        if win >= 3:
            kwargs["win_size"] = win
    try:
        return float(skimage_ssim(a, b, **kwargs))
    except Exception:
        return float("nan")


def make_dirs(root: Path) -> Dict[str, Path]:
    out = {
        "root": root,
        "frames": root / "frames",
        "single": root / "single",
        "overviews": root / "overviews",
        "gifs": root / "gifs",
        "metrics": root / "metrics",
        "plots": root / "plots",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def align_plot_arrays(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop/align 1D or 2D arrays to a shared shape."""
    if a.ndim == b.ndim == 1:
        n = min(a.shape[0], b.shape[0])
        return a[:n], b[:n]
    if a.ndim == b.ndim == 2:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        return a[:h, :w], b[:h, :w]
    # Fallback: flatten and align by shortest length
    af = a.reshape(-1)
    bf = b.reshape(-1)
    n = min(af.shape[0], bf.shape[0])
    return af[:n], bf[:n]


def save_triptych(true_arr: np.ndarray, pred_arr: np.ndarray, out_path: Path, title: str) -> None:
    diff_arr = np.abs(true_arr - pred_arr)
    vmin = float(min(true_arr.min(), pred_arr.min()))
    vmax = float(max(true_arr.max(), pred_arr.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    if true_arr.ndim == 1:
        x = np.arange(true_arr.shape[0])
        axes[0].plot(x, true_arr)
        axes[0].set_title("Actual")
        axes[0].set_ylim(vmin, vmax)
        axes[0].grid(alpha=0.25)

        axes[1].plot(x, pred_arr)
        axes[1].set_title("Predicted")
        axes[1].set_ylim(vmin, vmax)
        axes[1].grid(alpha=0.25)

        axes[2].plot(x, diff_arr, color="tab:red")
        axes[2].set_title("Absolute Diff")
        axes[2].grid(alpha=0.25)
    else:
        im0 = axes[0].imshow(true_arr, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("Actual")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(pred_arr, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Predicted")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(diff_arr, cmap="hot")
        axes[2].set_title("Absolute Diff")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_single(img: np.ndarray, out_path: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    if img.ndim == 1:
        ax.plot(np.arange(img.shape[0]), img, color="tab:blue")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    else:
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_gif(frames: List[Path], out_gif: Path, fps: int) -> None:
    if not frames:
        return
    imgs = [imageio.imread(p) for p in frames]
    # Matplotlib-exported PNGs can differ by a few pixels across timesteps
    # (e.g., colorbar tick label width). GIF encoders require identical shape.
    max_h = max(img.shape[0] for img in imgs)
    max_w = max(img.shape[1] for img in imgs)
    norm = []
    for img in imgs:
        h, w = img.shape[0], img.shape[1]
        if h == max_h and w == max_w:
            norm.append(img)
            continue
        pad_h = max_h - h
        pad_w = max_w - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if img.ndim == 2:
            padded = np.pad(img, ((top, bottom), (left, right)), mode="edge")
        else:
            padded = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")
        norm.append(padded)
    imageio.mimsave(out_gif, norm, fps=fps)


def evenly_spaced_indices(total: int, want: int) -> List[int]:
    if total <= 0:
        return []
    if want >= total:
        return list(range(total))
    idx = np.linspace(0, total - 1, want).round().astype(int).tolist()
    # preserve order while dropping accidental duplicates
    out = []
    seen = set()
    for i in idx:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def save_overview_grid(
    field_id: int,
    true_by_step: List[np.ndarray],
    pred_by_step: List[np.ndarray],
    chosen_idx: List[int],
    out_path: Path,
) -> None:
    n = len(chosen_idx)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # type: ignore[index]
    for col, t in enumerate(chosen_idx):
        t_img = true_by_step[t]
        p_img = pred_by_step[t]
        t_img, p_img = align_plot_arrays(t_img, p_img)
        vmin = float(min(t_img.min(), p_img.min()))
        vmax = float(max(t_img.max(), p_img.max()))
        if vmax <= vmin:
            vmax = vmin + 1e-8
        if t_img.ndim == 1:
            x = np.arange(t_img.shape[0])
            axes[0, col].plot(x, t_img)
            axes[0, col].set_ylim(vmin, vmax)
            axes[0, col].set_title(f"Actual t={t+1}")
            axes[0, col].grid(alpha=0.25)
            axes[1, col].plot(x, p_img)
            axes[1, col].set_ylim(vmin, vmax)
            axes[1, col].set_title(f"Pred t={t+1}")
            axes[1, col].grid(alpha=0.25)
        else:
            axes[0, col].imshow(t_img, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f"Actual t={t+1}")
            axes[0, col].axis("off")
            axes[1, col].imshow(p_img, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f"Pred t={t+1}")
            axes[1, col].axis("off")
    fig.suptitle(f"Field {field_id} Overview (Actual over Predicted)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(ts_metrics: List[Dict[str, float]], out_path: Path) -> None:
    t = np.array([int(x["timestep"]) for x in ts_metrics], dtype=int)
    mse = np.array([float(x["mse_mean"]) for x in ts_metrics], dtype=float)
    ssim_vals = np.array([float(x["ssim_mean"]) for x in ts_metrics], dtype=float)
    mse_cum = np.cumsum(mse) / np.arange(1, len(mse) + 1)
    # nan-aware cumulative mean for SSIM
    ssim_cum = []
    run = []
    for v in ssim_vals:
        if not np.isnan(v):
            run.append(v)
        ssim_cum.append(float(np.mean(run)) if run else np.nan)
    ssim_cum = np.array(ssim_cum, dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, mse, color="tab:red", label="MSE (step)")
    ax1.plot(t, mse_cum, color="tab:red", linestyle="--", label="MSE (cumulative mean)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("MSE", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(t, ssim_vals, color="tab:blue", label="SSIM (step)")
    ax2.plot(t, ssim_cum, color="tab:blue", linestyle="--", label="SSIM (cumulative mean)")
    ax2.set_ylabel("SSIM", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.suptitle("Rollout Metrics vs Timestep")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def rollout(cfg: EvalConfig, model: torch.nn.Module, sample: np.ndarray, out_dirs: Dict[str, Path]) -> None:
    # (T,D,H,W,C,F) -> (T,F,C,D,H,W)
    sample_t = torch.from_numpy(sample.transpose(0, 5, 4, 1, 2, 3)).float()
    T = int(sample_t.shape[0])
    n_steps = min(cfg.timesteps, T - cfg.ar_context)
    if n_steps <= 0:
        raise ValueError(f"timesteps/ar_context invalid for sample length T={T}.")

    ctx = sample_t[: cfg.ar_context].unsqueeze(0).to(cfg.device)  # (1,ar,F,C,D,H,W)
    num_fields = int(sample_t.shape[1])

    frame_lists: Dict[str, Dict[int, List[Path]]] = {
        "triptych": {f: [] for f in range(num_fields)},
        "actual": {f: [] for f in range(num_fields)},
        "pred": {f: [] for f in range(num_fields)},
        "diff": {f: [] for f in range(num_fields)},
    }
    per_field_true: Dict[int, List[np.ndarray]] = {f: [] for f in range(num_fields)}
    per_field_pred: Dict[int, List[np.ndarray]] = {f: [] for f in range(num_fields)}

    row_per_field_step: List[Dict[str, float | int | str]] = []
    row_per_step_agg: List[Dict[str, float | int | str]] = []

    for f in range(num_fields):
        (out_dirs["frames"] / f"field_{f}").mkdir(parents=True, exist_ok=True)
        (out_dirs["single"] / f"field_{f}" / "actual").mkdir(parents=True, exist_ok=True)
        (out_dirs["single"] / f"field_{f}" / "pred").mkdir(parents=True, exist_ok=True)
        (out_dirs["single"] / f"field_{f}" / "diff").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step in tqdm(range(n_steps), desc="rollout"):
            raw = model(ctx)
            pred = extract_pred(raw)  # (1,F,C,D,H,W)
            pred_cpu = pred.squeeze(0).cpu()
            true_cpu = sample_t[cfg.ar_context + step].cpu()

            step_mse_vals = []
            step_ssim_vals = []
            for f in range(num_fields):
                t2 = tensor_to_plot_array(true_cpu[f])
                p2 = tensor_to_plot_array(pred_cpu[f])
                t2, p2 = align_plot_arrays(t2, p2)
                d2 = np.abs(t2 - p2)

                mse_val = float(np.mean((t2 - p2) ** 2))
                ssim_val = safe_ssim(t2, p2)
                step_mse_vals.append(mse_val)
                if not np.isnan(ssim_val):
                    step_ssim_vals.append(ssim_val)

                row_per_field_step.append(
                    {
                        "dataset": cfg.dataset,
                        "checkpoint": str(cfg.checkpoint_path),
                        "model_size": cfg.model_size,
                        "ar_context": cfg.ar_context,
                        "timestep": step + 1,
                        "field": f,
                        "mse": mse_val,
                        "ssim": ssim_val,
                    }
                )

                trip = out_dirs["frames"] / f"field_{f}" / f"triptych_t{step+1:03d}.png"
                save_triptych(t2, p2, trip, f"{cfg.dataset} | field={f} | t={step+1}")
                frame_lists["triptych"][f].append(trip)

                a_out = out_dirs["single"] / f"field_{f}" / "actual" / f"actual_t{step+1:03d}.png"
                p_out = out_dirs["single"] / f"field_{f}" / "pred" / f"pred_t{step+1:03d}.png"
                d_out = out_dirs["single"] / f"field_{f}" / "diff" / f"diff_t{step+1:03d}.png"
                save_single(t2, a_out, f"Actual | field={f} | t={step+1}")
                save_single(p2, p_out, f"Predicted | field={f} | t={step+1}")
                save_single(d2, d_out, f"Diff | field={f} | t={step+1}", cmap="hot")
                frame_lists["actual"][f].append(a_out)
                frame_lists["pred"][f].append(p_out)
                frame_lists["diff"][f].append(d_out)
                per_field_true[f].append(t2)
                per_field_pred[f].append(p2)

            row_per_step_agg.append(
                {
                    "dataset": cfg.dataset,
                    "checkpoint": str(cfg.checkpoint_path),
                    "model_size": cfg.model_size,
                    "ar_context": cfg.ar_context,
                    "timestep": step + 1,
                    "mse_mean": float(np.mean(step_mse_vals)) if step_mse_vals else float("nan"),
                    "ssim_mean": float(np.mean(step_ssim_vals)) if step_ssim_vals else float("nan"),
                }
            )

            if cfg.ar_context == 1:
                ctx = pred.unsqueeze(1)
            else:
                ctx = torch.cat([ctx[:, 1:], pred.unsqueeze(1)], dim=1)

    # CSVs
    per_field_csv = out_dirs["metrics"] / "rollout_metrics_per_field.csv"
    with per_field_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "checkpoint", "model_size", "ar_context", "timestep", "field", "mse", "ssim"],
        )
        w.writeheader()
        w.writerows(row_per_field_step)

    per_step_csv = out_dirs["metrics"] / "rollout_metrics_per_timestep.csv"
    with per_step_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "checkpoint", "model_size", "ar_context", "timestep", "mse_mean", "ssim_mean"],
        )
        w.writeheader()
        w.writerows(row_per_step_agg)

    # Overview grids and GIFs
    chosen = evenly_spaced_indices(n_steps, cfg.subset_frames)
    for f in range(num_fields):
        overview_out = out_dirs["overviews"] / f"overview_field_{f}.png"
        save_overview_grid(f, per_field_true[f], per_field_pred[f], chosen, overview_out)
        build_gif(frame_lists["triptych"][f], out_dirs["gifs"] / f"field_{f}_triptych.gif", cfg.gif_fps)
        build_gif(frame_lists["actual"][f], out_dirs["gifs"] / f"field_{f}_actual.gif", cfg.gif_fps)
        build_gif(frame_lists["pred"][f], out_dirs["gifs"] / f"field_{f}_pred.gif", cfg.gif_fps)
        build_gif(frame_lists["diff"][f], out_dirs["gifs"] / f"field_{f}_diff.gif", cfg.gif_fps)

    save_metric_plot(row_per_step_agg, out_dirs["plots"] / "metrics_over_time.png")


def build_config(args: argparse.Namespace) -> EvalConfig:
    ckpt = parse_checkpoint(args.model_path, args.model_file)
    dpaths = default_dataset_paths()
    ds_path = Path(args.dataset_path).resolve() if args.dataset_path else dpaths[args.dataset]
    device = parse_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        dev_idx = 0 if device.index is None else device.index
        print(f"Using CUDA device: {device} ({torch.cuda.get_device_name(dev_idx)})")
    else:
        print("Using CPU")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.dataset}_{ckpt.stem}"
    out_root = Path(args.output_dir).resolve() / run_id
    return EvalConfig(
        checkpoint_path=ckpt,
        dataset=args.dataset,
        timesteps=args.timesteps,
        sample_idx=args.sample_idx,
        model_size=args.model_size,
        model_choice=args.model_choice,
        ar_context=args.ar_context,
        device=device,
        output_root=out_root,
        gif_fps=args.gif_fps,
        subset_frames=args.subset_frames,
        heads_xa=args.heads_xa,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        dataset_path=ds_path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone rollout evaluator (single-file MORPH pipeline).")
    p.add_argument("--model_path", required=True, help="Checkpoint path or checkpoint directory.")
    p.add_argument("--model_file", default=None, help="Checkpoint filename when --model_path is a directory.")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["MHD", "DR", "CFD1D", "CFD2D-IC", "CFD3D", "SW", "DR1D", "CFD2D", "CFD3D-TURB", "BE1D", "GSDR2D", "TGC3D", "FNS_KF_2D", "DR2D"],
    )
    p.add_argument("--dataset_path", default=None, help="Optional override for dataset root path.")
    p.add_argument("--timesteps", type=int, default=50, help="Rollout horizon (max timesteps).")
    p.add_argument("--sample_idx", type=int, default=0, help="Test trajectory index.")
    p.add_argument("--model_size", choices=list(MORPH_MODELS.keys()), default="Ti")
    p.add_argument("--model_choice", default="FM", help="FM or standalone model choice semantics from MORPH.")
    p.add_argument("--ar_context", type=int, default=1, help="Autoregressive context frames.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--output_dir", default=str(REPO_ROOT / "out" / "results" / "rollouts"))
    p.add_argument("--gif_fps", type=int, default=6)
    p.add_argument("--subset_frames", type=int, default=10, help="Frame count for overview grid plots.")
    p.add_argument("--heads_xa", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--emb_dropout", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    out_dirs = make_dirs(cfg.output_root)

    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Dataset: {cfg.dataset}")
    print(f"Dataset path: {cfg.dataset_path}")
    print(f"Output root: {cfg.output_root}")
    print("Loading test split...")
    test_data = DataloaderChaos.load_data(cfg.dataset, str(cfg.dataset_path), split="test")
    if cfg.sample_idx < 0 or cfg.sample_idx >= test_data.shape[0]:
        raise IndexError(f"sample_idx {cfg.sample_idx} out of range for test size {test_data.shape[0]}")

    model = load_model(cfg)
    sample = test_data[cfg.sample_idx]
    rollout(cfg, model, sample, out_dirs)
    print(f"Done. Results written to: {cfg.output_root}")


if __name__ == "__main__":
    main()
