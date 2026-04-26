#!/usr/bin/env python3
"""
Run MORPH inference + rollout visualization pipeline.

This script:
1) Calls MORPH/scripts/infer_MORPH.py (for parity with upstream flow).
2) Loads the same model/checkpoint and test dataset.
3) Performs autoregressive rollout for N steps.
4) Saves per-step single-frame, side-by-side, and diff PNGs.
5) Builds GIFs from those PNGs.
6) Records per-timestep MSE and SSIM in CSV.

Outputs are written under:
    out/results/rollouts/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

matplotlib.use("Agg")

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parent
MORPH_ROOT = REPO_ROOT / "MORPH"
sys.path.insert(0, str(MORPH_ROOT))

from src.utils.dataloaders.dataloaderchaos import DataloaderChaos  # noqa: E402
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression  # noqa: E402


MORPH_MODELS: Dict[str, List[int]] = {
    "Ti": [8, 256, 4, 4, 1024],
    "S": [8, 512, 8, 4, 2048],
    "M": [8, 768, 12, 8, 3072],
    "L": [8, 1024, 16, 16, 4096],
}

DATASETS = [
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


def dataset_paths() -> Dict[str, Path]:
    base = MORPH_ROOT / "datasets" / "normalized_revin"
    return {
        "MHD": base / DATASETS[1],
        "DR": base / DATASETS[0],
        "CFD1D": base / DATASETS[2],
        "CFD2D-IC": base / DATASETS[4],
        "CFD3D": base / DATASETS[5],
        "SW": base / DATASETS[3],
        "DR1D": base / DATASETS[6],
        "CFD2D": base / DATASETS[7],
        "CFD3D-TURB": base / DATASETS[8],
        "BE1D": base / DATASETS[9],
        "GSDR2D": base / DATASETS[10],
        "TGC3D": base / DATASETS[11],
        "FNS_KF_2D": base / DATASETS[12],
        "DR2D": base / DATASETS[0],
    }


def ensure_dirs(run_root: Path) -> Dict[str, Path]:
    paths = {
        "root": run_root,
        "single": run_root / "single_frames",
        "side_by_side": run_root / "side_by_side",
        "diff": run_root / "diffs",
        "gif": run_root / "gifs",
        "metrics": run_root / "metrics",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def to_hw(field_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert field tensor shapes like (C,D,H,W) or (C,H,W) or (H,W) to (H,W).
    Uses component 0 and center depth slice when needed.
    """
    x = field_tensor.detach().cpu()
    while x.ndim > 2 and 1 in x.shape:
        x = x.squeeze()
    if x.ndim == 4:  # C,D,H,W
        x = x[0, x.shape[1] // 2]
    elif x.ndim == 3:  # C,H,W or D,H,W
        x = x[0] if x.shape[0] <= 8 else x[x.shape[0] // 2]
    if x.ndim != 2:
        while x.ndim > 2:
            x = x[x.shape[0] // 2]
    return x.numpy()


def save_triptych_png(true_2d: np.ndarray, pred_2d: np.ndarray, out_file: Path, title: str) -> None:
    diff_2d = np.abs(true_2d - pred_2d)
    vmin = float(min(true_2d.min(), pred_2d.min()))
    vmax = float(max(true_2d.max(), pred_2d.max()))
    if vmax == vmin:
        vmax = vmin + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(true_2d, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Truth")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_2d, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff_2d, cmap="hot")
    axes[2].set_title("|Diff|")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def save_single_png(img_2d: np.ndarray, out_file: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img_2d, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def build_gif(frame_paths: List[Path], out_gif: Path, fps: int) -> None:
    if not frame_paths:
        return
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_gif, images, fps=fps)


def run_infer_script(args: argparse.Namespace, checkpoint_basename: str) -> None:
    infer_py = MORPH_ROOT / "scripts" / "infer_MORPH.py"
    cmd = [
        sys.executable,
        str(infer_py),
        "--model_choice",
        args.model_choice,
        "--model_size",
        args.model_size,
        "--checkpoint",
        checkpoint_basename,
        "--test_dataset",
        args.dataset,
        "--ar_order",
        str(args.ar_order),
        "--max_ar_order",
        str(args.ar_order),
        "--rollout_horizon",
        str(args.rollout_steps),
        "--test_sample",
        str(args.sample_idx),
        "--device_idx",
        str(args.device_idx),
    ]
    print("→ Calling infer_MORPH.py")
    subprocess.run(cmd, check=True, cwd=str(MORPH_ROOT))


def load_checkpoint_model(ckpt_path: Path, model_size: str, ar_order: int, device: torch.device) -> torch.nn.Module:
    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[model_size]
    model = ViT3DRegression(
        patch_size=8,
        dim=dim,
        depth=depth,
        heads=heads,
        heads_xa=32,
        mlp_dim=mlp_dim,
        max_components=3,
        conv_filter=filters,
        max_ar=ar_order,
        max_patches=4096,
        max_fields=3,
        dropout=0.1,
        emb_dropout=0.1,
        lora_r_attn=0,
        lora_r_mlp=0,
        lora_alpha=None,
        lora_p=0.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def rollout_and_record(
    model: torch.nn.Module,
    sample: np.ndarray,
    ar_order: int,
    rollout_steps: int,
    out_dirs: Dict[str, Path],
    fps: int,
    run_meta: Dict[str, str],
    device: torch.device,
) -> None:
    """
    sample shape: (T, D, H, W, C, F)
    model input:  (B, ar, F, C, D, H, W)
    """
    sample_t = torch.from_numpy(sample.transpose(0, 5, 4, 1, 2, 3)).float()  # (T,F,C,D,H,W)
    T = sample_t.shape[0]
    max_steps = min(rollout_steps, T - ar_order)
    if max_steps <= 0:
        raise ValueError(f"Not enough timesteps: T={T}, ar_order={ar_order}")

    ctx = sample_t[:ar_order].unsqueeze(0).to(device)
    rows: List[Dict[str, float | int | str]] = []
    sbs_frames: Dict[int, List[Path]] = {}
    single_pred_frames: Dict[int, List[Path]] = {}
    single_true_frames: Dict[int, List[Path]] = {}
    diff_frames: Dict[int, List[Path]] = {}

    with torch.no_grad():
        for step in tqdm(range(max_steps), desc="rollout"):
            out = model(ctx)
            pred = out[2] if isinstance(out, tuple) else out  # (B,F,C,D,H,W)
            pred_ = pred.squeeze(0).cpu()  # (F,C,D,H,W)
            true_ = sample_t[ar_order + step].cpu()  # (F,C,D,H,W)

            f_count = int(min(pred_.shape[0], true_.shape[0]))
            for fidx in range(f_count):
                p2 = to_hw(pred_[fidx])
                t2 = to_hw(true_[fidx])
                if p2.shape != t2.shape:
                    h = min(p2.shape[0], t2.shape[0])
                    w = min(p2.shape[1], t2.shape[1])
                    p2 = p2[:h, :w]
                    t2 = t2[:h, :w]

                mse_val = float(np.mean((p2 - t2) ** 2))
                data_range = float(max(t2.max(), p2.max()) - min(t2.min(), p2.min()))
                if data_range <= 0:
                    data_range = 1.0
                try:
                    ssim_val = float(ssim(t2, p2, data_range=data_range))
                except Exception:
                    ssim_val = float("nan")

                rows.append(
                    {
                        "run_id": run_meta["run_id"],
                        "dataset": run_meta["dataset"],
                        "model_file": run_meta["model_file"],
                        "model_size": run_meta["model_size"],
                        "ar_order": ar_order,
                        "timestep": step + 1,
                        "field": fidx,
                        "mse": mse_val,
                        "ssim": ssim_val,
                    }
                )

                sbs_out = out_dirs["side_by_side"] / f"field{fidx}_t{step+1:03d}.png"
                save_triptych_png(t2, p2, sbs_out, f"Field {fidx} | t={step+1}")
                sbs_frames.setdefault(fidx, []).append(sbs_out)

                pred_out = out_dirs["single"] / f"pred_field{fidx}_t{step+1:03d}.png"
                true_out = out_dirs["single"] / f"true_field{fidx}_t{step+1:03d}.png"
                diff_out = out_dirs["diff"] / f"diff_field{fidx}_t{step+1:03d}.png"
                save_single_png(p2, pred_out, f"Pred field {fidx} t={step+1}")
                save_single_png(t2, true_out, f"Truth field {fidx} t={step+1}")
                save_single_png(np.abs(t2 - p2), diff_out, f"|Diff| field {fidx} t={step+1}", cmap="hot")
                single_pred_frames.setdefault(fidx, []).append(pred_out)
                single_true_frames.setdefault(fidx, []).append(true_out)
                diff_frames.setdefault(fidx, []).append(diff_out)

            if ar_order == 1:
                ctx = pred.unsqueeze(1)
            else:
                ctx = torch.cat([ctx[:, 1:], pred.unsqueeze(1)], dim=1)

    # CSV metrics: per timestep and field + aggregate-by-timestep rows
    metrics_csv = out_dirs["metrics"] / "rollout_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_id", "dataset", "model_file", "model_size", "ar_order", "timestep", "field", "mse", "ssim"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    agg_csv = out_dirs["metrics"] / "rollout_metrics_aggregate.csv"
    by_ts: Dict[int, List[Dict[str, float | int | str]]] = {}
    for r in rows:
        by_ts.setdefault(int(r["timestep"]), []).append(r)
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_id", "dataset", "model_file", "model_size", "ar_order", "timestep", "mse_mean", "ssim_mean"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ts in sorted(by_ts):
            chunk = by_ts[ts]
            mse_vals = [float(x["mse"]) for x in chunk]
            ssim_vals = [float(x["ssim"]) for x in chunk if not np.isnan(float(x["ssim"]))]
            w.writerow(
                {
                    "run_id": run_meta["run_id"],
                    "dataset": run_meta["dataset"],
                    "model_file": run_meta["model_file"],
                    "model_size": run_meta["model_size"],
                    "ar_order": ar_order,
                    "timestep": ts,
                    "mse_mean": float(np.mean(mse_vals)),
                    "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
                }
            )

    for fidx, frames in sbs_frames.items():
        build_gif(frames, out_dirs["gif"] / f"field{fidx}_side_by_side.gif", fps)
    for fidx, frames in single_pred_frames.items():
        build_gif(frames, out_dirs["gif"] / f"field{fidx}_pred.gif", fps)
    for fidx, frames in single_true_frames.items():
        build_gif(frames, out_dirs["gif"] / f"field{fidx}_truth.gif", fps)
    for fidx, frames in diff_frames.items():
        build_gif(frames, out_dirs["gif"] / f"field{fidx}_diff.gif", fps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout + GIF + SSIM/MSE recorder for MORPH checkpoints.")
    p.add_argument("--model_dir", required=True, help="Directory containing checkpoint.")
    p.add_argument("--model_file", required=True, help="Checkpoint filename inside model_dir.")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["MHD", "DR", "CFD1D", "CFD2D-IC", "CFD3D", "SW", "DR1D", "CFD2D", "CFD3D-TURB", "BE1D", "GSDR2D", "TGC3D", "FNS_KF_2D", "DR2D"],
    )
    p.add_argument("--model_size", default="Ti", choices=list(MORPH_MODELS.keys()))
    p.add_argument("--model_choice", default="FM", help="Subdir key used by infer_MORPH.py under MORPH/models/")
    p.add_argument("--rollout_steps", type=int, default=50)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--ar_order", type=int, default=1)
    p.add_argument("--device_idx", type=int, default=0)
    p.add_argument("--gif_fps", type=int, default=6)
    p.add_argument("--output_root", default=str(REPO_ROOT / "out" / "results" / "rollouts"))
    p.add_argument("--skip_infer_call", action="store_true", help="Skip calling infer_MORPH.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.model_dir) / args.model_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.dataset}_{Path(args.model_file).stem}"
    out_dirs = ensure_dirs(Path(args.output_root) / run_id)

    # To call infer_MORPH.py with its existing interface, stage a symlink in MORPH/models/<model_choice>/.
    staged_dir = MORPH_ROOT / "models" / args.model_choice
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_ckpt = staged_dir / args.model_file
    if staged_ckpt.exists() or staged_ckpt.is_symlink():
        staged_ckpt.unlink()
    staged_ckpt.symlink_to(ckpt_path.resolve())

    try:
        if not args.skip_infer_call:
            run_infer_script(args, args.model_file)

        dpaths = dataset_paths()
        data = DataloaderChaos.load_data(args.dataset, str(dpaths[args.dataset]), split="test")
        if args.sample_idx < 0 or args.sample_idx >= data.shape[0]:
            raise IndexError(f"sample_idx {args.sample_idx} out of range for test size {data.shape[0]}")
        sample = data[args.sample_idx]

        device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
        model = load_checkpoint_model(ckpt_path, args.model_size, args.ar_order, device)

        rollout_and_record(
            model=model,
            sample=sample,
            ar_order=args.ar_order,
            rollout_steps=args.rollout_steps,
            out_dirs=out_dirs,
            fps=args.gif_fps,
            run_meta={
                "run_id": run_id,
                "dataset": args.dataset,
                "model_file": args.model_file,
                "model_size": args.model_size,
            },
            device=device,
        )
        print(f"→ Done. Output: {out_dirs['root']}")
    finally:
        # Clean staged symlink
        if staged_ckpt.is_symlink():
            staged_ckpt.unlink()


if __name__ == "__main__":
    main()
