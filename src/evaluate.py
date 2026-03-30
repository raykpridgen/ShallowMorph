#!/usr/bin/env python3
"""
Evaluate a fine-tuned MORPH checkpoint on shallow-water test trajectories.

Two evaluation modes:

  **single-step** (default)
    Teacher-forced next-frame prediction on the full test set.
    Reports MSE / MAE / RMSE (normalised) and VRMSE / NRMSE (denormalised).

  **rollout** (``--rollout-horizon K``)
    Starting from the first ``ar_order`` ground-truth frames, autoregressively
    predict ``K`` steps and compare with ground truth.  Reports per-horizon
    RMSE and saves the predicted sequence as ``.npy`` for visualisation.

Outputs (written to ``out/eval/<dataset_name>/``):
  - ``metrics_eval_<tag>.json``      — all metrics in structured form
  - ``rollout_pred_<tag>.npy``       — rollout predictions (T', H, W)
  - ``rollout_actual_<tag>.npy``     — corresponding ground truth
  - ``diff_t<N>.png``                — actual / predicted / diff triptych
  - ``grid_<tag>.png``               — evolution grid (actual vs predicted)
  - ``gif_actual_<tag>.gif``         — ground truth GIF
  - ``gif_rollout_<tag>.gif``        — rollout prediction GIF

Usage:
    python src/evaluate.py --checkpoint <path_to_best.pth> --ft-level1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATASETS_DIR,
    MODELS_DIR,
    MORPH_FM_FILENAMES,
    MORPH_MODELS,
    MORPH_ROOT,
    NORM_STATS_DIR,
    OUT_DIR,
    RESULTS_DIR,
    SW_DATASET_NAME,
    SW_DATASET_SPECS,
    get_logger,
    import_morph_modules,
    load_split_npy,
)
from visualize import (
    plot_diff_frame,
    plot_sequence_grid,
    render_sequence_gif,
)

_m = import_morph_modules()
DeviceManager          = _m["DeviceManager"]
Metrics3DCalculator    = _m["Metrics3DCalculator"]
RevIN                  = _m["RevIN"]
FastARDataPreparer     = _m["FastARDataPreparer"]
UPTF7                  = _m["UPTF7"]
ViT3DRegression        = _m["ViT3DRegression"]
del _m

log = get_logger("evaluate")

# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained MORPH checkpoint on shallow-water test data.",
    )

    # ── data ────────────────────────────────────────────────────────────
    p.add_argument("--dataset-dir", type=Path, default=DATASETS_DIR)
    p.add_argument("--dataset-name", type=str, default=SW_DATASET_NAME)
    p.add_argument("--dataset-specs", nargs=5, type=int,
                   metavar=("F", "C", "D", "H", "W"),
                   default=[SW_DATASET_SPECS[k] for k in "FCDHW"])

    # ── model / checkpoint ──────────────────────────────────────────────
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the .pth checkpoint (absolute or relative to MORPH/models/<dataset_name>/)")
    p.add_argument("--model-size", type=str, default="Ti",
                   choices=list(MORPH_MODELS.keys()))
    p.add_argument("--ft-level1", action="store_true")
    p.add_argument("--ft-level2", action="store_true")
    p.add_argument("--ft-level3", action="store_true")
    p.add_argument("--ft-level4", action="store_true")
    p.add_argument("--rank-lora-attn", type=int, default=16)
    p.add_argument("--rank-lora-mlp", type=int, default=12)
    p.add_argument("--lora-p", type=float, default=0.05)
    p.add_argument("--tf-reg", nargs=2, type=float, default=[0.1, 0.1],
                   metavar=("dropout", "emb_dropout"))
    p.add_argument("--heads-xa", type=int, default=32)

    # ── AR ──────────────────────────────────────────────────────────────
    p.add_argument("--ar-order", type=int, default=1)
    p.add_argument("--max-ar-order", type=int, default=1)

    # ── evaluation options ──────────────────────────────────────────────
    p.add_argument("--rollout-horizon", type=int, default=20,
                   help="Number of autoregressive rollout steps")
    p.add_argument("--test-sample", type=int, default=0,
                   help="Trajectory index in test set for detailed visualisation")
    p.add_argument("--n-vis-samples", type=int, default=3,
                   help="Number of test trajectories to produce diff plots for")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device-idx", type=int, default=0)
    p.add_argument("--parallel", type=str, choices=["dp", "no"], default="no")

    # ── output ──────────────────────────────────────────────────────────
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: out/eval/<dataset_name>/)")
    return p


# ───────────────────────────────────────────────────────────────────────────
# Thin Dataset wrapper
# ───────────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _to_uptf7(arr: np.ndarray, specs: list[int]) -> np.ndarray:
    F_, C, D, H, W = specs
    return UPTF7(
        dataset=arr, num_samples=arr.shape[0], traj_len=arr.shape[1],
        fields=F_, components=C, image_depth=D, image_height=H, image_width=W,
    ).transform()


def _extract_hw(frame_uptf7: np.ndarray | torch.Tensor) -> np.ndarray:
    """Extract a 2-D ``(H, W)`` slice from an UPTF7 frame ``(F, C, D, H, W)``."""
    if isinstance(frame_uptf7, torch.Tensor):
        frame_uptf7 = frame_uptf7.cpu().numpy()
    return frame_uptf7[0, 0, 0]  # F=0, C=0, D=0


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
    dropout, emb_dropout = args.tf_reg

    lora_r_attn = 0 if args.ft_level4 else args.rank_lora_attn
    lora_r_mlp  = 0 if args.ft_level4 else args.rank_lora_mlp
    lora_p      = 0 if args.ft_level4 else args.lora_p

    model = ViT3DRegression(
        patch_size=8, dim=dim, depth=depth,
        heads=heads, heads_xa=args.heads_xa, mlp_dim=mlp_dim,
        max_components=3, conv_filter=filters,
        max_ar=args.max_ar_order, max_patches=4096, max_fields=3,
        dropout=dropout, emb_dropout=emb_dropout,
        lora_r_attn=lora_r_attn, lora_r_mlp=lora_r_mlp,
        lora_alpha=None, lora_p=lora_p,
    ).to(device)
    return model


def _load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    sd = ckpt["model_state_dict"]
    target = model.module if isinstance(model, nn.DataParallel) else model
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    target.load_state_dict(sd, strict=True)


def _rollout(
    model: nn.Module,
    init_frame: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Autoregressive rollout from a single initial frame.

    *init_frame*: ``(F, C, D, H, W)``
    Returns a list of *horizon* predicted frames, each ``(F, C, D, H, W)``.
    """
    model.eval()
    preds: list[torch.Tensor] = []
    current = init_frame.unsqueeze(0)  # (1, F, C, D, H, W)

    with torch.no_grad():
        for _ in range(horizon):
            inp = current.unsqueeze(1).to(device)   # (1, 1, F, C, D, H, W)
            _, _, pred = model(inp)                  # (1, F, C, D, H, W)
            pred = pred.cpu()
            preds.append(pred.squeeze(0))
            current = pred

    return preds


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    t0 = time.time()

    dataset_name = args.dataset_name
    specs = args.dataset_specs

    # ── device ──────────────────────────────────────────────────────────
    devices = DeviceManager.list_devices()
    device = devices[args.device_idx] if devices else torch.device("cpu")

    # ── output directory ────────────────────────────────────────────────
    out_dir = args.out_dir or (OUT_DIR / "eval" / dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # ── resolve checkpoint path ─────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        candidate = MODELS_DIR / dataset_name / args.checkpoint
        if candidate.exists():
            ckpt_path = candidate
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    log.info("Checkpoint: %s", ckpt_path)

    # ── load test data ──────────────────────────────────────────────────
    log.info("Loading preprocessed splits from %s", args.dataset_dir)
    splits = load_split_npy(dataset_name, args.dataset_dir)
    test_raw = splits["test"]
    n_test, T = test_raw.shape[0], test_raw.shape[1]
    log.info("Test set: %d trajectories, T=%d", n_test, T)

    # ── UPTF7 + RevIN ──────────────────────────────────────────────────
    all_uptf7 = np.concatenate(
        [_to_uptf7(splits[s], specs) for s in ("train", "val", "test")], axis=0,
    )
    n_train = splits["train"].shape[0]
    n_val   = splits["val"].shape[0]

    norm_dir = NORM_STATS_DIR
    norm_dir.mkdir(parents=True, exist_ok=True)
    revin = RevIN(str(norm_dir))
    norm_prefix = f"norm_{dataset_name}"

    revin.compute_stats(all_uptf7, prefix=norm_prefix)
    all_norm = revin.normalize(all_uptf7, prefix=norm_prefix)

    test_norm = all_norm[n_train + n_val:]
    del all_uptf7, all_norm
    log.info("RevIN normalisation OK  (prefix=%s)", norm_prefix)

    # ── build + load model ──────────────────────────────────────────────
    model = _build_model(args, device)
    n_gpus = torch.cuda.device_count()
    if args.parallel == "dp" and n_gpus > 1:
        model = nn.DataParallel(model)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()
    log.info("Model loaded (%.3f M params)",
             sum(p.numel() for p in model.parameters()) / 1e6)

    # ═══════════════════════════════════════════════════════════════════
    #  SINGLE-STEP EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    log.info("── Single-step evaluation ──")
    order = (0, 1, 4, 5, 6, 3, 2)
    preparer = FastARDataPreparer(ar_order=args.ar_order)
    X_te, y_te = preparer.prepare(test_norm.transpose(*order))
    te_loader = DataLoader(PairDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False)

    out_all: list[torch.Tensor] = []
    tar_all: list[torch.Tensor] = []
    with torch.no_grad():
        for inp, tar in tqdm(te_loader, desc="Single-step eval", leave=False):
            inp = inp.to(device)
            _, _, out = model(inp)
            out_all.append(out.detach().cpu())
            tar_all.append(tar)

    out_cat = torch.cat(out_all, dim=0)
    tar_cat = torch.cat(tar_all, dim=0)

    mse  = F.mse_loss(out_cat, tar_cat, reduction="mean")
    mae  = F.l1_loss(out_cat, tar_cat, reduction="mean")
    rmse = mse ** 0.5

    T_minus_ar = T - args.ar_order
    out_rs = out_cat.reshape(n_test, T_minus_ar, *out_cat.shape[1:])
    tar_rs = tar_cat.reshape(n_test, T_minus_ar, *tar_cat.shape[1:])

    outputs_denorm = RevIN.denormalize_testeval(
        str(norm_dir), norm_prefix, out_rs, dataset=dataset_name,
    )
    targets_denorm = RevIN.denormalize_testeval(
        str(norm_dir), norm_prefix, tar_rs, dataset=dataset_name,
    )
    vrmse = Metrics3DCalculator.calculate_VRMSE(outputs_denorm, targets_denorm).mean()
    nrmse = Metrics3DCalculator.calculate_NRMSE(outputs_denorm, targets_denorm).mean()

    log.info("  MSE:   %.5f", mse.item())
    log.info("  MAE:   %.5f", mae.item())
    log.info("  RMSE:  %.5f", rmse.item())
    log.info("  VRMSE: %.5f", vrmse.item())
    log.info("  NRMSE: %.5f", nrmse.item())

    # ═══════════════════════════════════════════════════════════════════
    #  ROLLOUT EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    horizon = min(args.rollout_horizon, T - args.ar_order)
    log.info("── Rollout evaluation (horizon=%d) ──", horizon)

    test_uptf7 = _to_uptf7(test_raw, specs)  # unnormalised UPTF7 (N, T, F, C, D, H, W)
    test_norm_tensor = torch.from_numpy(test_norm).float()

    per_traj_rollout_rmse: list[float] = []
    per_horizon_rmse: np.ndarray = np.zeros(horizon, dtype=np.float64)

    vis_idx = args.test_sample
    rollout_pred_vis: np.ndarray | None = None
    rollout_actual_vis: np.ndarray | None = None

    for i in tqdm(range(n_test), desc="Rollout", leave=False):
        init_frame = test_norm_tensor[i, 0]  # (F, C, D, H, W) — first frame normalised
        preds = _rollout(model, init_frame, horizon, device)

        pred_stack = torch.stack(preds, dim=0)  # (K, F, C, D, H, W)
        true_stack = test_norm_tensor[i, 1: 1 + horizon]  # (K, F, C, D, H, W)

        per_step_mse = ((pred_stack - true_stack) ** 2).mean(dim=list(range(1, pred_stack.ndim)))
        per_step_rmse = per_step_mse.sqrt().numpy()
        per_horizon_rmse += per_step_rmse

        traj_rmse = float(per_step_rmse.mean())
        per_traj_rollout_rmse.append(traj_rmse)

        if i == vis_idx:
            actual_hw = np.array([_extract_hw(test_uptf7[i, t + 1]) for t in range(horizon)])
            pred_denorm = []
            for t_step in range(horizon):
                frame_n = preds[t_step].numpy()  # (F,C,D,H,W)  normalised
                pred_denorm.append(_extract_hw(frame_n))
            rollout_pred_vis = np.array(pred_denorm)
            rollout_actual_vis = actual_hw

    per_horizon_rmse /= n_test
    mean_rollout_rmse = float(np.mean(per_traj_rollout_rmse))

    log.info("  Mean rollout RMSE (normalised): %.5f", mean_rollout_rmse)
    log.info("  Per-horizon RMSE (first 5): %s",
             ", ".join(f"{v:.5f}" for v in per_horizon_rmse[:5]))

    # ═══════════════════════════════════════════════════════════════════
    #  VISUALISATION
    # ═══════════════════════════════════════════════════════════════════
    tag = f"s{vis_idx}_h{horizon}"
    log.info("── Generating visualisations ──")

    if rollout_actual_vis is not None and rollout_pred_vis is not None:
        np.save(out_dir / f"rollout_actual_{tag}.npy", rollout_actual_vis)
        np.save(out_dir / f"rollout_pred_{tag}.npy", rollout_pred_vis)
        log.info("  Saved rollout .npy arrays")

        try:
            render_sequence_gif(
                rollout_actual_vis, out_dir / f"gif_actual_{tag}.gif",
                title_prefix=f"Actual (traj {vis_idx})", fps=args.rollout_horizon // 2 or 5,
            )
            render_sequence_gif(
                rollout_pred_vis, out_dir / f"gif_rollout_{tag}.gif",
                title_prefix=f"Rollout (traj {vis_idx})", fps=args.rollout_horizon // 2 or 5,
            )
            log.info("  Saved GIFs")
        except Exception:
            log.warning("  Failed to generate GIFs", exc_info=True)

        try:
            diff_steps = [0, horizon // 4, horizon // 2, horizon - 1]
            for t_step in diff_steps:
                if t_step < rollout_actual_vis.shape[0]:
                    plot_diff_frame(
                        rollout_actual_vis[t_step],
                        rollout_pred_vis[t_step],
                        t_idx=t_step + 1,
                        label=f"Traj {vis_idx}",
                        save_path=out_dir / f"diff_t{t_step + 1:03d}_{tag}.png",
                    )
            log.info("  Saved diff triptychs")
        except Exception:
            log.warning("  Failed to generate diff triptychs", exc_info=True)

        try:
            plot_sequence_grid(
                rollout_actual_vis,
                rollout_pred_vis,
                n_cols=min(8, horizon),
                label=f"Traj {vis_idx} rollout",
                save_path=out_dir / f"grid_{tag}.png",
            )
            log.info("  Saved sequence grid")
        except Exception:
            log.warning("  Failed to generate sequence grid", exc_info=True)

    # ── Per-horizon RMSE plot ───────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, horizon + 1), per_horizon_rmse, marker="o", markersize=3)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("RMSE (normalised)")
    ax.set_title("Per-horizon rollout RMSE (averaged over test set)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"rollout_rmse_curve_{tag}.png"), dpi=150)
    plt.close(fig)
    log.info("  Saved rollout RMSE curve")

    # ═══════════════════════════════════════════════════════════════════
    #  METRICS OUTPUT
    # ═══════════════════════════════════════════════════════════════════
    metrics = {
        "checkpoint": str(ckpt_path),
        "model_size": args.model_size,
        "ar_order": args.ar_order,
        "max_ar_order": args.max_ar_order,
        "dataset_specs": specs,
        "n_test_traj": n_test,
        "single_step": {
            "MSE":   mse.item(),
            "MAE":   mae.item(),
            "RMSE":  rmse.item(),
            "VRMSE": vrmse.item(),
            "NRMSE": nrmse.item(),
        },
        "rollout": {
            "horizon": horizon,
            "mean_rmse": mean_rollout_rmse,
            "per_horizon_rmse": per_horizon_rmse.tolist(),
            "per_traj_rmse": per_traj_rollout_rmse,
        },
        "vis_sample_idx": vis_idx,
    }
    metrics_path = out_dir / f"metrics_eval_{tag}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    log.info("Metrics: %s", metrics_path)

    # ── Summary ─────────────────────────────────────────────────────────
    log.info("═══ SUMMARY ═══")
    log.info("  Single-step:  MSE=%.5f  RMSE=%.5f  VRMSE=%.5f  NRMSE=%.5f",
             mse.item(), rmse.item(), vrmse.item(), nrmse.item())
    log.info("  Rollout (%d steps):  mean RMSE=%.5f", horizon, mean_rollout_rmse)
    log.info("  Output dir: %s", out_dir)
    log.info("  Wall time: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
