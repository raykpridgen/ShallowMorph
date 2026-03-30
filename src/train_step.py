#!/usr/bin/env python3
"""
Single-step fine-tuning of MORPH on shallow-water data (train_step.md).

Consumes the pre-split .npy files produced by ``preprocess.py``, converts
them to UPTF7, normalises with RevIN, creates AR input/target pairs, and
fine-tunes a ViT3DRegression model with teacher-forced next-step MSE loss.

After training:
  - reloads the *best* checkpoint (lowest val loss)
  - evaluates on the held-out test set (normalised + denormalised metrics)
  - prints a summary and writes structured ``metrics.json``

Usage (minimal):
    python src/train_step.py --download_model --ft_level1

See ``python src/train_step.py --help`` for all options.
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
import matplotlib.pyplot as plt
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
    RESULTS_DIR,
    SW_DATASET_NAME,
    SW_DATASET_SPECS,
    get_logger,
    import_morph_modules,
    load_manifest,
    load_split_npy,
)

_m = import_morph_modules()
FastARDataPreparer          = _m["FastARDataPreparer"]
DeviceManager               = _m["DeviceManager"]
Metrics3DCalculator         = _m["Metrics3DCalculator"]
RevIN                       = _m["RevIN"]
SelectFineTuningParameters  = _m["SelectFineTuningParameters"]
Trainer                     = _m["Trainer"]
UPTF7                       = _m["UPTF7"]
ViT3DRegression             = _m["ViT3DRegression"]
del _m

log = get_logger("train_step")

# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune MORPH (single-step) on preprocessed shallow-water data.",
    )

    # ── data ────────────────────────────────────────────────────────────
    p.add_argument("--dataset-dir", type=Path, default=DATASETS_DIR,
                   help="Directory with shallow_water_{train,val,test}.npy")
    p.add_argument("--dataset-name", type=str, default=SW_DATASET_NAME)
    p.add_argument("--dataset-specs", nargs=5, type=int,
                   metavar=("F", "C", "D", "H", "W"),
                   default=[SW_DATASET_SPECS[k] for k in "FCDHW"])
    p.add_argument("--n-traj", type=int, default=None,
                   help="Limit number of training trajectories (default: all)")

    # ── model ───────────────────────────────────────────────────────────
    p.add_argument("--model-size", type=str, default="Ti",
                   choices=list(MORPH_MODELS.keys()))
    p.add_argument("--download-model", action="store_true",
                   help="Download FM weights from HuggingFace")
    p.add_argument("--ckpt-from", type=str, choices=["FM", "FT"], default="FM")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit checkpoint path (relative to MORPH/models/)")

    # ── fine-tune level ─────────────────────────────────────────────────
    p.add_argument("--ft-level1", action="store_true",
                   help="Level-1: LoRA + LayerNorm + PosEnc")
    p.add_argument("--ft-level2", action="store_true",
                   help="Level-2: + Encoder (conv, proj, xattn)")
    p.add_argument("--ft-level3", action="store_true",
                   help="Level-3: + Decoder linear")
    p.add_argument("--ft-level4", action="store_true",
                   help="Level-4: unfreeze everything")
    p.add_argument("--lr-level4", type=float, default=1e-4)
    p.add_argument("--wd-level4", type=float, default=0.0)

    # ── LoRA ────────────────────────────────────────────────────────────
    p.add_argument("--rank-lora-attn", type=int, default=16)
    p.add_argument("--rank-lora-mlp", type=int, default=12)
    p.add_argument("--lora-p", type=float, default=0.05)

    # ── AR ──────────────────────────────────────────────────────────────
    p.add_argument("--ar-order", type=int, default=1)
    p.add_argument("--max-ar-order", type=int, default=1)

    # ── training hyper-params ───────────────────────────────────────────
    p.add_argument("--n-epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-scheduler", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--tf-reg", nargs=2, type=float, default=[0.1, 0.1],
                   metavar=("dropout", "emb_dropout"))
    p.add_argument("--heads-xa", type=int, default=32)
    p.add_argument("--parallel", type=str, choices=["dp", "no"], default="dp")

    # ── evaluation ──────────────────────────────────────────────────────
    p.add_argument("--rollout-horizon", type=int, default=10)
    p.add_argument("--test-sample", type=int, default=0,
                   help="Trajectory index in test set for visualisation")

    # ── checkpointing / IO ──────────────────────────────────────────────
    p.add_argument("--overwrite-weights", action="store_true")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--save-batch-ckpt", action="store_true")
    p.add_argument("--save-batch-freq", type=int, default=1000)
    p.add_argument("--device-idx", type=int, default=0)

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
    """Convert ``(N, T, H, W, 1)`` → ``(N, T, F, C, D, H, W)``."""
    F_, C, D, H, W = specs
    return UPTF7(
        dataset=arr,
        num_samples=arr.shape[0],
        traj_len=arr.shape[1],
        fields=F_, components=C,
        image_depth=D, image_height=H, image_width=W,
    ).transform()


def _finetune_level(args: argparse.Namespace) -> int:
    if args.ft_level4:
        return 4
    if args.ft_level3:
        return 3
    if args.ft_level2:
        return 2
    if args.ft_level1:
        return 1
    raise ValueError("Select a fine-tuning level: --ft-level1, --ft-level2, --ft-level3, or --ft-level4")



# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    t_start = time.time()

    dataset_name = args.dataset_name
    specs = args.dataset_specs
    lev = _finetune_level(args)
    if lev == 4:
        args.rank_lora_attn = 0
        args.rank_lora_mlp = 0
        args.lora_p = 0

    # ── device ──────────────────────────────────────────────────────────
    devices = DeviceManager.list_devices()
    device = devices[args.device_idx] if devices else torch.device("cpu")

    # ── directory setup ─────────────────────────────────────────────────
    savepath_model = MODELS_DIR
    savepath_model.mkdir(parents=True, exist_ok=True)
    savepath_model_ds = savepath_model / dataset_name
    savepath_model_ds.mkdir(parents=True, exist_ok=True)

    savepath_results = RESULTS_DIR
    savepath_results.mkdir(parents=True, exist_ok=True)
    savepath_results_ds = savepath_results / dataset_name
    savepath_results_ds.mkdir(parents=True, exist_ok=True)

    norm_dir = NORM_STATS_DIR
    norm_dir.mkdir(parents=True, exist_ok=True)

    # ── load preprocessed splits ────────────────────────────────────────
    log.info("Loading preprocessed splits from %s", args.dataset_dir)
    splits = load_split_npy(dataset_name, args.dataset_dir)
    for name, arr in splits.items():
        log.info("  %-6s %s  dtype=%s", name, arr.shape, arr.dtype)

    # ── UPTF7 conversion (N,T,H,W,1) → (N,T,F,C,D,H,W) ───────────────
    log.info("Converting to UPTF7 format ...")
    train_uptf7 = _to_uptf7(splits["train"], specs)
    val_uptf7   = _to_uptf7(splits["val"], specs)
    test_uptf7  = _to_uptf7(splits["test"], specs)
    log.info("  train UPTF7: %s", train_uptf7.shape)

    # ── RevIN normalisation ─────────────────────────────────────────────
    # Compute stats on the *full* dataset (all splits concatenated) so
    # that train/val/test share the same per-trajectory normalisation.
    all_uptf7 = np.concatenate([train_uptf7, val_uptf7, test_uptf7], axis=0)
    n_train = train_uptf7.shape[0]
    n_val   = val_uptf7.shape[0]
    n_test  = test_uptf7.shape[0]

    revin = RevIN(str(norm_dir))
    norm_prefix = f"norm_{dataset_name}"
    revin.compute_stats(all_uptf7, prefix=norm_prefix)
    all_norm = revin.normalize(all_uptf7, prefix=norm_prefix)
    log.info("RevIN normalisation complete (prefix=%s)", norm_prefix)

    train_norm = all_norm[:n_train]
    val_norm   = all_norm[n_train : n_train + n_val]
    test_norm  = all_norm[n_train + n_val:]
    del all_uptf7, all_norm

    # ── AR data preparation ─────────────────────────────────────────────
    # FastARDataPreparer expects (N,T,D,H,W,C,F) internally
    # UPTF7 is (N,T,F,C,D,H,W) → transpose to (N,T,D,H,W,F,C)
    order = (0, 1, 4, 5, 6, 3, 2)  # (N,T,F,C,D,H,W) → (N,T,D,H,W,F,C)

    preparer = FastARDataPreparer(ar_order=args.ar_order)

    n_traj = args.n_traj if args.n_traj is not None else n_train
    n_traj = min(n_traj, n_train)
    log.info("Using %d / %d training trajectories", n_traj, n_train)

    X_tr, y_tr = preparer.prepare(train_norm[:n_traj].transpose(*order))
    X_va, y_va = preparer.prepare(val_norm.transpose(*order))
    X_te, y_te = preparer.prepare(test_norm.transpose(*order))
    log.info("AR pairs  train: X=%s y=%s", X_tr.shape, y_tr.shape)
    log.info("AR pairs  val:   X=%s y=%s", X_va.shape, y_va.shape)
    log.info("AR pairs  test:  X=%s y=%s", X_te.shape, y_te.shape)

    # Keep test_norm around for denormalised evaluation later
    test_norm_for_eval = test_norm.copy()
    del train_norm, val_norm, test_norm

    # ── DataLoaders ─────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    batch_size = args.batch_size
    if args.parallel == "dp" and n_gpus > 1:
        batch_size *= n_gpus

    tr_loader = DataLoader(PairDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(PairDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(PairDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    log.info("DataLoaders: train=%d  val=%d  test=%d batches",
             len(tr_loader), len(va_loader), len(te_loader))

    del X_tr, y_tr, X_va, y_va  # keep X_te, y_te for later if needed

    # ── Model ───────────────────────────────────────────────────────────
    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
    dropout, emb_dropout = args.tf_reg

    lora_r_attn = args.rank_lora_attn
    lora_r_mlp  = args.rank_lora_mlp

    ft_model = ViT3DRegression(
        patch_size=8, dim=dim, depth=depth,
        heads=heads, heads_xa=args.heads_xa, mlp_dim=mlp_dim,
        max_components=3, conv_filter=filters,
        max_ar=args.max_ar_order,
        max_patches=4096, max_fields=3,
        dropout=dropout, emb_dropout=emb_dropout,
        lora_r_attn=lora_r_attn, lora_r_mlp=lora_r_mlp,
        lora_alpha=None, lora_p=args.lora_p,
    ).to(device)

    n_params = sum(p.numel() for p in ft_model.parameters()) / 1e6
    log.info("Model params: %.3f M", n_params)

    if args.parallel == "dp" and n_gpus > 1:
        ft_model = nn.DataParallel(ft_model)
        log.info("Wrapped model in DataParallel (%d GPUs)", n_gpus)

    # ── Optimiser (via MORPH's SelectFineTuningParameters) ──────────────
    selector = SelectFineTuningParameters(ft_model, args)
    optimizer = selector.configure_levels()
    log.info("Level-%d fine-tuning | LR=%.2e  WD=%.2e",
             lev, optimizer.param_groups[0]["lr"],
             optimizer.param_groups[0]["weight_decay"])

    # ── Loss & scheduler ────────────────────────────────────────────────
    criterion = nn.MSELoss()
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

    # ── Load pretrained weights ─────────────────────────────────────────
    start_epoch = 0
    if args.ckpt_from == "FM":
        if args.download_model and args.checkpoint is None:
            from huggingface_hub import hf_hub_download
            fname = MORPH_FM_FILENAMES[args.model_size]
            log.info("Downloading FM weights: %s", fname)
            hf_hub_download(
                repo_id="mahindrautela/MORPH",
                filename=fname,
                subfolder="models/FM",
                repo_type="model",
                resume_download=True,
                local_dir=str(MORPH_ROOT),
                local_dir_use_symlinks=False,
            )
            ckpt_path = savepath_model / "FM" / fname
        elif args.checkpoint is not None:
            ckpt_path = savepath_model / "FM" / args.checkpoint
        else:
            fname = MORPH_FM_FILENAMES[args.model_size]
            ckpt_path = savepath_model / "FM" / fname

        log.info("Loading FM checkpoint: %s", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        state_dict = ckpt["model_state_dict"]

        target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model
        if state_dict and next(iter(state_dict)).startswith("module.") and args.parallel == "no":
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        strict = bool(args.ft_level4)
        missing, unexpected = target.load_state_dict(state_dict, strict=strict)
        log.info("Loaded FM weights (missing LoRA keys: %d, unexpected: %d)",
                 len(missing), len(unexpected))

    elif args.ckpt_from == "FT":
        resume_path = savepath_model_ds / args.checkpoint
        log.info("Resuming from FT checkpoint: %s", resume_path)
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=True)
        target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model
        sd = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        target.load_state_dict(sd, strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        log.info("Resumed at epoch %d", start_epoch)

    # ── Model name for checkpoint files ─────────────────────────────────
    model_tag = (
        f"ft_morph-{args.model_size}-{dataset_name}"
        f"-ar{args.ar_order}_max_ar{args.max_ar_order}"
        f"_lora{lora_r_attn}_ftlev{lev}"
        f"_lr{args.lr_level4}_wd{args.wd_level4}"
    )

    # ═══════════════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════
    model_path_prefix = str(savepath_model_ds / model_tag)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_path: str | None = None
    epochs_no_improve = 0

    log.info("Starting training for %d epochs (patience=%d) ...",
             args.n_epochs - start_epoch, args.patience)
    t_train_start = time.time()

    ft_model.train().to(device)

    for epoch in range(start_epoch, args.n_epochs):
        tr_loss = Trainer.train_singlestep(
            ft_model, tr_loader, criterion, optimizer, device,
            epoch, scheduler, model_path_prefix,
            args.save_batch_ckpt, args.save_batch_freq,
        )
        vl_loss = Trainer.validate_singlestep(
            ft_model, va_loader, criterion, device,
        )
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if args.lr_scheduler and scheduler is not None:
            scheduler.step(vl_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_min = (time.time() - t_train_start) / 60
        log.info(
            "Epoch %d/%d  train=%.5f  val=%.5f  LR=%.2e  (%.1f min)",
            epoch + 1, args.n_epochs, tr_loss, vl_loss, current_lr, elapsed_min,
        )

        # ── early stopping + best-checkpoint save ───────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0

            if (epoch + 1) % args.save_every == 0:
                ckpt_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": ft_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                if args.overwrite_weights:
                    ckpt_file = f"{model_path_prefix}_best.pth"
                else:
                    ckpt_file = f"{model_path_prefix}_ep{epoch + 1}.pth"
                torch.save(ckpt_dict, ckpt_file)
                best_ckpt_path = ckpt_file
                log.info("  ↳ new best val=%.5f  saved %s", vl_loss, ckpt_file)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                log.info("Early stopping at epoch %d (patience=%d)",
                         epoch + 1, args.patience)
                break

        # ── loss curve plot (overwritten each epoch) ────────────────────
        fig, ax = plt.subplots()
        ax.plot(train_losses, label="Train")
        ax.plot(val_losses, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        fig.savefig(str(savepath_results_ds / f"loss_{model_tag}.png"))
        plt.close(fig)

    train_minutes = (time.time() - t_train_start) / 60
    log.info("Training complete in %.1f min  (best val=%.5f @ epoch %d)",
             train_minutes, best_val_loss, best_epoch)

    # ═══════════════════════════════════════════════════════════════════
    #  RELOAD BEST CHECKPOINT FOR EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    if best_ckpt_path is not None and Path(best_ckpt_path).exists():
        log.info("Reloading best checkpoint for evaluation: %s", best_ckpt_path)
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model
        sd = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        target.load_state_dict(sd, strict=True)

    # ═══════════════════════════════════════════════════════════════════
    #  TEST EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    log.info("Evaluating on test set ...")
    ft_model.eval()

    out_all: list[torch.Tensor] = []
    tar_all: list[torch.Tensor] = []
    with torch.no_grad():
        for inp, tar in tqdm(te_loader, desc="Test eval", leave=False):
            inp = inp.to(device)
            _, _, out = ft_model(inp)
            out_all.append(out.detach().cpu())
            tar_all.append(tar)

    out_cat = torch.cat(out_all, dim=0)
    tar_cat = torch.cat(tar_all, dim=0)
    log.info("Predictions: %s  Targets: %s", out_cat.shape, tar_cat.shape)

    # ── normalised metrics ──────────────────────────────────────────────
    mse  = F.mse_loss(out_cat, tar_cat, reduction="mean")
    mae  = F.l1_loss(out_cat, tar_cat, reduction="mean")
    rmse = mse ** 0.5

    # ── denormalised metrics ────────────────────────────────────────────
    # Reshape flat (N*(T-1), F, C, D, H, W) back to (N_test, T-1, F, C, D, H, W)
    T = splits["test"].shape[1]
    T_minus_1 = T - args.ar_order
    reshape_target = (n_test, T_minus_1, *out_cat.shape[1:])

    out_rs = out_cat.reshape(reshape_target)
    tar_rs = tar_cat.reshape(reshape_target)

    outputs_denorm = RevIN.denormalize_testeval(
        str(norm_dir), norm_prefix, out_rs, dataset=dataset_name,
    )
    targets_denorm = RevIN.denormalize_testeval(
        str(norm_dir), norm_prefix, tar_rs, dataset=dataset_name,
    )

    vrmse = Metrics3DCalculator.calculate_VRMSE(outputs_denorm, targets_denorm).mean()
    nrmse = Metrics3DCalculator.calculate_NRMSE(outputs_denorm, targets_denorm).mean()

    # ── print summary ───────────────────────────────────────────────────
    log.info("═══ TEST RESULTS ═══")
    log.info("  MSE:   %.5f", mse.item())
    log.info("  MAE:   %.5f", mae.item())
    log.info("  RMSE:  %.5f", rmse.item())
    log.info("  VRMSE: %.5f", vrmse.item())
    log.info("  NRMSE: %.5f", nrmse.item())
    log.info("  Best val loss: %.5f  (epoch %d)", best_val_loss, best_epoch)
    log.info("  Checkpoint: %s", best_ckpt_path)

    # ── structured metrics file ─────────────────────────────────────────
    metrics = {
        "model_size": args.model_size,
        "ft_level": lev,
        "ar_order": args.ar_order,
        "max_ar_order": args.max_ar_order,
        "dataset_specs": specs,
        "n_train_traj": n_traj,
        "n_val_traj": n_val,
        "n_test_traj": n_test,
        "n_epochs_run": len(train_losses),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test": {
            "MSE":   mse.item(),
            "MAE":   mae.item(),
            "RMSE":  rmse.item(),
            "VRMSE": vrmse.item(),
            "NRMSE": nrmse.item(),
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint": best_ckpt_path,
        "training_minutes": round(train_minutes, 1),
    }
    metrics_path = savepath_results_ds / f"metrics_{model_tag}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    log.info("Metrics written to %s", metrics_path)

    # Also write a plain-text summary for quick reference
    txt_path = savepath_results_ds / f"metrics_{model_tag}.txt"
    txt_path.write_text(
        f"MSE: {mse.item():.5f}, MAE: {mae.item():.5f}, RMSE: {rmse.item():.5f}, "
        f"VRMSE: {vrmse.item():.5f}, NRMSE: {nrmse.item():.5f}\n"
        f"Best val loss: {best_val_loss:.5f} @ epoch {best_epoch}\n"
        f"Checkpoint: {best_ckpt_path}\n"
    )

    log.info("Done (total wall time: %.1f min)", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
