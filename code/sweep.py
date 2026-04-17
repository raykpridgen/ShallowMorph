"""
In-process MORPH finetuning sweep (single script, no subprocess).

Layout and I/O follow ``code/computer_setup.md``:

- Reads data from ``<MORPH>/datasets/normalized_revin/`` and FM weights from
  ``<MORPH>/models/FM/`` (same as ``MORPH/scripts/finetune_MORPH.py`` when that script
  lives under the MORPH clone).
- Writes this process under ``<repo>/out/sweep_{A|B}_{DATETIME}/``: ``models/`` (recovery
  + sweep B per-dataset bests), ``results/`` (metrics / plots mirror), ``sweep_metrics.csv``,
  and ``epoch_metrics.csv``. Stdout is unchanged so the job scheduler can capture it.

Loop order is optimized to avoid redundant I/O and rebuilds (see ``specs/plan.md``
for the Cartesian product of hyperparameters — order here does not change results):

1. ``ft_dataset`` — load raw train / val / test arrays once per dataset.
2. ``model_size`` — ``torch.load`` the FM checkpoint once per size (cached CPU dict).
3. ``ar_context`` — ``ViT3DRegression(max_ar=…)`` and ``FastARDataPreparer`` depend on
   context; build one model per (size, context) and apply cached FM weights.
4. ``train_frac`` — re-slice trajectories and re-run ``prepare()`` only (cheap vs HDF5).
5. ``n_epochs`` — reset weights to FM, new optimizer, run the epoch loop.

``Problematic`` CLI args from ``finetune_MORPH.py`` are resolved as follows:

- ``--ckpt_from`` / ``--checkpoint``: always start from FM for these sweeps; ``checkpoint``
  is the **basename** under ``<MORPH>/models/<model_choice>/`` (same contract as
  ``morph_wrap.morph_cli.finetune_argv``). Use ``ckpt_from='FT'`` only if you add resume
  logic later.
- Checkpoints: Sweep A writes only a per-run **recovery** checkpoint (overwritten each
  epoch, removed after a successful run). Sweep B additionally keeps ``best_ft_<DATASET>.pth``
  under the run's ``models/`` when validation improves the best seen for that dataset
  across the whole sweep; no writes to ``MORPH/models/<ft_dataset>/``.
- ``ar_order`` vs ``rollout_horizon``: training context is ``ar_order`` / ``max_ar_order``
  (kept equal). ``rollout_horizon`` only affects visualization length (default 50 here).
- ``batch_size``: ``None`` → same per-dataset defaults as upstream.
- ``patience``: set very large during sweeps so early stopping does not truncate planned
  epoch counts.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import time
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

# --- path layout: repo root + MORPH clone (matches MORPH/scripts/finetune_MORPH.py) ---
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_CODE_DIR, ".."))
MORPH_ROOT = os.path.join(REPO_ROOT, "MORPH")
if not os.path.isdir(MORPH_ROOT):
    raise RuntimeError(f"Expected MORPH at {MORPH_ROOT}")
sys.path.insert(0, MORPH_ROOT)

from config.data_config import DataConfig  # noqa: E402
from src.utils.data_preparation_fast import FastARDataPreparer  # noqa: E402
from src.utils.dataloaders.dataloaderchaos import DataloaderChaos  # noqa: E402
from src.utils.device_manager import DeviceManager  # noqa: E402
from src.utils.metrics_3d import Metrics3DCalculator  # noqa: E402
from src.utils.normalization import RevIN  # noqa: E402
from src.utils.select_fine_tuning_parameters import SelectFineTuningParameters  # noqa: E402
from src.utils.trainers import Trainer  # noqa: E402
from src.utils.visualize_predictions_3d_full import Visualize3DPredictions  # noqa: E402
from src.utils.visualize_rollouts_3d_full import Visualize3DRolloutPredictions  # noqa: E402
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression  # noqa: E402

# Optional: reuse trajectory caps from the subprocess driver (same semantics as README).
try:
    sys.path.insert(0, _CODE_DIR)
    from morph_wrap.sweep_config import TRAJECTORY_POOL  # type: ignore
except Exception:  # morph_wrap not on path or import error
    TRAJECTORY_POOL = None

# -----------------------------------------------------------------------------
# Sweep selection and grids (edit here)
# -----------------------------------------------------------------------------

SWEEP = "A"  # "A" or "B"

# If None, raw HDF5 lives under ``<MORPH_ROOT>/datasets/normalized_revin/...`` (upstream).
# Set to an explicit root only when your tree matches finetune's join layout.
DATASET_ROOT: str | None = None

MODEL_CHOICE = "FM"
PARALLEL = "dp"
DEVICE_IDX = 0

# Study datasets (MORPH ``ft_dataset`` names)
FT_DATASETS: List[str] = ["BE1D", "SW", "DR2D"]

SWEEP_A = {
    "train_size": [0.1, 0.5],  # fraction of train-split trajectories (or percent if > 1)
    "epoch_size": [10, 50],
    "model_type": ["Ti", "S"],
    "ar_context": [1, 5, 10],
}

SWEEP_B = {
    "train_size": [0.1, 0.25, 0.5, 1.0],
    "epoch_size": [10, 50, 100, 200],
    "model_type": ["Ti", "S", "L"],
}

# After sweep A: fill from ``morph_wrap.pick_context`` JSON or by hand.
SWEEP_B_CONTEXTS: Dict[str, int] = {
    "BE1D": 1,
    "SW": 5,
    "DR2D": 10,
}

FM_CHECKPOINT_BASENAME: Dict[str, str] = {
    "Ti": "morph_ti_fm.pth",
    "S": "morph_s_fm.pth",
    "M": "morph_m_fm.pth",
    "L": "morph_l_fm.pth",
}

MORPH_MODELS = {
    "Ti": [8, 256, 4, 4, 1024],
    "S": [8, 512, 8, 4, 2048],
    "M": [8, 768, 12, 8, 3072],
    "L": [8, 1024, 16, 16, 4096],
}

# I/O toggles
SAVE_METRICS = True
SAVE_LOSS_FIGURES = False  # upstream saves every epoch — expensive for large sweeps
RUN_VIZ = False  # single-step + rollout PNGs; enable for a few pilot runs

PATCH_SIZE = 8
# Constructed in upstream finetune for side effects / path hooks; keep import parity.
DataConfig(MORPH_ROOT, PATCH_SIZE)

SWEEP_METRICS_FIELDS: List[str] = [
    "run_id",
    "sweep",
    "dataset",
    "model_size",
    "ar_context",
    "train_frac",
    "n_traj",
    "n_epochs",
    "best_val_loss",
    "mae",
    "mse",
    "rmse",
    "nrmse",
    "vrmse",
    "duration_sec",
]
EPOCH_METRICS_FIELDS: List[str] = [
    "run_id",
    "sweep",
    "dataset",
    "model_size",
    "ar_context",
    "train_frac",
    "n_traj",
    "n_epochs",
    "epoch",
    "train_loss",
    "val_loss",
    "lr",
]


def _ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    if not os.path.isfile(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def _scalar_float(x: Any) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _make_run_id(
    sweep: str,
    ft_dataset: str,
    model_size: str,
    ar_context: int,
    train_frac: float,
    n_traj: int,
    n_epochs: int,
) -> str:
    tf = str(train_frac).replace(".", "p")
    return f"{sweep}_{ft_dataset}_{model_size}_ar{ar_context}_tf{tf}_ntr{n_traj}_ep{n_epochs}"


# -----------------------------------------------------------------------------
# Defaults mirroring finetune_MORPH argparse (non-swept knobs)
# -----------------------------------------------------------------------------


def base_train_args() -> Namespace:
    """Static hyperparameters and flags; swept fields are overwritten per run."""
    return Namespace(
        model_choice=MODEL_CHOICE,
        parallel=PARALLEL,
        device_idx=DEVICE_IDX,
        ckpt_from="FM",
        # LoRA / transformer defaults (match morph_cli / upstream argparse)
        rank_lora_attn=16,
        rank_lora_mlp=12,
        lora_p=0.05,
        tf_reg=[0.1, 0.1],
        heads_xa=32,
        test_sample=0,
        rollout_horizon=50,
        batch_size=None,
        lr_level4=1e-4,
        wd_level4=0.0,
        save_batch_ckpt=False,
        save_batch_freq=1000,
        # Level 3 == LoRA + encoder + decoder (requires all three flags in upstream script)
        ft_level1=True,
        ft_level2=True,
        ft_level3=True,
        ft_level4=False,
    )


def resolve_n_traj(ft_dataset: str, train_frac: float, train_rows: int) -> int:
    """``train_frac`` in (0,1] = fraction of loaded train trajectories; >1 treated as percent."""
    if TRAJECTORY_POOL is not None:
        key_map = {"BE1D": "BE1D", "SW": "SW2D", "DR2D": "DR2D"}
        pool_key = key_map.get(ft_dataset)
        if pool_key and pool_key in TRAJECTORY_POOL:
            cap = TRAJECTORY_POOL[pool_key]
            pct = train_frac * 100.0 if train_frac <= 1.0 else train_frac
            n = max(1, int(cap * pct / 100.0))
            return min(n, train_rows)
    if train_frac <= 1.0:
        n = max(1, int(train_rows * train_frac))
    else:
        n = max(1, int(train_rows * train_frac / 100.0))
    return min(n, train_rows)


def build_datapaths(dataset_root: str) -> Dict[str, str]:
    datasets = [
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
    base = os.path.join(dataset_root, "datasets", "normalized_revin")
    paths = {
        "MHD": os.path.join(base, datasets[1]),
        "DR": os.path.join(base, datasets[0]),
        "CFD1D": os.path.join(base, datasets[2]),
        "SW": os.path.join(base, datasets[3]),
        "CFD2D-IC": os.path.join(base, datasets[4]),
        "CFD3D": os.path.join(base, datasets[5]),
        "DR1D": os.path.join(base, datasets[6]),
        "CFD2D": os.path.join(base, datasets[7]),
        "CFD3D-TURB": os.path.join(base, datasets[8]),
        "BE1D": os.path.join(base, datasets[9]),
        "GSDR2D": os.path.join(base, datasets[10]),
        "TGC3D": os.path.join(base, datasets[11]),
        "FNS_KF_2D": os.path.join(base, datasets[12]),
        "DR2D": os.path.join(base, datasets[0]),
    }
    return paths


class _ArrayDataset(Dataset):
    __slots__ = ("X", "y")

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _finetune_level(args: Namespace) -> int:
    if args.ft_level4:
        return 4
    if args.ft_level1 and args.ft_level2 and args.ft_level3:
        return 3
    if args.ft_level1 and args.ft_level2:
        return 2
    if args.ft_level1:
        return 1
    raise ValueError("Select a fine-tuning level: ft_level1/2/3 or ft_level4")


def _load_fm_state_dict(savepath_model: str, model_choice: str, ckpt_basename: str) -> dict:
    path = os.path.join(savepath_model, model_choice, ckpt_basename)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt["model_state_dict"]


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if next(iter(state_dict)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _apply_fm_weights(ft_model: nn.Module, fm_state: dict, parallel: str) -> None:
    target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model
    sd = dict(fm_state)
    if parallel == "no":
        sd = _strip_module_prefix(sd)
    inc = target.load_state_dict(sd, strict=False)
    miss = [k for k in inc.missing_keys if k.endswith((".A", ".B")) or ".lora" in k]
    if miss:
        print("→ Missing keys (expected LoRA etc.):", miss[:8], "..." if len(miss) > 8 else "")
    if inc.unexpected_keys:
        print("→ Unexpected keys:", inc.unexpected_keys[:8], "..." if len(inc.unexpected_keys) > 8 else "")


def run_one_finetune(
    *,
    args: Namespace,
    run_id: str,
    run_dir: str,
    sweep_key: str,
    ft_dataset: str,
    train_frac: float,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    fm_state_cpu: dict,
    device: torch.device,
    savepath_results: str,
    loadpath_muvar: str,
    n_traj: int,
    n_epochs: int,
    sweep_metrics_csv: str,
    epoch_metrics_csv: str,
    dataset_best_val: Dict[str, float],
    save_run_best_weights: bool,
) -> None:
    """One full finetune + metrics (+ optional viz). Mutates nothing in the numpy buffers."""
    run_t0 = time.time()
    run_models = os.path.join(run_dir, "models")
    trainer_stub_dir = os.path.join(run_models, ".trainer", run_id)
    os.makedirs(trainer_stub_dir, exist_ok=True)
    recovery_path = os.path.join(run_models, f"recovery_{run_id}.pth")
    run_best_path = os.path.join(run_models, f"_run_best_{run_id}.pth")

    args.ft_dataset = ft_dataset
    args.n_epochs = n_epochs
    args.n_traj = n_traj
    args.checkpoint = FM_CHECKPOINT_BASENAME[args.model_size]

    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
    dropout, emb_dropout = args.tf_reg
    lev = _finetune_level(args)

    batch_sizes = {
        "DR1D": 384 // 2,
        "CFD2D": 64 // 2,
        "CFD3D-TURB": 16 // 2,
        "BE1D": 384 // 2,
        "GSDR2D": 64 // 2,
        "TGC3D": 16 // 2,
        "FNS_KF_2D": 64 // 2,
        "SW": 64 // 2,
        "DR2D": 64 // 2,
    }
    batch_size = args.batch_size if args.batch_size is not None else batch_sizes[ft_dataset]

    norm_prefix = f"stats_{ft_dataset.lower()}"

    max_patches, max_fields, max_components = 4096, 3, 3
    ft_model = ViT3DRegression(
        patch_size=PATCH_SIZE,
        dim=dim,
        depth=depth,
        heads=heads,
        heads_xa=args.heads_xa,
        mlp_dim=mlp_dim,
        max_components=max_components,
        conv_filter=filters,
        max_ar=args.max_ar_order,
        max_patches=max_patches,
        max_fields=max_fields,
        dropout=dropout,
        emb_dropout=emb_dropout,
        lora_r_attn=args.rank_lora_attn,
        lora_r_mlp=args.rank_lora_mlp,
        lora_alpha=None,
        lora_p=args.lora_p,
    ).to(device)

    n_gpus = torch.cuda.device_count()
    if args.parallel == "dp" and n_gpus > 1:
        ft_model = nn.DataParallel(ft_model)
        batch_size = n_gpus * batch_size

    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1")

    preparer = FastARDataPreparer(ar_order=args.ar_order)
    print(f"→ [{ft_dataset}] n_traj={n_traj}, ar={args.ar_order}, epochs={n_epochs}")

    X_tr, y_tr = preparer.prepare(train_data[0:n_traj])
    X_va, y_va = preparer.prepare(val_data[0 : int(n_traj * 0.125)])
    X_te, y_te = preparer.prepare(test_data)
    assert X_tr.shape[0] == n_traj * (train_data.shape[1] - args.ar_order), (
        "Shape mismatch (check ar_order vs time length)"
    )

    ft_tr_loader = DataLoader(_ArrayDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    ft_va_loader = DataLoader(_ArrayDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    ft_te_loader = DataLoader(_ArrayDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    selector = SelectFineTuningParameters(ft_model, args)
    optimizer = selector.configure_levels()
    ft_model.train().to(device)

    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    if args.ckpt_from != "FM":
        raise NotImplementedError("In-process sweep always resets from FM; use FT only with resume support.")

    _apply_fm_weights(ft_model, fm_state_cpu, args.parallel)

    # ``Trainer`` only touches ``model_path`` when ``save_batch_ckpt`` is true (upstream).
    model_path = os.path.join(trainer_stub_dir, "trainer_stub")

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    ep_st = time.time()
    start_epoch = 0
    run_ok = False

    try:
        for epoch in range(start_epoch, n_epochs):
            tr_loss = Trainer.train_singlestep(
                ft_model,
                ft_tr_loader,
                criterion,
                optimizer,
                device,
                epoch,
                scheduler,
                model_path,
                args.save_batch_ckpt,
                args.save_batch_freq,
            )
            vl_loss = Trainer.validate_singlestep(ft_model, ft_va_loader, criterion, device)
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            scheduler.step(vl_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Time = {(time.time() - ep_st) / 60:.2f} min., LR:{current_lr:.6f}, "
                f"Epoch {epoch + 1}/{n_epochs} | Train:{tr_loss:.5f}, Val:{vl_loss:.5f}"
            )

            _append_csv_row(
                epoch_metrics_csv,
                EPOCH_METRICS_FIELDS,
                {
                    "run_id": run_id,
                    "sweep": sweep_key,
                    "dataset": ft_dataset,
                    "model_size": args.model_size,
                    "ar_context": args.ar_order,
                    "train_frac": train_frac,
                    "n_traj": n_traj,
                    "n_epochs": n_epochs,
                    "epoch": epoch + 1,
                    "train_loss": f"{tr_loss:.8f}",
                    "val_loss": f"{vl_loss:.8f}",
                    "lr": f"{current_lr:.8f}",
                },
            )

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                epochs_no_improve = 0
                if save_run_best_weights:
                    ckpt_dict = {
                        "epoch": epoch + 1,
                        "run_id": run_id,
                        "sweep": sweep_key,
                        "ft_dataset": ft_dataset,
                        "val_loss": vl_loss,
                        "model_state_dict": ft_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    torch.save(ckpt_dict, run_best_path)
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")

            rec = {
                "epoch": epoch + 1,
                "run_id": run_id,
                "model_state_dict": ft_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(rec, recovery_path)

            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping triggered: validation did not improve for {args.patience} epochs."
                )
                break

            if SAVE_LOSS_FIGURES:
                fig, ax = plt.subplots()
                ax.plot(train_losses, label="Train")
                ax.plot(val_losses, label="Val")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                fig.savefig(
                    os.path.join(
                        savepath_results,
                        (
                            f"loss_{args.model_choice}_{ft_dataset}_max_ar_{args.max_ar_order}_"
                            f"rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}.png"
                        ),
                    )
                )
                plt.close(fig)

        run_ok = True
    finally:
        if run_ok and os.path.isfile(recovery_path):
            try:
                os.remove(recovery_path)
                print(f"→ Removed recovery checkpoint (run finished): {recovery_path}")
            except OSError as e:
                print(f"→ Warning: could not remove recovery file: {e}")

    del X_tr, y_tr, X_va, y_va

    # --- metrics (normalized space) ---
    ft_model.eval()
    out_all, tar_all = [], []
    with torch.no_grad():
        for inp, tar in tqdm(ft_te_loader, desc="test"):
            inp = inp.to(device)
            out = ft_model(inp)
            out_all.append(out.detach().cpu())
            tar_all.append(tar)
    out_all = torch.concat(out_all, dim=0)
    tar_all = torch.concat(tar_all, dim=0)
    mse = F.mse_loss(out_all, tar_all, reduction="mean")
    mae = F.l1_loss(out_all, tar_all, reduction="mean")
    rmse = mse**0.5

    td_out = torch.from_numpy(test_data[:, 1:])
    td_out = td_out.permute(0, 1, 6, 5, 2, 3, 4)
    out_all_rs = out_all.reshape(td_out.shape)
    tar_all_rs = tar_all.reshape(td_out.shape)

    outputs_denorm = RevIN.denormalize_testeval(
        loadpath_muvar, norm_prefix, out_all_rs, dataset=ft_dataset
    )
    targets_denorm = RevIN.denormalize_testeval(
        loadpath_muvar, norm_prefix, tar_all_rs, dataset=ft_dataset
    )
    vrmse = Metrics3DCalculator.calculate_VRMSE(outputs_denorm, targets_denorm).mean()
    nrmse = Metrics3DCalculator.calculate_NRMSE(outputs_denorm, targets_denorm).mean()
    print(
        f"→ RMSE: {rmse:.5f}, MAE: {mae:.5f}, MSE: {mse:.5f} VRMSE: {vrmse:.5f}, NRMSE: {nrmse:.5f}"
    )

    duration = time.time() - run_t0

    if SAVE_METRICS:
        savepath_results_ = os.path.join(savepath_results, ft_dataset)
        os.makedirs(savepath_results_, exist_ok=True)
        metrics_str = (
            f" MAE: {mae:.5f}, MSE: {mse:.5f}, RMSE: {rmse:.5f}, NRMSE: {nrmse:.5f}, VRMSE: {vrmse:.5f}"
        )
        metrics_name = os.path.join(
            savepath_results_,
            (
                f"metrics_MORPH-{args.model_size}_{args.model_choice}_ar{args.max_ar_order}_"
                f"tot-trajs{n_traj}_tot-eps{n_epochs}_rank-lora{args.rank_lora_attn}_ftlevel{lev}_"
                f"lr{args.lr_level4}_wd{args.wd_level4}.txt"
            ),
        )
        with open(metrics_name, "w", encoding="utf-8") as f:
            f.write(metrics_str)
        print(f"→ Metrics written to {metrics_name}")

    _append_csv_row(
        sweep_metrics_csv,
        SWEEP_METRICS_FIELDS,
        {
            "run_id": run_id,
            "sweep": sweep_key,
            "dataset": ft_dataset,
            "model_size": args.model_size,
            "ar_context": args.ar_order,
            "train_frac": train_frac,
            "n_traj": n_traj,
            "n_epochs": n_epochs,
            "best_val_loss": f"{best_val_loss:.8f}",
            "mae": f"{_scalar_float(mae):.8f}",
            "mse": f"{_scalar_float(mse):.8f}",
            "rmse": f"{_scalar_float(rmse):.8f}",
            "nrmse": f"{_scalar_float(nrmse):.8f}",
            "vrmse": f"{_scalar_float(vrmse):.8f}",
            "duration_sec": f"{duration:.3f}",
        },
    )

    if save_run_best_weights and os.path.isfile(run_best_path):
        prev = dataset_best_val.get(ft_dataset, float("inf"))
        if best_val_loss < prev:
            dest = os.path.join(run_models, f"best_ft_{ft_dataset}.pth")
            ck = torch.load(run_best_path, map_location="cpu", weights_only=True)
            ck["sweep"] = sweep_key
            ck["run_id"] = run_id
            ck["ft_dataset"] = ft_dataset
            ck["selection_val_loss"] = float(best_val_loss)
            torch.save(ck, dest)
            dataset_best_val[ft_dataset] = best_val_loss
            print(f"→ Dataset-wide best checkpoint for {ft_dataset}: {dest} (val={best_val_loss:.6f})")
        try:
            os.remove(run_best_path)
        except OSError:
            pass

    if os.path.isdir(trainer_stub_dir):
        shutil.rmtree(trainer_stub_dir, ignore_errors=True)

    if RUN_VIZ:
        sim = test_data[args.test_sample]
        sim_rs = np.transpose(sim, (0, 5, 4, 1, 2, 3)).astype(np.float32)
        sim_tensor = torch.from_numpy(sim_rs).unsqueeze(0)
        field_names = [f"field-{fi}" for fi in range(1, sim_tensor.shape[2] + 1)]
        slice_dim = "d" if ft_dataset not in ["CFD1D", "DR1D"] else "1d"
        savepath_results_ = os.path.join(savepath_results, ft_dataset)
        os.makedirs(savepath_results_, exist_ok=True)

        viz = Visualize3DPredictions(ft_model, sim_tensor, device)
        figurename = (
            f"ft_st_MORPH-{args.model_size}_{args.model_choice}_ar{args.max_ar_order}_chAll_"
            f"samp{args.test_sample}_tot-trajs{n_traj}_tot-eps{n_epochs}_"
            f"rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_t"
        )
        for t in range(args.rollout_horizon):
            viz.visualize_predictions(
                time_step=t,
                component=0,
                slice_dim=slice_dim,
                save_path=savepath_results_,
                figname=f"{figurename}{t}.png",
            )

        viz_roll = Visualize3DRolloutPredictions(
            model=ft_model,
            test_dataset=sim_tensor,
            device=device,
            field_names=field_names,
            component_names=["d", "h", "w"],
        )
        figurename = (
            f"ft_ro_MORPH-{args.model_size}_{args.model_choice}_ar{args.max_ar_order}_tAll_"
            f"samp{args.test_sample}_tot-trajs{n_traj}_tot-eps{n_epochs}_"
            f"rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_field"
        )
        for fi in range(sim_tensor.shape[2]):
            viz_roll.visualize_rollout(
                start_step=0,
                num_steps=args.rollout_horizon,
                field=fi,
                component=0,
                slice_dim=slice_dim,
                save_path=savepath_results_,
                figname=f"{figurename}{fi}.png",
            )

    del ft_model, optimizer, ft_tr_loader, ft_va_loader, ft_te_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    sweep_conf = SWEEP_A if SWEEP == "A" else SWEEP_B
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO_ROOT, "out", f"sweep_{SWEEP}_{stamp}")
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)

    sweep_metrics_csv = os.path.join(run_dir, "sweep_metrics.csv")
    epoch_metrics_csv = os.path.join(run_dir, "epoch_metrics.csv")
    _ensure_csv_header(sweep_metrics_csv, SWEEP_METRICS_FIELDS)
    _ensure_csv_header(epoch_metrics_csv, EPOCH_METRICS_FIELDS)

    meta_path = os.path.join(run_dir, "sweep_setup.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(
            f"sweep={SWEEP}\n"
            f"started_utc={stamp}\n"
            f"REPO_ROOT={REPO_ROOT}\n"
            f"MORPH_ROOT={MORPH_ROOT}\n"
            f"DATASET_ROOT_config={repr(DATASET_ROOT)}\n"
            f"DATASET_ROOT_effective={DATASET_ROOT or MORPH_ROOT}\n"
            f"fm_models_dir={os.path.join(MORPH_ROOT, 'models', MODEL_CHOICE)}\n"
        )

    dataset_root = DATASET_ROOT if DATASET_ROOT is not None else MORPH_ROOT
    datapaths = build_datapaths(dataset_root)

    devices = DeviceManager.list_devices()
    device = devices[DEVICE_IDX] if devices else torch.device("cpu")

    fm_models_root = os.path.join(MORPH_ROOT, "models")
    savepath_results = os.path.join(run_dir, "results")
    loadpath_muvar = os.path.join(MORPH_ROOT, "data")

    data_module = DataloaderChaos()
    fm_cache: Dict[str, dict] = {}
    dataset_best_val: Dict[str, float] = {ds: float("inf") for ds in FT_DATASETS}
    save_run_best_weights = SWEEP == "B"

    print(f"→ Sweep output directory: {run_dir}")
    print(f"→ Loading FM checkpoints from {os.path.join(fm_models_root, MODEL_CHOICE)}")

    for ft_dataset in FT_DATASETS:
        print(f"\n=== Dataset {ft_dataset}: load raw arrays once ===")
        path = datapaths[ft_dataset]
        train_data, val_data = data_module.load_data(ft_dataset, path, split="train")
        test_data = data_module.load_data(ft_dataset, path, split="test")
        print(
            f"[{ft_dataset}] train {train_data.shape}, val {val_data.shape}, test {test_data.shape}"
        )

        for model_size in sweep_conf["model_type"]:
            ckpt_file = FM_CHECKPOINT_BASENAME.get(model_size)
            if not ckpt_file:
                raise KeyError(f"No FM checkpoint basename for model_size={model_size}")
            if model_size not in fm_cache:
                print(f"→ Cache FM weights: {ckpt_file}")
                fm_cache[model_size] = _load_fm_state_dict(fm_models_root, MODEL_CHOICE, ckpt_file)
            fm_state = fm_cache[model_size]

            contexts = (
                sweep_conf["ar_context"] if SWEEP == "A" else [SWEEP_B_CONTEXTS[ft_dataset]]
            )

            for ar_context in contexts:
                args = base_train_args()
                args.model_size = model_size
                args.ar_order = ar_context
                args.max_ar_order = ar_context
                args.patience = 10**6
                if SWEEP == "A":
                    args.save_every = 10**9
                    args.overwrite_weights = False
                else:
                    args.save_every = 1
                    args.overwrite_weights = True

                for train_frac in sweep_conf["train_size"]:
                    n_traj = resolve_n_traj(ft_dataset, train_frac, train_data.shape[0])
                    for n_epochs in sweep_conf["epoch_size"]:
                        run_id = _make_run_id(
                            SWEEP, ft_dataset, model_size, ar_context, train_frac, n_traj, n_epochs
                        )
                        print(
                            f"\n--- Run run_id={run_id} | SWEEP={SWEEP} ds={ft_dataset} "
                            f"model={model_size} context={ar_context} train_frac={train_frac} "
                            f"n_traj={n_traj} epochs={n_epochs} ---"
                        )
                        run_one_finetune(
                            args=args,
                            run_id=run_id,
                            run_dir=run_dir,
                            sweep_key=SWEEP,
                            ft_dataset=ft_dataset,
                            train_frac=train_frac,
                            train_data=train_data,
                            val_data=val_data,
                            test_data=test_data,
                            fm_state_cpu=fm_state,
                            device=device,
                            savepath_results=savepath_results,
                            loadpath_muvar=loadpath_muvar,
                            n_traj=n_traj,
                            n_epochs=n_epochs,
                            sweep_metrics_csv=sweep_metrics_csv,
                            epoch_metrics_csv=epoch_metrics_csv,
                            dataset_best_val=dataset_best_val,
                            save_run_best_weights=save_run_best_weights,
                        )

        del train_data, val_data, test_data


if __name__ == "__main__":
    main()
