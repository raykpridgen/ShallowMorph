"""Build argv for MORPH ``finetune_MORPH.py`` (and optional ``infer_MORPH.py``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

from morph_wrap.sweep_config import SweepJob, lr_for


def finetune_argv(
    job: SweepJob,
    *,
    morph_root: Path,
    dataset_root: Path,
    fm_checkpoint: str,
    model_choice: str = "FM",
    ckpt_from: str = "FM",
    parallel: str = "dp",
    patience: int = 10_000,
    device_idx: int = 0,  # pass value from morph_wrap.device_resolve.resolve_training_device_index
    ft_level1: bool = True,
    ft_level2: bool = True,
    ft_level3: bool = True,
    rank_lora_attn: int = 16,
    rank_lora_mlp: int = 12,
    rollout_horizon: int = 10,
    extra_args: Sequence[str] | None = None,
) -> List[str]:
    """
    MORPH finetune script expects ``--checkpoint`` as basename under
    ``models/<model_choice>/`` when ``ckpt_from=='FM'``.

    Multi-frame context uses MORPH's ``FastARDataPreparer``:
    ``--ar_order`` and ``--max_ar_order`` match ``job.context_frames`` (I3-A).

    Fine-tuning levels: pass ``--ft_level1 --ft_level2 --ft_level3`` so the script
    selects encoder+decoder partial unfreeze (see ``select_fine_tuning_parameters``).
    """
    script = morph_root / "scripts" / "finetune_MORPH.py"
    # Pilot LRs (I5-C): see ``LR_TABLE`` / manifest column ``lr_pilot_table`` and env
    # ``MORPH_PILOT_LR``; MORPH level 1–3 optimizer LRs live in
    # ``select_fine_tuning_parameters.py`` unless you patch them to read the env.

    argv: List[str] = [
        os.fspath(script),
        "--dataset_root",
        os.fspath(dataset_root),
        "--model_choice",
        model_choice,
        "--model_size",
        job.morph_model_size,
        "--ckpt_from",
        ckpt_from,
        "--checkpoint",
        fm_checkpoint,
        "--ft_dataset",
        job.morph_ft_dataset,
        "--n_epochs",
        str(job.epochs),
        "--n_traj",
        str(job.n_traj),
        "--ar_order",
        str(job.context_frames),
        "--max_ar_order",
        str(job.context_frames),
        "--parallel",
        parallel,
        "--device_idx",
        str(device_idx),
        "--patience",
        str(patience),
        "--rollout_horizon",
        str(rollout_horizon),
        "--rank_lora_attn",
        str(rank_lora_attn),
        "--rank_lora_mlp",
        str(rank_lora_mlp),
    ]

    if ft_level1:
        argv.append("--ft_level1")
    if ft_level2:
        argv.append("--ft_level2")
    if ft_level3:
        argv.append("--ft_level3")

    # Pass sweep metadata into env for a thin logging hook inside MORPH (optional).
    # Finetune script ignores unknown env vars.
    if extra_args:
        argv.extend(extra_args)

    return argv


def finetune_env(
    job: SweepJob,
    repo_out: Path,
    *,
    pilot_lr_override: str | None = None,
) -> dict:
    """Environment additions for optional logging wrappers."""
    pilot_lr = (
        pilot_lr_override
        if pilot_lr_override is not None
        else str(lr_for(job.dataset, job.model, job.context_frames))
    )
    return {
        **os.environ,
        "MORPH_SWEEP_NAME": f"sweep{job.sweep}",
        "MORPH_RUN_ID": job.run_id,
        "MORPH_SWEEP_OUT": os.fspath(repo_out),
        "MORPH_TRAIN_FRAC": str(job.train_frac),
        "MORPH_CONTEXT_FRAMES": str(job.context_frames),
        "MORPH_PILOT_LR": pilot_lr,
    }
