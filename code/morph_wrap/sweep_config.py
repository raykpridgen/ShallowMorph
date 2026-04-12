"""
Single source of truth for sweep grids and MORPH CLI mapping.

Trajectory pool: used to set --n_traj ≈ train_frac% of this cap. After your first
local load, set these to match ``train_data.shape[0]`` from ``finetune_MORPH.py``
for each dataset (train split size).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple

SweepName = Literal["A", "B"]
DatasetKey = Literal["BE1D", "SW2D", "DR2D"]
ModelKey = Literal["tiny", "small", "large"]

# Order matches specs/plan.md nested loops.
DATASET_ORDER: Tuple[DatasetKey, ...] = ("BE1D", "SW2D", "DR2D")

MORPH_FT_DATASET: Dict[DatasetKey, str] = {
    "BE1D": "BE1D",
    "SW2D": "SW",
    "DR2D": "DR2D",
}

MORPH_MODEL_SIZE: Dict[ModelKey, str] = {
    "tiny": "Ti",
    "small": "S",
    "large": "L",
}

# Basename under MORPH/models/<model_choice>/ when ckpt_from == FM (edit to match your files).
FM_CHECKPOINT_BASENAME: Dict[ModelKey, str] = {
    "tiny": "morph_ti_fm.pth",
    "small": "morph_s_fm.pth",
    "large": "morph_l_fm.pth",
}

# Upper bound on train trajectories for --n_traj = max(1, int(pool * train_frac / 100)).
TRAJECTORY_POOL: Dict[DatasetKey, int] = {
    "BE1D": 7000,
    "SW2D": 700,
    "DR2D": 700,
}

# Canonical shape strings for runs.csv (from specs/plan.md / design §2.2).
DATA_DIMS: Dict[DatasetKey, str] = {
    "BE1D": "(10000,201,1024) raw tensor / MORPH inflated",
    "SW2D": "(101,128,128,1) per group HDF5",
    "DR2D": "(101,128,128,2) per group HDF5 (diffusion–reaction 2D)",
}

# Learning rates after pilot (I5-C). Keys: (dataset, model_key, context_frames).
# Replace with pilot results; default is a single conservative LR everywhere.
def default_lr_table() -> Dict[Tuple[DatasetKey, ModelKey, int], float]:
    out: Dict[Tuple[DatasetKey, ModelKey, int], float] = {}
    for ds in DATASET_ORDER:
        for mk in ("tiny", "small", "large"):
            for n in (1, 5, 10):
                out[(ds, mk, n)] = 1e-4
    return out


LR_TABLE: Dict[Tuple[DatasetKey, ModelKey, int], float] = default_lr_table()


def fm_checkpoint_for(model: ModelKey, override: str | None = None) -> str:
    if override:
        return override
    return FM_CHECKPOINT_BASENAME[model]


def lr_for(dataset: DatasetKey, model: ModelKey, context_frames: int) -> float:
    key = (dataset, model, context_frames)
    if key in LR_TABLE:
        return LR_TABLE[key]
    # Sweep B may use context not in {1,5,10}; fall back per dataset+model.
    for n in (context_frames, 10, 5, 1):
        k = (dataset, model, n)
        if k in LR_TABLE:
            return LR_TABLE[k]
    return 1e-4


@dataclass(frozen=True)
class SweepJob:
    sweep: SweepName
    combo_index: int
    combo_total: int
    dataset: DatasetKey
    model: ModelKey
    train_frac: int
    epochs: int
    context_frames: int
    run_id: str

    @property
    def morph_ft_dataset(self) -> str:
        return MORPH_FT_DATASET[self.dataset]

    @property
    def morph_model_size(self) -> str:
        return MORPH_MODEL_SIZE[self.model]

    @property
    def n_traj(self) -> int:
        pool = TRAJECTORY_POOL[self.dataset]
        return max(1, int(pool * self.train_frac / 100.0))


def make_run_id(
    sweep: SweepName,
    dataset: DatasetKey,
    model: ModelKey,
    train_frac: int,
    epochs: int,
    context_frames: int,
) -> str:
    morph_sz = MORPH_MODEL_SIZE[model]
    return (
        f"{sweep}_{dataset}_{model}_p{train_frac}"
        f"_e{epochs}_n{context_frames}_{morph_sz}"
    )


def sweep_a_jobs() -> List[SweepJob]:
    models: Tuple[ModelKey, ...] = ("tiny", "small")
    train_fracs = (10, 50)
    epochs_list = (10, 50)
    contexts = (1, 5, 10)

    flat: List[Tuple[DatasetKey, ModelKey, int, int, int]] = []
    for ds in DATASET_ORDER:
        for mk in models:
            for tf in train_fracs:
                for ep in epochs_list:
                    for n in contexts:
                        flat.append((ds, mk, tf, ep, n))
    total = len(flat)
    out: List[SweepJob] = []
    for i, (ds, mk, tf, ep, n) in enumerate(flat):
        out.append(
            SweepJob(
                sweep="A",
                combo_index=i + 1,
                combo_total=total,
                dataset=ds,
                model=mk,
                train_frac=tf,
                epochs=ep,
                context_frames=n,
                run_id=make_run_id("A", ds, mk, tf, ep, n),
            )
        )
    return out


def sweep_b_jobs(
    context_by_dataset: Dict[DatasetKey, int],
    *,
    models: Optional[Tuple[ModelKey, ...]] = None,
) -> List[SweepJob]:
    if models is None:
        models = ("tiny", "small", "large")
    train_fracs = (10, 25, 50, 100)
    epochs_list = (10, 50, 100, 200)

    flat: List[Tuple[DatasetKey, ModelKey, int, int, int]] = []
    for ds in DATASET_ORDER:
        if ds not in context_by_dataset:
            raise KeyError(
                f"Missing context_frames for dataset {ds} in context_by_dataset"
            )
        n_ctx = context_by_dataset[ds]
        for mk in models:
            for tf in train_fracs:
                for ep in epochs_list:
                    flat.append((ds, mk, tf, ep, n_ctx))
    total = len(flat)
    out: List[SweepJob] = []
    for i, (ds, mk, tf, ep, n_ctx) in enumerate(flat):
        out.append(
            SweepJob(
                sweep="B",
                combo_index=i + 1,
                combo_total=total,
                dataset=ds,
                model=mk,
                train_frac=tf,
                epochs=ep,
                context_frames=n_ctx,
                run_id=make_run_id("B", ds, mk, tf, ep, n_ctx),
            )
        )
    return out


def staged_models(stage: Literal["small", "large", "all"]) -> FrozenSet[ModelKey]:
    if stage == "small":
        return frozenset({"tiny", "small"})
    if stage == "large":
        return frozenset({"large"})
    return frozenset({"tiny", "small", "large"})


def filter_jobs_by_models(
    jobs: List[SweepJob], keep: FrozenSet[ModelKey]
) -> List[SweepJob]:
    filtered = [j for j in jobs if j.model in keep]
    total = len(filtered)
    out: List[SweepJob] = []
    for i, j in enumerate(filtered):
        out.append(
            SweepJob(
                sweep=j.sweep,
                combo_index=i + 1,
                combo_total=total,
                dataset=j.dataset,
                model=j.model,
                train_frac=j.train_frac,
                epochs=j.epochs,
                context_frames=j.context_frames,
                run_id=j.run_id,
            )
        )
    return out
