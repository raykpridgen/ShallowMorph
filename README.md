# MORPH surrogate sweeps

This repository wraps [LANL MORPH](https://github.com/lanl/MORPH) (`./MORPH`) to run **reproducible finetuning sweeps** for a surrogate-modeling study on three PDE datasets. Study design, metrics, and design decisions live under `specs/`.

## Study overview

**Goal:** Measure how well MORPH finetunes as a **next-step surrogate** when varying **temporal context** (autoregressive history length), **training-set size**, **epochs**, and **model size**, across:

| This repo | MORPH `ft_dataset` | Source (DARUS) |
|-----------|-------------------|----------------|
| Burgers 1D | `BE1D` | [datafile 268187](https://darus.uni-stuttgart.de/api/access/datafile/268187) |
| Shallow water 2D | `SW` | [datafile 133021](https://darus.uni-stuttgart.de/api/access/datafile/133021) |
| Diffusion–reaction 2D | `DR2D` | [datafile 133017](https://darus.uni-stuttgart.de/api/access/datafile/133017) |

**Sweep A (72 jobs):** context ∈ {1, 5, 10} frames, train fraction {10%, 50%}, epochs {10, 50}, models {tiny, small}.

**Sweep B (144 jobs):** train fraction {10%, 25%, 50%, 100%}, epochs {10, 50, 100, 200}, models {tiny, small, large}, with **per-dataset** best context chosen after Sweep A (see `specs/design.md` §6.3). By default the driver runs **tiny + small** first; add `--include-large` for all **large** jobs when you have budget.

**Training target:** single next-frame prediction with sliding context; inference uses the same context length as training (`specs/issues.md` I3).

Details: `specs/plan.md`, `specs/design.md`, `specs/issues.md`.

## Repository layout

```
morph/
├── MORPH/                 # upstream MORPH (env, scripts, datasets tree)
├── code/morph_wrap/       # sweep driver + config
├── scripts/hpc/           # Slurm example + optional module loader
├── specs/                 # plan, design, decisions
├── requirements.txt       # notes: stdlib-only wrap; Conda for MORPH
└── out/                   # manifests, metrics (gitignored as needed)
```

Local patches under `MORPH/scripts/` include **`SW` in finetune** and a correct training-sample assert when `--ar_order` > 1.

## Prerequisites

1. **MORPH environment** — follow `MORPH/README.md` and `MORPH/docs` (e.g. `conda env create -f MORPH/environment.yml`, PyTorch with CUDA as needed).
2. **Foundation checkpoints** — place FM weights under `MORPH/models/FM/` with basenames matching `code/morph_wrap/sweep_config.py` (`FM_CHECKPOINT_BASENAME`), or pass `--fm-checkpoint` to use one file for all sizes while testing.
3. **GPU** — finetune defaults assume CUDA; use `finetune_MORPH.py --device_idx` / `--parallel` as in upstream docs.

## Preparing data

MORPH finetuning expects data under a **dataset root** (usually the MORPH project root) in the **normalized / RevIN** layout used by the official dataloaders, e.g. paths like:

`MORPH/datasets/normalized_revin/<dataset_folder>/train/…`

**Steps (high level):**

1. **Download** the archives from DARUS (links above). HDF5 layout and shapes are summarized in `specs/plan.md` (e.g. BE1D `tensor`, SW and DR2D groups `0000/data`).
2. **Preprocess** with `python code/preprocess.py` (`--be1d`, `--sw`, `--dr2d`) or follow `MORPH/scripts/data_normalization_revin.py`; loaders: `dataloader_be1d.py`, `dataloader_sw2d.py`, `dataloader_dr.py` (DR2D). Upstream notes are in `MORPH/docs`.
3. **Point the sweep** at that tree with `--dataset-root` (defaults to `./MORPH` if data lives inside the clone).

**Trajectory pool:** `TRAJECTORY_POOL` in `code/morph_wrap/sweep_config.py` should match the **number of trajectories in the train split** you actually load (`train_data.shape[0]` in `finetune_MORPH.py`). The sweep sets `--n_traj` to `max(1, int(pool * train_frac / 100))`. If `pool` is too large, `n_traj` can exceed the real train size.

**Norm stats:** MORPH expects RevIN / `stats_*` files under `MORPH/data/` per dataset; follow upstream preparation so finetuning does not fail on missing statistics.

## Configure the sweep

Edit `code/morph_wrap/sweep_config.py`:

- **`TRAJECTORY_POOL`** — train-split trajectory counts (see above).
- **`FM_CHECKPOINT_BASENAME`** — FM checkpoint filename per model size (`Ti` / `S` / `L`).
- **`LR_TABLE`** — pilot learning rates after your **per-(dataset, model, context)** search (`specs/issues.md` I5); the manifest logs these; MORPH’s level-1–3 optimizer LRs are still defined in `MORPH/src/utils/select_fine_tuning_parameters.py` unless you patch them to read env (e.g. `MORPH_PILOT_LR` set by the driver).

## Quick start

From this repo root (activate your MORPH conda env first):

```bash
# List commands for the first N jobs (no training)
PYTHONPATH=code python -m morph_wrap.run_sweep --sweep A --dry-run --limit 5

# Full Sweep A manifest + print all 72 commands
PYTHONPATH=code python -m morph_wrap.run_sweep --sweep A --dry-run

# Run jobs (sequential; each invokes MORPH/finetune_MORPH.py)
PYTHONPATH=code python -m morph_wrap.run_sweep --sweep A --execute \
  --morph-root MORPH --dataset-root MORPH
```

After you have rollout metrics in a CSV (columns at least `run_id`, `rollout_step`, `mse`, `dataset`, `context_frames`):

```bash
PYTHONPATH=code python -m morph_wrap.pick_context \
  --rollout-csv out/sweepA_rollout.csv \
  --out out/sweep_b_context.json
```

Sweep B (default: **96** tiny+small jobs; **144** with `--include-large`):

```bash
PYTHONPATH=code python -m morph_wrap.run_sweep --sweep B --dry-run \
  --context-json out/sweep_b_context.json
```

Example context JSON: `code/morph_wrap/examples/sweep_b_context.example.json`.

Manifests are written under `out/sweep_*_manifest_*.csv` (command, `lr_pilot_table`, `fm_checkpoint_basename`, etc.).

## HPC / batch jobs (Slurm and similar)

The sweep driver is **stdout/stderr friendly**: finetune subprocesses **inherit** your terminal (or Slurm’s `#SBATCH -o/-e` files). Failures also append **structured JSON lines** to `out/sweep_failures.jsonl` (override with `--failure-log`).

### Dependencies

- **`morph_wrap`** uses only the Python **standard library** (see root `requirements.txt`).  
- **Training** needs the full **MORPH** stack (PyTorch, etc.) via `MORPH/environment.yml` or your site’s modules + venv.

### Optional environment modules

Many clusters use Lmod / Environment Modules. Set a space-separated list and source the helper **before** `conda activate`:

```bash
export MORPH_HPC_MODULES="cuda/12.1 cudnn/9.0 gcc/11"
source scripts/hpc/load_modules.sh
```

If `module` is missing (e.g. laptop), the script **warns and continues**. If a listed module **fails to load**, the script **exits 1** with a clear stderr message.

### One job per finetune (recommended for long runs)

1. On a login node, generate a **fixed manifest** (bounds your array size). Do **not** use `--limit` if the job array must cover the full sweep.

```bash
export PYTHONPATH=code
python -m morph_wrap.run_sweep --sweep A --dry-run \
  --manifest-csv out/sweep_A_manifest.csv
# Array size: number of data rows = (wc -l < manifest.csv) - 1
```

2. Submit a **job array** so each task runs **one** row (1-based task id):

```bash
export PYTHONUNBUFFERED=1   # timely logs when redirecting stdout
python -m morph_wrap.run_sweep \
  --manifest-csv out/sweep_A_manifest.csv \
  --use-slurm-array-task-id --array-base 1 \
  --execute --morph-root MORPH --dataset-root MORPH \
  --out-dir out \
  --summary-log out/sweep_slurm_summary.log
```

Or pass an explicit index: `--execute-job-index 17 --array-base 1`.

3. Optional prolog (same conda env as training):

```bash
python -m morph_wrap.env_check --require-cuda
```

### Sequential batch on one node

```bash
python -m morph_wrap.run_sweep --sweep A --execute \
  --continue-on-failure \
  --failure-log out/sweep_failures.jsonl \
  --summary-log out/sweep_batch_summary.log
```

Non-zero exit after the batch if **any** job failed (`--continue-on-failure`); without it, the driver stops at the **first** failure.

### Site template

See `scripts/hpc/slurm_array_example.sbatch` and `scripts/hpc/load_modules.sh` (edit paths, partition, conda, array range).

### GPU index (Slurm / CUDA visibility)

By default, `run_sweep` uses **`--device-index auto`**: after optional `MORPH_DEVICE_IDX`, it sets MORPH’s `--device_idx` from **`torch.cuda.current_device()`**, with **`torch.cuda.device_count()`** used for validation and logging. On typical Slurm GPU jobs, the scheduler sets `CUDA_VISIBLE_DEVICES`, PyTorch exposes **one** logical GPU as `cuda:0`, and `current_device()` is **0** — that is the index passed to finetune.

Override when needed: `--device-index 1` or `export MORPH_DEVICE_IDX=1` (env wins over torch only when using `auto`). Suppress the resolution line with **`--quiet-device`**.

## Documentation map

| File | Contents |
|------|----------|
| `specs/plan.md` | Datasets, sweep grids, metrics sketch |
| `specs/design.md` | Implementation-facing design, CSV schemas, checklist |
| `specs/issues.md` | Locked design choices (wrapper, context, staging, …) |

## Citation

If you use MORPH, cite the upstream paper linked from `MORPH/README.md` and respect LANL / dataset licenses for your sources.
