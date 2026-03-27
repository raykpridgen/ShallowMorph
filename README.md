# MORPH Shallow-Water Fine-Tuning Pipeline

Fine-tune [MORPH](https://huggingface.co/mahindrautela/MORPH) (a Vision-Transformer-based foundation model for PDE surrogates) on 2-D shallow-water simulation data. The pipeline is modular: each stage is a standalone script that reads the previous stage's outputs and writes well-defined artifacts for the next.

## Goals

- **Learn next-step dynamics** of water-height evolution on a 128 x 128 grid by fine-tuning MORPH's pretrained weights with LoRA-style adaptation.
- **Evaluate both single-step accuracy and multi-step rollout stability**, tracking per-horizon error to detect drift.
- **Produce publication-ready visualisations** — animated GIFs, actual-vs-predicted triptychs, and strided evolution grids.

## Project layout

```
morph/
├── README.md                   ← this file
├── src/
│   ├── utils.py                ← shared constants, data loading, MORPH import helpers
│   ├── preprocess.py           ← Step 1: pack & split raw data
│   ├── train_step.py           ← Step 2: single-step fine-tuning
│   ├── evaluate.py             ← Step 3: inference, metrics, visualisation
│   └── visualize.py            ← plotting library + standalone CLI
├── specs/                      ← design documents
│   ├── master/plan.md
│   ├── preprocess.md
│   ├── train.md / train_step.md / train_seq.md
│   ├── evaluation.md
│   └── visualize.md
├── MORPH/                      ← MORPH repository (submodule / clone)
│   ├── models/FM/              ← foundation-model checkpoints
│   ├── datasets/SW2d/          ← per-trajectory .npy files (1000 trajectories)
│   ├── datasets/               ← packed splits land here after preprocessing
│   ├── data/                   ← RevIN normalisation stats
│   ├── experiments/results/    ← training metrics & loss curves
│   └── src/utils/              ← MORPH internals (model, trainers, normalization …)
└── out/                        ← visualisation outputs
    └── eval/<dataset_name>/    ← evaluate.py outputs
```

## Quick start

```bash
# 0. Activate the conda environment with PyTorch
conda activate ml_base

# 1. Preprocess — pack 1000 trajectories, split 80/10/10
python src/preprocess.py

# 2. Train — download FM weights, fine-tune with LoRA (level 1)
python src/train_step.py --download-model --ft-level1

# 3. Evaluate — load best checkpoint, compute metrics, generate plots
python src/evaluate.py \
  --checkpoint <name_of_best.pth> \
  --ft-level1 --rollout-horizon 50
```

## Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA (the `ml_base` conda environment)
- numpy, matplotlib, tqdm, Pillow, scikit-learn, huggingface_hub
- h5py (only needed if loading from raw HDF5)

The MORPH repository should be cloned or symlinked at `morph/MORPH/`. Per-trajectory data files (`0000__data.npy` … `0999__data.npy`) should already exist in `MORPH/datasets/SW2d/`.

---

## Detailed workflow

### Step 1 — Preprocessing (`src/preprocess.py`)

Packs the 1000 individual trajectory files into consolidated NumPy arrays and splits them at the trajectory level to prevent temporal leakage.

**What it does:**

1. Loads all `????__data.npy` files from `MORPH/datasets/SW2d/` (or from a raw HDF5 via `--h5`).
2. Validates every trajectory: shape `(101, 128, 128, 1)`, `float32`, all finite.
3. Performs a group-wise train/val/test split (default 80/10/10, seeded for reproducibility).
4. Saves four `.npy` files and a JSON manifest.

**Outputs** (in `MORPH/datasets/`):

| File | Shape | Purpose |
|------|-------|---------|
| `shallow_water_train.npy` | (800, 101, 128, 128, 1) | Training split |
| `shallow_water_val.npy` | (100, 101, 128, 128, 1) | Validation split |
| `shallow_water_test.npy` | (100, 101, 128, 128, 1) | Test split |
| `shallow_water_all.npy` | (1000, 101, 128, 128, 1) | Combined (for compat with `finetune_MORPH_general.py`) |
| `shallow_water_manifest.json` | — | Records N, T, H, W, split seed, dataset_specs |

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sw2d-dir` | `MORPH/datasets/SW2d` | Source directory of per-trajectory files |
| `--h5` | — | Alternative: raw HDF5 file |
| `--out-dir` | `MORPH/datasets` | Where to write outputs |
| `--name` | `shallow_water` | Filename prefix |
| `--seed` | `42` | Split RNG seed |
| `--train-ratio` | `0.8` | Fraction for training |
| `--val-ratio` | `0.1` | Fraction for validation |
| `--test-ratio` | `0.1` | Fraction for testing |
| `--skip-all` | off | Skip writing the combined `_all.npy` |

**Example:**

```bash
# Default (reads SW2d folder, writes to MORPH/datasets/)
python src/preprocess.py

# Custom split
python src/preprocess.py --train-ratio 0.9 --val-ratio 0.05 --test-ratio 0.05
```

---

### Step 2 — Training (`src/train_step.py`)

Single-step (teacher-forced) fine-tuning of MORPH. Learns the mapping from an `ar_order`-length context window to the next frame, using MSE loss on RevIN-normalised data.

**What it does:**

1. Loads the preprocessed splits produced by Step 1.
2. Converts to MORPH's UPTF7 tensor layout `(N, T, F, C, D, H, W)`.
3. Computes RevIN normalisation statistics over all trajectories (saved to `MORPH/data/` for later use by evaluation).
4. Creates autoregressive (input, target) pairs via sliding window.
5. Builds a `ViT3DRegression` model, loads foundation-model weights (optionally downloaded from HuggingFace), and applies the chosen fine-tuning level.
6. Trains with early stopping; saves checkpoints only when validation loss improves.
7. After training, **reloads the best checkpoint** and evaluates on the test set.
8. Writes structured metrics (JSON + plain text) and a loss-curve plot.

**Outputs:**

| Artifact | Location |
|----------|----------|
| Best checkpoint | `MORPH/models/shallow_water/ft_morph-..._best.pth` |
| RevIN stats | `MORPH/data/norm_shallow_water_mu.npy`, `…_var.npy` |
| Metrics (JSON) | `MORPH/experiments/results/test/shallow_water/metrics_*.json` |
| Metrics (text) | `…/metrics_*.txt` |
| Loss curve | `…/loss_*.png` |

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--download-model` | off | Download FM checkpoint from HuggingFace |
| `--model-size` | `Ti` | `Ti`, `S`, `M`, or `L` |
| `--ckpt-from` | `FM` | `FM` (foundation model) or `FT` (resume from fine-tuned) |
| `--checkpoint` | — | Explicit `.pth` path for `FT` resume |
| `--ft-level1` | off | LoRA + LayerNorm + positional encoding |
| `--ft-level2` | off | + Encoder (conv, proj, cross-attention) |
| `--ft-level3` | off | + Decoder linear |
| `--ft-level4` | off | Full model (all parameters unfrozen) |
| `--ar-order` | `1` | Context window length |
| `--max-ar-order` | `1` | Model's maximum AR order |
| `--n-epochs` | `150` | Maximum training epochs |
| `--batch-size` | `8` | Per-GPU batch size |
| `--lr` | `1e-4` | Learning rate |
| `--lr-scheduler` | off | ReduceLROnPlateau |
| `--patience` | `10` | Early-stopping patience |
| `--n-traj` | all | Limit training trajectories |
| `--overwrite-weights` | off | Save as `_best.pth` instead of `_ep{N}.pth` |

**Recommended first run:**

```bash
python src/train_step.py \
  --download-model \
  --ft-level1 \
  --model-size Ti \
  --n-epochs 50 \
  --patience 10 \
  --lr-scheduler \
  --overwrite-weights
```

**Resuming from a fine-tuned checkpoint:**

```bash
python src/train_step.py \
  --ckpt-from FT \
  --checkpoint ft_morph-Ti-shallow_water-..._best.pth \
  --ft-level1 --ft-level2 \
  --n-epochs 100
```

---

### Step 3 — Evaluation (`src/evaluate.py`)

Standalone inference script that loads a trained checkpoint and produces comprehensive metrics and visualisations. Designed to run independently from training.

**What it does:**

1. Loads preprocessed test data and recomputes RevIN stats (using the same train+val+test concatenation order as training, so stats are consistent).
2. **Single-step evaluation**: teacher-forced next-frame prediction on all test pairs.
   - Reports MSE, MAE, RMSE (normalised) and VRMSE, NRMSE (denormalised).
3. **Rollout evaluation**: starting from the first ground-truth frame, autoregressively predicts `K` steps.
   - Tracks per-horizon RMSE averaged over all test trajectories.
   - Reveals drift and stability characteristics.
4. Generates visualisations for a selected test trajectory:
   - GIFs of actual and predicted rollout sequences
   - Actual / Predicted / Diff triptych at key timesteps
   - Multi-column evolution grid (actual vs predicted)
   - Per-horizon RMSE curve plot
5. Saves predictions as `.npy` for further analysis or custom plotting.

**Outputs** (in `out/eval/shallow_water/`):

| File | Description |
|------|-------------|
| `metrics_eval_s0_h50.json` | All metrics (single-step + rollout) |
| `rollout_actual_s0_h50.npy` | Ground-truth frames `(K, H, W)` |
| `rollout_pred_s0_h50.npy` | Predicted frames `(K, H, W)` |
| `gif_actual_s0_h50.gif` | Animated ground truth |
| `gif_rollout_s0_h50.gif` | Animated rollout predictions |
| `diff_t001_s0_h50.png` … | Actual / predicted / diff triptych |
| `grid_s0_h50.png` | Strided evolution grid |
| `rollout_rmse_curve_s0_h50.png` | Error vs rollout step |

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | Path to `.pth` (absolute or relative to `MORPH/models/<name>/`) |
| `--rollout-horizon` | `20` | Number of autoregressive steps |
| `--test-sample` | `0` | Test trajectory index for detailed visualisation |
| `--out-dir` | `out/eval/<name>/` | Output directory |
| `--ft-level1` … | — | Must match the architecture used during training |
| `--model-size` | `Ti` | Must match training |

**Example:**

```bash
python src/evaluate.py \
  --checkpoint ft_morph-Ti-shallow_water-ar1_max_ar1_lora16_ftlev1_lr0.0001_wd0.0_best.pth \
  --ft-level1 \
  --rollout-horizon 50 \
  --test-sample 5
```

---

### Visualisation (`src/visualize.py`)

Both a **library** of plotting functions (imported by `evaluate.py`) and a **standalone CLI** for ad-hoc visualisation of any sequence.

**Library functions** (importable):

| Function | Input | Output |
|----------|-------|--------|
| `render_sequence_gif(seq, path, ...)` | `(T, H, W)` numpy array | GIF file |
| `plot_diff_frame(actual, pred, ...)` | Two `(H, W)` frames | PNG with actual/predicted/diff panels |
| `plot_sequence_grid(actual_seq, pred_seq, ...)` | Two `(T, H, W)` arrays | PNG grid with stride |

**CLI modes:**

```bash
# Animate a .npy predicted rollout
python src/visualize.py --npy out/eval/shallow_water/rollout_pred_s0_h50.npy

# Animate a raw HDF5 group
python src/visualize.py --h5 ../data/shallow_data.h5 --group 0042

# Single frame from .npy
python src/visualize.py --npy rollout.npy --t 10 --out frame10.png
```

---

## Shallow-water dataset specs

| Parameter | Value |
|-----------|-------|
| Trajectories (N) | 1000 |
| Timesteps per trajectory (T) | 101 |
| Spatial resolution (H x W) | 128 x 128 |
| Fields (F) | 1 (water height) |
| Components (C) | 1 |
| Depth (D) | 1 (2-D data) |
| `--dataset-specs` | `1 1 1 128 128` |

## MORPH model sizes

| Size | Params | dim | heads | depth | mlp_dim |
|------|--------|-----|-------|-------|---------|
| Ti | ~10.5 M | 256 | 4 | 4 | 1024 |
| S | ~33 M | 512 | 8 | 4 | 2048 |
| M | ~86 M | 768 | 12 | 8 | 3072 |
| L | ~200 M | 1024 | 16 | 16 | 4096 |

## Fine-tuning levels

| Level | What is unfrozen | Typical use |
|-------|-----------------|-------------|
| 1 | LoRA (A/B), LayerNorms, positional encoding | Start here — cheap, effective |
| 1+2 | + Convolutional encoder, projection, cross-attention | If level 1 plateaus |
| 1+2+3 | + Decoder linear layer | More capacity |
| 4 | Full model (all parameters) | Last resort; risk of overfitting |

## Escalation path (if rollouts drift)

1. Increase `--ar-order` (e.g. 2 or 4) with a matching `--max-ar-order`
2. Add fine-tuning depth (level 2, then 3)
3. Regularise: keep LoRA rank moderate, add `--lr-scheduler`
4. Escalate to sequence training (`specs/train_seq.md` — not yet implemented)
