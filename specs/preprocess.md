# Preprocessing spec (Shallow Water → MORPH)

## Goal

Prepare the shallow-water simulation trajectories into the tensor format MORPH expects for fine-tuning/inference, with **minimal cleaning** and a **group-wise split** to avoid leakage across time-correlated sequences.

This project currently uses two relevant data layouts:

- **Raw HDF5 “group” format** (used by `src/visualize_shallow_water.py`):
  - HDF5 keys like `'0000'..'0999'`
  - Each group contains dataset `data` shaped roughly `(T, X, Y, 1)` (water height)
- **Processed SW2d folder format** (generated in `MORPH/datasets/SW2d`):
  - One trajectory per file: `0000__data.npy` … `0999__data.npy`
  - Observed shape/dtype (sampled): **`(T,H,W,1) = (101,128,128,1)`**, `float32`
  - Optional coordinate arrays per trajectory:
    - `0000__grid__t.npy` shape `(101,)`
    - `0000__grid__x.npy` shape `(128,)`
    - `0000__grid__y.npy` shape `(128,)`
- **MORPH fine-tune `.npy` format** (used by `MORPH/scripts/finetune_MORPH_general.py`):
  - Loads a `.npy` array, then converts to **UPTF7** (N,T,F,C,D,H,W)
  - `dataset_specs` provides `(F, C, D, H, W)`

The preprocessing job is to bridge these, producing a `.npy` array with consistent ordering and dimensions.

## Data model and canonical shapes

### Canonical physical field(s)

- **Default**: 1 field (water height), so F=1
- Components: C=1 (single scalar per field)
- Depth: D=1 (2D spatial data)
- Spatial: H=X, W=Y

So MORPH expects each timestep frame shaped **(F, C, D, H, W) = (1,1,1,H,W)** after UPTF7.

### Recommended raw-to-npy intermediate shape

Store dataset as a numpy array:

- `dataset.npy`: shape `(N, T, H, W)` or `(N, T, H, W, 1)`

Then, at loading time (or during export), reshape/expand dims to match the finetune script’s assumptions:

- If you export (N,T,H,W), you can expand to (N,T,H,W,1) for “field” and later inflate to UPTF7.

### SW2d-as-generated: what you have right now
Your current processed set in `MORPH/datasets/SW2d` is already in the per-trajectory form:

- \(N=1000\) trajectories
- Each trajectory file is \( (T,H,W,1) = (101,128,128,1) \)

This is “clean” and consistent with the shallow-water HDF5 layout; the main remaining preprocessing step (if you want to use `finetune_MORPH_general.py` unchanged) is to **pack these 1000 files into one array** with shape `(N,T,H,W,1)` and save as a single `.npy`.

## Group-wise splitting (required)

Because each group `0xxx` is a single coherent simulation trajectory, you should split at the **group level**:

- **Train**: a set of groups (e.g., 80%)
- **Val**: disjoint groups (e.g., 10%)
- **Test**: disjoint groups (e.g., 10%)

### Implementation details

- Enumerate keys once: `keys = sorted(f.keys())`
- Decide split indices using a fixed RNG seed for reproducibility.
- Ensure no overlap of keys across splits.
- Export **three files** (preferred) or one file plus a split index manifest:
  - `shallow_train.npy`
  - `shallow_val.npy`
  - `shallow_test.npy`

If you prefer a single `.npy` (because `finetune_MORPH_general.py` currently does its own random split), you can still do it, but group-wise splitting in preprocessing is better and avoids mixing correlated sequences.

## Minimal “cleaning” checks (do not over-clean)

Do only sanity checks that prevent silent shape/NaN bugs:

- **Shape consistency**: all groups must have the same `(T,H,W,1)` (or `(T,H,W)`).
- **Finite values**: assert `np.isfinite(data).all()` or drop/repair only if you find rare NaNs.
- **Dtype**: cast to `np.float32`.

No reordering, smoothing, clipping, or interpolation unless you discover a concrete issue.

## Normalization strategy (align with MORPH)

`finetune_MORPH_general.py` uses **RevIN** stats computed over the full loaded dataset in UPTF7 format:

- It computes stats with a prefix like `norm_<dataset_name>`
- It normalizes trajectories before AR preparation
- It denormalizes outputs for evaluation metrics

### Implementation guidance

- Keep preprocessing **unnormalized**; let the finetune/infer scripts compute RevIN stats consistently.
- Ensure the file naming (`--dataset_name`) is stable so normalization prefixes are stable across runs.

## Converting HDF5 groups to `.npy` (export procedure)

Export procedure should:

- Read each group’s `data`
- Squeeze final singleton channel if needed
- Stack into a single array: `(N, T, H, W)` (or keep channel `(N,T,H,W,1)`)

## Packing `MORPH/datasets/SW2d/*.npy` into a single training file (recommended for current finetune script)
Because `MORPH/scripts/finetune_MORPH_general.py` currently expects a **single** `.npy` that it can load with `np.load(...)`, the simplest path is:

- Read all `????__data.npy` files in sorted order
- Stack into `all_data` with shape `(N,T,H,W,1)`
- Save e.g. `MORPH/datasets/SW2d.npy` (or `SW2d_all.npy`)

Implementation notes:
- Preserve sorted filename order so trajectory indices match group ids (`'0000'..'0999'`).
- Keep dtype `float32`.
- You can ignore `grid__{t,x,y}` for training; keep them for visualization and sanity checks.

After packing, your `dataset_specs` should be:
- `--dataset_specs 1 1 1 128 128`

and the finetune script will infer:
- `N = dataset.shape[0] = 1000`
- `T = dataset.shape[1] = 101`

### Edge cases

- If `data` is `(T,X,Y,1)` and your visualization script calls them `(X,Y)`, set:
  - H=X, W=Y
- If some groups differ in `T`, choose one:
  - **Trim** to min `T` (recommended for simplicity), or
  - **Pad** (not recommended unless you mask loss)

## Mapping to MORPH `dataset_specs`

When running finetune/infer, set:

- `--dataset_specs 1 1 1 H W`
- `--ar_order 1` initially (one-step conditioning), then try higher if stable
- `--max_ar_order` must be \ge `ar_order` (and should match the checkpoint capabilities)

## Outputs produced by preprocessing

In `specs/` terms, PP is considered complete when you have:

- Either:
  - **Packed**: `MORPH/datasets/<name>.npy` shaped `(N,T,H,W,1)` (preferred for current `finetune_MORPH_general.py`), or
  - **Folder form**: `MORPH/datasets/SW2d/????__data.npy` (what you already have), plus a small packing step before training
- A small markdown note (optional) recording:
  - N, T, H, W
  - split seed and proportions
  - chosen `dataset_specs`

