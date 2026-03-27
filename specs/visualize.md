# Visualization spec (Qualitative evaluation for Shallow Water surrogate)

## Goal
Provide visual diagnostics that work for both:

- **real (ground-truth) sequences** from the dataset
- **given/predicted** sequences from MORPH (single-step and rollout)

The objective is to quickly spot:
- spatial artifacts (ringing, checkerboard, boundary issues)
- systematic bias (over/under-shooting heights)
- temporal instability / drift during rollouts

## Data contract for plotting
Assume you can access sequences shaped as:

- **Ground truth**: \((T, H, W)\) or \((T, H, W, 1)\)
- **Predicted** (single-step): one frame \((H,W)\)
- **Predicted rollout**: \((T', H, W)\)

If your MORPH tensors are in UPTF7 layout \((T,F,C,D,H,W)\) for one trajectory, select:
- `F=0`, `C=0`, `D=0` to get a 2D frame \((H,W)\)

## Visualizations to implement

## 1) “Basic GIF of evolution” (already exists)
You already have `src/visualize_shallow_water.py` which renders:
- a single timestep PNG, or
- a full-sequence GIF over all `T`

### Requirements to satisfy the spec
- Ensure the script can be pointed at:
  - a **real** dataset group (`--h5`, `--group`)
  - a **predicted** rollout saved to disk (see below)
- Ensure consistent color scaling:
  - Support a fixed `vmin/vmax` or `--global-scale` for comparability across frames.

### Predicted sequence support (implementation approach)
Store predicted rollouts to a portable format:
- either a small `.npy` of shape `(T',H,W)` (preferred)
- or an HDF5 that mimics the raw format (group/data)

Then reuse the GIF routine on that output.

## 2) Single-frame diff plot (Actual / Predicted / Diff)
Plot one row of three images for a selected timestep:

- **Panel 1**: actual frame \(x_{t}\)
- **Panel 2**: predicted frame \(\hat{x}_{t}\)
- **Panel 3**: difference \(\hat{x}_{t} - x_{t}\) (signed) or \(|\hat{x}_{t}-x_{t}|\)

### Plotting details
- Use the same `vmin/vmax` for panels 1 and 2.
- Use a diverging colormap (e.g., `coolwarm`) for signed diff and center at 0.
- Add colorbars for interpretability.
- Title should include:
  - dataset/group id
  - timestep index
  - metric summary for that frame (optional: per-frame MAE/RMSE)

### When to generate
- At least for:
  - a representative “easy” trajectory
  - the worst-performing trajectory by rollout error (if you compute it)

## 3) Entire sequence diff evolution (Actual vs Predicted over time)
Plot a grid that shows temporal evolution compactly:

- **Top row**: actual frames at selected timesteps
- **Bottom row**: predicted frames at the same timesteps
- (Optional third row): diff frames

### “Increase plot step between timesteps”
If \(T\) is large, choose a stride:
- `stride = max(1, T // n_cols)`
so you plot `n_cols` frames evenly across the trajectory.

### Recommended layout
- Choose `n_cols` in [6, 10] for readability.
- Keep color scale fixed across all frames in the grid for fair visual comparison.

## Connecting to MORPH’s existing visualization utilities
`MORPH/scripts/finetune_MORPH_general.py` already calls:
- `Visualize3DPredictions` (next-step qualitative checks)
- `Visualize3DRolloutPredictions` (multi-step qualitative checks)

### Requirements to align with shallow water
Because shallow water is 2D:
- use `slice_dim='d'` and `component=0`, and ensure \(D=1\) so slicing is well-defined
- treat “field” as water height (field 0)

If you want the visualization to match `src/visualize_shallow_water.py`:
- extract the 2D `(H,W)` arrays and reuse its colormap/scaling conventions.

## Output organization (recommended)
Write visual outputs into a run-specific folder, e.g.:
- `out/vis/<dataset_name>/<run_id>/`

Include:
- `gif_actual_<group>.gif`
- `gif_rollout_pred_<group>.gif`
- `diff_frame_tXXXX.png`
- `grid_sequence_strideS.png`

## “Done” criteria
Visualization is complete when:
- you can render a **ground-truth** GIF from a real group
- you can render a **predicted rollout** GIF for the same trajectory
- you have a **single-frame triptych** (actual/pred/diff)
- you have a **sequence grid** (actual vs predicted, with stride support)

