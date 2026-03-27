# Evaluation spec (Metrics, model saving, and inference workflow)

## Goal
Evaluate fine-tuned MORPH on held-out shallow-water trajectories with:

- **comprehensive metric logging during training**
- a **final best-summary** at the end
- a **saved model checkpoint** usable for inference
- a **separate inference/eval mode** that loads the saved checkpoint and runs manual/visual checks

This spec aligns with the current evaluation flow in `MORPH/scripts/finetune_MORPH_general.py`.

## Evaluation data contract
Evaluation must use a **held-out set of trajectories** (group-disjoint), shaped the same as training trajectories and normalized with the same RevIN stats prefix.

Key requirement: your unit of split should be **trajectory/group**, not individual timesteps, to avoid temporal leakage.

## Metrics to compute

### Normalized-scale metrics (fast, training-aligned)
Computed directly on model outputs and targets in normalized space:
- **MSE**: `F.mse_loss(out, target, reduction='mean')`
- **MAE**: `F.l1_loss(out, target, reduction='mean')`
- **RMSE**: `sqrt(MSE)`

These are useful for monitoring learning progress but may be less interpretable physically.

### Denormalized-scale metrics (physics-interpretable)
After reshaping predictions back to \((N, T-1, F, C, D, H, W)\), denormalize with RevIN stats:
- `outputs_denorm = RevIN.denormalize_testeval(..., out_all_rs, dataset=dataset_name)`
- `targets_denorm = RevIN.denormalize_testeval(..., tar_all_rs, dataset=dataset_name)`

Then compute:
- **NRMSE**
- **VRMSE**

These are already implemented via `Metrics3DCalculator` in the current script.

## What to log “as training runs”
During training epochs, track and persist:
- `train_loss` per epoch
- `val_loss` per epoch
- `best_val_loss` and the epoch where it occurs
- (optional) current LR if using scheduler

Recommended persistence:
- Write a single `metrics.json` or `metrics.csv` per run that appends each epoch.
- Continue saving the plotted loss curve image, but the structured log is the source of truth.

Even without code changes, you can treat:
- the printed logs + saved loss plot
as the “comprehensive data” baseline, but structured logs are preferable.

## “Print best outcome at the end”
Define “best outcome” as:
- minimum `val_loss` (normalized MSE) across epochs
and print:
- best val loss
- corresponding epoch
- associated checkpoint filename
- test metrics for that checkpoint (if you evaluate after training ends)

Current behavior:
- checkpoints are saved only when validation improves
- metrics are computed after training loop (using whatever weights are currently in memory)

To guarantee “best checkpoint” test metrics, ensure evaluation loads the best-saved checkpoint (if training stops after non-improving epochs).

## Saving model for inference
Inference needs:
- the `.pth` checkpoint (state dict + optimizer state optional)
- the RevIN stats files written under `MORPH/data/` (mu/var with `norm_<dataset_name>` prefix)
- the dataset specs \(F,C,D,H,W\) and AR settings used in training

Minimum artifact set:
- `models/<dataset_name>/<model_name>... .pth`
- `data/norm_<dataset_name>*` (whatever files RevIN writes)
- a run record (text/markdown) capturing:
  - model size
  - max_ar_order / ar_order
  - dataset specs
  - checkpoint name

## Separate inference mode / script
There are two good inference modes:

### Mode 1: “single-step” inference
Given a ground-truth context window (length `ar_order`), predict the next frame and compare.
This is useful for diagnosing per-step accuracy and bias.

### Mode 2: “rollout” inference
Given the first `ar_order` frames of a trajectory, predict forward for \(K\) steps using the sliding window of predictions.
This is the real surrogate use case; it reveals drift and stability issues.

In the current repo, `finetune_MORPH_general.py` already produces:
- single-step visualization via `Visualize3DPredictions`
- rollout visualization via `Visualize3DRolloutPredictions`

So the “separate script / mode” requirement can be satisfied by:
- using `MORPH/scripts/infer_MORPH.py` (if you standardize on it), or
- running `finetune_MORPH_general.py` in a dedicated inference-only path (recommended future cleanup)

## Evaluation protocol checklist

### Before running evaluation
- Ensure **test set** trajectories are group-disjoint from train/val.
- Confirm `dataset_name` matches the RevIN prefix used during training.
- Confirm `dataset_specs` and AR settings match the trained model.

### During evaluation
- Compute normalized metrics (MSE/MAE/RMSE).
- Reshape outputs to \((N, T-1, F, C, D, H, W)\).
- Denormalize and compute NRMSE/VRMSE.
- Save a metrics text file (already implemented).

### After evaluation
- Store a “best run summary” artifact including:
  - best validation score
  - test metrics
  - checkpoint path
  - a pointer to the visualization outputs (see V spec)

## Common failure modes (and what to check)
- **Shape mismatch**: confirm the `.npy` exported shape matches the expected `(N,T,...)`.
- **Wrong H/W ordering**: ensure spatial axes aren’t swapped when exporting from HDF5.
- **AR mismatch**: using `ar_order` in data prep different from what the model expects (`max_ar_order` / checkpoint training).
- **Normalization drift**: accidentally recomputing RevIN stats on a different dataset partition; keep prefix stable and consistent.

