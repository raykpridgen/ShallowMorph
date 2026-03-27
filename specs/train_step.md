# Training (step) — `train_step.md`

## Goal
Fine-tune MORPH as a shallow-water surrogate by learning a **single-step** dynamics map:

\[
(x_{t-ar+1}, \dots, x_t)\ \rightarrow\ \hat{x}_{t+1}
\]

Then use that trained model to **roll out** multiple steps at inference time using a sliding-window autoregressive loop.

This is the recommended **first** method because it is stable, sample-efficient, and matches the current repository training implementation (`MORPH/scripts/finetune_MORPH_general.py`).

## When to use this method (precedence)
Use **step training first** in almost all cases.

Escalate to `train_seq.md` only if:
- single-step metrics are good, **but rollouts drift quickly** (exposure bias),
- or you specifically care about long-horizon stability and can afford more expensive training.

## Data and shapes (as used by current code)
Assume a dataset of trajectories:
- \(N\) simulations
- \(T\) timesteps per simulation
- shallow water height field (typically \(F=1, C=1, D=1, H=128, W=128\))

The fine-tune pipeline (current code) normalizes and prepares autoregressive pairs:
- **Inputs** `X`: windows of length `ar_order`
- **Targets** `y`: the next frame

Conceptually:
- `X[i] = (x_{t-ar+1}, …, x_t)`
- `y[i] = x_{t+1}`

## Autoregressive order (`ar_order`) selection
Definitions:
- `ar_order`: number of previous frames provided as conditioning context.
- `max_ar_order`: the model’s configured maximum AR order; must satisfy `max_ar_order >= ar_order`.

Recommended starting point for SW2d:
- Start with `ar_order = 1`, `max_ar_order = 1`
- If rollouts drift early, try `ar_order = 2` (or 4) **if** your chosen checkpoint/model supports it.

## Training objective (loss)
Train to minimize **next-step prediction error**:
- baseline: **MSE** on normalized scale (RevIN-normalized)

This is “teacher forcing”: the conditioning frames come from the ground-truth trajectory during training.

### Why this supports rollout
A rollout is just repeated application of the one-step model, feeding predictions back in:
- Step 1 predicts \(x_{t+1}\)
- Step 2 predicts \(x_{t+2}\) from a window that includes \(\hat{x}_{t+1}\)
- etc.

So single-step training learns the local dynamics; rollout evaluation tests accumulated error and stability.

## Rollout procedure (inference/evaluation)
Given initial context frames \(x_0..x_{ar-1}\), roll forward for \(K\) steps:

- Initialize the context window \(W_0 = (x_0, …, x_{ar-1})\)
- For step \(s=0..K-1\):
  - \(\hat{x}_{ar+s} = f_\theta(W_s)\)
  - Update window by dropping oldest frame and appending prediction:
    - \(W_{s+1} = (W_s[1:], \hat{x}_{ar+s})\)

Rollout diagnostics you should track:
- error vs horizon \(k\) (e.g., RMSE at each predicted step)
- stability (does it blow up, saturate, or slowly drift?)

## What to do if rollout drift is bad
Treat this as a decision ladder, from cheapest to most involved:

- **Increase context**: raise `ar_order` (often the best first fix)
- **Regularize / constrain**: keep fine-tuning light (LoRA level) to avoid overfitting
- **Train on harder samples**: include more diverse trajectories; ensure splits are group-wise
- **Escalate to sequence training**: use `train_seq.md` (multi-step loss / scheduled sampling)

## Suggested run recipe (initial SW2d attempt)
- Model size: `Ti` or `S`
- Fine-tuning: `--ft_level1` (LoRA-style adaptation)
- AR: `--ar_order 1 --max_ar_order 1`
- Loss: MSE
- Evaluate with rollouts: `--rollout_horizon` 10–50 for quick iteration

## “Done” criteria for step method
You can consider the step method successful when:
- one-step validation loss improves and stabilizes
- rollouts remain qualitatively stable for your target horizon (or degrade gracefully)
- rollout error grows slowly with horizon rather than exploding immediately

