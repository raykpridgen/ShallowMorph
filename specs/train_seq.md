# Training (sequence) — `train_seq.md`

## Goal
Reduce rollout drift by training MORPH not just for one-step accuracy, but for **multi-step rollout performance**. This targets the main failure mode of step-trained models: **exposure bias** (training conditions on ground truth; inference conditions on model predictions).

This method is **second precedence**: use it after `train_step.md` if rollouts drift too fast for your target horizon.

## When to use this method (precedence)
Use sequence training if:
- `train_step.md` yields good one-step metrics, but rollout error grows rapidly
- you care about medium/long horizons (e.g., 50–100 steps) and stability matters more than per-step sharpness

Keep step training as your baseline: sequence training should be a targeted fix, not the default starting point.

## Two practical sequence-training approaches

## 1) Multi-step rollout loss (direct)
Train using short rollouts during training and compute loss across the predicted horizon.

### Setup
Choose:
- `ar_order` (context length)
- rollout training horizon `K_train` (e.g., 5 then 10; avoid jumping straight to 50)
- (optional) discount factor \(\gamma \in (0,1]\) to emphasize near-term accuracy

### Training loop (conceptual)
For each trajectory segment with ground-truth frames \((x_{t-ar+1}, …, x_{t+K_train})\):

- Initialize window \(W_0=(x_{t-ar+1}, …, x_t)\)
- For \(k=1..K_{train}\):
  - Predict \(\hat{x}_{t+k} = f_\theta(W_{k-1})\)
  - Update window using the prediction (same as inference):
    - \(W_k = (W_{k-1}[1:], \hat{x}_{t+k})\)

### Loss definition
Compute loss over the horizon:

- **Sum**:
  \[
  \mathcal{L}=\sum_{k=1}^{K_{train}} \|\hat{x}_{t+k} - x_{t+k}\|^2
  \]
- **Discounted**:
  \[
  \mathcal{L}=\sum_{k=1}^{K_{train}} \gamma^{k-1}\|\hat{x}_{t+k} - x_{t+k}\|^2
  \]

Use the same base per-frame error as step training (MSE on normalized scale) initially.

### Why this helps rollouts
The model is explicitly trained on the distribution of inputs it will see at inference time: windows that include its own predictions.

### Costs / trade-offs
- More compute and memory (you unroll the model \(K_{train}\) times per sample)
- Optimization can become harder; start with small `K_train`
- Risk of oversmoothing if you optimize long horizons aggressively

## 2) Scheduled sampling (hybrid)
Keep the single-step objective but gradually replace some ground-truth inputs in the AR window with the model’s own predictions during training.

### How it works
At each step, when updating the context window:
- with probability \(p\), feed \(\hat{x}\) (model prediction)
- with probability \(1-p\), feed \(x\) (ground truth)

Schedule \(p\) from low to higher over training:
- start \(p \approx 0\) (pure teacher forcing)
- increase to \(p \approx 0.25\) or \(0.5\) as training stabilizes

### Benefits
- Often more stable than full multi-step loss
- Lets you keep mostly the existing training pipeline structure

### Drawbacks
- Adds another hyperparameter (the schedule)
- Still doesn’t directly optimize long-horizon error unless you also compute multi-step losses

## Practical “escalation plan” from step → sequence
Use this ordered playbook:

- **Step baseline** (`train_step.md`): train single-step, measure rollout error vs horizon.
- If drift grows fast:
  - First try **higher `ar_order`** (cheap and frequently effective).
  - If still bad:
    - Add **short-horizon** `K_train = 5` multi-step loss, keep \(\gamma \le 1\).
    - Or use **scheduled sampling** with a gentle schedule.
  - If still bad:
    - Increase `K_train` gradually (5 → 10 → 20).

You should not jump to very long training horizons immediately; it typically destabilizes training.

## Metrics to judge whether seq training worked
Evaluate in the way you care about:
- Rollout RMSE/NRMSE vs horizon \(k\)
- Qualitative stability: no blow-up, no rapid saturation, reasonable wave propagation

Compare against step baseline at the same horizon.

## Recommended defaults (first seq attempt)
For SW2d:
- Keep `ar_order` the same as your best step run
- Set `K_train = 5`
- Use discounted loss with \(\gamma \in [0.9, 1.0]\)
- Keep fine-tuning light (LoRA / lower LR) because the unrolled loss is more sensitive

## “Done” criteria for seq method
Sequence training is successful when, relative to the step baseline:
- rollout error grows more slowly with horizon
- qualitative rollouts remain stable for longer
- one-step accuracy does not collapse (some trade-off is acceptable if rollout is the goal)

## CLI Interface
- Include a progress bar indicating training progress
- Include metrics showcasing latest and best so far
- Include all paramters and config context in this menu screen to maintain continuity
