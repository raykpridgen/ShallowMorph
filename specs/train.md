# Training spec (Fine-tune MORPH on Shallow Water)

## Goal
Fine-tune MORPH (ViT3DRegression) as a **surrogate** for shallow water evolution by learning next-step dynamics (and optionally multi-step rollouts) from full trajectories, while respecting MORPH’s **autoregressive conditioning** and expected tensor layout.

This training plan intentionally has two methods, in **order of precedence**:

- **Step method (start here)**: `specs/train_step.md`
  - Train next-step prediction (teacher forcing), then evaluate with rollouts.
  - Matches the current implementation pattern in `MORPH/scripts/finetune_MORPH_general.py`.
- **Sequence method (use if rollout drift is too fast)**: `specs/train_seq.md`
  - Train to reduce rollout drift using multi-step loss and/or scheduled sampling.


## details
- **Step training details** (data contract, AR windowing, losses, rollout eval): `specs/train_step.md`
- **Sequence training details** (multi-step loss / scheduled sampling, escalation plan): `specs/train_seq.md`

