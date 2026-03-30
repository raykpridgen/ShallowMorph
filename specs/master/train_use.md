 [-h] [--dataset-dir DATASET_DIR] [--dataset-name DATASET_NAME]
                     [--dataset-specs F C D H W] [--n-traj N_TRAJ] [--model-size {Ti,S,M,L}]
                     [--download-model] [--ckpt-from {FM,FT}] [--checkpoint CHECKPOINT] [--ft-level1]
                     [--ft-level2] [--ft-level3] [--ft-level4] [--lr-level4 LR_LEVEL4]
                     [--wd-level4 WD_LEVEL4] [--rank-lora-attn RANK_LORA_ATTN]
                     [--rank-lora-mlp RANK_LORA_MLP] [--lora-p LORA_P] [--ar-order AR_ORDER]
                     [--max-ar-order MAX_AR_ORDER] [--n-epochs N_EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
                     [--weight-decay WEIGHT_DECAY] [--lr-scheduler] [--patience PATIENCE]
                     [--tf-reg dropout emb_dropout] [--heads-xa HEADS_XA] [--parallel {dp,no}]
                     [--rollout-horizon ROLLOUT_HORIZON] [--test-sample TEST_SAMPLE] [--overwrite-weights]
                     [--save-every SAVE_EVERY] [--save-batch-ckpt] [--save-batch-freq SAVE_BATCH_FREQ]
                     [--device-idx DEVICE_IDX]

Fine-tune MORPH (single-step) on preprocessed shallow-water data.

options:
  -h, --help            show this help message and exit
  --dataset-dir DATASET_DIR
                        Directory with shallow_water_{train,val,test}.npy
  --dataset-name DATASET_NAME
  --dataset-specs F C D H W
  --n-traj N_TRAJ       Limit number of training trajectories (default: all)
  --model-size {Ti,S,M,L}
  --download-model      Download FM weights from HuggingFace
  --ckpt-from {FM,FT}
  --checkpoint CHECKPOINT
                        Explicit checkpoint path (relative to MORPH/models/)
  --ft-level1           Level-1: LoRA + LayerNorm + PosEnc
  --ft-level2           Level-2: + Encoder (conv, proj, xattn)
  --ft-level3           Level-3: + Decoder linear
  --ft-level4           Level-4: unfreeze everything
  --lr-level4 LR_LEVEL4
  --wd-level4 WD_LEVEL4
  --rank-lora-attn RANK_LORA_ATTN
  --rank-lora-mlp RANK_LORA_MLP
  --lora-p LORA_P
  --ar-order AR_ORDER
  --max-ar-order MAX_AR_ORDER
  --n-epochs N_EPOCHS
  --batch-size BATCH_SIZE
  --lr LR
  --weight-decay WEIGHT_DECAY
  --lr-scheduler
  --patience PATIENCE
  --tf-reg dropout emb_dropout
  --heads-xa HEADS_XA
  --parallel {dp,no}
  --rollout-horizon ROLLOUT_HORIZON
  --test-sample TEST_SAMPLE
                        Trajectory index in test set for visualisation
  --overwrite-weights
  --save-every SAVE_EVERY
  --save-batch-ckpt
  --save-batch-freq SAVE_BATCH_FREQ
  --device-idx DEVICE_IDX




python src/train_step.py \
> --dataset-dir MORPH/datasets/ \
> --dataset-specs 1 1 1 128 128 \
> --model-size S \
> --ft-level1 \
> --n-epochs 20 \
> --batch-size 24 \
> --lr 0.001 \
> --patience 6 \



python src/train_step.py \
    --download-model \
    --model-size Ti \
    --ft-decoder-only \
    --n-epochs 150 \
    --batch-size 8 \
    --lr-scheduler \
    --patience 10 \
    --parallel dp \
    --device-idx 0


python src/evaluate.py \
    --checkpoint
  MORPH/models/shallow_water/ft_morph-Ti-shallow_water-ar1_max_ar1_lora16_ftlev5_lr0.0001_wd0.0_best.pth \
    --model-size Ti \
    --rollout-horizon 20 \
    --batch-size 16 \
    --device-idx 0
