# usage: finetune_MORPH_general.py 


[-h] --dataset DATASET
[--dataset_name DATASET_NAME]
[--dataset_specs F C D H W]
[--model_choice MODEL_CHOICE]
[--download_model] [--model_size {Ti,S,M,L}]
--ckpt_from {FM,FT} [--checkpoint CHECKPOINT]
[--ft_level1] [--ft_level2] [--ft_level3]
[--ft_level4] [--lr_level4 LR_LEVEL4]
[--wd_level4 WD_LEVEL4] [--parallel {dp,no}]
[--rank_lora_attn RANK_LORA_ATTN]
[--rank_lora_mlp RANK_LORA_MLP]
[--lora_p LORA_P] [--n_epochs N_EPOCHS]
[--n_traj N_TRAJ]
[--rollout_horizon ROLLOUT_HORIZON]
[--tf_reg dropout emb_dropout]
[--heads_xa HEADS_XA] [--ar_order AR_ORDER]
[--max_ar_order MAX_AR_ORDER]
[--test_sample TEST_SAMPLE]
[--device_idx DEVICE_IDX]
[--patience PATIENCE]
[--batch_size BATCH_SIZE] [--lr LR]
[--weight_decay WEIGHT_DECAY]
[--lr_scheduler] [--overwrite_weights]
[--save_every SAVE_EVERY] [--save_batch_ckpt]
[--save_batch_freq SAVE_BATCH_FREQ]

## Run inference on trained ViT3D model

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to saved .npy/.pth
  --dataset_name DATASET_NAME
                        Name of the dataset
  --dataset_specs F C D H W
                        Dataset specs
  --model_choice MODEL_CHOICE
                        Model to finetune
  --download_model      Download model weights
  --model_size {Ti,S,M,L}
                        choose from Ti, S, M, L
  --ckpt_from {FM,FT}   Checkpoint information from FM or previous FT
  --checkpoint CHECKPOINT
                        Path to saved .pth state dict if download_model is
                        False
  --ft_level1           Level-1 finetuning (LoRA, PE, LN)
  --ft_level2           Level-2 finetuning (Encoder)
  --ft_level3           Level-3 finetuning (Decoder)
  --ft_level4           All model parameters
  --lr_level4 LR_LEVEL4
                        Learning rate for level-4 finetuning
  --wd_level4 WD_LEVEL4
                        Weight decay for level-4 finetuning
  --parallel {dp,no}    DataParallel vs No parallelization
  --rank_lora_attn RANK_LORA_ATTN
                        Rank of attention layers in transformer module
  --rank_lora_mlp RANK_LORA_MLP
                        Rank of MLP layers in transformer module
  --lora_p LORA_P       Dropout inside LoRA layers
  --n_epochs N_EPOCHS   Fine-tuning epochs
  --n_traj N_TRAJ       Fine-tuning trajectories
  --rollout_horizon ROLLOUT_HORIZON
                        Visualization: single step & rollouts
  --tf_reg dropout emb_dropout
                        Transformer regularization: dropouts
  --heads_xa HEADS_XA   Number of heads of cross attention
  --ar_order AR_ORDER   Autoregressive order of the data
  --max_ar_order MAX_AR_ORDER
                        Max autoregressive order for the model
  --test_sample TEST_SAMPLE
                        Sample to plot from the test set
  --device_idx DEVICE_IDX
                        CUDA device index to run on
  --patience PATIENCE   Early stopping criteria
  --batch_size BATCH_SIZE
                        Batch size for finetuning
  --lr LR               Learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay
  --lr_scheduler        Use LR scheduler
  --overwrite_weights   Over-ride previous checkpoints (saves storage)
  --save_every SAVE_EVERY
                        Save epochs at intervals
  --save_batch_ckpt     Save batch checkpoints
  --save_batch_freq SAVE_BATCH_FREQ
                        Batch checkpoints frequency
