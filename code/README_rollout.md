# Rollout Inference Scripts

This directory contains scripts for performing extended rollouts on trained MORPH models and generating GIF visualizations with SSIM and MSE tracking.

## Scripts

### `rollout_inference.py`
Main script that performs rollout inference on a single model and generates visualizations.

**Features:**
- Extended autoregressive rollouts (50+ timesteps)
- Side-by-side comparison visualizations (ground truth vs prediction vs difference)
- GIF generation for each field
- SSIM and MSE calculation per timestep
- CSV export of metrics following specs/plan.md format

**Usage:**
```bash
# Basic usage
python code/rollout_inference.py \
    --model_dir out/models \
    --model_file best_ft_DR2D.pth \
    --dataset DR2D \
    --rollout_steps 50

# Advanced usage with custom parameters
python code/rollout_inference.py \
    --model_dir out/models \
    --model_file recovery_A_SW_S_ar1_tf0p5_ntr350_epmax200.pth \
    --dataset SW \
    --rollout_steps 100 \
    --sample_idx 1 \
    --ar_order 1 \
    --model_size S \
    --gif_fps 10 \
    --output_dir out/results/rollouts
```

### `batch_rollout.py`
Batch processing script to run rollouts on multiple models automatically.

**Features:**
- Automatic dataset and model size inference from filenames
- Pattern-based model file selection
- Parallel processing support
- Progress tracking and error handling
- Summary reporting

**Usage:**
```bash
# Process all best_* models
python code/batch_rollout.py --model_dir out/models --rollout_steps 50

# Process specific pattern
python code/batch_rollout.py \
    --model_dir out/models \
    --pattern "*DR2D*.pth" \
    --dataset DR2D \
    --rollout_steps 75

# Dry run to see what would be processed
python code/batch_rollout.py \
    --model_dir out/models \
    --pattern "*.pth" \
    --dry_run
```

## Dependencies

### Required Python packages:
```bash
conda activate ml_base
pip install pytorch-msssim imageio pillow
```

### Optional (for better performance):
```bash
pip install tqdm matplotlib seaborn
```

## Output Structure

All outputs are saved to `out/results/rollouts/rollout_YYYYMMDD_HHMMSS/`:

```
rollout_20240426_153000/
├── gifs/                           # Generated GIF animations
│   ├── field_0_rollout.gif
│   ├── field_1_rollout.gif
│   └── ...
├── frames/                         # Individual PNG frames
│   ├── field_0_step_000.png
│   ├── field_0_step_001.png
│   └── ...
├── side_by_side/                   # Side-by-side comparisons
├── diffs/                          # Difference visualizations  
└── metrics/                        # CSV files with metrics
    └── rollout_metrics_YYYYMMDD_HHMMSS.csv
```

## Metrics CSV Format

The CSV output follows the specification in `specs/plan.md` under "### Record":

```csv
# Rollout Metrics
# Model: best_ft_DR2D.pth
# Dataset: DR2D
# Rollout Steps: 50
# Generated: 20240426_153000

timestep,mse_aggregate,ssim_aggregate,mse_field_0,mse_field_1,ssim_field_0,ssim_field_1
0,0.001234,0.9876,0.001200,0.001268,0.9880,0.9872
1,0.001456,0.9834,0.001420,0.001492,0.9840,0.9828
...
```

## Model File Examples

The scripts work with model files from your sweep outputs:

```bash
# Examples from out/models/
best_ft_DR2D.pth                    # Best model for DR2D dataset
best_ft_BE1D.pth                    # Best model for BE1D dataset
recovery_A_SW_S_ar1_tf0p5_ntr350_epmax200.pth  # Recovery checkpoint
```

## Environment Setup

Ensure you're using the correct conda environment:

```bash
conda activate ml_base
export PYTHONPATH="${PYTHONPATH}:/home/raykp/sp26/ml/morph/MORPH"
```

## Troubleshooting

### Common Issues:

1. **ImportError: No module named 'pytorch_msssim'**
   ```bash
   pip install pytorch-msssim
   ```

2. **CUDA out of memory**
   - Reduce `--rollout_steps`
   - Use smaller model (`--model_size Ti`)
   - Reduce batch processing

3. **Dataset not found**
   - Ensure MORPH/datasets/normalized_revin/ contains the dataset
   - Check dataset name matches available options

4. **Model loading errors**
   - Verify model file exists and is not corrupted
   - Check model_size parameter matches the model architecture

### Performance Tips:

- Use `--ar_order 1` for faster processing (unless testing multi-context models)
- Set `--gif_fps 3` for smaller GIF files
- Use `--max_models 5` in batch processing for testing
- Monitor GPU memory usage with `nvidia-smi`

## Integration with Sweep Results

These scripts are designed to work with the output from `sweep.py`:

1. Run your parameter sweep: `python code/sweep.py`
2. Best models are saved to `out/models/best_ft_*.pth`
3. Run rollout inference: `python code/batch_rollout.py --model_dir out/models`
4. Analyze results in `out/results/rollouts/`

This creates a complete pipeline from training → evaluation → visualization as specified in `specs/plan.md`.