#!/usr/bin/env python3
"""
Extended rollout inference script that calls infer_MORPH.py functionality 
and adds GIF generation with SSIM/MSE tracking per timestep.

Usage:
    python code/rollout_inference.py --model_dir out/models --model_file best_ft_DR2D.pth --dataset DR2D --rollout_steps 50
"""

import os
import sys
import argparse
import subprocess
import csv
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import imageio

# Ensure we can import MORPH modules
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_CODE_DIR, ".."))
MORPH_ROOT = os.path.join(REPO_ROOT, "MORPH")
sys.path.insert(0, MORPH_ROOT)

from src.utils.device_manager import DeviceManager
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
from src.utils.metrics_3d import Metrics3DCalculator
from src.utils.visualize_predictions_3d_full import Visualize3DPredictions
from src.utils.visualize_rollouts_3d_full import Visualize3DRolloutPredictions
from src.utils.data_preparation_fast import FastARDataPreparer
from config.data_config import DataConfig
from src.utils.dataloaders.dataloaderchaos import DataloaderChaos
from src.utils.normalization import RevIN

# Try to import SSIM - install with: pip install pytorch-msssim
try:
    from pytorch_msssim import ssim
    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-msssim not available. SSIM metrics will be skipped.")
    print("Install with: pip install pytorch-msssim")
    SSIM_AVAILABLE = False

# Model configurations
MORPH_MODELS = {
    'Ti': [8, 256,  4,  4, 1024],
    'S' : [8, 512,  8,  4, 2048],
    'M' : [8, 768, 12,  8, 3072],
    'L' : [8, 1024,16, 16, 4096]
}

def create_output_directories(base_dir: str) -> dict:
    """Create organized output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dirs = {
        'base': os.path.join(base_dir, f"rollout_{timestamp}"),
        'gifs': os.path.join(base_dir, f"rollout_{timestamp}", "gifs"),
        'frames': os.path.join(base_dir, f"rollout_{timestamp}", "frames"),
        'side_by_side': os.path.join(base_dir, f"rollout_{timestamp}", "side_by_side"),
        'diffs': os.path.join(base_dir, f"rollout_{timestamp}", "diffs"),
        'metrics': os.path.join(base_dir, f"rollout_{timestamp}", "metrics"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def load_model_from_checkpoint(checkpoint_path: str, model_size: str, device: torch.device, 
                              max_ar: int = 1, max_patches: int = 4096, max_fields: int = 3, 
                              max_components: int = 3) -> nn.Module:
    """Load model from checkpoint with proper configuration."""
    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[model_size]
    
    model = ViT3DRegression(
        patch_size=8, dim=dim, depth=depth, heads=heads, heads_xa=32,
        mlp_dim=mlp_dim, max_components=max_components, conv_filter=filters,
        max_ar=max_ar, max_patches=max_patches, max_fields=max_fields,
        dropout=0.1, emb_dropout=0.1, lora_r_attn=0, lora_r_mlp=0,
        lora_alpha=None, lora_p=0.0
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Remove 'module.' prefix if present (DataParallel)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

def calculate_ssim_safe(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate SSIM safely, handling edge cases."""
    if not SSIM_AVAILABLE:
        return float('nan')
    
    try:
        # Ensure tensors are in correct format for SSIM
        if len(pred.shape) == 2:  # 2D case
            pred = pred.unsqueeze(0).unsqueeze(0)    # Add batch and channel dims
            target = target.unsqueeze(0).unsqueeze(0)
        elif len(pred.shape) == 3:  # 3D case, assume (C, H, W)
            pred = pred.unsqueeze(0)    # Add batch dim
            target = target.unsqueeze(0)
        
        # SSIM requires values in range [0, 1] or [-1, 1]
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        ssim_val = ssim(pred_norm, target_norm, data_range=1.0)
        return float(ssim_val.item())
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        return float('nan')

def create_comparison_frame(pred_slice: np.ndarray, true_slice: np.ndarray, 
                          timestep: int, field_name: str = "Field", 
                          colormap: str = 'viridis') -> np.ndarray:
    """Create side-by-side comparison frame with difference."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Calculate shared color range
    vmin = min(pred_slice.min(), true_slice.min())
    vmax = max(pred_slice.max(), true_slice.max())
    
    # True
    im1 = axes[0].imshow(true_slice, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Ground Truth t={timestep}', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Predicted
    im2 = axes[1].imshow(pred_slice, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Prediction t={timestep}', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference
    diff = np.abs(true_slice - pred_slice)
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'|Difference| t={timestep}', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'{field_name} - Timestep {timestep}', fontsize=16)
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    frame_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_array = frame_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return frame_array

def perform_extended_rollout(model: nn.Module, test_data: np.ndarray, device: torch.device,
                           rollout_steps: int, sample_idx: int = 0, ar_order: int = 1) -> dict:
    """Perform extended rollout with detailed tracking."""
    model.eval()
    
    # Select sample and prepare initial context
    sample = test_data[sample_idx]  # Shape: (T, D, H, W, C, F)
    T, D, H, W, C, F = sample.shape
    
    # Convert to model input format: (T, F, C, D, H, W)
    sample_tensor = torch.from_numpy(sample.transpose(0, 5, 4, 1, 2, 3)).float()
    
    # Initialize with ar_order frames
    current_context = sample_tensor[:ar_order].unsqueeze(0).to(device)  # (1, ar_order, F, C, D, H, W)
    
    predictions = []
    ground_truths = []
    
    print(f"Starting rollout for {rollout_steps} steps...")
    
    with torch.no_grad():
        for step in tqdm(range(rollout_steps), desc="Rollout"):
            # Forward pass
            try:
                # Model expects (B, ar_order, F, C, D, H, W)
                model_output = model(current_context)
                
                # Handle different model output formats
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        pred = model_output[2]  # (B, F, C, D, H, W)
                    else:
                        pred = model_output[-1]
                else:
                    pred = model_output
                
                predictions.append(pred.cpu())
                
                # Get ground truth if available
                if ar_order + step < T:
                    gt = sample_tensor[ar_order + step].unsqueeze(0)  # (1, F, C, D, H, W)
                    ground_truths.append(gt)
                else:
                    # No more ground truth available
                    break
                
                # Update context for next step (sliding window)
                if ar_order == 1:
                    current_context = pred.unsqueeze(1)  # (B, 1, F, C, D, H, W)
                else:
                    # Slide window: remove oldest, add newest
                    current_context = torch.cat([
                        current_context[:, 1:],  # Remove first timestep
                        pred.unsqueeze(1)        # Add prediction as new timestep
                    ], dim=1)
                    
            except Exception as e:
                print(f"Error at step {step}: {e}")
                break
    
    return {
        'predictions': torch.cat(predictions, dim=0) if predictions else torch.empty(0),
        'ground_truths': torch.cat(ground_truths, dim=0) if ground_truths else torch.empty(0),
        'num_steps': len(predictions)
    }

def calculate_metrics_per_timestep(predictions: torch.Tensor, ground_truths: torch.Tensor) -> dict:
    """Calculate MSE and SSIM for each timestep."""
    num_steps = min(len(predictions), len(ground_truths))
    
    metrics = {
        'timesteps': list(range(num_steps)),
        'mse_per_step': [],
        'ssim_per_step': [],
        'mse_per_field': [],  # Per field averages
        'ssim_per_field': []
    }
    
    print("Calculating metrics per timestep...")
    
    for step in tqdm(range(num_steps), desc="Metrics"):
        pred = predictions[step]  # (F, C, D, H, W)
        true = ground_truths[step]
        
        # Overall MSE for this timestep
        mse_step = F.mse_loss(pred, true, reduction='mean').item()
        metrics['mse_per_step'].append(mse_step)
        
        # Per-field metrics
        F_dim = pred.shape[0]
        field_mse = []
        field_ssim = []
        
        for field_idx in range(F_dim):
            pred_field = pred[field_idx]  # (C, D, H, W)
            true_field = true[field_idx]
            
            # MSE for this field
            mse_field = F.mse_loss(pred_field, true_field, reduction='mean').item()
            field_mse.append(mse_field)
            
            # SSIM for this field (use first component if multiple)
            pred_slice = pred_field[0] if len(pred_field.shape) > 2 else pred_field
            true_slice = true_field[0] if len(true_field.shape) > 2 else true_field
            
            # For 3D data, take middle slice
            if len(pred_slice.shape) == 3:
                mid_idx = pred_slice.shape[0] // 2
                pred_slice = pred_slice[mid_idx]
                true_slice = true_slice[mid_idx]
            
            ssim_field = calculate_ssim_safe(pred_slice, true_slice)
            field_ssim.append(ssim_field)
        
        metrics['mse_per_field'].append(field_mse)
        metrics['ssim_per_field'].append(field_ssim)
        
        # Average SSIM across fields for this timestep
        valid_ssim = [s for s in field_ssim if not np.isnan(s)]
        avg_ssim = np.mean(valid_ssim) if valid_ssim else float('nan')
        metrics['ssim_per_step'].append(avg_ssim)
    
    return metrics

def create_visualization_frames(predictions: torch.Tensor, ground_truths: torch.Tensor,
                              output_dirs: dict, dataset_name: str) -> list:
    """Create individual frames for GIF generation."""
    num_steps = min(len(predictions), len(ground_truths))
    F_dim = predictions.shape[1] if len(predictions) > 0 else 0
    
    frame_files = {f'field_{f}': [] for f in range(F_dim)}
    
    print("Creating visualization frames...")
    
    for step in tqdm(range(num_steps), desc="Frames"):
        pred = predictions[step]  # (F, C, D, H, W)
        true = ground_truths[step]
        
        for field_idx in range(F_dim):
            pred_field = pred[field_idx]  # (C, D, H, W)
            true_field = true[field_idx]
            
            # Extract 2D slice for visualization
            if len(pred_field.shape) == 3:  # (C, D, H, W) - take first component
                pred_slice = pred_field[0]
                true_slice = true_field[0]
                
                # For 3D data, take middle slice along depth
                if len(pred_slice.shape) == 3:
                    mid_idx = pred_slice.shape[0] // 2
                    pred_slice = pred_slice[mid_idx].numpy()
                    true_slice = true_slice[mid_idx].numpy()
                else:
                    pred_slice = pred_slice.numpy()
                    true_slice = true_slice.numpy()
            else:
                pred_slice = pred_field.numpy()
                true_slice = true_field.numpy()
            
            # Create comparison frame
            colormap = ['viridis', 'plasma', 'inferno'][field_idx % 3]
            frame_array = create_comparison_frame(
                pred_slice, true_slice, step, 
                f"{dataset_name} Field {field_idx}", colormap
            )
            
            # Save frame
            frame_filename = os.path.join(
                output_dirs['frames'], 
                f"field_{field_idx}_step_{step:03d}.png"
            )
            Image.fromarray(frame_array).save(frame_filename)
            frame_files[f'field_{field_idx}'].append(frame_filename)
    
    return frame_files

def create_gifs_from_frames(frame_files: dict, output_dirs: dict, fps: int = 5):
    """Create GIFs from saved frames."""
    print("Creating GIFs...")
    
    for field_key, files in frame_files.items():
        if not files:
            continue
        
        gif_filename = os.path.join(output_dirs['gifs'], f"{field_key}_rollout.gif")
        
        # Load images and create GIF
        images = []
        for file_path in tqdm(files, desc=f"Loading {field_key}"):
            img = Image.open(file_path)
            images.append(np.array(img))
        
        if images:
            imageio.mimsave(gif_filename, images, fps=fps)
            print(f"Created GIF: {gif_filename}")

def save_metrics_to_csv(metrics: dict, output_dirs: dict, model_info: dict):
    """Save metrics to CSV file following specs format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dirs['metrics'], f"rollout_metrics_{timestamp}.csv")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header with model info
        writer.writerow(['# Rollout Metrics'])
        writer.writerow(['# Model:', model_info.get('model_file', 'Unknown')])
        writer.writerow(['# Dataset:', model_info.get('dataset', 'Unknown')])
        writer.writerow(['# Rollout Steps:', len(metrics['timesteps'])])
        writer.writerow(['# Generated:', timestamp])
        writer.writerow([])  # Empty row
        
        # Metrics header
        writer.writerow(['timestep', 'mse_aggregate', 'ssim_aggregate'] + 
                       [f'mse_field_{i}' for i in range(len(metrics['mse_per_field'][0]) if metrics['mse_per_field'] else 0)] +
                       [f'ssim_field_{i}' for i in range(len(metrics['ssim_per_field'][0]) if metrics['ssim_per_field'] else 0)])
        
        # Data rows
        for i, timestep in enumerate(metrics['timesteps']):
            row = [
                timestep,
                metrics['mse_per_step'][i] if i < len(metrics['mse_per_step']) else '',
                metrics['ssim_per_step'][i] if i < len(metrics['ssim_per_step']) else ''
            ]
            
            # Add per-field metrics
            if i < len(metrics['mse_per_field']):
                row.extend(metrics['mse_per_field'][i])
            if i < len(metrics['ssim_per_field']):
                row.extend(metrics['ssim_per_field'][i])
            
            writer.writerow(row)
    
    print(f"Metrics saved to: {csv_filename}")
    return csv_filename

def main():
    parser = argparse.ArgumentParser(description="Extended rollout inference with GIF generation")
    
    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model files (e.g., out/models)')
    parser.add_argument('--model_file', type=str, required=True,
                       help='Model checkpoint filename (e.g., best_ft_DR2D.pth)')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['MHD','DR','CFD1D','CFD2D-IC','CFD3D','SW','DR1D','CFD2D',
                               'CFD3D-TURB', 'BE1D', 'GSDR2D', 'TGC3D', 'FNS_KF_2D', 'DR2D'],
                       help='Dataset name for loading test data')
    
    # Optional arguments
    parser.add_argument('--rollout_steps', type=int, default=50,
                       help='Number of rollout steps to perform')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Test sample index to use for rollout')
    parser.add_argument('--device_idx', type=int, default=0,
                       help='CUDA device index')
    parser.add_argument('--output_dir', type=str, default='out/results/rollouts',
                       help='Base output directory')
    parser.add_argument('--gif_fps', type=int, default=5,
                       help='GIF frames per second')
    parser.add_argument('--ar_order', type=int, default=1,
                       help='Autoregressive order (context length)')
    parser.add_argument('--model_size', type=str, 
                       choices=list(MORPH_MODELS.keys()), default='Ti',
                       help='Model size (inferred from checkpoint if not specified)')
    
    args = parser.parse_args()
    
    # Validate inputs
    model_path = os.path.join(args.model_dir, args.model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Setup device
    devices = DeviceManager.list_devices()
    device = devices[args.device_idx] if devices else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    output_dirs = create_output_directories(args.output_dir)
    print(f"Output directory: {output_dirs['base']}")
    
    # Initialize data config
    patch_size = 8
    DataConfig(MORPH_ROOT, patch_size)
    
    # Load test data
    print(f"Loading test data for {args.dataset}...")
    data_module = DataloaderChaos()
    
    # Build dataset paths (same as infer_MORPH.py)
    datasets = ["DR2d_data_pdebench","MHD3d_data_thewell","1dcfd_pdebench","2dSW_pdebench",
                "2dcfd_ic_pdebench","3dcfd_pdebench","1ddr_pdebench","2dcfd_pdebench",
                "3dcfd_turb_pdebench","1dbe_pdebench","2dgrayscottdr_thewell",
                "3dturbgravitycool_thewell","2dFNS_KF_pdegym"]
    
    datapaths = {
        'MHD': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[1]),
        'DR' : os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[0]),
        'CFD1D' : os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[2]),
        'CFD2D-IC': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[4]),
        'CFD3D': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[5]),
        'SW': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[3]),
        'DR1D': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[6]),
        'CFD2D': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[7]),
        'CFD3D-TURB': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[8]),
        'BE1D': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[9]),
        'GSDR2D': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[10]),
        'TGC3D': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[11]),
        'FNS_KF_2D': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[12]),
        'DR2D': os.path.join(MORPH_ROOT,'datasets', 'normalized_revin', datasets[0])
    }
    
    test_data = data_module.load_data(args.dataset, datapaths[args.dataset], split='test')
    try:
        print(f"Test data shape: {test_data.shape}")
    except Exception as e:
        print(f"Error: {args.dataset} path could not be found at {datapaths[args.dataset]}\n")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model_from_checkpoint(
        model_path, args.model_size, device, 
        max_ar=args.ar_order
    )
    
    # Model info for logging
    model_info = {
        'model_file': args.model_file,
        'dataset': args.dataset,
        'model_size': args.model_size,
        'ar_order': args.ar_order
    }
    
    # Perform rollout
    print(f"Performing rollout for sample {args.sample_idx}...")
    rollout_results = perform_extended_rollout(
        model, test_data, device, args.rollout_steps, 
        args.sample_idx, args.ar_order
    )
    
    if rollout_results['num_steps'] == 0:
        print("No successful rollout steps completed.")
        return
    
    print(f"Completed {rollout_results['num_steps']} rollout steps")
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics_per_timestep(
        rollout_results['predictions'], 
        rollout_results['ground_truths']
    )
    
    # Create visualizations and frames
    frame_files = create_visualization_frames(
        rollout_results['predictions'], 
        rollout_results['ground_truths'],
        output_dirs, args.dataset
    )
    
    # Create GIFs
    create_gifs_from_frames(frame_files, output_dirs, args.gif_fps)
    
    # Save metrics
    csv_file = save_metrics_to_csv(metrics, output_dirs, model_info)
    
    # Summary
    print("\n" + "="*50)
    print("ROLLOUT COMPLETE")
    print("="*50)
    print(f"Model: {args.model_file}")
    print(f"Dataset: {args.dataset}")
    print(f"Rollout steps: {rollout_results['num_steps']}")
    print(f"Output directory: {output_dirs['base']}")
    print(f"GIFs created: {len(frame_files)} fields")
    print(f"Metrics saved: {csv_file}")
    
    if metrics['mse_per_step']:
        avg_mse = np.mean(metrics['mse_per_step'])
        print(f"Average MSE: {avg_mse:.6f}")
    
    if SSIM_AVAILABLE and metrics['ssim_per_step']:
        valid_ssim = [s for s in metrics['ssim_per_step'] if not np.isnan(s)]
        if valid_ssim:
            avg_ssim = np.mean(valid_ssim)
            print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()