#!/usr/bin/env python3
"""
Batch rollout script to run rollout_inference.py on multiple models.

Usage:
    python code/batch_rollout.py --model_dir out/models --rollout_steps 50
    python code/batch_rollout.py --model_dir out/models --pattern "*DR2D*.pth" --dataset DR2D
"""

import os
import sys
import argparse
import subprocess
import glob
from pathlib import Path
import time

def find_model_files(model_dir: str, pattern: str = "*.pth") -> list:
    """Find model files matching pattern."""
    search_path = os.path.join(model_dir, pattern)
    model_files = glob.glob(search_path)
    return [os.path.basename(f) for f in model_files]

def infer_dataset_from_filename(filename: str) -> str:
    """Infer dataset name from model filename."""
    # Common patterns in sweep model names
    dataset_mapping = {
        'BE1D': 'BE1D',
        'SW': 'SW', 
        'DR2D': 'DR2D',
        'DR': 'DR2D',  # Sometimes DR maps to DR2D
        'CFD1D': 'CFD1D',
        'CFD2D': 'CFD2D',
        'CFD3D': 'CFD3D',
        'MHD': 'MHD',
        'GSDR2D': 'GSDR2D',
        'TGC3D': 'TGC3D',
        'FNS_KF_2D': 'FNS_KF_2D'
    }
    
    filename_upper = filename.upper()
    
    # Look for dataset indicators in filename
    for key, dataset in dataset_mapping.items():
        if key in filename_upper:
            return dataset
    
    # Default fallback
    print(f"Warning: Could not infer dataset from filename {filename}, using DR2D")
    return 'DR2D'

def infer_model_size_from_filename(filename: str) -> str:
    """Infer model size from filename."""
    filename_upper = filename.upper()
    
    if '_TI_' in filename_upper or '_TI.' in filename_upper:
        return 'Ti'
    elif '_S_' in filename_upper or '_S.' in filename_upper:
        return 'S'
    elif '_M_' in filename_upper or '_M.' in filename_upper:
        return 'M'
    elif '_L_' in filename_upper or '_L.' in filename_upper:
        return 'L'
    
    # Default fallback
    return 'Ti'

def run_rollout_inference(model_dir: str, model_file: str, dataset: str, 
                         rollout_steps: int, model_size: str, 
                         device_idx: int, gif_fps: int, ar_order: int,
                         output_dir: str) -> tuple:
    """Run rollout inference for a single model."""
    
    script_path = os.path.join(os.path.dirname(__file__), 'rollout_inference.py')
    
    cmd = [
        'python', script_path,
        '--model_dir', model_dir,
        '--model_file', model_file,
        '--dataset', dataset,
        '--rollout_steps', str(rollout_steps),
        '--model_size', model_size,
        '--device_idx', str(device_idx),
        '--gif_fps', str(gif_fps),
        '--ar_order', str(ar_order),
        '--output_dir', output_dir
    ]
    
    print(f"\n{'='*60}")
    print(f"Running rollout for: {model_file}")
    print(f"Dataset: {dataset}, Model Size: {model_size}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run with conda environment if specified
        if 'CONDA_DEFAULT_ENV' in os.environ:
            conda_cmd = ['conda', 'run', '-n', 'ml_base'] + cmd
            result = subprocess.run(conda_cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.1f}s): {model_file}")
            return True, f"Success in {duration:.1f}s"
        else:
            print(f"❌ FAILED ({duration:.1f}s): {model_file}")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False, f"Failed: {result.stderr[-200:]}"
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {model_file}")
        return False, "Timeout after 1 hour"
    except Exception as e:
        print(f"💥 ERROR: {model_file} - {e}")
        return False, f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Batch rollout inference on multiple models")
    
    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model files')
    
    # Optional arguments
    parser.add_argument('--pattern', type=str, default='best_*.pth',
                       help='File pattern to match model files (default: best_*.pth)')
    parser.add_argument('--dataset', type=str, 
                       choices=['MHD','DR','CFD1D','CFD2D-IC','CFD3D','SW','DR1D','CFD2D',
                               'CFD3D-TURB', 'BE1D', 'GSDR2D', 'TGC3D', 'FNS_KF_2D', 'DR2D'],
                       help='Dataset name (if not specified, will infer from filename)')
    parser.add_argument('--rollout_steps', type=int, default=50,
                       help='Number of rollout steps')
    parser.add_argument('--device_idx', type=int, default=0,
                       help='CUDA device index')
    parser.add_argument('--gif_fps', type=int, default=5,
                       help='GIF frames per second')
    parser.add_argument('--ar_order', type=int, default=1,
                       help='Autoregressive order')
    parser.add_argument('--output_dir', type=str, default='out/results/rollouts',
                       help='Base output directory')
    parser.add_argument('--max_models', type=int, default=None,
                       help='Maximum number of models to process (for testing)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without executing')
    
    args = parser.parse_args()
    
    # Validate model directory
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    # Find model files
    model_files = find_model_files(args.model_dir, args.pattern)
    
    if not model_files:
        print(f"No model files found matching pattern '{args.pattern}' in {args.model_dir}")
        return
    
    # Limit number of models if specified
    if args.max_models:
        model_files = model_files[:args.max_models]
    
    print(f"Found {len(model_files)} model files to process")
    
    # Process each model
    results = []
    successful = 0
    failed = 0
    
    for i, model_file in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] Processing: {model_file}")
        
        # Infer parameters if not specified
        dataset = args.dataset if args.dataset else infer_dataset_from_filename(model_file)
        model_size = infer_model_size_from_filename(model_file)
        
        if args.dry_run:
            print(f"DRY RUN - Would process {model_file} with dataset={dataset}, model_size={model_size}")
            continue
        
        # Run rollout inference
        success, message = run_rollout_inference(
            args.model_dir, model_file, dataset, args.rollout_steps,
            model_size, args.device_idx, args.gif_fps, args.ar_order,
            args.output_dir
        )
        
        results.append({
            'model_file': model_file,
            'dataset': dataset,
            'model_size': model_size,
            'success': success,
            'message': message
        })
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH ROLLOUT SUMMARY")
    print(f"{'='*60}")
    print(f"Total models processed: {len(model_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed models:")
        for result in results:
            if not result['success']:
                print(f"  ❌ {result['model_file']}: {result['message']}")
    
    print(f"\nOutputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()