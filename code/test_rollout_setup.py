#!/usr/bin/env python3
"""
Test script to validate rollout inference setup and dependencies.

Usage:
    python code/test_rollout_setup.py
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_conda_env():
    """Check if we're in the correct conda environment."""
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"Current conda environment: {env_name}")
    
    if env_name != 'ml_base':
        print("⚠️  Warning: Expected 'ml_base' conda environment")
        print("   Run: conda activate ml_base")
    else:
        print("✅ Correct conda environment")
    
    return env_name == 'ml_base'

def check_python_packages():
    """Check if required Python packages are available."""
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'), 
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('imageio', 'ImageIO'),
        ('tqdm', 'tqdm')
    ]
    
    optional_packages = [
        ('pytorch_msssim', 'pytorch-msssim (for SSIM calculations)')
    ]
    
    print("\nChecking required packages:")
    all_required_ok = True
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Install with: pip install {package}")
            all_required_ok = False
    
    print("\nChecking optional packages:")
    optional_ok = True
    
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - Install with: pip install {package.replace('_', '-')}")
            optional_ok = False
    
    return all_required_ok, optional_ok

def check_morph_structure():
    """Check if MORPH directory structure is correct."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    morph_root = os.path.join(repo_root, "MORPH")
    
    print(f"\nChecking MORPH structure:")
    print(f"Repository root: {repo_root}")
    print(f"MORPH root: {morph_root}")
    
    required_paths = [
        "MORPH",
        "MORPH/src/utils/vit_conv_xatt_axialatt2.py",
        "MORPH/src/utils/dataloaders/dataloaderchaos.py",
        "MORPH/src/utils/visualize_rollouts_3d_full.py",
        "MORPH/scripts/infer_MORPH.py"
    ]
    
    all_paths_ok = True
    for rel_path in required_paths:
        full_path = os.path.join(repo_root, rel_path)
        if os.path.exists(full_path):
            print(f"✅ {rel_path}")
        else:
            print(f"❌ {rel_path}")
            all_paths_ok = False
    
    return all_paths_ok

def check_output_directories():
    """Check and create output directories if needed."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    required_dirs = [
        "out",
        "out/results",
        "out/results/rollouts"
    ]
    
    print("\nChecking/creating output directories:")
    
    for rel_dir in required_dirs:
        full_dir = os.path.join(repo_root, rel_dir)
        if os.path.exists(full_dir):
            print(f"✅ {rel_dir} (exists)")
        else:
            try:
                os.makedirs(full_dir, exist_ok=True)
                print(f"✅ {rel_dir} (created)")
            except Exception as e:
                print(f"❌ {rel_dir} (failed to create: {e})")
                return False
    
    return True

def check_model_files():
    """Check for example model files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    model_dirs = [
        "out/models",
        "out/*/models",  # sweep output structure
        "MORPH/models"   # pretrained models
    ]
    
    print("\nLooking for model files:")
    found_models = []
    
    for model_pattern in model_dirs:
        search_path = os.path.join(repo_root, model_pattern)
        if '*' in model_pattern:
            import glob
            matching_dirs = glob.glob(search_path)
            for dir_path in matching_dirs:
                if os.path.isdir(dir_path):
                    pth_files = [f for f in os.listdir(dir_path) if f.endswith('.pth')]
                    if pth_files:
                        found_models.extend([(dir_path, f) for f in pth_files[:3]])  # Max 3 per dir
        else:
            if os.path.isdir(search_path):
                pth_files = [f for f in os.listdir(search_path) if f.endswith('.pth')]
                if pth_files:
                    found_models.extend([(search_path, f) for f in pth_files[:3]])  # Max 3 per dir
    
    if found_models:
        print(f"✅ Found {len(found_models)} model files:")
        for model_dir, model_file in found_models[:5]:  # Show max 5
            rel_path = os.path.relpath(model_dir, repo_root)
            print(f"   {rel_path}/{model_file}")
        if len(found_models) > 5:
            print(f"   ... and {len(found_models) - 5} more")
    else:
        print("⚠️  No .pth model files found")
        print("   Run training first or check model directories")
    
    return len(found_models) > 0

def test_imports():
    """Test importing key MORPH modules."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    morph_root = os.path.join(repo_root, "MORPH")
    
    print("\nTesting MORPH imports:")
    
    # Add MORPH to path
    sys.path.insert(0, morph_root)
    
    test_imports = [
        ('src.utils.vit_conv_xatt_axialatt2', 'ViT3DRegression'),
        ('src.utils.dataloaders.dataloaderchaos', 'DataloaderChaos'),
        ('src.utils.visualize_rollouts_3d_full', 'Visualize3DRolloutPredictions'),
        ('config.data_config', 'DataConfig')
    ]
    
    import_success = True
    
    for module_name, class_name in test_imports:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name}")
            else:
                print(f"⚠️  {module_name} (missing {class_name})")
        except ImportError as e:
            print(f"❌ {module_name} - {e}")
            import_success = False
    
    return import_success

def print_usage_examples():
    """Print example usage commands."""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Single model rollout:")
    print("   python code/rollout_inference.py \\")
    print("       --model_dir out/models \\")
    print("       --model_file best_ft_DR2D.pth \\")
    print("       --dataset DR2D \\")
    print("       --rollout_steps 50")
    
    print("\n2. Batch processing:")
    print("   python code/batch_rollout.py \\")
    print("       --model_dir out/models \\")
    print("       --rollout_steps 50")
    
    print("\n3. Test run (dry run):")
    print("   python code/batch_rollout.py \\")
    print("       --model_dir out/models \\")
    print("       --dry_run")
    
    print(f"\n4. View outputs:")
    print("   ls -la out/results/rollouts/")

def main():
    print("MORPH Rollout Inference Setup Test")
    print("="*50)
    
    # Run all checks
    env_ok = check_conda_env()
    required_ok, optional_ok = check_python_packages()
    morph_ok = check_morph_structure()
    dirs_ok = check_output_directories()
    models_ok = check_model_files()
    imports_ok = test_imports()
    
    # Summary
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    checks = [
        ("Conda Environment", env_ok),
        ("Required Packages", required_ok),
        ("Optional Packages", optional_ok),
        ("MORPH Structure", morph_ok),
        ("Output Directories", dirs_ok),
        ("Model Files", models_ok),
        ("MORPH Imports", imports_ok)
    ]
    
    all_critical_ok = all([required_ok, morph_ok, dirs_ok, imports_ok])
    
    for check_name, status in checks:
        status_icon = "✅" if status else ("⚠️ " if check_name in ["Conda Environment", "Optional Packages", "Model Files"] else "❌")
        print(f"{status_icon} {check_name}")
    
    if all_critical_ok:
        print(f"\n🎉 Setup looks good! You can run rollout inference.")
        if not optional_ok:
            print("   Note: Install optional packages for full functionality")
        if not models_ok:
            print("   Note: No model files found - run training first")
    else:
        print(f"\n❌ Setup has critical issues. Please fix the problems above.")
    
    print_usage_examples()

if __name__ == "__main__":
    main()