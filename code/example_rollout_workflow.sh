#!/bin/bash
# Example workflow for MORPH rollout inference
# This script demonstrates the complete workflow from setup to visualization

echo "MORPH Rollout Inference Workflow Example"
echo "========================================"

# Activate conda environment
echo "1. Activating conda environment..."
conda activate ml_base

# Test setup
echo "2. Testing setup..."
python code/test_rollout_setup.py

echo ""
echo "3. Example commands (uncomment to run):"

# Single model example
echo "# Single model rollout example:"
echo "# python code/rollout_inference.py \\"
echo "#     --model_dir out/models \\"
echo "#     --model_file best_ft_DR2D.pth \\"
echo "#     --dataset DR2D \\"
echo "#     --rollout_steps 50 \\"
echo "#     --gif_fps 5"

echo ""

# Batch processing example
echo "# Batch processing example:"
echo "# python code/batch_rollout.py \\"
echo "#     --model_dir out/models \\"
echo "#     --pattern 'best_*.pth' \\"
echo "#     --rollout_steps 50 \\"
echo "#     --max_models 3"

echo ""

# Dry run example
echo "# Test what would be processed (dry run):"
echo "# python code/batch_rollout.py \\"
echo "#     --model_dir out/models \\"
echo "#     --pattern '*.pth' \\"
echo "#     --dry_run"

echo ""
echo "4. After running, check outputs:"
echo "   ls -la out/results/rollouts/"
echo "   # View GIFs, metrics CSV, and frame images"

echo ""
echo "5. For more details, see:"
echo "   code/README_rollout.md"