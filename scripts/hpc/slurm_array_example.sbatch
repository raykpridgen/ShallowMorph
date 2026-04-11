#!/bin/bash
#SBATCH -J morph-sweep
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=1-72
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#
# 1) Edit array bounds to match your manifest row count (see ``wc -l manifest.csv`` minus header).
# 2) Set REPO, MANIFEST, CONDA_SH, MODULES, and partition/account flags for your site.
# 3) Generate the manifest once (login node):
#      cd "$REPO" && PYTHONPATH=code python -m morph_wrap.run_sweep --sweep A --dry-run \
#        --manifest-csv "$REPO/out/sweep_A_manifest.csv"
#
# Logs: Slurm writes stdout/stderr to logs/; failures also append JSON lines to
#       ``out/sweep_failures.jsonl`` when using ``run_sweep`` below.

set -euo pipefail

REPO="${REPO:-/path/to/morph}"
MANIFEST="${MANIFEST:-${REPO}/out/sweep_A_manifest.csv}"
# Space-separated module names (optional):
export MORPH_HPC_MODULES="${MORPH_HPC_MODULES:-}"
# e.g. source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate morph
CONDA_SH="${CONDA_SH:-}"
ENV_NAME="${ENV_NAME:-pytorch_py38_env}"

cd "${REPO}"
mkdir -p logs out

if [[ -n "${MORPH_HPC_MODULES}" ]]; then
  # shellcheck source=load_modules.sh
  source "${REPO}/scripts/hpc/load_modules.sh"
fi

if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
fi

export PYTHONPATH="${REPO}/code${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

python -m morph_wrap.env_check --require-cuda || exit 3

python -m morph_wrap.run_sweep \
  --manifest-csv "${MANIFEST}" \
  --use-slurm-array-task-id \
  --array-base 1 \
  --execute \
  --morph-root "${REPO}/MORPH" \
  --dataset-root "${REPO}/MORPH" \
  --out-dir "${REPO}/out" \
  --summary-log "${REPO}/out/sweep_slurm_summary.log"
