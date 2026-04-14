#!/bin/bash

#SBATCH --job-name="rkp_morph_sweep"

#SBATCH --output="results.%j.%N.out"

#SBATCH --partition=compute

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --account=ccu108

#SBATCH --export=ALL

#SBATCH -t 00:10:00



module purge

module load cpu/0.15.4

module load gcc/10.2.0

source ../venv/bin/activate
cd ../code
python3 -m morph_wrap.run_sweep --sweep A --execute --execute-job-index 4 --device-index auto --manifest-csv ../out/sweep_A_manifest_72.csv
