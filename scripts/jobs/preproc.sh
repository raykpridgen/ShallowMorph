#!/bin/bash

#SBATCH --job-name="rkp_morph_preproc"

#SBATCH --output="preproc.%j.%N.out"

#SBATCH --partition=compute

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=100G

#SBATCH --account=ccu108

#SBATCH --export=ALL

#SBATCH -t 00:30:00



module purge

module load cpu/0.15.4

module load gcc/10.2.0

source ../venv/bin/activate
cd ../MORPH/datasets/raw
python ../../../code/preprocess.py --morph-root ../.. --be1d be1d.hdf5 --dr2d dr2d.h5 --sw sw2d.hdf5 --force
