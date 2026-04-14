#!/bin/bash

#SBATCH --job-name="data_loading_pde_rkp"

#SBATCH --output="rkp_data.%j.%N.out"

#SBATCH --partition=shared

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=2G

#SBATCH --account=ccu108

#SBATCH --export=ALL

#SBATCH -t 00:01:00



module purge

module load cpu/0.15.4
module load gcc/10.2.0
module load openmpi/4.0.4
module load slurm/expanse/23.02.7
module load python/3.8.5
module load py-six
module load py-mpi4py
module load py-h5py

python inspect.py
