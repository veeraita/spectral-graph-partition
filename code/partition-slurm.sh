#!/usr/bin/env bash

#SBATCH --time=0-05:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=5
#SBATCH -o slurm-%A.log

srun python spectral_partition.py $1 $2
