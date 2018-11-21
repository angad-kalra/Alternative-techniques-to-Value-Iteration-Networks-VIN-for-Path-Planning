#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

#SBATCH -p cidsegpu1

#SBATCH --gres=gpu:1

#SBATCH -t 0-72:00
##SBATCH -A arkalra
#SBATCH -o slurm.%j-GRU-moore10k-k15-f3.out
#SBATCH -e slurm.%j-GRU-moore10k-k15-f3.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arkalra@asu.edu