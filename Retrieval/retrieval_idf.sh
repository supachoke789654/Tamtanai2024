#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 120:00:00
#SBATCH -A lt200301
#SBATCH -J retrieval

module load Mamba/23.11.0
conda activate retrieval

python retrieval_idf.py