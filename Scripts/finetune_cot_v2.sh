#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 2 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 120:00:00
#SBATCH -A lt200301
#SBATCH -J finetune

module load Mamba/23.11.0
conda activate tamtanai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetune_cot_v2.py