#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 2 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 120:00:00
#SBATCH -A lt200301
#SBATCH -J e2e_1

module load Mamba/23.11.0-0
conda activate retrieval
pip install numpy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python end_to_end-new_testdataset.py --n_documents 1 --new_testdataset 1