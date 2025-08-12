#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 120:00:00
#SBATCH -A lt200301
#SBATCH -J reranker

module load Mamba/23.11.0
conda activate retrieval

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node 4 \
-m FlagEmbedding.reranker.run \
--output_dir /project/lt200301-edubot/Capstone-TamTanai/models/bge-reranker-v2-m3-finetune-with_similar=5_keyword=5_2nd \
--model_name_or_path /project/lt200301-edubot/Capstone-TamTanai/models/bge-reranker-v2-m3 \
--train_data /project/lt200301-edubot/Capstone-TamTanai/reranker_training_dataset/reranker_training_dataset_with_similar=5_keyword=5_2nd_train.jsonl \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--train_group_size 6 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 10 