import FlagEmbedding

#torchrun --nproc_per_node {number of gpus} \ 
#-m FlagEmbedding.reranker.run \
#--output_dir {path to save model} \
#--model_name_or_path BAAI/bge-reranker-base \
#--train_data ./toy_finetune_data.jsonl \
#--learning_rate 6e-5 \
#--fp16 \
#--num_train_epochs 5 \
#--per_device_train_batch_size {batch size; set 1 for toy data} \
#--gradient_accumulation_steps 4 \
#--dataloader_drop_last True \
#--train_group_size 16 \
#--max_len 512 \
#--weight_decay 0.01 \
#--logging_steps 10 

model_name = "bge-reranker-v2-m3-finetune-with-neg=other-document"
dataset_path = "/project/lt200301-edubot/Capstone-TamTanai/reranker_training_dataset/reranker_training_dataset_without_other_documents.jsonl"
torchrun --nproc_per_node {4} \
-m FlagEmbedding.reranker.run \
--output_dir {f"/project/lt200301-edubot/Capstone-TamTanai/models/{model_name}"} \
--model_name_or_path "/project/lt200301-edubot/Capstone-TamTanai/models/bge-reranker-v2-m3" \
--train_data dataset_path\
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size y{16} \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--train_group_size 16 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 10 