import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "<base model path>"
finetune_weight_path = "<adapter path>"
save_dir = "<export merge weight path>"

# tokenizer = AutoTokenizer.from_pretrained(finetune_weight_path)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side="left",
    add_eos_token=False,
    add_bos_token=False,
)
tokenizer.model_max_length = 4096
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=bnb_config,
    device_map = 'auto',
    local_files_only=True,
    # pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.float16,
    use_cache=False,
)

model = PeftModel.from_pretrained(model, finetune_weight_path, device_map='auto')

print('load model completed\n')

model = model.merge_and_unload()

print('merge model completed\n')

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print('save model completed\n')


