import os
import time
import torch
import pandas as pd
import bitsandbytes as bnb

from tqdm import tqdm
from torch.nn import DataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE} device")


def find_best_model(model_name):
    log_path = f'/project/lt200301-edubot/Capstone-TamTanai/logs/{model_name}.csv'
    log = pd.read_csv(log_path)
    step = int(log.iloc[log['eval_loss'].argmin()]['step'])
    return f'/project/lt200301-edubot/Capstone-TamTanai/models/{model_name}/checkpoint-{step}'


# SeaLLMs-v3-7B-Chat-finetune-dataset-v1 Llama-3-Typhoon-1.5-8B-finetune-dataset-v1
MODEL_NAME = 'typhoon2-qwen2.5-7b-instruct-finetune-dataset-v1-exam-thailegal-with-v1+exam+thailegal-validation-set' # Llama-3-Typhoon-1.5-8B-finetune-v1-rslora,#Llama-3-Typhoon-1.5-8B
FIND_BEST_MODEL = True
MODEL_NAME_OR_PATH = f"/project/lt200301-edubot/Capstone-TamTanai/models/{MODEL_NAME}"
if FIND_BEST_MODEL:
    MODEL_NAME_OR_PATH = find_best_model(MODEL_NAME)
EVALUATE_RESULT_PATH = f"/project/lt200301-edubot/Capstone-TamTanai/inference_result/inference_{MODEL_NAME}_fewshot.csv" #exam, exam_doc ,exam_doc_only_question.csv, cot , general , irac ,irac_v2, exam_cot, exam_irac

print(f'Inferencing model {MODEL_NAME}')

# Load both LLM model and tokenizer
def load_LLM_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        quantization_config=bnb_config,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=True,
        max_position_embeddings=8192
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        model_max_length=8192
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_LLM_and_tokenizer()
model.config.use_cache = True

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model).module


def generate_answer_with_timer(text: str):
    try:
        start_time = time.time()
        encoded_input = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(DEVICE)
        with torch.cuda.amp.autocast():
            generate_kwargs = dict(
                {"input_ids": encoded_input},
                do_sample=True,
                max_new_tokens=512,
                top_p=0.9,
                #top_k=10,
                temperature=0.1,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        encoded_output = model.generate(**generate_kwargs)
        response = tokenizer.decode(encoded_output[0][len(encoded_input[0]):], skip_special_tokens=True)
        response_time = time.time() - start_time
        return response, response_time
    except:
        print("Encounter error!")
        return None, 0.0


test_data = pd.read_csv("../asset/dataset/testdata_fewshot.csv")  #testdata.csv, exam_test_data_with_document_and_prompt_v2.csv, generalQA.csv, testdata_cot.csv, testdata_irac.csv, testdata_irac_v2.csv, testdata_cot_v2.csv, exam_test_data_with_document_and_irac_prompt.csv
print("Finished reading data")

answers = []
times = []
for i, (index, data_point) in tqdm(enumerate(test_data.iterrows()), total=test_data.shape[0]):
    prompt = data_point["prompt_fewshot"] #prompt , prompt_with_doc, prompt_with_doc_only_question, prompt_irac, prompt_cot
    output = generate_answer_with_timer(prompt)
    answers.append(output[0])
    times.append(output[1])

test_data["response_finetune"] = answers
test_data["time_finetune"] = times

test_data.to_csv(EVALUATE_RESULT_PATH, index=False, encoding='utf-8')
print("Finished inference test dataset")