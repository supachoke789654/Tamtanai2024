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


class LLM:

    def __init__(
        self,
        model_name,
        max_input_tokens=8192,
        top_p=0.9,
        temperature=0.1,
        repetition_penalty=1.1,
        max_new_tokens=512
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_path = self._find_best_model()
        self.max_input_tokens = max_input_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _find_best_model(self):
        log_path = f'/project/lt200301-edubot/Capstone-TamTanai/logs/{self.model_name}.csv'
        log = pd.read_csv(log_path)
        step = int(log.iloc[log['eval_loss'].argmin()]['step'])
        return f'/project/lt200301-edubot/Capstone-TamTanai/models/{self.model_name}/checkpoint-{step}'

    def _load_model_and_tokenizer(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            max_position_embeddings=self.max_input_tokens
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            model_max_length=self.max_input_tokens
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.use_cache = True
        if torch.cuda.device_count() > 1:
            model = DataParallel(model).module
        return model, tokenizer

    def __call__(self, text):
        encoded_input = self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(self.device)
        with torch.cuda.amp.autocast():
            generate_kwargs = dict(
                {"input_ids": encoded_input},
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            encoded_output = self.model.generate(**generate_kwargs)
        response = self.tokenizer.decode(encoded_output[0][len(encoded_input[0]):], skip_special_tokens=True)
        return response


class BaseLLM:

    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_path = self._get_model_path()
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _get_model_path(self):
        return f'/project/lt200301-edubot/Capstone-TamTanai/models/{self.model_name}'

    def _load_model_and_tokenizer(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            max_position_embeddings=8192
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            model_max_length=8192
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.use_cache = True
        if torch.cuda.device_count() > 1:
            model = DataParallel(model).module
        return model, tokenizer

    def __call__(self, text):
        encoded_input = self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(self.device)
        with torch.cuda.amp.autocast():
            generate_kwargs = dict(
                {"input_ids": encoded_input},
                do_sample=True,
                max_new_tokens=512,
                top_p=0.9,
                temperature=0.1,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            encoded_output = self.model.generate(**generate_kwargs)
        response = self.tokenizer.decode(encoded_output[0][len(encoded_input[0]):], skip_special_tokens=True)
        return response
