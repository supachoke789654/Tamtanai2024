import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Detector:

    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        self.model.eval()
        self.model.to(self.device)

    def _load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        return tokenizer, model

    def __call__(self, text, return_prob=True):
        encodings = self.tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        prediction = bool(prediction)
        probability = torch.softmax(logits, dim=-1).cpu().numpy()[0][1]
        probability = float(probability)
        if return_prob:
            return prediction, probability
        return prediction
