import os
import sys
sys.path.append('/home/shanwibo/Capstone-TamTanai')

import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
from tqdm import tqdm

from src.main import LegalAssistant


parser = argparse.ArgumentParser()
parser.add_argument('--n_documents', default=1, type=int, help='Number of legal documents')
args = parser.parse_args()
n_documents = args.n_documents

INFERENCE_RESULT_PATH = f'/project/lt200301-edubot/Capstone-TamTanai/inference_result/inference_end_to_end_n_documents={n_documents}.csv'

if n_documents == 1:
    max_input_tokens = 8192
elif n_documents <= 2:
    max_input_tokens = 8192 * 2
else:
    max_input_tokens = 8192 * 4

df = pd.read_csv('../asset/dataset/testdata.csv')

legal_assistant = LegalAssistant(max_input_tokens=max_input_tokens)

result_df = []
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    result = legal_assistant(row['question'], n_documents=n_documents)
    if result['retrieval'] is None:
        del result['retrieval']
    result = pd.json_normalize(result, sep='_').to_dict(orient='records')[0]
    result.update({
        'question': row['question'],
        'answer': row['answer']
    })
    result_df.append(result)
result_df = pd.DataFrame(result_df)
result_df = result_df[['question', 'answer', 'processing_time', 'domain_detection_result', 'domain_detection_probability',
                       'domain_detection_time_usage', 'retrieval_result', 'retrieval_reranking_score', 'retrieval_time_usage',
                       'generation_result', 'generation_time_usage']]
result_df.to_csv(INFERENCE_RESULT_PATH, index=False, encoding='utf-8')
