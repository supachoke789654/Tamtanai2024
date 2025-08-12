import time

from .domain_detection import Detector
from .retrieval import Retriever
from .llm import LLM, BaseLLM

import warnings
warnings.filterwarnings('ignore')


LLM_INPUT_TEMPLATE = '''
<s><|im_start|>system
{system_prompt}
</s><|im_start|>user
{user_query}
</s><|im_start|>assistant
'''

DEFAULT_SYSTEM_PROMPT_TEMPLATE = '''
คุณคือนักกฎหมายที่จะตอบคำถามเกี่ยวกับกฎหมาย จงตอบคำถามโดยใช้ความรู้ที่ให้ดังต่อไปนี้
ถ้าหากคุณไม่รู้คำตอบ ให้ตอบว่าไม่รู้ อย่าสร้างคำตอบขึ้นมาเอง
ความรู้ที่ให้:
{context}
'''

DEFAULT_BASE_SYSTEM_PROMPT_TEMPLATE = '''
คุณคือนักกฎหมายที่จะตอบคำถามทั่วไปด้วยความสุภาพ
'''

WARMUP_INPUT = 'การขับรถชนคนโดยไม่ตั้งใจผิดกฎหมายไหม'


class LegalAssistant:

    def __init__(
        self,
        # Domain Detection part
        domain_detection_model_path='/project/lt200301-edubot/Capstone-TamTanai/models/finetuned_mpnet',
        # Retrieval part
        specific_case_path='/home/shanwibo/Capstone-TamTanai/notebooks/specific_case_knowledge.txt',
        general_source_dir='/home/shanwibo/Capstone-TamTanai/asset/documentation',
        keyword_search=True,
        idf=True,
        case_number=True,
        law_number=True,
        law_name=True,
        context_search=True,
        embedding_model_path='/project/lt200301-edubot/Capstone-TamTanai/models/multilingual-e5-large',
        persist_directory=None,
        vector_store='faiss',
        similarity_threshold=0.6,
        n_similar_documents=40,
        reranker_path='/project/lt200301-edubot/Capstone-TamTanai/models/bge-reranker-v2-m3-finetune-with_similar=5_keyword=5_2nd',
        # Fintuned LLM part
        system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        finetuned_llm_name='typhoon2-qwen2.5-7b-instruct-finetune-dataset-v1-exam-thailegal-with-v1+exam+thailegal-validation-set',
        max_input_tokens=8192,
        top_p=0.9,
        temperature=0.1,
        repetition_penalty=1.1,
        max_new_tokens=512,
        # Base LLM part
        base_system_prompt_template=DEFAULT_BASE_SYSTEM_PROMPT_TEMPLATE,
        base_llm_name='typhoon2-qwen2.5-7b-instruct'
    ):
        self.domain_detector = Detector(model_path=domain_detection_model_path)
        retrieval_kwargs = {
            'specific_case_path': specific_case_path,
            'general_source_dir': general_source_dir,
            'keyword_search': keyword_search,
            'idf': idf,
            'case_number': case_number,
            'law_number': law_number,
            'law_name': law_name,
            'context_search': context_search,
            'embedding_model_path': embedding_model_path,
            'persist_directory': persist_directory,
            'vector_store': vector_store,
            'similarity_threshold': similarity_threshold,
            'n_similar_documents': n_similar_documents,
            'reranker_path': reranker_path,
        }
        self.retriever = Retriever(**retrieval_kwargs)
        finetuned_llm_kwargs = {
            'model_name': finetuned_llm_name,
            'max_input_tokens': max_input_tokens,
            'top_p': top_p,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
            'max_new_tokens': max_new_tokens
        }
        self.llm = LLM(**finetuned_llm_kwargs)
        self.base_llm = BaseLLM(model_name=base_llm_name)
        self.system_prompt_template = system_prompt_template
        self.base_system_prompt_template = base_system_prompt_template
        self._warmup()

    def _warmup(self):
        self.domain_detector(WARMUP_INPUT)
        self.retriever.search(WARMUP_INPUT)

    def __call__(self, text, n_documents=1):
        processing_start = time.time()
        domain_detection_start = time.time()
        legal_domain, legal_domain_prob = self.domain_detector(text, return_prob=True)
        domain_detection_end = time.time()
        domain_detection = {
            'result': legal_domain,
            'probability': legal_domain_prob,
            'time_usage': domain_detection_end - domain_detection_start
        }
        if legal_domain:
            retrieval_start = time.time()
            retrieval_results = self.retriever.search(text, n=n_documents)
            retrieval_end = time.time()
            retrieval = {
                'result': [retrieval_result['document'].page_content for retrieval_result in retrieval_results],
                'source': [retrieval_result['document'].metadata['source'] for retrieval_result in retrieval_results],
                'reranking_score': [retrieval_result['reranking_score'] for retrieval_result in retrieval_results],
                'time_usage': retrieval_end - retrieval_start
            }
            context = '\n\n\n'.join([retrieval_result['document'].page_content for retrieval_result in retrieval_results])
            system_prompt = self.system_prompt_template.format(context=context)
            llm_input = LLM_INPUT_TEMPLATE.format(system_prompt=system_prompt, user_query=text)
            generation_start = time.time()
            response = self.llm(llm_input)
            if '</s>' in response:
                response = response[: response.index('</s>')]
            generation_end = time.time()
            generation = {
                'result': response,
                'time_usage': generation_end - generation_start
            }
        else:
            retrieval = None
            llm_input = LLM_INPUT_TEMPLATE.format(system_prompt=self.base_system_prompt_template, user_query=text)
            generation_start = time.time()
            response = self.base_llm(llm_input)
            generation_end = time.time()
            generation = {
                'result': response,
                'time_usage': generation_end - generation_start
            }
        processing_end = time.time()
        
        return {
            'domain_detection': domain_detection,
            'retrieval': retrieval,
            'generation': generation,
            'processing_time': processing_end - processing_start
        }
