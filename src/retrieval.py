import os
import sys
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import nltk
from nltk.corpus import stopwords
from pythainlp import word_tokenize, pos_tag
from pythainlp.corpus.common import thai_stopwords

import torch
from FlagEmbedding import FlagReranker
from langchain.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader


INGEST_THREADS = os.cpu_count() or 8
DOCUMENT_MAP = {
    '.txt': TextLoader,
    '.md': TextLoader,
    '.pdf': PDFMinerLoader,
    '.csv': CSVLoader,
    '.xls': UnstructuredExcelLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.docx': Docx2txtLoader,
    '.doc': Docx2txtLoader,
}
ROOT_DIR = os.path.dirname(os.getcwd())

ENDL = '\n'

CASE_MAPPER = {
    'law_doc-84-89.txt': '761/2566',
    'law_doc-44-46.txt': '1301/2566',
    'law_doc-54-57.txt': '1225/2566',
    'law_doc-12-13.txt': '2525/2566',
    'law_doc-40-43.txt': '1305/2566',
    'law_doc-14-15.txt': '2085/2566',
    'law_doc-64-69.txt': '1090/2566',
    'law_doc-1-5.txt': '2610/2566',
    'law_doc-78-81.txt': '882/2566',
    'law_doc-82-83.txt': '835/2566',
    'law_doc-35-39.txt': '1306/2566',
    'law_doc-16-20.txt': '1574/2566',
    'law_doc-32-34.txt': '1373/2566',
    'law_doc-74-77.txt': '934/2566',
    'law_doc-6-11.txt': '2609/2566',
    'law_doc-90-92.txt': '756/2566',
    'law_doc-47-53.txt': '1300/2566',
    'law_doc-58-63.txt': '1101/2566',
    'law_doc-70-73.txt': '1003/2566',
    'law_doc-21-31.txt': '1542/2566',
}
CASE_MAPPER_REVERSE = {f'คดี {case}': filename for filename, case in CASE_MAPPER.items()}

THAI_STOPWORDS = set(thai_stopwords())
STOPWORDS = set(stopwords.words())

KEY_TAGS = ['NCMN', 'NCNM', 'NPRP', 'NONM', 'NLBL', 'NTTL']


def load_single_document(file_path, law_number=True, law_name=True):
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class == TextLoader:
        loader = TextLoader(file_path, encoding='utf-8')
    elif loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError('Document type is undefined')
    document = loader.load()[0]
    if law_number:
        extracted_law_numbers = extract_law_numbers(file_path)
        document.metadata['law_numbers'] = extracted_law_numbers
    if law_name:
        extracted_law_name = extract_law_name(file_path)
        document.metadata['law_name'] = extracted_law_name
    return document


def extract_law_numbers(file_path):
    pattern = r'มาตรา (\d+(?:/\d+)?)(?: (ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ|เอกาทศ|ทวาทศ|เตรส|จตุทศ|ปัณรส|โสฬส|สัตตรส|อัฏฐารส))?'
    extracted_law_numbers = []
    with open(file_path) as file:
        for line in file:
            if line.startswith('มาตรา'):
                match = re.search(pattern, line.strip())
                if match:
                    number = match.group(1)
                    suffix = match.group(2) if match.group(2) else ''
                    law_number = f'{number} {suffix}'.strip()
                    if '(ยกเลิก)' in line:
                        continue
                    extracted_law_numbers.append(law_number)
    return extracted_law_numbers


def extract_law_name(file_path):
    return file_path.split('/')[-2].strip().split(' ')[0]


def load_document_batch(file_paths):
    with ThreadPoolExecutor(len(file_paths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in file_paths]
        data_list = [future.result() for future in futures]
        return (data_list, file_paths)


def load_documents(source_dir, chunk_size=1000, chunk_overlap=200):
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)
    
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = max(round(len(paths) / n_workers), 1)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        for i in range(0, len(paths), chunksize):
            file_paths = paths[i : (i + chunksize)]
            future = executor.submit(load_document_batch, file_paths)
            futures.append(future)
        for future in as_completed(futures):
            contents, _ = future.result()
            docs.extend(contents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(docs)
    return documents


def load_specific_case_documents(specific_case_path):
    with open(specific_case_path, 'r', encoding='utf-8') as f:
        content = f.read()
    contents = content.split('\n\n')
    specific_case_documents = [
        Document(
            page_content=f'{ENDL.join(content.split(ENDL)[1:])}',
            metadata={
                'source': f'docs/{CASE_MAPPER_REVERSE[content.split(ENDL)[0]]}',
                'category': 'specific',
            },
        )
        for content in contents
    ]
    return specific_case_documents


def load_general_documents(general_source_dir):
    general_documents = load_documents(source_dir=general_source_dir, chunk_size=10e14, chunk_overlap=0)
    for i in range(len(general_documents)):
        general_documents[i].metadata['source'] = general_documents[i].metadata['source'].replace(f'{ROOT_DIR}/', '')
        general_documents[i].metadata['category'] = 'general'
    return general_documents


class Retriever:

    def __init__(self, specific_case_path=None, general_source_dir=None, keyword_search=True,
                 idf=True, case_number=True, law_number=True, law_name=True, context_search=True,
                 embedding_model_path='/project/lt200301-edubot/Capstone-TamTanai/models/multilingual-e5-large',
                 persist_directory='Capstone-TamTanai/src/retrieval/vectordb', vector_store='faiss',
                 similarity_threshold=0.55, n_similar_documents=50,
                 reranker_path='/project/lt200301-edubot/Capstone-TamTanai/models/bge-reranker-v2-m3-finetune-with_similar=5_keyword=5_2nd'):
        self.documents = []
        if specific_case_path:
            self.documents += load_specific_case_documents(specific_case_path)
        if general_source_dir:
            self.documents += load_general_documents(general_source_dir)
        self._keyword_search = keyword_search
        self.idf = idf
        self.case_number = case_number
        self.law_number = law_number
        self.law_name = law_name
        if self.law_name:
            self.law_names = self._load_law_name()
        self._context_search = context_search
        self.embedding_model_path = embedding_model_path
        self.persist_directory = persist_directory
        self.vector_store = vector_store
        # self.similarity_threshold = similarity_threshold
        self.distance_threshold = 1 - similarity_threshold
        self.n_similar_documents = n_similar_documents
        self.reranker_path = reranker_path
        if self._context_search:
            self.embedding_model = self._load_embedding_model()
            self.vector_database = self._load_vector_database()
        self.reranker = self._load_reranker()

    def _load_law_name(self):
        law_names = []
        for document in self.documents:
            if 'law_name' in document.metadata and document.metadata['law_name'] not in law_names and \
            document.metadata['law_name'] != '.ipynb_checkpoints':
                law_names.append(document.metadata['law_name'])
        return law_names
            
    def _load_embedding_model(self):
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_path, model_kwargs={'device': device_type})
        return embeddings 

    def _load_vector_database(self):
        if self.persist_directory and os.path.isdir(self.persist_directory):
            if self.vector_store == 'faiss':
                vector_db = FAISS.load_local(self.persist_directory, self.embedding_model,
                                             allow_dangerous_deserialization=True)
            elif self.vector_store == 'chroma':
                vector_db = Chroma(embedding_function=self.embedding_model, persist_directory=self.persist_directory)
                vector_db.persist()
            else:
                raise NotImplementedError(f'Embedding Algorithm {self.vector_store} is not supported yet.')
        else:
            if self.vector_store == 'faiss':
                vector_db = FAISS.from_documents(documents=self.documents, embedding=self.embedding_model)
                if self.persist_directory:
                    vector_db.save_local(self.persist_directory)
            elif self.vector_store == 'chroma':
                vector_db = Chroma.from_documents(documents=self.documents, embedding=self.embedding_model,
                                                  persist_directory=self.persist_directory)
                if self.persist_directory:
                    vector_db.persist()
            else:
                raise NotImplementedError(f'Embedding Algorithm {self.vector_store} is not supported yet.')
        return vector_db

    def _load_reranker(self):
        return FlagReranker(self.reranker_path, use_fp16=True)

    def search(self, query, n=1, similarity_threshold=None, k=None):
        retrieved_documents = []
        if self._keyword_search:
            retrieved_documents += self.keyword_search(query)
        if self._context_search:
            if similarity_threshold is None:
                similarity_threshold = 1 - self.distance_threshold
            if k is None:
                k = self.n_similar_documents
            retrieved_documents += self.context_search(query, similarity_threshold, k)
        retrieved_documents = self.remove_duplicated_documents(retrieved_documents)
        reranked_documents, reranking_scores = self.rerank(query, retrieved_documents, n)
        return [{'document': reranked_document, 'reranking_score': reranking_score}
                 for reranked_document, reranking_score in zip(reranked_documents, reranking_scores)]
    
    def keyword_search(self, query):
        keywords = self.extract_keyword(query)
        if not keywords:
            return []
        keyword_searched_documents = []
        found_case_number = False
        found_law_number = False
        found_law_name = False
        if self.case_number:
            found_case_number, case_numbers = self.find_case_number(query)
        if self.law_number:
            found_law_number, law_numbers = self.find_law_number(query)
        if self.law_name:
            found_law_name, law_names = self.find_law_name(query)

        for document in self.documents:
            if found_case_number:
                for case_number in case_numbers:
                    pattern = re.compile(re.escape(case_number))
                    if pattern.search(document.page_content):
                        matched_keywords = self.keyword_matcher(document, keywords)
                        if len(matched_keywords) >= min(3, len(keywords)):
                            keyword_searched_documents.append(document)
                            continue
            law_number_documents = []
            if found_law_number:
                for law_number in law_numbers:
                    if 'law_numbers' in document.metadata and law_number in document.metadata['law_numbers']:
                        law_number_documents.append(document)
            law_name_documents = []
            if found_law_name:
                for law_name in law_names:
                    if 'law_name' in document.metadata and law_name == document.metadata['law_name']:
                        law_name_documents.append(document)

            if found_law_number and not found_law_name:
                law_number_and_name_documents = law_number_documents
            if not found_law_number and found_law_name:
                law_number_and_name_documents = law_name_documents
            if found_law_number and found_law_name:
                law_number_and_name_documents = [document for document in law_number_documents if document in law_name_documents]
            else:
                matched_keywords = self.keyword_matcher(document, keywords)
                if len(matched_keywords) >= min(2, len(keywords)):
                    keyword_searched_documents.append(document)
        return keyword_searched_documents
        

    def context_search(self, query, similarity_threshold=None, k=None):
        if similarity_threshold is None:
            similarity_threshold = 1 - self.distance_threshold
        if k is None:
            k = self.n_similar_documents
        score_threshold = 1 - similarity_threshold
        retriever = self.vector_database.as_retriever(search_type='similarity',
                                                      search_kwargs={'score_threshold': score_threshold,
                                                                     'k': k})
        retrieved_documents = retriever.get_relevant_documents(query)
        return retrieved_documents
        

    def rerank(self, query, documents, n=1):
        if not documents:
            return [], []
        scores = self.reranker.compute_score([[query, document.page_content] for document in documents], normalize=True)
        sorted_pairs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_documents = zip(*sorted_pairs)
        sorted_scores = list(sorted_scores)
        sorted_documents = list(sorted_documents)
        return sorted_documents[:n], sorted_scores[:n]

    def extract_keyword(self, query):
        tokens = word_tokenize(query, engine='newmm', keep_whitespace=False)
        pos_tags = pos_tag(tokens)
        noun_pos_tags = []
        for e in pos_tags:
            if e[1] in KEY_TAGS:
                noun_pos_tags.append(e[0])
        noun_pos_tags = self.remove_stopwords(noun_pos_tags)
        if self.idf:
            noun_pos_tags = [word for word in noun_pos_tags if word != 'มาตรา' or 'มาตรา' not in word]
        noun_pos_tags = list(set(noun_pos_tags))
        return noun_pos_tags

    def remove_stopwords(self, text):
        res = [
            word.lower()
            for word in text
            if (word not in THAI_STOPWORDS and word not in STOPWORDS)
        ]
        return res

    def find_case_number(self, text):
        pattern = re.compile(r'(?<!\d)(\d{1,5}/\d{4})(?!\d)')
        match = re.findall(pattern, text)
        if pattern.search(text) and all(e in mapper.values() for e in match):
            return [True, match]
        else:
            return [False, '']
    
    def find_law_number(self, text):
        pattern = r'มาตรา ?(\d+(?:/\d+)?)(?: ?(ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ|เอกาทศ|ทวาทศ|เตรส|จตุทศ|ปัณรส|โสฬส|สัตตรส|อัฏฐารส))?'
        match = re.findall(pattern, text)
        result = []
        for m in match:
            result.append(f'{m[0]} {m[1]}'.strip())
        if result:
            return [True, result]
        else:
            return [False, '']

    def find_law_name(self, text):
        result = []
        for law_name in self.law_names:
            if law_name in text and law_name not in result:
                result.append(law_name)
        if result:
            return [True, result]
        else:
            return [False, '']

    def keyword_matcher(self, document, keywords):
        matched_keywords = []
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword))
            if pattern.search(document.page_content):
                matched_keywords.append(keyword)
        return matched_keywords

    def remove_duplicated_documents(self, documents):
        sources = [document.metadata['source'] for document in documents]
        unique_indices = []
        seen = set()
        for index, value in enumerate(sources):
            if value not in seen:
                unique_indices.append(index)
                seen.add(value)
        return list(np.array(documents)[unique_indices])
