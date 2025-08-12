import numpy as np
import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words
import os
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import time
import json

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS


EMBEDDING_MODEL = '/project/lt200301-edubot/Capstone-TamTanai/models/multilingual-e5-large'
RERANKING_MODEL = '/project/lt200301-edubot/Capstone-TamTanai/models/bge-reranker-v2-m3' # bge-reranker-v2-m3 gte-multilingual-reranker-base


import os
INGEST_THREADS = os.cpu_count() or 8
def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class == TextLoader:
        loader = TextLoader(file_path, encoding="utf-8")
    elif loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)
def loadDocuments(
    source_dir: str, chunk_size=1000, chunk_overlap=200
) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    # chunksize = round(len(paths) / n_workers)
    chunksize = max(round(len(paths) / n_workers), 1)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents: list[Document]
    documents = text_splitter.split_documents(docs)
    # documents = char_data_splitter(docs, chunk_size, chunk_overlap)
    return documents


endl = "\n"


#load_dotenv()
import re
# root_dir = "."
# sys.path.append(root_dir)  # if import module in this project error
if os.name != "nt":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
# %% [markdown]
###**setup var**

#%%
#chunk_size = 2000
# chunk_overlap = 200
# embedding_algorithm = "faiss"
# source_directory = f"{root_dir}/ir-service/docs"
# persist_directory = f"{root_dir}/ir-service/tmp/embeddings/{embedding_algorithm}"
# print(root_dir)
# print(persist_directory)

# Original mapper dictionary
mapper = {
    "law_doc-84-89.txt": "761/2566",
    "law_doc-44-46.txt": "1301/2566",
    "law_doc-54-57.txt": "1225/2566",
    "law_doc-12-13.txt": "2525/2566",
    "law_doc-40-43.txt": "1305/2566",
    "law_doc-14-15.txt": "2085/2566",
    "law_doc-64-69.txt": "1090/2566",
    "law_doc-1-5.txt": "2610/2566",
    "law_doc-78-81.txt": "882/2566",
    "law_doc-82-83.txt": "835/2566",
    "law_doc-35-39.txt": "1306/2566",
    "law_doc-16-20.txt": "1574/2566",
    "law_doc-32-34.txt": "1373/2566",
    "law_doc-74-77.txt": "934/2566",
    "law_doc-6-11.txt": "2609/2566",
    "law_doc-90-92.txt": "756/2566",
    "law_doc-47-53.txt": "1300/2566",
    "law_doc-58-63.txt": "1101/2566",
    "law_doc-70-73.txt": "1003/2566",
    "law_doc-21-31.txt": "1542/2566",
}

# Reverse the mapping and format it
mapper_reverse = {f"คดี {case}": filename for filename, case in mapper.items()}

endl = "\n"
# print(root_dir)
exclude_pattern = re.compile(r"[^ก-๙]+")  # |[^0-9a-zA-Z]+


import sys
root_dir = os.path.dirname(os.getcwd())
# sys.path.append(os.path.join(root_dir, 'deployment/ir-trt-service'))


contents = None
documents_specific = None
documents_general = None
documents = None


if contents is None:
    with open("/home/shanwibo/Capstone-TamTanai/notebooks/specific_case_knowledge.txt", "r", encoding="utf-8") as f:
        content = f.read()
    contents = content.split("\n\n")

    if documents_specific is None:
        documents_specific = [
            Document(
                page_content=f"{endl.join(c.split(endl)[1:])}",
                metadata={
                    "source": f"docs/{mapper_reverse[c.split(endl)[0]]}",
                    "category": "specific",
                },
            )
            for c in contents
        ]
if documents_general is None:
    # documents_general = loadDocuments(
    #         source_dir=f"/home/shanwibo/Capstone-TamTanai/asset/documentation", chunk_size=10e14, chunk_overlap=0
    #     )
    documents_general = loadDocuments(
            source_dir=f"/home/shanwibo/Capstone-TamTanai/asset/documentation", chunk_size=10e14, chunk_overlap=0
        )
    for i in range(len(documents_general)):
        documents_general[i].metadata["source"] = (
                documents_general[i].metadata["source"].replace(f"{root_dir}/", "")
            )
        documents_general[i].metadata["category"] = "general"
if documents is None:
    documents = documents_general + documents_specific


from langchain_community.embeddings import SentenceTransformerEmbeddings
#reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
def load_embedding_model(embedding_model_name=EMBEDDING_MODEL):

    if torch.cuda.is_available():
        device_type = "cuda"
    # elif torch.backends.mps.is_available():
    #     device_type = "mps"
    else:
        device_type = "cpu"
    #embeddings = SentenceTransformerEmbeddings(model_name="infloat/multilingual-e5-large-instruct", model_kwargs={"trust_remote_code":True})
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device_type}
    )
    return embeddings

def embed_database(
    documents,
    persist_directory,
    embedding_model_name=EMBEDDING_MODEL,
    vector_store="faiss",
):

    embeddings = load_embedding_model(embedding_model_name)

    # Embedding temp exists
    if os.path.isdir(persist_directory):
        if vector_store == "faiss":
            vectordb = FAISS.load_local(
                persist_directory, embeddings,allow_dangerous_deserialization=True
            )
        elif vector_store == "chroma":
            vectordb = Chroma(
                embedding_function=embeddings, persist_directory=persist_directory
            )
            vectordb.persist()
        else:
            raise NotImplementedError(
                f"Embedding Algorithm {vector_store} is not supported/implemented"
            )

    # Create embeddings if not exists
    else:
        if vector_store == "faiss":
            vectordb = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            vectordb.save_local(persist_directory)
        elif vector_store == "chroma":
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            vectordb.persist()
        else:
            raise NotImplementedError(
                f"Embedding Algorithm {vector_store} is not supported/implemented"
            )

    return vectordb

def reranker(question,doc):
    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
    check = []
    for i in range (len(doc)) :
        check.append([question,doc[i].page_content])
    score = reranker.compute_score(check)
    return doc[score.index(max(score))].page_content


persist_directory = "./vectordb"
vectordb = embed_database(documents=documents, persist_directory=persist_directory)


from pythainlp import word_tokenize, pos_tag
from pythainlp.corpus.common import thai_stopwords
import nltk
from nltk.corpus import stopwords


if os.name != "nt":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
# %% [markdown]
###**setup var**

#%%
#chunk_size = 2000
# chunk_overlap = 200
# embedding_algorithm = "faiss"
# source_directory = f"{root_dir}/ir-service/docs"
# persist_directory = f"{root_dir}/ir-service/tmp/embeddings/{embedding_algorithm}"
# print(root_dir)
# print(persist_directory)

# Original mapper dictionary
mapper = {
    "law_doc-84-89.txt": "761/2566",
    "law_doc-44-46.txt": "1301/2566",
    "law_doc-54-57.txt": "1225/2566",
    "law_doc-12-13.txt": "2525/2566",
    "law_doc-40-43.txt": "1305/2566",
    "law_doc-14-15.txt": "2085/2566",
    "law_doc-64-69.txt": "1090/2566",
    "law_doc-1-5.txt": "2610/2566",
    "law_doc-78-81.txt": "882/2566",
    "law_doc-82-83.txt": "835/2566",
    "law_doc-35-39.txt": "1306/2566",
    "law_doc-16-20.txt": "1574/2566",
    "law_doc-32-34.txt": "1373/2566",
    "law_doc-74-77.txt": "934/2566",
    "law_doc-6-11.txt": "2609/2566",
    "law_doc-90-92.txt": "756/2566",
    "law_doc-47-53.txt": "1300/2566",
    "law_doc-58-63.txt": "1101/2566",
    "law_doc-70-73.txt": "1003/2566",
    "law_doc-21-31.txt": "1542/2566",
}

# Reverse the mapping and format it
mapper_reverse = {f"คดี {case}": filename for filename, case in mapper.items()}

endl = "\n"
# print(root_dir)
exclude_pattern = re.compile(r"[^ก-๙]+")  # |[^0-9a-zA-Z]+


def is_exclude(text):
    return bool(exclude_pattern.search(text))


key_tags = ["NCMN", "NCNM", "NPRP", "NONM", "NLBL", "NTTL"]

thaistopwords = list(thai_stopwords())
nltk.download("stopwords")


def remove_stopwords(text):
    res = [
        word.lower()
        for word in text
        if (word not in thaistopwords and word not in stopwords.words())
    ]
    return res


def keyword_search(question, idf):
    tokens = word_tokenize(question, engine="newmm", keep_whitespace=False)
    pos_tags = pos_tag(tokens)
    noun_pos_tags = []
    for e in pos_tags:
        if e[1] in key_tags:
            noun_pos_tags.append(e[0])
    noun_pos_tags = remove_stopwords(noun_pos_tags)
    if idf:
        # noun_pos_tags = [word for word in noun_pos_tags if word != 'มาตรา' or 'มาตรา' not in word]
        noun_pos_tags = [word for word in noun_pos_tags if word not in ['มาตรา', 'บุคคล', 'พระราชบัญญัติ', 'ศาล', 'ใด']]
    noun_pos_tags = list(set(noun_pos_tags))
    return noun_pos_tags


# %%
def find_case_number(text):
    pattern = re.compile(r"(?<!\d)(\d{1,5}/\d{4})(?!\d)")
    match = re.findall(pattern, text)
    if pattern.search(text) and all(e in mapper.values() for e in match):
        return [True, match]
    else:
        return [False, ""]


def keyword_matcher(doc, keywords):
    matched_keywords = []
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword))
        if pattern.search(doc.page_content):
            matched_keywords.append(keyword)
    return matched_keywords


def filter_docs_by_keywords(docs, keywords, question):
    filtered_docs = []
    matches = []
    for doc in docs:
        matched_keywords = []
        if find_case_number(question)[0]:
            case_num = find_case_number(question)[1]
            for num in case_num:
                pattern = re.compile(re.escape(num))
                if pattern.search(doc.page_content):
                    matched_keywords = keyword_matcher(doc, keywords)
                    if len(matched_keywords) >= min(3, len(keywords)):
                        matches.append(matched_keywords)
                        filtered_docs.append(doc)
            continue
        matched_keywords = keyword_matcher(doc, keywords)
        if len(matched_keywords) >= min(2, len(keywords)):
            matches.append(matched_keywords)
            filtered_docs.append(doc)
    return filtered_docs, matches


# %%
def parse_source_docs(source_docs):
    if source_docs is not None:
        results = []
        for res in source_docs:
            if res.metadata["source"].split("/")[-1] in mapper:
                context = f"""คดีหมายเลข {mapper[res.metadata["source"].split("/")[-1]]}\n{res.page_content}"""
                results.append(context)
            else:
                results.append(res.page_content)
        # srcs = [f"""<<<{res.metadata["source"].split("/")[-1]}>>>\n<<<case #{mapper[res.metadata["source"].split("/")[-1]]}>>>\n{res.page_content}""" for res in source_docs]
        result = "\n\n".join(results)
        return result
    else:
        return []


def parse_matched_keywords(matched_keywords):
    if matched_keywords is not None:
        result = "\n".join(str(keyword) + "," for keyword in matched_keywords)
    else:
        result = []
    return result


def retriever(question, documents, vector_database, idf=False):
    global co
    try:
        if question in ["", "-", None]:
            raise Exception("No question")
        ti = time.time()
    
        # keywords search
        keywords = keyword_search(question, idf)
        keywords_filtered_docs, matched_keywords = filter_docs_by_keywords(
            documents, keywords, question
        )
        #if len(keywords_filtered_docs) == 0:
           # return {
             #   "time": 0,
            #    "question": question,
             #   "reranked_docs": "",
           # }
    
        # context search
        retrieved_docs = []
        if not find_case_number(question)[0]:
          retriever= vector_database.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.6})
          retrieved_docs = retriever.get_relevant_documents(question)
        #if len(retrieved_docs) == 0: return []
        #else : return retrieved_docs
            #return {
            #    "time": tf - ti,
            #    "question": question,
            #    "reranked_docs": "",
           # }
    
        # rerank
        relevant_src_docs = keywords_filtered_docs + retrieved_docs
        if len(relevant_src_docs) == 0 : return []
        else : return relevant_src_docs
        #max_relevant_doc = reranker(question,relevant_src_docs)
        #return max_relevant_doc
        if len(relevant_src_docs) == 0:
            return {
                "time": tf - ti,
                "question": question,
                "reranked_docs": "",
            }
        relevant_docs = [doc.page_content for doc in relevant_src_docs]
        if co is None:
            co = cohere.Client(os.getenv("COHERE"))
        rerank_hits = co.rerank(
            query=question,
            documents=relevant_docs,
            model="rerank-multilingual-v2.0",
            top_n=1,
        )
        results = [relevant_src_docs[hit.index] for hit in rerank_hits.results]
        parse_reranked_docs = parse_source_docs(results)
        tf = time.time()
    
        del keywords, keywords_filtered_docs, matched_keywords, retrieved_docs, relevant_src_docs, relevant_docs, rerank_hits, results
    
        # return f"""> Time: {tf-ti}\n\n> Question: {question}\n\n> Answer: {result}\n\n> Source docs:\n{relevant_source_docs}"""
        return {
            "time": tf - ti,
            "question": question,
            "reranked_docs": parse_reranked_docs,
        }
    except Exception as e:
        print(f"{question} @{e}")
        return {
            "error": str(e),
            "source_doc": [],
            "response": "",
            "time": "",
            "source": "",
        }


df = pd.read_csv('../asset/dataset/dataset.csv')


from FlagEmbedding import FlagReranker
reranker = FlagReranker(RERANKING_MODEL, use_fp16=True)


hits = []
reranked_hits = []

start = time.time()
for _, row in df.iterrows():
    hit = False
    retrieved_documents = retriever(row['question'], documents, vectordb, idf=True)
    for retrieved_document in retrieved_documents:
        if '/'.join(row['source'].split('/')[1:]) in retrieved_document.metadata['source']:
            hit = True
            break
    hits.append(hit)

    reranked_hit = False
    scores = reranker.compute_score([[row['question'], retrieved_document.page_content] for retrieved_document in retrieved_documents], normalize=True)
    idx = np.argmax(scores)
    retrieved_document = retrieved_documents[idx]
    if '/'.join(row['source'].split('/')[1:]) in retrieved_document.metadata['source']:
        reranked_hit = True
    reranked_hits.append(reranked_hit)
end = time.time()


result = {
    'recall': sum(hits) / len(hits),
    'accuracy': sum(reranked_hits) / len(reranked_hits),
    'time': end-start
}
with open(f'result_idf2_{EMBEDDING_MODEL.split("/")[-1]}_{RERANKING_MODEL.split("/")[-1]}.json', 'w') as file:
    json.dump(result, file, indent=4)
