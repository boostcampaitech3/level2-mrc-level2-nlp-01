import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

import argparse
from pprint import pprint
import konlpy.tag
from transformers import AutoTokenizer
from importlib import import_module
import json
from collections import OrderedDict

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25SparseRetrieval:
    def __init__(
        self,
        retrieval_path,
        vectorizer_parameters,
        tokenize_fn, 
        output_path = None,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        num_clusters = 64
    ) -> NoReturn:

        """
        Arguments:
            retrieval_path: 
                이전 BM25SparseRetrieval 결과값이 저장되어 있는 경로입니다.
                지정을 해주면 이전 결과값을 불러옵니다.
                "" 이면 새롭게 BM25SparseRetrieval 생성합니다.

            vectorizer_parameters:
                BM25SparseRetrieval 의 parameter 값들입니다.
            
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            output_path:
                결과를 저장하는 경입니다.
            
            data_path:
                데이터가 보관되어 있는 경로입니다.
            
            context_path:
                Passage들이 묶여있는 파일명입니다.
            
            data_path/context_path가 존재해야합니다.
        Summary:
            Passage 파일을 불러오고 BM25SparseRetrieval 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.
        self.num_clusters = num_clusters
        self.tokenize_fn = tokenize_fn
        self.idf = None
        self.k1 = None
        self.b = None

        self.get_sparse_embedding(retrieval_path, tokenize_fn, vectorizer_parameters, output_path)

    def get_sparse_embedding(self, retrieval_path, tokenize_fn, vectorizer_parameters, output_path) -> NoReturn:

        """
        Summary:
            retriever_path 존재하면 해당 sparse retrieval loading
            retriever_path 존재하지 않으면,
                1) self.vectorizer 호출
                2) Passage Embedding을 만들고
                3) TFIDF와 Embedding을 pickle로 저장합니다.
        """


        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding_bm25.bin"
        encoder_name = f"encoder_bm25.bin"
        idf_encoder_name = f"idf_encoder_bm25.bin"
        idf_path_name = f"idf_path_bm25.bin"

        if retrieval_path:
            print(f'Initializing sparse retriever on {retrieval_path}')
            emd_path = os.path.join(retrieval_path, pickle_name)
            encoder_path = os.path.join(retrieval_path, encoder_name)
            idf_encoder_path = os.path.join(retrieval_path, idf_encoder_name)
            idf_path = os.path.join(retrieval_path, idf_path_name)

            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(encoder_path, "rb") as file:
                self.encoder = pickle.load(file)
            with open(idf_encoder_path, "rb") as file:
                self.idf_encoder = pickle.load(file)
            with open(idf_path, "rb") as file:
                self.idf = pickle.load(file)

        else:
            print(f'Initializing new sparse retriever. Please wait...')
            emd_path = os.path.join(output_path, pickle_name)
            encoder_path = os.path.join(output_path, encoder_name)
            idf_encoder_path = os.path.join(output_path, idf_encoder_name)
            idf_path = os.path.join(output_path, idf_path_name)

            self.encoder = TfidfVectorizer(
                tokenizer=tokenize_fn,
                use_idf = False,
                norm = None,
                **vectorizer_parameters
            )
            self.idf_encoder = TfidfVectorizer(
                tokenizer=tokenize_fn,
                norm=None,
                smooth_idf=False,
                **vectorizer_parameters
            )

            print("Build sparse embedding. Please wait...")

            self.p_embedding = self.encoder.fit_transform(self.contexts)
            self.idf_encoder.fit(self.contexts)
            self.idf = self.idf_encoder.idf_

            print(self.p_embedding.shape)

            with open(idf_path, "wb") as f:
                pickle.dump(self.idf, f)
            with open(encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(idf_encoder_path, "wb") as file:
                pickle.dump(self.idf_encoder, file)
            print(f"Saving Passage embedding & Sparse Vectorizer on {output_path}")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, k1 = 1.6, b = 0.75
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]
        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        self.k1 = k1
        self.b = b

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            total_dict = OrderedDict()
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                context_lst = [self.contexts[pid] for pid in doc_indices[idx]]
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "scores": doc_scores[idx],
                    "context_lst": context_lst
                }
                tmp2 = {
                    "question": example["question"],
                    "context_lst": context_lst,
                    "scores": doc_scores[idx],
                }

                
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                    tmp2["original_context"] = example["context"]
                    tmp2["answers"] = example["answers"]

                    answer_text_lst = example["answers"]["text"]

                    ctx_idx_lst = []
                    check = 0
                    for answer in answer_text_lst:
                        for idx_, context in enumerate(context_lst):
                            if tmp2["original_context"] == context:
                                check = 1
                            if answer in context:
                                ctx_idx_lst.append(idx_)
                    tmp2["answer_exact_context"] = check
                    tmp["answer_exact_context"] = check
                    tmp2["answer_context"] = ctx_idx_lst

                    if len(ctx_idx_lst) > 0:
                        tmp["answers_in"] = 1
                        tmp2["answer_in"] = 1
                    else:
                        tmp["answers_in"] = 0
                        tmp2["answer_in"] = 0

                total.append(tmp)
                total_dict[idx] = tmp2

            cqas = pd.DataFrame(total)
            return cqas, total_dict
            

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        with timer("transform"):
            query_vec = self.encoder.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            p_embedding = self.p_embedding.tocsc()
            k1 = self.k1
            b = self.b
            len_p = np.zeros(len(self.contexts))
            for idx, context in enumerate(self.contexts):
                len_p[idx] = len(context)

            avdl = np.mean(len_p)
            p_emb_for_q = p_embedding[:, query_vec.indices]
            denom = p_emb_for_q + (k1 * (1 - b + b * len_p / avdl))[:, None]

            #idf = self.idf[None, query_vec.indices] - 1.0
            idf = self.idf[None, query_vec.indices] - 1.0
            numer = p_emb_for_q.multiply(np.broadcast_to(idf, p_emb_for_q.shape)) * (k1 + 1)
            result = (numer / denom).sum(1).A1

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]

        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        query_vecs = self.encoder.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        doc_scores = []
        doc_indices = []

        p_embedding = self.p_embedding.tocsc()

        self.results = []

        for query_vec in tqdm(query_vecs):

            k1 = self.k1
            b = self.b
            len_p = np.zeros(len(self.contexts))

            for idx, context in enumerate(self.contexts):
                len_p[idx] = len(context)

            avdl = np.mean(len_p)
            p_emb_for_q = p_embedding[:, query_vec.indices]
            denom = p_emb_for_q + (k1 * (1 - b + b * len_p / avdl))[:, None]

            idf = self.idf[None, query_vec.indices] - 1.0
            numer = p_emb_for_q.multiply(np.broadcast_to(idf, p_emb_for_q.shape)) * (k1 + 1)
            result = (numer / denom).sum(1).A1
            #result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            sorted_result_idx = np.argsort(result)[::-1]
            doc_score, doc_indice = result[sorted_result_idx].tolist()[:k], sorted_result_idx.tolist()[:k]
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)

        return doc_scores, doc_indices


