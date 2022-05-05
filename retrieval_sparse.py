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
import utils
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


class SparseRetrieval:
    def __init__(
        self,
        retrieval_path,
        retrieval_type,
        retrieval_parameters,
        tokenize_fn,
        output_path,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        num_clusters = 64
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # 순서대로 중복 제거
        # ('key1', 'key2', 'key3', 'key1', 'key4', 'key2') -> ['key1', 'key2', 'key3', 'key4']
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.
        self.num_clusters = num_clusters

        self.vectorizer_type = retrieval_type
        self.tokenize_fn = tokenize_fn

        self.get_sparse_embedding(retrieval_path, retrieval_type, tokenize_fn, retrieval_parameters, output_path)

    def get_sparse_embedding(self, retriever_path, vectorizer_type, tokenize_fn, vectorizer_parameters, output_path) -> NoReturn:

        """
        Summary:
            retriever_path 존재하면 해당 sparse retrieval loading
            retriever_path 존재하지 않으면,
                1) self.vectorizer 호출
                2) Passage Embedding을 만들고
                3) TFIDF와 Embedding을 pickle로 저장합니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        vectorizer_name = f"vectorizer.bin"

        if retriever_path:
            print(f'Initializing sparse retriever on {retriever_path}')
            emb_path = os.path.join(retriever_path, pickle_name)
            vectorizer_path = os.path.join(retriever_path, vectorizer_name)

            if os.path.isfile(emb_path) and os.path.isfile(vectorizer_path):
                with open(emb_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                with open(vectorizer_path, "rb") as file:
                    self.vectorizer = pickle.load(file)
                print(f"Passage embedding & Sparse Vectorizer Loaded from {retriever_path}")
            elif not os.path.isfile(emb_path) and os.path.isfile(vectorizer_path):
                with open(vectorizer_path, "rb") as file:
                    self.vectorizer = pickle.load(file)
        else:
            print(f'Initializing new sparse retriever. Please wait...')
            emb_path = os.path.join(output_path, pickle_name)
            vectorizer_path = os.path.join(output_path, vectorizer_name)

            # Transform by vectorizer
            if hasattr(import_module("sklearn.feature_extraction.text"), vectorizer_type):
                vectorizer_class = getattr(import_module("sklearn.feature_extraction.text"), vectorizer_type)
                self.vectorizer = vectorizer_class(tokenizer=tokenize_fn, **vectorizer_parameters)
                print(f'{self.vectorizer}')

            elif hasattr(import_module("retriever"), vectorizer_type):
                vectorizer_class = getattr(import_module("retriever"), vectorizer_type)
                self.vectorizer = vectorizer_class(self.contexts, tokenize_fn)
                print(f'{self.vectorizer}')
            else:
                raise Exception(f"Use correct tokenizer type : Current tokenizer : {vectorizer_type}")
            print(f'Initializing retriever Complete : {self.vectorizer}')

            print("Build passage embedding. Please wait...")
            if self.vectorizer_type == "TfidfVectorizer":
                self.p_embedding = self.vectorizer.fit_transform(self.contexts)
                print(self.p_embedding.shape)
                with open(emb_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                with open(vectorizer_path, "wb") as file:
                    pickle.dump(self.vectorizer, file)
                print(f"Saving Passage embedding & Sparse Vectorizer to {output_path}")
            elif self.vectorizer_type == "BM25":
                with open(vectorizer_path, "wb") as file:
                    pickle.dump(self.vectorizer, file)
                print(f"Saving Sparse Vectorizer to {output_path}")


    def build_faiss(self) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """
        num_clusters = self.num_clusters
        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
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
        if self.vectorizer_type == 'TfidfVectorizer':
            assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

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
                    "context_lst": context_lst
                }
                tmp2 = {
                    "question": example["question"],
                    "context_lst": context_lst
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                    tmp2["original_context"] = example["context"]
                    tmp2["answers"] = example["answers"]

                    answer_text_lst = example["answers"]["text"]

                    ctx_idx_lst = []
                    for answer in answer_text_lst:
                        for idx_, context in enumerate(context_lst):
                            if answer in context:
                                ctx_idx_lst.append(idx_)

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
        if self.vectorizer_type == "TfidfVectorizer":
            with timer("transform"):
                query_vec = self.vectorizer.transform([query])
            assert (
                np.sum(query_vec) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            with timer("query ex search"):
                result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()

            sorted_result = np.argsort(result.squeeze())[::-1]
            doc_score = result.squeeze()[sorted_result].tolist()[:k]
            doc_indices = sorted_result.tolist()[:k]
            return doc_score, doc_indices

        if self.vectorizer_type == "BM25":
            tokenized_query = self.vectorizer.tokenizer(query)
            result = self.vectorizer.get_scores(tokenized_query)

            sorted_result = np.argsort(result)[::-1]
            doc_score = result[sorted_result].tolist()[:k]
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

        if self.vectorizer_type == 'TfidfVectorizer':
            query_vec = self.vectorizer.transform(queries)
            assert (
                np.sum(query_vec) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return doc_scores, doc_indices

        if self.vectorizer_type == 'BM25':
            query_vec = [self.vectorizer.tokenizer(q) for q in queries]
            doc_scores = []
            doc_indices = []
            for tokenized_query in tqdm(query_vec):
                result_ = self.vectorizer.get_scores(tokenized_query)
                sorted_result_ = np.argsort(result_)[::-1]
                doc_scores.append(result_[sorted_result_].tolist()[:k])
                doc_indices.append(sorted_result_.tolist()[:k])
            return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
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
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.vectorizer.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
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

        query_vecs = self.vectorizer.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


if __name__ == "__main__":
    MYDICT = {'key': 'value'}

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--retriever_path",
        default="",
        metavar="", type=str, help=""
    )
    parser.add_argument(
        "--config_retrieval",
        default="./config/retrieval_config.json",
        metavar="./config/retrieval_config.json", type=str, help=""
    )
    parser.add_argument(
        "--vectorizer_type",
        default="TfidfVectorizer",
        metavar="TfidfVectorizer", type=str, help=""
    )
    parser.add_argument('--vectorizer_parameters',
                        type=json.loads, default=MYDICT)
    parser.add_argument(
        "--tokenizer_type",
        default="AutoTokenizer",
        metavar="AutoTokenizer", type=str, help=""
    )
    parser.add_argument(
        "--dataset_name", default = "./data/train_dataset",
        metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default ="bert-base-multilingual-cased",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument(
        "--top_k",
        default =10,
        metavar=10,
        type=int,
        help="",
    )
    parser.add_argument("--data_path",default = "./data",
                        metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path",
        default = "wikipedia_documents.json",
        metavar="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument(
        "--output_path",
        default="./retriever_result",
        metavar="./retriever_result", type=str, help=""
    )

    parser.add_argument("--use_faiss", default=False, metavar=False, type=bool, help="")
    parser.add_argument("--num_clusters", default=64, metavar=64, type=int, help="")

    args = parser.parse_args()
    config = utils.read_json(args.config_retrieval)
    parser.set_defaults(**config)
    args = parser.parse_args()

    pprint(vars(args))

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    if hasattr(import_module("transformers"), args.tokenizer_type):
        tokenizer_type = getattr(import_module("transformers"), args.tokenizer_type)
        tokenizer = tokenizer_type.from_pretrained(args.model_name_or_path, use_fast=False, )
        print(f'{args.tokenizer_type}')
    elif hasattr(import_module("konlpy.tag"), args.tokenizer_type):
        tokenizer = getattr(import_module("konlpy.tag"), args.tokenizer_type)()
        print(f'{args.tokenizer_type}')
    else:
        raise Exception(f"Use correct tokenizer type - {args.tokenizer_type}")
    print(tokenizer)

    #
    if args.tokenizer_type == "AutoTokenizer":
        output_path = args.output_path + f'/{args.vectorizer_type}_{args.model_name_or_path}_{args.context_path}'
    else:
        output_path = args.output_path + f'/{args.vectorizer_type}_{args.tokenizer_type}_{args.context_path}'
    output_path = utils.increment_directory(output_path)
    print(f'output_path directory: {output_path}')

    save_config_path = os.path.join(output_path, 'config.json')
    with open(save_config_path, 'wt') as f:
        json.dump(vars(args), f, indent=4)


    retriever = SparseRetrieval(
        retrieval_path = args.retriever_path,
        retrieval_type=args.vectorizer_type,
        retrieval_parameters=args.vectorizer_parameters,
        tokenize_fn=tokenizer.tokenize if args.tokenizer_type == "AutoTokenizer" else tokenizer.morphs,
        output_path=output_path,
        data_path=args.data_path,
        context_path=args.context_path,
        num_clusters = args.num_clusters
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query, topk = args.top_k)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df, result_dict = retriever.retrieve_faiss(full_ds, topk = args.top_k)
            # df["correct"] = df["original_context"] == df["context"]
            print("correct retrieval result by faiss", df["answers_in"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df, result_dict = retriever.retrieve(full_ds, topk = args.top_k)
            result = f'total documents : {len(df)} + answer_in_documents : {df["answers_in"].sum()} + accuracy : {df["answers_in"].sum()/len(df)}'
            print(result)
            result_txt = os.path.join(output_path, 'result.txt')
            with open(result_txt, "w") as f:
                f.write(result)
            save_result_path = os.path.join(output_path, 'result.json')
            with open(save_result_path, 'wt', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=4, ensure_ascii=False)

        # with timer("single query by exhaustive search"):
        #     scores, indices = retriever.retrieve(query, topk=args.top_k)
