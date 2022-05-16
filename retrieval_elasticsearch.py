import os
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
from datasets import Dataset, concatenate_datasets, load_from_disk

import argparse
import utils
from pprint import pprint
import pandas as pd
from importlib import import_module
import json
from collections import OrderedDict
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class ElasticSearchClient:
    def __init__(self, index, output_path, data_path, context_path):
        self._ES_URL = 'https://localhost:9200'
        self.es_client = Elasticsearch(self._ES_URL, timeout = 30, max_retries=10, retry_on_timeout=True)
        self.index = index
        print(f'Ping Elasticsearch Server : {self.es_client.ping}')

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, k1=1.6, b=0.75
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

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i + 1} passage with score {doc_scores[i]:4f}")
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
        query_ = {
            "query": {
                "match": {"document_text": query},
            }
        }
        res = self.es_client.search(index=self.index, body=query_, size=k)
        doc_score = []
        doc_indices = []

        for result in res['hits']['hits']:
            _id = result['_id']
            _score = result['_score']
            doc_score.append(_score)
            doc_indices.append(int(_id))

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
        """
        total_doc_scores = []
        total_doc_indices = []
        for query_ in tqdm(queries):
            doc_score, doc_indices = self.get_relevant_doc(query_, k)
            total_doc_scores.append(doc_score)
            total_doc_indices.append(doc_indices)

        return total_doc_scores, total_doc_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name",
        type=str,
        default="wikipedia_documents_no_shingle",
        help="Elasticsearch  index",
    )
    parser.add_argument("--data_path", default="../data",
                        metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path",
        default="wikipedia_documents.json",
        metavar="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument(
        "--output_path",
        default="./retriever_result",
        metavar="./retriever_result", type=str, help=""
    )
    parser.add_argument(
        "--dataset_name", default = "../data/train_dataset",
        metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--top_k",
        default=20,
        metavar=20,
        type=int,
        help="",
    )

    args = parser.parse_args()
    pprint(vars(args))

    output_path = args.output_path + '/elasticsearch_' + f'{args.index_name}'
    output_path = utils.increment_directory(output_path)
    print(f'output_path directory: {output_path}')

    retriever = ElasticSearchClient(
        index = args.index_name,
        output_path=output_path,
        data_path=args.data_path,
        context_path=args.context_path,
    )

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

    # query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"


    with timer("bulk query by exhaustive search"):
        df, result_dict = retriever.retrieve(full_ds, topk = args.top_k)
        result = f'total queries : {len(df)} + answer_in_documents : {df["answers_in"].sum()} + accuracy : {df["answers_in"].sum()/len(df)} \n'
        result_2 = f'total queries : {len(df)} + exact_contenxt : {df["answer_exact_context"].sum()} + accuracy : {df["answer_exact_context"].sum() / len(df)}'
        result += result_2
        print(f'result : {result}')
        result_txt = os.path.join(output_path, 'result.txt')
        with open(result_txt, "w") as f:
            f.write(result)
        save_result_path = os.path.join(output_path, 'result.json')
        with open(save_result_path, 'wt', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

