import os
import time
from contextlib import contextmanager

from datasets import Dataset, concatenate_datasets, load_from_disk

import argparse
import utils
from pprint import pprint
from importlib import import_module
import json

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

if __name__ == "__main__":
    MYDICT = {'key': 'value'}

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--retriever_path",
        default="",
        metavar="", type=str, help=""
    )
    parser.add_argument(
        "--config_retriever",
        default="./config/retrieval_config.json",
        metavar="./config/retrieval_config.json", type=str, help=""
    )
    parser.add_argument(
        "--retriever_type",
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
    config = utils.read_json(args.config_retriever)
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

    if args.tokenizer_type == "AutoTokenizer":
        output_path = args.output_path + f'/{args.retriever_type}_{args.model_name_or_path}_{args.top_k}_{args.context_path}'
    else:
        output_path = args.output_path + f'/{args.retriever_type}_{args.tokenizer_type}_{args.top_k}_{args.context_path}'
    output_path = utils.increment_directory(output_path)
    print(f'output_path directory: {output_path}')

    save_config_path = os.path.join(output_path, 'config.json')
    with open(save_config_path, 'wt') as f:
        json.dump(vars(args), f, indent=4)

    retriever_dict = {
        'TfidfVectorizer' : 'TfidfSparseRetrieval',
        'BM25' : 'BM25SparseRetrieval'
    }

    retriever_class = getattr(import_module("retriever"), retriever_dict[args.retriever_type])

    retriever = retriever_class(
        retrieval_path = args.retriever_path,
        vectorizer_parameters=args.vectorizer_parameters,
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

        # with timer("single query by exhaustive search"):
        #     scores, indices = retriever.retrieve(query, topk=args.top_k)
