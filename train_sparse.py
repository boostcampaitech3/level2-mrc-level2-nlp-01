from pprint import pprint
import os


def run():
    file_lst = ["./config/bert-base-multilingual-cased/retrieval_config_4.json",
                "./config/bert-base-multilingual-cased/retrieval_config_5.json",
                "./config/bert-base-multilingual-cased/retrieval_config_6.json",
                "./config/bert-base-multilingual-cased/retrieval_config_7.json",
                "./config/bert-base-multilingual-cased/retrieval_config_8.json",
                "./config/klue_roberta-large/retrieval_config_1.json",
                "./config/klue_roberta-large/retrieval_config_2.json",
                "./config/klue_roberta-large/retrieval_config_3.json",
                "./config/klue_roberta-large/retrieval_config_4.json",
                "./config/klue_roberta-large/retrieval_config_5.json",
                "./config/klue_roberta-large/retrieval_config_6.json",
                "./config/klue_roberta-large/retrieval_config_7.json",
                "./config/klue_roberta-large/retrieval_config_8.json",
                "./config/mecab/retrieval_config_1.json",
                "./config/mecab/retrieval_config_2.json",
                "./config/mecab/retrieval_config_3.json",
                "./config/mecab/retrieval_config_4.json",
                "./config/mecab/retrieval_config_5.json",
                "./config/mecab/retrieval_config_6.json",
                "./config/mecab/retrieval_config_7.json",
                "./config/mecab/retrieval_config_8.json",
                ]

    for file in file_lst:
        cmd = f"python retrieval_sparse.py --config_retrieval {file}"
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    run()