from elasticsearch import Elasticsearch
from tqdm import tqdm
import utils
import argparse
import json

class ElasticSearchClient:
    def __init__(self, args):
        self._ES_URL = "localhost:9200"
        self.es_client = Elasticsearch(self._ES_URL, timeout = 30, max_retries=10, retry_on_timeout=True)
        self.index = args.index_name
        print(f'Ping Elasticsearch Server : {self.es_client.ping}')

        print(f'Checking elastic Search index')
        self.create_index(self.index, args)
        self.build_wiki(self.index, args.data_path)

    def create_index(self, index, args):
        if self.es_client.indices.exists(index):
            print(f"Delete Existing Index : {index}")
            self.es_client.indices.delete(index=args.index_name)
            print(f"Creating Index : {index}")
            self.es_client.indices.create(index=index, body=args.elasticsearch_config)
            print(f"Finishing Creating  Index : {index}")
        else:
            print(f"Creating Index : {index}")
            self.es_client.indices.create(index=index, body=args.elasticsearch_config)
            print(f"Finishing Creating  Index : {index}")

    def load_wiki(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # 순서대로 중복 제거
        # ('key1', 'key2', 'key3', 'key1', 'key4', 'key2') -> ['key1', 'key2', 'key3', 'key4']
        contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        wiki_contexts = [
            {"document_text": contexts[i]} for i in range(len(contexts))
        ]

        return wiki_contexts

    def build_wiki(self, index, data_path):
        wiki_doc_lst = self.load_wiki(data_path)
        # Inserting wiki data
        for i, rec in enumerate(tqdm(wiki_doc_lst)):
            try:
                self.es_client.index(index=index, id=i, body=rec)
            except:
                print(f"Unable to load document {i}.")

        n_records = self.es_client.count(index=index)["count"]
        print(f"Succesfully loaded {n_records} into {index}")
        return 1

def main(args):
    print("Setting elasticsearch Server and Index")
    if ElasticSearchClient(args):
        print("Finish")
    else:
        print("Error Occurred")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_elasticsearch",
        type=str,
        default="./config/elasticsearch/elasticsearch_config_1.json",
        help="Elastic search configuration file",
    )

    args = parser.parse_args()
    config = utils.read_json(args.config_elasticsearch)
    parser.set_defaults(**config)
    args = parser.parse_args()

    main(args)