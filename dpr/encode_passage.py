from base64 import encode
import logging
import sys
import pickle
from tqdm import tqdm
from typing import Dict, List, NoReturn

import numpy as np
from arguments import DataTrainingArguments, InferenceArguments, ModelArguments
from torch.utils.data import DataLoader, Dataset
from datasets import (
    DatasetDict,
    load_from_disk,
    load_metric,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    set_seed,
)
from DPR import *

logger = logging.getLogger(__name__)

device = torch.device('cuda')
print(device)

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, InferenceArguments)
    )
    model_args, data_args, inference_args = parser.parse_json_file('./model_config/eval_config.json')

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    with open(data_args.dataset_name, 'rb') as f:
        datasets = pickle.load(f)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )

    q_encoder = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    p_encoder = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # DPR model
    model = BiEncoder(q_encoder, p_encoder)
    model.load_state_dict(torch.load(inference_args.checkpoint_path))

    # only use passage encoder
    sub_model = model.ctx_model

    encode_passages(data_args, datasets, tokenizer, sub_model)

def encode_passages(
    data_args: DataTrainingArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names
    print(model.embeddings)

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    train_passage_dataset = datasets["train"]["context"]
    valid_passage_dataset = datasets["validation"]["context"]

    last_checkpoint, max_seq_length = None, data_args.max_seq_length

    tokenized_train_passages = tokenizer(
                train_passage_dataset,
                truncation=True,
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
    tokenized_valid_passages = tokenizer(
                valid_passage_dataset,
                truncation=True,
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                padding="max_length" if data_args.pad_to_max_length else False,
            )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=False
    )

    logger.info("*** Evaluate ***")

    class PassageDataset(Dataset):
        def __init__(self, dataset):
            super(PassageDataset, self).__init__()
            self.dataset = dataset
            self.input_ids = self.dataset['input_ids']
            self.token_type_ids = self.dataset['token_type_ids']
            self.attention_mask = self.dataset['attention_mask']
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, index):
            return {'input_ids' : torch.tensor(self.input_ids[index]),
                    'token_type_ids' : torch.tensor(self.token_type_ids[index]),
                    'attention_mask' : torch.tensor(self.attention_mask[index])}

    train_passage_dataset = PassageDataset(tokenized_train_passages)
    valid_passage_dataset = PassageDataset(tokenized_valid_passages)
    
    train_passage_dataloader = DataLoader(dataset=train_passage_dataset,
                                  batch_size=4,
                                  collate_fn=data_collator)

    valid_passage_dataloader = DataLoader(dataset=valid_passage_dataset,
                                  batch_size=4,
                                  collate_fn=data_collator)
    
    model.to(device)
    model.eval()
    passage_list = []
    for batch in tqdm(train_passage_dataloader):
        _, p_output = model(input_ids=batch['input_ids'].to(device),
                            token_type_ids=batch['token_type_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device)).to_tuple()
        print(p_output)
        p_output = p_output.detach().cpu()
        passage_list.append(p_output)

    passage_tensor = torch.cat(passage_list)
    print(passage_tensor)
    print(passage_tensor.shape)
    with open('./passage_tensor.pkl', 'wb') as f:
        pickle.dump(passage_tensor, f)


if __name__ == "__main__":
    main()