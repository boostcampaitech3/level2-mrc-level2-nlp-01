import logging
import sys
import pickle
from typing import Dict, List, NoReturn
from tqdm import tqdm

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

    # question encoder
    sub_model = model.question_model

    run_dense_retrieval(data_args, datasets, tokenizer, sub_model)

def run_dense_retrieval(
    data_args: DataTrainingArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]

    last_checkpoint, max_seq_length = None, data_args.max_seq_length

    # 전처리를 진행합니다.
    def prepare_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name],
            truncation=True,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        num_labels = len(examples[question_column_name])
        labels = [i for i in range(num_labels)]
        tokenized_examples['labels'] = labels
        
        return tokenized_examples
    
    train_dataset = datasets["train"]
    valid_dataset = datasets["validation"]
    
    # Train Feature 생성
    train_dataset = train_dataset.map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation Feature 생성
    valid_dataset = valid_dataset.map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
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

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=8,
                                  collate_fn=data_collator)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=8,
                                  collate_fn=data_collator)
    
    scores_list = []
    with open('./passage_tensor.pkl', 'rb') as f:
        p_output = pickle.load(f)
    model.to(device)
    model.eval()
    for batch in tqdm(train_dataloader):
        _, q_output = model(input_ids=batch['input_ids'].to(device),
                            token_type_ids=batch['token_type_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device)).to_tuple()
        q_output = q_output.detach().cpu()
        score = dot_product_scores(q_vectors=q_output, ctx_vectors=p_output)
        scores_list.append(score)
    score_tensor = torch.cat(scores_list, dim=0)
    with open('./training_scores.pkl', 'wb') as f:
        pickle.dump(score_tensor, f)
    print(score_tensor)
    print(score_tensor.shape)

    #### eval dataset & eval example - predictions.json 생성됨
    """if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)"""


if __name__ == "__main__":
    main()
