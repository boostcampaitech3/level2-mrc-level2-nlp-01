# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dense Passage Retriever의 training을 위한 'Trainer'의 subclass 코드 입니다.
"""

from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import Union, Optional, List, Dict
from transformers.file_utils import PaddingStrategy
import datasets
import torch
from DPR import BiEncoderNllLoss

# Bi-Encoder의 input을 만들기 위한 data
@dataclass
class DataCollatorWithPaddingForDPR:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 여기서, question 및 passage를 나누는 역할을 한다.
        question_features = []
        passages_features = []
        batch = {}
        labels = []
        for label, feature in enumerate(features):
            passages_feature = {}
            question_feature = {}
            passages_feature['input_ids'] = feature.pop('passages_input_ids')
            passages_feature['token_type_ids'] = feature.pop('passages_token_type_ids')
            passages_feature['attention_mask'] = feature.pop('passages_attention_mask')

            question_feature['input_ids'] = feature.pop('questions_input_ids')
            question_feature['token_type_ids'] = feature.pop('questions_token_type_ids')
            question_feature['attention_mask'] = feature.pop('questions_attention_mask')

            question_features.append(question_feature)
            passages_features.append(passages_feature)
            labels.append(label)
            
        batch['labels'] = torch.tensor(labels)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        question_batch = self.tokenizer.pad(
            question_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        passage_batch = self.tokenizer.pad(
            passages_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        for key in ['input_ids', 'token_type_ids', 'attention_mask']:
            batch['questions_' + key] = question_batch[key]
            batch['passages_' + key] = passage_batch[key]

        return batch

# Huggingface의 Trainer를 상속받아 DPR을 위한 Trainer를 생성합니다.
class DensePassageRetrievalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        q_outputs, p_outputs = model(**inputs)
        loss_fct = BiEncoderNllLoss()
        loss, outputs = loss_fct.calc(q_outputs, p_outputs)

        # trainer에서 eval 계산 후 첫 항을 제외하는 코드가 있어서 이렇게 구현하였음.
        # trainer.py의 2033줄 logits = outputs[1:] 참조
        if return_outputs:
            outputs_shape = outputs.shape[0]
            outputs = torch.cat([torch.zeros([1, outputs_shape]).to(outputs.device), outputs], dim=0)

        return (loss, outputs) if return_outputs else loss