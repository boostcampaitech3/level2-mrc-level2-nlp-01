# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from biencoder_data import BiEncoderSample, BiEncoderBatch
from dpr_utils import Tensorizer
from dpr_utils import CheckpointState

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    query vector에 대한 context vector의 score
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """
    Bi-Encoder model 구성요소. question encoder와 passage encoder를 캡슐화하여 묶는다.
    현재는 in-batch negative만 가능함
    """

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        last_hidden_state = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    last_hidden_state, pooled_output = sub_model(
                        input_ids=ids,
                        token_type_ids=segments,
                        attention_mask=attn_mask
                    ).to_tuple()

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                last_hidden_state, pooled_output = sub_model(
                    input_ids=ids,
                    token_type_ids=segments,
                    attention_mask=attn_mask
                ).to_tuple()

        return last_hidden_state, pooled_output

    def forward(
        self,
        questions_input_ids: T,
        questions_token_type_ids: T,
        questions_attention_mask: T, # 여기까지는 question
        passages_input_ids: T,
        passages_token_type_ids: T,
        passages_attention_mask: T, # 여기까지는 context
        labels=None, #이건 안쓰임. compute_metrics를 사용하기 위해 넣음.
        encoder_type: str = None
    ) -> Tuple[T, T]:
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        q_hidden, q_pooled_out = self.get_representation(
            sub_model=q_encoder,
            ids=questions_input_ids,
            segments=questions_token_type_ids,
            attn_mask=questions_attention_mask,
            fix_encoder=self.fix_q_encoder
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        ctx_hidden, ctx_pooled_out = self.get_representation(
            sub_model=ctx_encoder,
            ids=passages_input_ids,
            segments=passages_token_type_ids,
            attn_mask=passages_attention_mask,
            fix_encoder=self.fix_ctx_encoder
        )

        return q_pooled_out, ctx_pooled_out

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        주어진 question 및 context vector의 list에 대한 nll loss를 계산한다.
        현재 hard_negative_idx_per_question은 사용이 불가하다. 그렇지만 loss를 변형하여 사용하면 된다.
        :return: loss값 및 batch당 맞춘 개수의 tuple 반환
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)
        positive_idx_per_question = [i for i in range(q_num)]

        loss = F.nll_loss(
            softmax_scores, # prediction probability
            torch.tensor(positive_idx_per_question).to(softmax_scores.device), # label
            reduction="mean",
        )

        # max_score, max_idxs = torch.max(softmax_scores, 1) # top-1 score 및 index
        # correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, softmax_scores

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores