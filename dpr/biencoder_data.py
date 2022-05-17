from typing import List
from dataclasses import dataclass
from torch import Tensor as T

@dataclass
class BiEncoderPassage:
    text: str
    title: str

@dataclass
class BiEncoderSample:
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]

@dataclass
class BiEncoderBatch:
    question_ids: T
    question_segments: T
    context_ids: T
    ctx_segments: T
    is_positive: List[int]
    hard_negatives: List[List[int]]
    encoder_type: str