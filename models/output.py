import torch
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union


@dataclass
class QuestionAnsweringV2ModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    is_impossible_logits: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
