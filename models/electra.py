import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers import (
    ElectraPreTrainedModel,
    ElectraModel,
    ElectraConfig,
)
from models.output import QuestionAnsweringV2ModelOutput


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/electra/modeling_electra.py
# https://github.com/jinmang2/RetroReader/blob/main/retro_reader/models/modeling_electra.py


class ElectraForV2QuestionAnswering(ElectraPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.impossible_outputs = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(config.hidden_size, 1))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        is_impossibles: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringV2ModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]

        sentence_output = sequence_output[:, 0, :]  # (batch_size, hidden_size)
        sentence_logits = self.impossible_outputs(sentence_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if is_impossibles is not None:
            loss_fct = BCEWithLogitsLoss()
            is_impossible_loss = loss_fct(
                sentence_logits.view(-1,), is_impossibles.float().view(-1,)
            )
            total_loss += is_impossible_loss * 0.1

        if not return_dict:
            output = (start_logits, end_logits,) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringV2ModelOutput(
            loss=total_loss,
            is_impossible_logits=sentence_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

