import os
import json
import datasets
from datasets import Dataset
from typing import Optional, List, Tuple
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, PredictionOutput


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function


    def compute_loss(self, model, inputs, return_outputs=False):
        label_smoothing_factor = self.args.label_smoothing_factor
        if label_smoothing_factor != 0:
            is_impossibles = inputs.pop("is_impossibles")
            start_positions = inputs.pop("start_positions")
            end_positions = inputs.pop("end_positions")

        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if label_smoothing_factor != 0 :
            ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.model_max_length)
            bce_loss = nn.BCEWithLogitsLoss()

            span_loss = (ce_loss(outputs["start_logits"], start_positions) + ce_loss(outputs["end_logits"], end_positions)) / 2
            
            is_impossibles = is_impossibles.float()
            is_impossibles = torch.where(is_impossibles == 1.0, is_impossibles - label_smoothing_factor,
                is_impossibles + label_smoothing_factor
            )

            flag_loss = bce_loss(outputs["is_impossible_logits"].view(-1, ), is_impossibles)
            loss = span_loss + flag_loss * self.model.is_impossible_ratio
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss


    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset
        eval_examples = self.eval_examples
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()),
            )

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
        )

        checkpoint_dir = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, self.args, checkpoint_dir
            )
            metrics = self.compute_metrics(eval_preds)

            ## Wandb log를 위해서 prefix 수정
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return metrics

    def predict(
        self,
        test_dataset: Dataset,
        test_examples: Dataset,
        ignore_keys: Optional[List[str]] = None,
    ) -> PredictionOutput:

        test_dataloader = self.get_test_dataloader(test_dataset)

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"], columns=list(test_dataset.features.keys()),
            )

        output = self.prediction_loop(
            test_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )

        checkpoint_dir = None
        test_predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args, checkpoint_dir
        )

        return test_predictions
