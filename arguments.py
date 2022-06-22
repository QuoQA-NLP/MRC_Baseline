from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    PLM: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    save_path: str = field(
        default="checkpoints", metadata={"help": "Path to save checkpoint from fine tune model"},
    )
    model_category: str = field(default="base", metadata={"help": "Model category(base, retro)"})


@dataclass
class DataTrainingArguments:
    max_length: int = field(
        default=512, metadata={"help": "Max length of input sequence"},
    )
    stride: int = field(
        default=128, metadata={"help": "stride width for overflow token mappings"},
    )
    data_path: str = field(
        default="QuoQA-NLP/train-only", metadata={"help": "Huggingface dataset name"}
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    report_to: Optional[str] = field(default="wandb",)
    model_name: Optional[str] = field(
        default="base",
        metadata={
            "help": "model class if class is base, it returns AutoModelForQuestionAnswering class"
        },
    )
    max_answer_length: Optional[int] = field(
        default=30, metadata={"help": "Maximum length of answer after post processing"}
    )
    use_validation: bool = field(
        default=False, metadata={"help": "Use validation dataset"},
    )



@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default="wandb.env", metadata={"help": "input your dotenv path"},
    )
    project_name: Optional[str] = field(
        default="AiChallenge - MRC", metadata={"help": "project name"},
    )
    group_name: Optional[str] = field(
        default="reproduction", metadata={"help": "group name"},
    )


@dataclass
class InferenceArguments:
    file_name: Optional[str] = field(
        default="base.csv", metadata={"help": "The csv file for test dataset"}
    )
    dotenv_path: Optional[str] = field(
        default="wandb.env", metadata={"help": "input your dotenv path"},
    )
    use_ensemable: Optional[bool] = field(
        default=False, metadata={"help": "flag wheter to use ensemable"}
    )
    best_exact_threshold: Optional[float] = field(
        default=0.0, metadata={"help": "exact threshold"}
    )