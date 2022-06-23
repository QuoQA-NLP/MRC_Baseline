import os
import json
import wandb
import torch
import random
import numpy as np
import importlib
import copy
import multiprocessing
from dotenv import load_dotenv
from datasets import load_dataset
from utils.loader import Loader
from utils.metric import Metric
from utils.encoder import Encoder
from utils.preprocessor import Preprocessor
from utils.postprocessor import post_process_function
from trainer import QuestionAnsweringTrainer
from arguments import ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    HfArgumentParser,
    DataCollatorWithPadding,
    T5Tokenizer,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    # -- Loading datasets

    with open("questino_ids.json", "r") as f :
       question_id_orders = json.load(f)
    
    loader = Loader("/DATA")
    dset = loader.load_train_data(question_id_orders=question_id_orders)
    print(dset)

    CPU_COUNT = 6
    MODEL_CATEGORY = model_args.model_category  

    # -- Preprocessing
    preprocessor = Preprocessor(model_category=MODEL_CATEGORY)

    # -- Tokenizing & Encoding
    train_dset = copy.deepcopy(dset["train"])
    train_dset = train_dset.map(preprocessor.preprocess_train, batched=True, num_proc=CPU_COUNT)

    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)
    encoder = Encoder(tokenizer, stride=data_args.stride, max_length=data_args.max_length)

    train_dset = train_dset.map(
        encoder.prepare_train_features,
        batched=True,
        num_proc=CPU_COUNT,
        remove_columns=train_dset.column_names,
    )

    if training_args.use_validation:

        validation_dset = copy.deepcopy(dset["validation"])
        dset["validation"] = dset["validation"].map(
            preprocessor.preprocess_validation, batched=True, num_proc=CPU_COUNT
        )

        validation_dset = validation_dset.map(
            encoder.prepare_validation_features,
            batched=True,
            num_proc=CPU_COUNT,
            remove_columns=validation_dset.column_names,
        )

    # -- Config & Model Class
    config = AutoConfig.from_pretrained(model_args.PLM)

    MODEL_NAME = training_args.model_name

    model_category = importlib.import_module("models." + MODEL_CATEGORY)
    model_class = getattr(model_category, MODEL_NAME)

    # -- Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    # -- Model
    model = model_class.from_pretrained(model_args.PLM, config=config)

    # # -- Wandb
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    if training_args.max_steps == -1:
        name = f"EP:{training_args.num_train_epochs}_"
    else:
        name = f"MS:{training_args.max_steps}_"

    name += f"LR:{training_args.learning_rate}_BS:{training_args.per_device_train_batch_size}_WR:{training_args.warmup_ratio}_WD:{training_args.weight_decay}_"
    name += MODEL_NAME
    name += model_args.PLM

    wandb.init(
        entity="quoqa-nlp",
        project=logging_args.project_name,
        group=logging_args.group_name,
        name=name,
    )
    wandb.config.update(training_args)

    metric = Metric()
    compute_metric = metric.compute_metrics

    if training_args.use_validation:
        trainer = QuestionAnsweringTrainer(  # the instantiated 🤗 Transformers model to be trained
            model=model,  # model
            args=training_args,  # training arguments, defined above
            train_dataset=train_dset,  # training dataset
            eval_dataset=validation_dset,  # evaluation dataset
            eval_examples=dset["validation"],  # raw validation dataset
            data_collator=data_collator,  # collator
            tokenizer=tokenizer,  # tokenizer
            compute_metrics=compute_metric,  # define metrics function
            post_process_function=post_process_function,  # post process function
        )
    elif not training_args.use_validation:
        trainer = QuestionAnsweringTrainer(  # the instantiated 🤗 Transformers model to be trained
            model=model,  # model
            args=training_args,  # training arguments, defined above
            train_dataset=train_dset,  # training dataset
            data_collator=data_collator,  # collator
            tokenizer=tokenizer,  # tokenizer
            compute_metrics=compute_metric,  # define metrics function
            post_process_function=post_process_function,  # post process function
        )

    # -- Training
    if training_args.do_train:
        train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # -- Evalute
    if training_args.use_validation:
        evaluation_metrics = trainer.evaluate()

        trainer.log_metrics("eval", evaluation_metrics)
        trainer.save_metrics("eval", evaluation_metrics)

    trainer.save_model(model_args.save_path)
    wandb.finish()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()
