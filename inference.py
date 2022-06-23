import os
import wandb
import torch
import random
import numpy as np
import pandas as pd
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
from arguments import ModelArguments, DataTrainingArguments, MyTrainingArguments, InferenceArguments
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    HfArgumentParser,
    DataCollatorWithPadding,
)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, InferenceArguments)
    )
    model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    # -- Loading datasets
    loader = Loader("/DATA")
    dset = loader.load_test_data()
    print(dset)

    CPU_COUNT = 6
    MODEL_CATEGORY = model_args.model_category  ## roberta, t5, electra, bert, retro

    # -- Tokenizing & Encoding
    test_dset = copy.deepcopy(dset["test"])

    if inference_args.use_ensemable :
        checkpoint_dir = model_args.PLM
        files = os.listdir(checkpoint_dir)
        checkpoint_list = [os.path.join(checkpoint_dir, f) for f in files if os.path.isdir(os.path.join(checkpoint_dir, f))]
        PLM = checkpoint_list[0]
    else :
        PLM = model_args.PLM

    tokenizer = AutoTokenizer.from_pretrained(PLM)
    encoder = Encoder(tokenizer, stride=data_args.stride, max_length=data_args.max_length)

    test_dset = test_dset.map(
        encoder.prepare_validation_features,
        batched=True,
        num_proc=CPU_COUNT,
        remove_columns=test_dset.column_names,
    )

    # -- Config & Model Class
    MODEL_CATEGORY = model_args.model_category  ## roberta, t5, electra, bert, retro
    MODEL_NAME = training_args.model_name       ## RobertaForV2QuestionAnswering ...

    if MODEL_NAME == "base":
        model_class = AutoModelForQuestionAnswering
    else:
        model_category = importlib.import_module("models." + MODEL_CATEGORY)
        model_class = getattr(model_category, MODEL_NAME)

    # -- Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    # -- Ensemable checkpoints
    USE_ENSEMABLE = inference_args.use_ensemable
    EXACT_THRESHOLD = 0.0 if USE_ENSEMABLE else inference_args.best_exact_threshold
    if  USE_ENSEMABLE :
        checkpoint_num = len(checkpoint_list)

        is_impossible_logits_list, start_logits_list, end_logits_list = [], [], []
        
        for i in tqdm(range(checkpoint_num)) :
            sub_plm = checkpoint_list[i]
            config = AutoConfig.from_pretrained(sub_plm)
            model = model_class.from_pretrained(sub_plm, config=config)

            # -- Trainer
            trainer = QuestionAnsweringTrainer(  # the instantiated 🤗 Transformers model to be trained
                model=model,  # model
                args=training_args,  # training arguments, defined above
                data_collator=data_collator,  # collator
                tokenizer=tokenizer,  # tokenizer
                post_process_function=post_process_function,  # post process function
            )

            logits = trainer.predict_logits(test_dataset=test_dset)
            all_is_impossible_logits, all_start_logits, all_end_logits = logits

            is_impossible_logits_list.append(all_is_impossible_logits)
            start_logits_list.append(all_start_logits)
            end_logits_list.append(all_end_logits)

        is_impossible_logits = np.mean(is_impossible_logits_list, axis=0)
        start_logits = np.mean(start_logits_list, axis=0)
        end_logits = np.mean(end_logits_list, axis=0)

        mean_predictions = (is_impossible_logits, start_logits, end_logits)
        predictions = trainer.predict(test_dataset=test_dset, test_examples=dset["test"], predictions=mean_predictions)
    
    # -- Predictions single model
    else :
        config = AutoConfig.from_pretrained(PLM)
        model = model_class.from_pretrained(PLM, config=config)

        # -- Trainer
        trainer = QuestionAnsweringTrainer(  # the instantiated 🤗 Transformers model to be trained
            model=model,  # model
            args=training_args,  # training arguments, defined above
            data_collator=data_collator,  # collator
            tokenizer=tokenizer,  # tokenizer
            post_process_function=post_process_function,  # post process function
        )

        # --Inference
        predictions = trainer.predict(test_dataset=test_dset, test_examples=dset["test"])

    mapping = {}
    for pred in predictions:
        quid = pred["id"]
        text = pred["prediction_text"]
        flag = pred["no_answer_probability"]

        if flag > EXACT_THRESHOLD :
            mapping[quid] = ""
        else:
            mapping[quid] = text

    # --Submission
    submission_df = pd.read_csv("/DATA/sample_submission.csv")
    question_ids = submission_df["question_id"]
    answer_texts = []

    for quid in question_ids:
        answer_texts.append(mapping[quid])

    submission_df["answer_text"] = answer_texts
    submission_df.to_csv(
        os.path.join('/RESULT', inference_args.file_name), index=False
    )


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
