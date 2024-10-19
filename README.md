# 2022 NIPA AI Competition: Machine Reading Comprehension - Team: QuoQA


## Project Overview

The task is to find the answer to a question from a given text. Both answerable and unanswerable questions are present, and performance is evaluated based on Exact Match criteria. The machine reading comprehension is in the form of Extractive Question Answering, where the goal is to locate the answer span within the context. 

**One of the biggest challenges of the competition was that it wasn't just a contest between startups and other companies, but also that the computational hardware for both training and inference was limited to a 10-core CPU and an Nvidia T4 GPU**. The repository below provides instructions on how to replicate the 2nd place solution from the NIPA AI competition.

## Methodology and Reproduction Commands

- The key idea is to determine both the answerability of the question and extract the answer string simultaneously using a single Transformer backbone model.
- The [total loss](./models/roberta.py) is weighted average of answerability loss and the loss calculated from matching the start and end position of the answer string.
- To train within the limited GPU VRAM resources, we applied techniques such as Gradient Accumulation and Gradient Checkpointing, which helped us to replicate performance in the platform's hardware provided by NIPA.

**Train Script**
`bash running_train_only.sh`

**Inference Script**
`bash running_inference_only.sh`


## Pretrained Language Model

We utilized RoBERTa-Large model proposed in KLUE: Korean Language Understanding Evaluation(2021) (arXiv:2105.09680). We used model weight in huggingface [🔗 klue/roberta-large](https://huggingface.co/klue/roberta-large).

The reasons for selecting the RoBERTa model are as follows:

1. We confirmed that the RoBERTa backbone performed well on benchmarks like the [SQuAD v2.0 Benchmark](https://paperswithcode.com/sota/question-answering-on-squad20), [KLUE Benchmark](https://klue-benchmark.com/tasks/72/leaderboard/task), which focus on unanswerable questions and answer strings.
2. Based on the number of trainable parameters and layers, we anticipated that the RoBERTa-large model would have a comparative advantage in deep learning training compared to base-size models like KPFBert-base.
3. When our team divided the training dataset into five folds and calculated the evaluation score, the klue/roberta-large model demonstrated the best performance.

- Pretrained Corpora (62GB)
    - MODU Corpus
        - Korean Corpus containing formal articles and colloquial text released by the National Institute of Korean Language
    - CC-100-Kor
        - Korean portion of the multilingual web crawled corpora used for training XLM-R
    - NAMUWIKI
        - Korean web-based encyclopedia
    - NEWSCRAWL
        - Collection of 12,800,000 news articles from 2011 to 2020
    - PETITION
        - Blue House National Petition: collection of public petitions
- Tokenizer
    - 32K Vocab Size
    - Morpheme-based subword tokenization
    - Pre-tokenize raw-text into morphemes and then apply BPE
- Model Structure
    - 24 transformer layers
    - 337M trainable parameters
    - Dynamic / WWM Masking



## Dataset

```
../DATA
|    +- sample_submission.csv
|    +- test.json
|    +- train.json
```

    - Convert `train.json` into the Huggingface `datasets.Dataset` class.
    - Fine-tune the `RobertaForV2QuestionAnswering` model using the train dataset converted into the `Dataset` class.
    - Convert `test.json` into the Huggingface `datasets.Dataset` class.
    - Generate the `FINAL_SUBMISSION.csv` file based on the previously fine-tuned `RobertaForV2QuestionAnswering` model.

## Competition Hardware for Training & Inference, provided by NIPA

`CPU 10C, Nvidia T4 GPU x 1, 90MEM, 1TB`

## Directory Structure

```
USER/
├── running_train_only.sh
├── running_inference_only.sh
├── train.py
├── inference.py
├── trainer.py
├── arguments.py
├── question_ids.json
├── README.md
├── requirements.txt
├── .gitignore
│
├── models
│   ├── roberta.py
│   ├── output.py
│   ├── bart.py
│   ├── bert.py
│   └── electra.py
│
├── utils
│   ├── encoder.py
│   ├── loader.py
│   ├── preprocessor.py
│   ├── postprocessor.py
│   └── metric.py
│
├── exps
│   ├── checkpoint-125/
│   ├── checkpoint-250/
│   ├── checkpoint-375/
│   ├── checkpoint-500/
│   ├── checkpoint-625/
│   ├── checkpoint-750/
│   └── checkpoint-875/
│
├── mecab-0.996-ko-0.9.2/
│
├── mecab-ko-dic-2.1.1-20180720/
│
└── RESULT
    ├── final_submission.csv
    └── checkpoint-875
        ├── pytorch_model.bin
        ├── config.json
        ├── optimizer.pt
        ├── rng_state.pth
        ├── scheduler.pt
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        ├── trainer_state.json
        ├── training_args.bin
        └── vocab.txt
```


- `running_train_only.sh`

  - This is a shell script file used for training the model.
  - Please refer to the arguments below for the required training parameters.

- `running_inference_only.sh`

  - This is a shell script file used for inference with the model's weight files.
  - Please refer to the arguments below for the necessary inference parameters.

- `train.py`

  - This code runs the model training process.
  - The saved model checkpoint weight files are located in the `exps/` folder.
  - The final model weight files used for inference are stored in the `RESULT/` folder.

- `inference.py`

  - This code performs predictions using the trained model weights and saves the predicted results in a CSV file.
  - The final submission file is stored in the `RESULT/` folder.

- `trainer.py`

  - This file implements a custom trainer by inheriting Huggingface's `Trainer` class.
  - The functions `compute_loss`, `evaluate`, and `predict` have been customized.

- `arguments.py`

  - This file defines the classes for the arguments needed for training and inference.
  - It specifies the types of arguments, default values, and help messages.

- `question_ids.json`

  - To ensure reproducibility, this file stores the ID list of the training data used during training and arranges the data from `/DATA/train.json` accordingly.

- `models/`

  - This directory contains the files implementing model classes.
  - The final model used is the `RobertaForV2QuestionAnswering` class from `roberta.py`.
  - In addition, `output.py` implements the output class for the model.

- `utils/`

  - This directory contains files for dataset preprocessing, input/output processing for the model, and evaluation metrics.
  
  - `encoder.py`
    - Defines the `Encoder` class, which tokenizes data and calculates the `is_impossible` flag and answer indices.
  
  - `loader.py`
    - Contains a class that loads raw JSON data from the `/DATA` directory (for train and test) and converts it into a format compatible with Huggingface's `Datasets` class.
  
  - `preprocessor.py`
    - Defines the `Preprocessor` class to handle cases where there is no answer or when there are multiple answers.
  
  - `postprocessor.py`
    - Defines functions to generate the final predictions based on model outputs and formats them for output.
    - Uses the Konlpy's morphological analyzer `mecab` for morphological analysis and removes unnecessary particles and special characters (mecab version: 0.996/ko-0.9.2).
  
  - `metric.py`
    - Defines the `Metric` class for evaluating the model.

- `exps/`

  - This directory stores the model checkpoints generated during training when running `train.py`.

- `RESULT/`

  - This directory stores the final model checkpoint weight files after training with `train.py`.

  - It also stores the results of the model's predictions on the test data generated by `inference.py`.

  - `final_submission.csv`
    - The submission file containing the final predictions is saved here.
    
  - `checkpoint-875/`
    - Directory where the final model weights are saved.
    - `pytorch_model.bin`
        - File containing the model's weights.
    - `config.json`
        - File containing general information and paths related to the model.
    - `optimizer.pt`
        - File containing the optimizer weights.
    - `rng_state.pth`
        - File containing Python, NumPy, and CPU state information.
    - `scheduler.pt`
        - File containing the scheduler weights.
    - `special_tokens_map.json`
        - File containing the special tokens used by the tokenizer.
    - `tokenizer_config.json`
        - File containing information about the special tokens, class, and model name used by the tokenizer.
    - `tokenizer.json`
        - File containing each vocabulary ID used by the tokenizer.
    - `trainer_state.json`
        - File containing logs on learning rate, loss, evaluation information, etc., for each log step.
    - `training_args.bin`
        - File containing the training arguments.
    - `vocab.txt`
        - File containing the characters handled by the tokenizer.


## Arguments

### Explanation of Arguments for `running_train_only.sh`

|        Argument         | Description                                                 |
| :---------------------: | :---------------------------------------------------------- |
|        do_train         | Determines whether to train the model.                      |
|       group_name        | Specifies the wandb group name.                             |
|        data_path        | Selects the Nipa dataset.                                   |
|     use_validation      | Determines whether to perform validation.                   |
|           PLM           | Specifies the model PLM to use.                             |
|     model_category      | Selects the file to use from the `models` folder.           |
|       model_name        | Specifies the detailed class from the selected file in `model_category`. |
|       max_length        | Specifies the maximum sequence length.                      |
|      save_strategy      | Specifies the save strategy, such as by step or epoch.       |
|    save_total_limit     | Sets the maximum number of checkpoints to save.             |
|      learning_rate      | Specifies the learning rate for training.                   |
| per_device_train_batch_size | Sets the train batch size.                               |
| per_device_eval_batch_size  | Sets the evaluation batch size.                          |
| gradient_accumulation_steps | Specifies the number of gradient accumulation steps.     |
|   gradient_checkpointing  | Determines whether to enable gradient checkpointing.       |
|        max_steps        | Specifies the maximum number of training steps.             |


### Explanation of Arguments for `running_inference_only.sh`

|      Argument     | Description                                                 |
| :---------------: | :---------------------------------------------------------- |
|    do_predict     | Determines whether to run predictions on the given data.     |
|       PLM         | Loads the desired pre-trained model weights.                 |
|  model_category   | Selects the file to use from the `models` folder.            |
|    model_name     | Specifies the detailed class from the selected file in `model_category`. |
|    max_length     | Specifies the maximum sequence length.                      |
|    output_dir     | Sets the path where prediction outputs will be saved.       |
|    file_name      | Specifies the file name for the prediction outputs.         |