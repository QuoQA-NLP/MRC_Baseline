# 2022년 인공지능 온라인 경진대회 / 문서 검색 효율화를 위한 기계독해 문제 - 팀: QuoQA


## 프로젝트 개요


텍스트와 질문이 주어졌을 때 본문에서 질문의 답을 찾는 과제입니다. 답변이 불가능한 경우와 답변이 가능한 경우가 모두 존재하며, Exact Match를 기준으로 성능을 평가합니다. 기계독해는 Extractive Question Answering 형식으로써 Context 안에서 Answer Span을 찾는 것이 목표입니다.


## 사용방법론 및 재현 명령어

- 해당 문단으로 답변 가능성을 판단하는 것과 답변 문자열을 추출하는 과정을 Transformer backbone model 단일 모델로써 동시에 수행하는 것이 핵심입니다.
- 답변 가능성을 기준으로 산출한 loss와 문자열 시작점, 끝점이 일치하는지를 기준으로 산출한 loss를 가중평균합하여 [total loss](./models/roberta.py)를 산정합니다.
- 한정된 GPU VRAM 자원에서 훈련시키기 위하여 Gradient Accumulation, Gradient Checkpoint를 사용했으며 이를 통해 성능 향상을 이뤄냈습니다.


**훈련 명령어**
`bash running_train_only.sh`

**추론 명령어**
`bash running_inference_only.sh`


## 기학습가중치(Pretrained Language Model)

KLUE: Korean Language Understanding Evaluation(2021)에서 공개한 roberta-large 모델을 사용했습니다. (arXiv:2105.09680)

RoBERTa 모델을 선정한 이유는 다음과 같습니다.
1. 답변 불가 항목과 응답 문자열을 벤치마크로 삼은 [SQuAD v2.0 Benchmark](https://paperswithcode.com/sota/question-answering-on-squad20), [KLUE Benchmark](https://klue-benchmark.com/tasks/72/leaderboard/task)에서 Roberta Backbone이 성능이 좋다는 것을 확인했습니다.
2. #Trainable Params와 Num Layers를 따졌을 때 RoBERTa-large 모델이 KPFBert-base 등과 같은 base size 모델에 비해서 딥러닝 학습에 비교우위가 있을 것으로 예상했습니다.
3. 팀 자체적으로 Train Dataset을 5 Fold로 나눠서 Evaluation Score을 산출했을 때 klue/roberta-large가 성능이 제일 우수하게 나왔습니다.

구체적으로 Huggingface에 업로드된 모델 가중치를 사용했습니다: [🔗 klue/roberta-large](https://huggingface.co/klue/roberta-large)

해당 pre-trained weight는 2021년 06월 15일에 공개되었습니다. 해당 PLM은 다음과 같은 데이터셋, 토크나이저, 모델 구조를 바탕으로 훈련이 되었습니다.

- Pretrained Corpora (총 62GB)
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



## 데이터셋

```
../DATA
|    +- sample_submission.csv
|    +- test.json
|    +- train.json
```

    - 'train.json'를 Huggingface의 datasets.Dataset 클래스로 변환한다.
    - Dataset 클래스로 변환된 train dataset을 바탕으로 RobertaForV2QuestionAnswering을 파인튜닝을 진행한다.
    - 'test.json'를 Huggingface의 datasets.Dataset 클래스로 변환한다.
    - 앞서 Finetuning한 RobertaForV2QuestionAnswering 모델을 바탕으로 'FINAL_SUBMISSION.csv' 파일을 생성한다.
    
## 하드웨어

`CPU 10C, Nvidia T4 GPU x 1, 90MEM, 1TB`

## 디렉토리 구조

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
│   ├── checkpoint-125/ *하단 상세 기술*
│   ├── checkpoint-250/
│   ├── checkpoint-375/
│   ├── checkpoint-500/
│   ├── checkpoint-625/
│   ├── checkpoint-750/
│   └── checkpoint-875/
│
└── RESULT * Output 상세설명 *
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

  - 모델 학습하기 위한 shell script 파일입니다.
  - 훈련에 필요한 argument는 아래를 참조하시기 바랍니다.

- `running_inference_only.sh`

  - 모델 가중치 파일로 추론하기 위한 shell script 파일입니다.
  - 추론에 필요한 argument는 아래를 참조하시기 바랍니다.

- `train.py`

  - 모델 학습을 실행하는 코드입니다.
  - 저장된 model checkpoint 가중치 파일은 `exps/` 폴더에 있습니다.
  - 최종 추론에 쓰이는 모델 가중치 파일은 `RESULT/` 폴더에 있습니다.

- `inference.py`

  - 학습된 model 가중치를 통해 prediction하고, 예측한 결과를 csv 파일로 저장하는 코드입니다.
  - 저장된 최종 submission 파일은 `RESULT/` 폴더에 있습니다.

- `trainer.py`

  - Huggingface의 Trainer class를 상속받아 trainer를 구현한 파일입니다.
  - compute_loss, evaluate, predict 함수를 custom하게 변경했습니다. 

- `arguments.py`

  - 학습 및 추론에 필요한 arguments 관련 class를 정의한 파일입니다.
  - arguments의 종류, 기본값, help message 등을 정의했습니다.

- `question_ids.json`

  - 재현을 위해서 학습할 때의 train data의 id 리스트를 저장하고 이를 이용해서 /DATA/train.json에 있는 데이터를 정렬하였습니다.

- `models/`

  - 모델 class를 구현한 파일들이 있는 디렉토리입니다.
	- 최종 모델은 `roberta.py`에 있는 RobertaForV2QuestionAnswering class만 사용했습니다.
	- 이 외에 `output.py`에서 모델 출력물 class를 구현했습니다.

- `utils/`
	  - 데이터셋 전처리, 모델 입력 데이터 전/후처리, 평가지표 파일들이 있는 디렉토리입니다.
	- `encoder.py`
		- 데이터를 tokenize하고 is_impossible, 정답 index 등을 구하는 Encoder class를 정의한 파일입니다.
	- `loader.py`
	    - train, test 데이터가 있는 /DATA 디렉토리에서 json 파일인 원시 데이터를 불러오고 Huggingface의 Datasets 클래스에 맞게 형식을 변형하는 클래스가 있는 파일입니다.
	- `preprocessor.py`
		- 정답이 없는 경우, 정답이 2개 이상인 경우를 처리하는 Preprocessor class를 정의한 파일입니다.
	- `postprocessor.py`
		- 모델 출력을 기반으로 최종 prediction을 구하고 포맷에 맞춰 출력하는 함수를 정의한 파일입니다.
		- Konlpy의 형태소 분석기 mecab을 활용하여 형태소 분석 후, 끝에 조사 및 앞뒤에 특수 문자 제거 (mecab version: mecab of 0.996/ko-0.9.2)
	- `metric.py`
		- 모델을 평가하기 위한 평가지표 Metric class를 정의한 파일입니다.


- `exps/`

    - train.py를 실행할 시, 훈련될 때마다 생성되는 모델 checkpoint를 저장하는 디렉토리입니다.

- `RESULT/`
    
    - train.py를 통해 학습된 최종 모델 checkpoint 가중치 파일을 저장하는 디렉토리입니다.

    - inference.py를 통해 Test data에 대해서 모델이 예측한 결과를 저장하는 디렉토리입니다.

    - `final_submission.csv`
        - 최종 예측값이 저장된 submission 파일입니다.
        
    - `checkpoint-875/`
        - 최종 모델 가중치가 저장된 디렉토리입니다.
        - `pytorch_model.bin`
            - 모델 가중치가 저장된 파일입니다.
        - `config.json`
            - 모델에 대한 전반적인 특징 및 경로가 적혀있는 파일입니다.
        - `optimizer.pt`
            - optimizer weight를 담은 파일입니다.
        - `rng_state.pth`
            - python, numpy, cpu 정보를 담은 파일입니다.
        - `scheduler.pt`
            - scheduler weight를 담은 파일입니다.
        - `special_tokens_map.json`
            - tokenizer에서 사용하는 special token을 담은 파일입니다.
        - `tokenizer_config.json`
            - tokenizer의 special token, class 및 모델 이름 정보 담은 파일입니다.
        - `tokenizer.json`
            - tokenizer의 각 vocab id 정보를 담은 파일입니다.
        - `trainer_state.json`
            - 각 log step 당, learning rate나 loss, eval 정보 등을 담은 파일입니다.
        - `training_args.bin`
            - train argument를 담은 파일입니다.
        - `vocab.txt`
            - tokenizer에 다루는 문자들을 담은 파일입니다.


## Arguments

### running_train_only.sh Argument 설명

|      argument       | description                                                                                   |
| :-----------------: | :-------------------------------------------------------------------------------------------- | 
| do_train|모델을 훈련할지 여부 결정합니다.|
| group_name|wandb 그룹 이름 지정합니다.|
|data_path|Nipa dataset 선택합니다.|
|use_validation|validation을 수행할지 여부 결정|
|PLM|모델 PLM 결정합니다.|
|model_category|models 폴더 안에 사용할 파일 선택합니다.|
|model_name|model_category에서 선택한 파일에서 세부 class 선택합니다.|
|max_length|최대 길이 지정합니다.|
|save_strategy|step or epoch 기준 등으로 저장하는 방식을 정합니다.|
|save_total_limit|최대 checkpoint 저장 갯수를 지정합니다.|
|learning_rate|훈련 learning rate를 지정합니다.|
|per_device_train_batch_size|train batch size를 지정합니다.|
|per_device_eval_batch_size|eval batch size를 지정합니다.|
|gradient_accumulation_steps|gradient accumulation 수를 정합니다.|
|gradient_checkpointing|gradient checkpoint 여부를 정합니다.|
|max_steps|학습 최대 step을 지정합니다.|


### running_inference_only.sh Argument 설명

|      argument       | description                                                                                   |
| :-----------------: | :-------------------------------------------------------------------------------------------- | 
| do_predict|주어진 데이터에 대해 예측할지 말지를 결정합니다.|
|PLM|원하는 가중치 모델을 가져옵니다.|
|model_category|models 폴더 안에 사용할 파일 선택합니다.|
|model_name|model_category에서 선택한 파일에서 세부 class 선택합니다.|
|max_length|최대 길이 지정합니다.|
|output_dir|예측값을 저장할 경로를 설정합니다.|
|file_name|예측값에 대한 파일 이름을 지정합니다.|