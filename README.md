# 2022년 인공지능 온라인 경진대회 / 문서 검색 효율화를 위한 기계독해 문제

## Hardware

- `GPU : Tesla V100 32GB`

## Project Description

```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```

## Dataset

- dataset 설명

```
data
|    +- train_pororo_sub.csv
|    +- test_pororo_sub.csv
|    +- train.csv
|    +- test.csv
```

    - 'train_pororo_sub.csv'를 활용하여 `RBERT`, `KLUE/RoBERTa-large` 학습을 진행한다.
    - 'test_pororo_sub.csv'를 활용하여 `RBERT`, `KLUE/RoBERTa-large` 모델을 바탕으로 'submission.csv' 파일을 생성한다.
    - 'train.csv'를 활용하여 `RE Improved Baseline` 학습을 진행한다.
    - 'test.csv'를 활용하여 `RE Improved Baseline` 모델을 바탕으로 'submission.csv' 파일을 생성한다.

- Dataset 통계
  - train dataset : 총 32470개
  - test dataset : 7765개 (label은 전부 100으로 처리되어 있습니다.)
- Data 예시 (`train.csv`)
  - `id`, `sentence`, `subject_entity`, `object_entity`, `label`, `source`로 구성된 csv 파일
  - `sentence example` : <Something>는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다. (문장)
  - `subject_entity example` : {'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'} (단어, 시작 idx, 끝 idx, 타입)
  - `object_entity example` : {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} (단어, 시작 idx, 끝 idx, 타입)
  - `label example` : no_relation (관계),
  - `source example` : wikipedia (출처)
- Relation class에 대한 정보는 다음과 같습니다.
  ![1](https://user-images.githubusercontent.com/53552847/136692171-30942eec-fb83-4175-aa8d-13559ae2caf1.PNG)

## code

#### 디렉토리 구조

```
│  running_train_only.sh
│  train.py
│  inference.py
│  trainer.py
│  arguments.py
│  upload_to_hub.py
│  README.md
│  requirements.txt
│  .gitignore
│
├─models
│      roberta.py
│      output.py
│      bart.py
│      bert.py
│      electra.py
│
├─data
│      sample_submission.csv
│
├─notebooks
│      RE_improved_baseline.ipynb
│      roberta_with_lstm.ipynb
│      tmp_sub.ipynb
│      train_with_pororo.ipynb
│
└─utils
        encoder.py
        metric.py
        postprocessor.py
        preprocessor.py
```

- `running_train_only.sh`

  - 모델 학습하기 위한 shell command 파일입니다.
  - 훈련에 필요한 argument는 파일 안에 저장돼있습니다.

- `train.py`

  - 모델 학습 코드 파일입니다.
  - 저장된 model관련 파일은 `results` 폴더에 있습니다.

- `trainer.py`

  - Huggingface의 Trainer class를 상속받아 구현함.
  - 저장된 최종 submission 파일은 `results` 폴더에 있습니다.

- `inference.py`

  - 학습된 model 가중치를 통해 prediction하며, 예측한 결과를 csv 파일로 저장해줍니다.
  - 저장된 최종 submission 파일은 `results` 폴더에 있습니다.

- `logs`

  - 텐서보드 로그가 담기는 폴더 입니다.

- `prediction`

  - `inference.py` 를 통해 model이 예측한 정답 `submission.csv` 파일이 저장되는 폴더 입니다.

- `results`

  - `train.py`를 통해 설정된 step 마다 model이 저장되는 폴더 입니다.

- `best_model `

  - 학습중 evaluation이 best인 model이 저장 됩니다.

- `dict_label_to_num.pkl`

  - 문자로 되어 있는 label을 숫자로 변환 시킬 dictionary 정보가 저장되어 있습니다.

- `dict_num_to_label.pkl`
  - 숫자로 되어 있는 label을 원본 문자로 변환 시킬 dictionary 정보가 저장되어 있습니다.

## Implementation

In Terminal

- Install Requirements

```python
pip install -r requirements.txt
```

- training

```
python train.py
```

- inference

```
python inference.py
```

## Arguments Usage

- RBERT

| Argument               | type  | Default                         | Explanation                                  |
| ---------------------- | ----- | ------------------------------- | -------------------------------------------- |
| batch_size             | int   | 40                              | 학습&예측에 사용될 batch size                |
| num_folds              | int   | 5                               | Stratified KFold의 fold 개수                 |
| num_train_epochs       | int   | 5                               | 학습 epoch                                   |
| loss                   | str   | focalloss                       | loss function                                |
| gamma                  | float | 1.0                             | focalloss 사용시 gamma 값                    |
| optimizer              | str   | adamp                           | 학습 optimizer                               |
| scheduler              | str   | get_cosine_schedule_with_warmup | learning rate를 조절하는 scheduler           |
| learning_rate          | float | 0.00005                         | 초기 learning rate 값                        |
| weight_decay           | float | 0.01                            | Loss function에 Weigth가 커질 경우 패널티 값 |
| warmup_step            | int   | 500                             |
| debug                  | bool  | false                           | 디버그 모드일 경우 True                      |
| dropout_rate           | float | 0.1                             | dropout 비율                                 |
| save_steps             | int   | 100                             | 모델 저장 step 수                            |
| evaluation_steps       | int   | 100                             | evaluation할 step 수                         |
| metric_for_best_model  | str   | eval/loss                       | 최고 성능을 가늠하는 metric                  |
| load_best_model_at_end | bool  | True                            |

- RE Improved Baseline

| Argument                    | type  | Default                         |
| --------------------------- | ----- | ------------------------------- |
| batch_size                  | int   | 16                              |
| num_folds                   | int   | 5                               |
| num_train_epochs            | int   | 5                               |
| loss                        | str   | focalloss                       |
| gamma                       | float | 1.0                             |
| optimizer                   | str   | adamp                           |
| scheduler                   | str   | get_cosine_schedule_with_warmup |
| learning_rate               | float | 0.00005                         |
| weight_decay                | float | 0.01                            |
| gradient_accumulation_steps | int   | 2                               |
| max_grad_norm               | float | 1.0                             |
| warmup_ratio                | float | 0.1                             |
| warmup_step                 | int   | 500                             |
| debug                       | bool  | false                           |
| dropout_rate                | float | 0.1                             |
| save_steps                  | int   | 100                             |
| evaluation_steps            | int   | 100                             |
| metric_for_best_model       | str   | f1                              |
| load_best_model_at_end      | bool  | True                            |

- Concat Model

| Argument               | type  | Default            |
| ---------------------- | ----- | ------------------ |
| model                  | str   | CustomModel        |
| num_labels             | int   | 30                 |
| num_workers            | int   | 4                  |
| max_token_length       | int   | 132                |
| stopwords              | list  | [.]                |
| pretrained_model_name  | str   | klue/roberta-large |
| fine_tuning_method     | str   | concat             |
| batch_size             | int   | 40                 |
| num_folds              | int   | 5                  |
| num_train_epochs       | int   | 3                  |
| loss                   | str   | focalloss          |
| gamma                  | int   | 0.5                |
| optimizer              | str   | adamp              |
| learning_rate          | float | 0.00005            |
| weight_decay           | float | 0.01               |
| warmup_steps           | int   | 300                |
| debug                  | bool  | false              |
| dropout_rate           | float | 0.1                |
| save_steps             | int   | 100                |
| evaluation_strategy    | str   | steps              |
| evaluation_steps       | int   | 500                |
| metric_for_best_model  | str   | accuracy           |
| load_best_model_at_end | bool  | true               |