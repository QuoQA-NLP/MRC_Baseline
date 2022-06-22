import os
import json
import torch
import torch.nn as nn
import numpy as np
import collections
from datasets import Dataset
from typing import Optional, Tuple
from transformers import EvalPrediction
from tqdm import tqdm

from konlpy.tag import Mecab


def postprocess_qa_predictions(
    raw_datasets: Dataset,
    encoded_datasets: Dataset,
    predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_answer_length: int = 30,
):

    all_is_impossible_logits, all_start_logits, all_end_logits = predictions

    ## key : 데이터 id ("6566495-0-0", "6566495-0-1", ....) & value : 데이터 index (1,2,3, ...)
    dataid_to_index = {k: i for i, k in enumerate(raw_datasets["question_id"])}

    encoded_data_per_id = collections.defaultdict(list)
    for i, encoded_data in enumerate(encoded_datasets):

        data_id = encoded_data["question_id"]  ## 데이터 id
        index = dataid_to_index[data_id]  ## 데이터 index

        ## overflow token mappings로 인해서 문서의 길이가 긴 경우에는 여러 개의 데이터로 나누었기 때문에 필요한 작업
        encoded_data_per_id[index].append(i)  ## Key : 인코딩 이전 데이터 인덱스 & Value : 인코딩 이후 데이터 인덱스

    all_predictions = {}

    for data_index, raw_data in enumerate(tqdm(raw_datasets)):

        data_id = raw_data["question_id"]
        encoded_data_indices = encoded_data_per_id[data_index]

        min_null_prediction = None
        prelim_predictions = []

        for encoded_data_id in encoded_data_indices:

            start_logits = all_start_logits[encoded_data_id]
            end_logits = all_end_logits[encoded_data_id]
            is_impossible_logits = all_is_impossible_logits[encoded_data_id]

            offset_mapping = encoded_datasets[encoded_data_id]["offset_mapping"]

            ## 시작위치, 끝 위치 (토큰 기준) 모두 0이면 정답이 없다는 것을 의미
            feature_null_score = start_logits[0] + end_logits[0]

            ## 하나의 context에서 여러개의 데이터가 생성이 된다면 그 중에서 가장 작은 CLS 값을 경계값으로 지정한다.
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:

                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": float(feature_null_score),
                    "is_impossible_logits": float(is_impossible_logits),
                }

            start_indices = np.argsort(start_logits)[-10:][
                ::-1
            ].tolist()  ## start_logits 기준 가장 점수 높은 10개 선정
            end_indices = np.argsort(end_logits)[-10:][
                ::-1
            ].tolist()  ## end_logits 기준 가장 점수 높은 10개 선정

            context_start = encoded_datasets[encoded_data_id]["context_start"]
            context_end = encoded_datasets[encoded_data_id]["context_end"]

            ## 선정된 start, end index들 대상으로 모든 경우의 수를 고려
            for start_index in start_indices:
                for end_index in end_indices:

                    ## start_index 및 end_index가 모두 context 내에 있는 것만을 고려
                    if context_start <= start_index and end_index <= context_end:

                        ## end position이 start position보다 앞에 있는 경우는 생략
                        if end_index < start_index:
                            continue

                        ## 정답 문구의 길이가 너무 긴 경우는 생략
                        answer_text = offset_mapping[end_index][1] - offset_mapping[start_index][0]
                        if answer_text > max_answer_length:
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "is_impossible_logits": float(is_impossible_logits),
                                "score": float(start_logits[start_index] + end_logits[end_index]),
                            }
                        )

        ## 선택된 start, end index들이 없는 경우 => 해당 context에는 질문에 대한 답이 없다는 것을 의미
        if len(prelim_predictions) == 0:
            min_null_prediction["prediction_text"] = ""
            all_predictions[data_id] = min_null_prediction
            continue

        ## 가장 점수가 높은 상위 10개를 선정해서 점수 기준으로 정렬
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:10]

        ## 가장 점수가 높은 것 보다 cls가 점수가 높은 경우 => 해당 context에는 질문에 대한 답이 없다는 것을 의미
        if predictions[0]["score"] < min_null_prediction["score"]:
            min_null_prediction["prediction_text"] = ""
            all_predictions[data_id] = min_null_prediction
        else:
            ## offset을 이용해서 context에서 정답 문구를 추출
            start_id = predictions[0]["offsets"][0]
            end_id = predictions[0]["offsets"][1]

            context = raw_data["context"]
            if context[start_id:end_id].strip()=="":
              predictions[0]["prediction_text"] = ""
            else:
              mecab = Mecab()
              predict = pos_processing_josa(context, context[start_id:end_id], start_id, mecab)
              predictions[0]["prediction_text"] = predict
            # predictions[0]["prediction_text"] = context[start_id:end_id]
            all_predictions[data_id] = predictions[0]

    return all_predictions


def pos_processing_josa(context, answer_text, start_id, analysis):
    stride = 20
    end_id = len(answer_text) + start_id
    before_text = ""
    after_text = ""
    if len(context[:start_id]) >= stride:
        before_text = context[start_id - stride : start_id]
        if answer_text in before_text:
            before_text = context[before_text.find(answer_text) + len(answer_text) + 1 : start_id]

    else:
        before_text = context[:start_id]
        if answer_text in before_text:
            before_text = context[before_text.find(answer_text) + len(answer_text) + 1 : start_id]

    if len(context[end_id:]) >= stride:
        after_text = context[end_id : end_id + stride]
        if answer_text in after_text:
            after_text = context[end_id : after_text.find(answer_text) - len(answer_text)]

    else:
        after_text = context[end_id:]
        if answer_text in after_text:
            after_text = context[end_id : after_text.find(answer_text) - len(answer_text)]

    tmp_text = before_text + answer_text + after_text
    pos_tag = analysis.pos(tmp_text)
    an = "".join(answer_text.split())
    t = ""
    idx = 0
    # context에 대해서 mecab을 통한 품사 태깅합니다.
    for iz in range(len(pos_tag)):
        t += pos_tag[iz][0]
        if an in t:
            idx = iz
            break

    # 조사가 포함된 품사가 있다면 해당되는 단어를 제거합니다.
    if pos_tag[idx][1] in {
        "JX",
        "JKB",
        "JKO",
        "JKS",
        "ETM",
        "VCP",
        "JC",
        "VCP+EC",
        "SS",
        "Josa",
        "Verb",
        "JKG",
    }:
        count = 0
        for idx_char, char in enumerate(pos_tag[idx][0]):
            count += 1
            if pos_tag[idx][0][idx_char] == an[-1]:
                break

        if pos_tag[idx][1] in "JKB" and pos_tag[idx][0] == "로":
            answer_text = answer_text
        elif pos_tag[idx][1] in "JX" and pos_tag[idx][0] == "야":
            answer_text = answer_text
        else:
            answer_text = answer_text[: len(answer_text) - count]

    pos_front = analysis.pos(answer_text)
    if len(pos_front) == 0:
        return answer_text.strip()
    idx = 0
    if pos_front[idx][1] in {
        "JX",
        "JKB",
        "JKO",
        "JKS",
        "ETM",
        "VCP",
        "JC",
        "VCP+EC",
        "SS",
        "SY",
        "Josa",
    }:
        count = 0
        for idx_char, char in enumerate(pos_front[idx][0]):
            count += 1
            if pos_front[idx][0][idx_char] == an[-1]:
                break
        if pos_tag[idx][0] == "$":
            answer_text = answer_text
        else:
            answer_text = answer_text[count:]
    return answer_text.strip()


def post_process_function(
    raw_datasets, encoded_datasets, predictions, training_args, checkpoint_dir
):
    """
        predictions
            1. type : dict
            2. key : example id
            3. value : example 예측 결과
                1) offset,
                2) score,
                3) start_logit,
                4) end_logit
                5) prediction_text
    """
    predictions = postprocess_qa_predictions(
        raw_datasets=raw_datasets,  # Tokenization 되지 않은 dataset
        encoded_datasets=encoded_datasets,  # Tokenization 된 dataset
        predictions=predictions,  # start logits과 the end logits을 나타내는 two arrays
        max_answer_length=training_args.max_answer_length,  # 후처리 이후 정답 문구의 최대 길이
    )

    output_dir = training_args.output_dir

    if checkpoint_dir is not None:
        ## 추론 결과를 json 파일 형태로 checkpoint directory에 저장
        dir_path = os.path.join(output_dir, checkpoint_dir)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        prediction_file = os.path.join(dir_path, "predictions.json")
        with open(prediction_file, "w") as f:
            json.dump(predictions, f, ensure_ascii=False)

    ## Metric을 구할 수 있도록 predictions 형태를 맞춘다.
    formatted_predictions = [
        {
            "id": k,
            "prediction_text": v["prediction_text"],
            "no_answer_probability": v["is_impossible_logits"],
        }
        for k, v in predictions.items()
    ]

    ## test dataset 대상
    if training_args.do_predict:
        return formatted_predictions

    ## validation dataset 대상 / 정답의 갯수가 여러 개인 경우 start, text 모두 리스트 형식으로 저장한다.
    elif training_args.do_eval:
        references = [
            {
                "id": ex["question_id"],
                "answers": {
                    "answer_start": ex["answers"]["answer_start"],
                    "text": ex["answers"]["text"],
                },
            }
            for ex in raw_datasets
        ]

        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
