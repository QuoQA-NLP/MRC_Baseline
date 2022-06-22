class Encoder:
    def __init__(self, tokenizer, stride=128, max_length=512, has_cls=True):
        self.tokenizer = tokenizer
        self.stride = stride
        self.max_length = max_length
        self.has_cls = has_cls

    def prepare_train_features(self, datasets):

        is_impossibles = datasets["is_impossible"]

        model_inputs = self.tokenizer(
            datasets["question"],
            datasets["context"],
            return_token_type_ids=False,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation="only_second",
            stride=self.stride,
            max_length=self.max_length,
        )

        sample_mappings = model_inputs.pop("overflow_to_sample_mapping")
        offset_mappings = model_inputs.pop("offset_mapping")

        ## QuestionAnswering 모델의 label
        model_inputs["start_positions"] = []
        model_inputs["end_positions"] = []
        model_inputs["is_impossibles"] = []

        for i in range(len(sample_mappings)):

            ## tokenizing 이전의 dataset index
            sample_id = sample_mappings[i]
            impossible_flag = 1 if is_impossibles[sample_id] == True else 0

            input_ids = model_inputs["input_ids"][i]
            if self.has_cls:
                start_token_id = self.tokenizer.cls_token_id
            else:
                start_token_id = self.tokenizer.eos_token_id
            cls_index = input_ids.index(start_token_id)
            mapping = offset_mappings[i]

            ## character 관점의 정답 시작 위치
            answer_start = datasets["answer_start"][sample_id]
            answer_text = datasets["answer_text"][sample_id]

            ## 정답이 없는 경우
            if len(answer_text) == 0:
                ## start_position, end_postion 모두 0으로 예측하도록 모델을 학습
                model_inputs["start_positions"].append(cls_index)
                model_inputs["end_positions"].append(cls_index)
                model_inputs["is_impossibles"].append(1)
            else:
                ## character 관점의 정답 끝 위치
                answer_end = answer_start + len(answer_text)

                ## [cls] question [sep] context [sep] 에서 question, context를 구분할 수 있는 sequence_id 리스트
                ## [cls], [sep] : None / question : 0 / context : 1
                sequence_ids = model_inputs.sequence_ids(i)

                ## 1이 시작되는 시작 토큰, 끝 토큰을 파악 (Context안에 정답이 있으므로)
                start_token, end_token = self.get_positions(sequence_ids, value=1)

                token_start_index = start_token
                token_end_index = end_token

                ## 1이 시작되는 토큰의 시작점 <= 정답의 시작 & 정답의 끝 <= 1이 끝나는 토큰의 마지막점이 성립되어야 context 내에 정답이 존재한다는 뜻
                if (
                    mapping[token_start_index][0] <= answer_start
                    and answer_end <= mapping[token_end_index][1]
                ):

                    ## 토큰 관점으로 시작 지점, 끝 지점 이동
                    while (
                        token_start_index < len(sequence_ids)
                        and mapping[token_start_index][0] <= answer_start
                    ):
                        token_start_index += 1

                    while answer_end <= mapping[token_end_index][1]:
                        token_end_index -= 1

                    model_inputs["start_positions"].append(token_start_index - 1)
                    model_inputs["end_positions"].append(token_end_index + 1)
                    model_inputs["is_impossibles"].append(impossible_flag)

                else:
                    model_inputs["start_positions"].append(cls_index)
                    model_inputs["end_positions"].append(cls_index)
                    model_inputs["is_impossibles"].append(1)

        return model_inputs

    def prepare_validation_features(self, datasets):

        model_inputs = self.tokenizer(
            datasets["question"],
            datasets["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )

        sample_mappings = model_inputs.pop("overflow_to_sample_mapping")

        model_inputs["question_id"] = []
        model_inputs["context_start"] = []
        model_inputs["context_end"] = []

        ## evaluate을 위해서는 start_position, end_position이 불필요
        for i in range(len(sample_mappings)):

            sample_id = sample_mappings[i]
            model_inputs["question_id"].append(datasets["question_id"][sample_id])

            sequence_ids = model_inputs.sequence_ids(i)
            start_token, end_token = self.get_positions(sequence_ids, value=1)

            model_inputs["context_start"].append(start_token)
            model_inputs["context_end"].append(end_token)

        return model_inputs

    def get_positions(self, vector, value=1):
        start_token = 1
        end_token = len(vector) - 1

        while start_token < len(vector) and vector[start_token] != value:
            start_token += 1

        while end_token > 0 and vector[end_token] != value:
            end_token -= 1

        return start_token, end_token
