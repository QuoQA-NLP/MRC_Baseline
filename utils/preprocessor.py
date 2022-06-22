class Preprocessor:
    def __init__(self, model_category):
        self.model_category = model_category

    def preprocess_train(self, datasets):

        flags = datasets["is_impossible"]

        answer_texts = []
        answer_starts = []

        ## 정답의 갯수가 여러개면 가장 첫번째를 label로 결정
        for i in range(len(datasets["paragraph_id"])):
            answer = datasets["answers"][i]

            ## is_impossible이 True인 경우는 정답을 ''으로 변경
            flag = flags[i]
            if flag == True:
                answer_texts.append("")
                answer_starts.append(0)
            else:
                answer_texts.append(answer["text"][0])
                answer_starts.append(answer["answer_start"][0])

        datasets["answer_text"] = answer_texts
        datasets["answer_start"] = answer_starts
        return datasets

    def preprocess_validation(self, datasets):

        flags = datasets["is_impossible"]
        answers = []

        for i in range(len(datasets["paragraph_id"])):
            answer = datasets["answers"][i]

            ## Squad V2 코드 분석 결과 is_impossible인 경우 빈 리스트를 전달해야 No Answer를 제대로 처리
            flag = flags[i]
            if flag == True:
                answer = {"text": [], "answer_start": []}

            answers.append(answer)

        datasets["answers"] = answers
        return datasets
