
import os
import json
from datasets import Dataset, DatasetDict

class Loader :

    def __init__(self, data_path) :
        self.data_path = data_path

    def get_orders(self, dataset, question_id_orders) :
        mapping = {}
        for i, d in enumerate(dataset) :
            quid = d["question_id"]
            mapping[quid] = i

        orders = []
        for quid in question_id_orders :
            orders.append(mapping[quid])
        return orders

    def load_train_data(self, question_id_orders) :

        with open(os.path.join(self.data_path, "train.json"), "r") as f :
            train_dset = json.load(f)
        train_dset = train_dset["data"]

        content_ids = []
        paragraph_ids = []
        questions = []
        question_ids = []
        titles = []
        questions = []
        contexts = []
        answers = []
        is_impossibles = []

        for d in train_dset :
            content_id = d["content_id"]
            title = d["title"]

            paragraphs = d["paragraphs"]

            for p in paragraphs :

                p_id = p["paragraph_id"]
                context = p["context"]
                qas = p["qas"]

                for qa in qas :
                    q_id = qa["question_id"]
                    q = qa["question"]
                    a = qa["answers"]
                    f = qa["is_impossible"]

                    content_ids.append(content_id)
                    titles.append(title)
                    paragraph_ids.append(p_id)
                    question_ids.append(q_id)
                    contexts.append(context)
                    questions.append(q)
                    answers.append(a)
                    is_impossibles.append(f)

        dataset = Dataset.from_dict({"content_id" : content_ids,
            "paragraph_id" : paragraph_ids,
            "title" : titles,
            "question_id" : question_ids, 
            "question" : questions,
            "context" : contexts,
            "answers" : answers, 
            "is_impossible" : is_impossibles }
        )

        orders = self.get_orders(dataset, question_id_orders)
        dataset = dataset.select(orders)

        return DatasetDict({"train" : dataset})

    def load_test_data(self, ) :

        with open(os.path.join(self.data_path, "test.json"), "r") as f :
            test_dset = json.load(f)

        test_dset = test_dset["data"]

        content_ids = []
        paragraph_ids = []
        questions = []
        question_ids = []
        titles = []
        questions = []
        contexts = []

        for d in test_dset :
            content_id = d["content_id"]
            title = d["title"]

            paragraphs = d["paragraphs"]

            for p in paragraphs :

                p_id = p["paragraph_id"]
                context = p["context"]
                qas = p["qas"]

                for qa in qas :
                    q_id = qa["question_id"]
                    q = qa["question"]
            
                    content_ids.append(content_id)
                    titles.append(title)
                    paragraph_ids.append(p_id)
                    question_ids.append(q_id)
                    contexts.append(context)
                    questions.append(q)

        dataset = Dataset.from_dict({"content_id" : content_ids,
            "paragraph_id" : paragraph_ids,
            "title" : titles,
            "question_id" : question_ids, 
            "question" : questions,
            "context" : contexts}
        )
        return dataset


                

                