from datasets import load_metric
from transformers import EvalPrediction


class Metric:

    """
    # https://github.com/huggingface/datasets/blob/master/metrics/squad_v2/squad_v2.py
    # https://huggingface.co/spaces/evaluate-metric/squad_v2
    Computes SQuAD v2 scores (F1 and EM).
    Args:
        predictions: List of triple for question-answers to score with the following elements:
            - the question-answer 'id' field as given in the references (see below)
            - the text of the answer
            - the probability that the question has no answer
        references: List of question-answers dictionaries with the following key-values:
                - 'id': id of the question-answer pair (see above),
                - 'answers': a list of Dict {'text': text of the answer as a string}
        no_answer_threshold: float
            Probability threshold to decide that a question has no answer.
    Returns:
        'exact': Exact match (the normalized answer exactly match the gold answer)
        'f1': The F-score of predicted tokens versus the gold answer
        'total': Number of score considered
        'HasAns_exact': Exact match (the normalized answer exactly match the gold answer)
        'HasAns_f1': The F-score of predicted tokens versus the gold answer
        'HasAns_total': Number of score considered
        'NoAns_exact': Exact match (the normalized answer exactly match the gold answer)
        'NoAns_f1': The F-score of predicted tokens versus the gold answer
        'NoAns_total': Number of score considered
        'best_exact': Best exact match (with varying threshold)
        'best_exact_thresh': No-answer probability threshold associated to the best exact match
        'best_f1': Best F1 (with varying threshold)
        'best_f1_thresh': No-answer probability threshold associated to the best F1
    Examples:
        >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
        >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
        >>> squad_v2_metric = datasets.load_metric("squad_v2")
        >>> results = squad_v2_metric.compute(predictions=predictions, references=references)
        >>> print(results)
        {'exact': 100.0, 'f1': 100.0, 'total': 1, 'HasAns_exact': 100.0, 'HasAns_f1': 100.0, 'HasAns_total': 1, 'best_exact': 100.0, 'best_exact_thresh': 0.0, 'best_f1': 100.0, 'best_f1_thresh': 0.0}
    """

    def __init__(self,):
        self.metric_v1 = load_metric("squad")
        self.metric_v2 = load_metric("squad_v2")

    def compute_metrics(self, pred: EvalPrediction):
        results = self.metric_v2.compute(
            predictions=pred.predictions, references=pred.label_ids, no_answer_threshold=0.0
        )
        return results
