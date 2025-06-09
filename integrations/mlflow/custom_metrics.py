import numpy as np 
import pandas as pd 
import mlflow 
from mlflow.metrics import make_metric, MetricValue 

from evaluation.metrics import ExactMatch 


def exact_match_eval_fn(predictions: pd.Series,  targets: pd.Series) -> MetricValue:
    exact_match = ExactMatch()
    scores = [
        exact_match.calculate(pred, target)
        for pred, target in zip(predictions, targets)
    ]

    return MetricValue(
        scores=scores, 
        aggregate_results={
            "mean": np.mean(scores), 
            "variance": np.var(scores), 
            "median": np.median(scores)
        }
    )


exact_match_metric = make_metric(
    eval_fn=exact_match_eval_fn, 
    greater_is_better=True 
)
