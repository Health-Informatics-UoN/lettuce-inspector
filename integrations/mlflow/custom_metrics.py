import numpy as np 
import pandas as pd 
import mlflow 
from mlflow.metrics import make_metric, MetricValue, EvaluationMetric 

from evaluation.evaltypes import SingleResultMetric 
from evaluation.metrics import (
    ExactMatch, 
    UncasedMatch, 
    FuzzyMatchRatio
) 


def make_custom_metric(
    metric: SingleResultMetric, 
    metric_name: str = None, 
    greater_is_better: bool = True 
) -> EvaluationMetric: 
    """
    Convert a Lettuce metric to an MLflow EvaluationMetric.
    
    Args:
        metric: The Lettuce metric to convert
        metric_name: Optional custom name for the metric
        
    Returns:
        MLflow EvaluationMetric that can be used with mlflow.evaluate()
    """
    name = metric_name or metric.__class__.__name__

    def eval_fn(predictions: pd.Series, targets: pd.Series) -> float:
        scores = [
            metric.calculate(pred, target)
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

    return make_metric(
        eval_fn=eval_fn, 
        greater_is_better=greater_is_better, 
        name=name 
    )


def make_simple_string_comparison_metrics() -> list:
    simple_metrics = {
        "ExactMatch": ExactMatch,
        "UncasedMatch": UncasedMatch,
        "FuzzyMatchRatio": FuzzyMatchRatio
    }
    mlflow_metrics = []
    for metric_name in simple_metrics:
        metric_instance = None
        metric_instance = simple_metrics[metric_name]()
        mlflow_metrics.append(make_custom_metric(metric_instance))
    return mlflow_metrics
    