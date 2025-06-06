import mlflow
from typing import Dict, List, Any
import json

class EvaluationMLflowLogger:
    """MLflow integration for Lettuce evaluation framework"""

    def __init__(self, experiment_name: str = "lettuce-evaluation"):
        mlflow.set_experiment(experiment_name)

    def log_evaluation_run(self, evaluation_framework):
        """Log a complete evaluation run to MLflow"""
        with mlflow.start_run(run_name=evaluation_framework.name):
            mlflow.log_param("description", evaluation_framework._description)
            mlflow.log_param("dataset_size", len(evaluation_framework.input_data))

            for test_result in evaluation_framework.evaluation_results:
                for test_name, results in test_result.items():
                    with mlflow.start_run(run_name=test_name, nested=True):
                        # Log metric descriptions
                        desc_str = "; ".join(results["metric_descriptions"])
                        mlflow.log_param("metric_descriptions", desc_str)

                        # Log aggregate metrics only
                        self._log_aggregate_metrics(results["results"])

            # Log the complete results as artifact
            mlflow.log_dict(evaluation_framework.evaluation_results, "full_results.json")

    def _log_aggregate_metrics(self, results: List[Dict[str, float]]):
        """Calculate and log aggregate metrics"""
        metric_totals = {}

        for result in results:
            for metric_name, score in result.items():
                metric_totals.setdefault(metric_name, []).append(score)

        for metric_name, scores in metric_totals.items():
            mlflow.log_metric(f"avg_{metric_name}", sum(scores) / len(scores))
            mlflow.log_metric(f"min_{metric_name}", min(scores))
            mlflow.log_metric(f"max_{metric_name}", max(scores))