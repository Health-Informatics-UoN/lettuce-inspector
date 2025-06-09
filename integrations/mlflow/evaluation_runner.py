from typing import List 
import pandas as pd 
import mlflow 
from mlflow.models import infer_signature
from mlflow.metrics import MetricValue

from evaluation.evaltypes import SingleResultPipeline
from integrations.mlflow.pipeline_wrapper import MLflowPipelineWrapper 


class MLflowEvaluationRunner(): 
    def __init__(
        self, 
        experiment_name: str, 
        tracking_uri: str
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri 
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_model(self, model: MLflowPipelineWrapper): 
        mlflow.pyfunc.log_model(
            artifact_path="llm_model",
            python_model=model
        )

    def log_dataset(self): 
        pass 

    def log_metrics(self): 
        pass 

    def run_evaluation(
        self, 
        pipeline: SingleResultPipeline, 
        pipeline_type: str, 
        eval_df: pd.DataFrame, 
        metrics: List[MetricValue]
    ): 
        with mlflow.start_run() as run: 
            model = MLflowPipelineWrapper(pipeline, pipeline_type)
            model
            
            input_example = eval_df.iloc[0]

            self.log_dataset()
            
            self.log_model(model)
            
            result = mlflow.evaluate(
                model=str(model_path),         
                data=eval_df,                 
                targets="targets",               
                extra_metrics=metrics, 
                evaluator_config={
                    "col_mapping": {
                        "predictions": "predictions", 
                        "targets": "targets"          
                    }
                }
            )