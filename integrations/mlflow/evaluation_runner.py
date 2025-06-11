from typing import List, Optional
import pandas as pd 
import mlflow 
from mlflow.metrics import MetricValue

from evaluation.evaltypes import SingleResultPipeline
from integrations.mlflow.pipeline_wrapper import MLflowPipelineWrapper 
from integrations.mlflow.utils import generate_pip_requirements


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

    @staticmethod
    def log_dataset(eval_data: pd.DataFrame):
        """
        Log the dataset to the mlflow UI. 
        """
        pd_dataset = mlflow.data.from_pandas(
            eval_data, targets="targets"
        )
        mlflow.log_input(pd_dataset, context="evaluation")
        return pd_dataset 

    @staticmethod
    def log_pipeline(
        pipeline: SingleResultPipeline, 
        pipeline_name: str, 
        pipeline_type: str, 
        input_example: Optional[pd.DataFrame] = None, 
        signature: Optional[mlflow.models.ModelSignature] = None,
        pip_requirements: Optional[dict] = None,
        code_paths: Optional[List[str]] = None, 
        registered_model_name: Optional[str] = None    
    ) -> mlflow.models.model.ModelInfo: 
        """
        Log a lettuce pipeline as a MLflow model. 
        """
        wrapped_model = MLflowPipelineWrapper(pipeline, pipeline_type)

        if signature is None and input_example is not None: 
            output_example = wrapped_model.predict(input_example)
            signature = mlflow.models.infer_signature(input_example, output_example)

        if pip_requirements is None:
            pip_requirements = generate_pip_requirements()

        model_info = mlflow.pyfunc.log_model(
            artifact_path=pipeline_name, 
            python_model=wrapped_model, 
            input_example=input_example, 
            signature=signature, 
            pip_requirements=pip_requirements, 
            code_paths=code_paths,
            registered_model_name=registered_model_name 
        )

        if hasattr(pipeline, "prompt_template_text"):
            mlflow.log_text(
                pipeline.prompt_template_text,
                "prompt_template.jinja2"
            )
        
        if hasattr(pipeline, "_template_vars"):
            mlflow.log_param("template_vars", pipeline._template_vars)
        
        if hasattr(pipeline, "_top_k"):
            mlflow.log_param("top_k", pipeline._top_k)
        
        if hasattr(pipeline, "_embedding_model"):
            mlflow.log_param(
                "embedding_model",
                pipeline._embedding_model.__class__.__name__
            )

        return model_info 
    
    def run_evaluation(
        self, 
        pipeline: SingleResultPipeline, 
        pipeline_name: str, 
        pipeline_type: str, 
        eval_df: pd.DataFrame, 
        metrics: List[MetricValue]
    ): 
        with mlflow.start_run() as run: 
            pd_dataset = MLflowEvaluationRunner.log_dataset(eval_df)
           
            model_info = MLflowEvaluationRunner.log_pipeline(
                pipeline, 
                pipeline_name=pipeline_name, 
                pipeline_type=pipeline_type, 
                input_example=eval_df.iloc[0]
            )
            
            result = mlflow.evaluate(
                model=model_info.uri(),         
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
