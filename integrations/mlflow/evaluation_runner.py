from typing import List, Optional
from pathlib import Path
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
import mlflow 
from mlflow.metrics import MetricValue

from integrations.mlflow.config import BasePipelineConfig, LLMPipelineConfig, EmbeddingPipelineConfig, RAGPipelineConfig 
from integrations.mlflow.pipeline_wrapper import MLflowBasePipeline, MLflowLLMPipeline, MLflowEmbeddingPipeline, MLflowRAGPipeline
from integrations.mlflow.plotting import plot_boxplot, plot_violinplot
from integrations.mlflow.utils import generate_pip_requirements


class MLflowEvaluationRunner(): 
    def __init__(
        self, 
        experiment_name: str, 
        tracking_uri: str
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri 

    def _log_dataset(self, eval_data: pd.DataFrame):
        """
        Log the dataset to the mlflow UI. 
        """
        pd_dataset = mlflow.data.from_pandas(
            eval_data, targets="targets"
        )
        mlflow.log_input(pd_dataset, context="evaluation")
        return pd_dataset 

    def _validate_and_prepare_config(
        self, 
        pipeline, 
        pipeline_type, 
        pipeline_config, 
        pipeline_yaml
    ): 
        provided_inputs = sum([
            pipeline is not None,
            pipeline_config is not None,
            pipeline_yaml is not None
        ])
        
        if provided_inputs != 1:
            raise ValueError(
                "Provide exactly one of: pipeline, pipeline_config, or pipeline_yaml"
            )
        
        if pipeline_yaml:
            config_map = {
                "llm": LLMPipelineConfig,
                "embedding": EmbeddingPipelineConfig,
                "rag": RAGPipelineConfig
            }
            config_class = config_map.get(pipeline_type)
            if not config_class:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")  
               
            pipeline_config = config_class.from_yaml(pipeline_yaml)

        elif pipeline: 
            pipeline_config = pipeline.config 
        
        return pipeline_config 
    
    def _instantiate_pipeline_from_config(self, config, pipeline_type): 
        pipeline_map = {
            "llm": MLflowLLMPipeline,
            "embedding": MLflowEmbeddingPipeline,
            "rag": MLflowRAGPipeline
        }
        
        pipeline_class = pipeline_map.get(pipeline_type)
        if not pipeline_class:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
            
        return pipeline_class(config)

    
    def _log_pipeline(
        self, 
        pipeline: MLflowBasePipeline, 
        pipeline_name: str, 
        input_example: Optional[pd.DataFrame], 
        code_paths: Optional[List[str]] = None, 
    ) -> mlflow.models.model.ModelInfo: 
        """
        Log a lettuce pipeline as a MLflow model. 
        """
        output_example = pipeline.predict(input_example)
        signature = mlflow.models.infer_signature(input_example, output_example)

        pip_requirements = generate_pip_requirements()

        model_info = mlflow.pyfunc.log_model(
            artifact_path=pipeline_name, 
            python_model=pipeline, 
            input_example=input_example, 
            signature=signature, 
            pip_requirements=pip_requirements, 
            code_paths=code_paths,
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
        pipeline: MLflowBasePipeline = None, 
        pipeline_config: Optional[BasePipelineConfig] = None,
        pipeline_yaml: Optional[str] = None,
        pipeline_name: str = None,
        pipeline_type: str = None,
        eval_df: pd.DataFrame = None, 
        metrics: List[MetricValue] = None, 
        code_paths: Optional[List[str]] = None
    ):
        """
        Run evaluation on a pipeline.
        
        Provide ONE of:
        - pipeline: 
            Already instantiated pipeline object      
        - pipeline_config: 
            Configuration object
        - pipeline_yaml: 
            Path to YAML configuration file
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        config = self._validate_and_prepare_config(
            pipeline, 
            pipeline_type, 
            pipeline_config, 
            pipeline_yaml 
        )

        if pipeline is None: 
            pipeline = self._instantiate_pipeline_from_config(config, pipeline_type)

        with mlflow.start_run() as run: 
            mlflow.log_dict(config.to_dict(), "pipeline_config.yaml")

            _ = self._log_dataset(eval_df)
           
            input_example = eval_df[["input_data"]].head(5) if eval_df.shape[0] >= 5 else eval_df[["input_data"]] 
            model_info = self._log_pipeline(
                pipeline, 
                pipeline_name=pipeline_name, 
                input_example=input_example, 
                code_paths=code_paths
            )
            results = mlflow.evaluate(
                model=model_info.model_uri,         
                data=eval_df,                 
                targets="targets",               
                extra_metrics=metrics, 
                evaluator_config={
                    "col_mapping": {
                        "inputs": "input_data",
                        "predictions": "predictions", 
                        "targets": "targets"          
                    }
                }
            )

            self.log_metric_distribution_figures(results.tables["eval_results_table"], metrics)

        return run 
        
    def evaluate_against_baseline(
        self, 
        baseline_model_run_id = None,  
        static_result = None     
    ): 
        """
        See MLflow docs - Model Evaluation section.

        Used to compare a candidate pipeline model against a baseline pipeline. 
        """
        pass 

    def log_metric_distribution_figures(
        self,
        eval_results: pd.DataFrame, 
        metrics: List[MetricValue]
    ):
        """
        Plot the distribution of the each metric and log the plots as a artefact to mlflow. 

        Input: 
            eval_results: pandas.DataFrame 

            metrics: List[MetricValue]


        Returns: 
            None 
        """
        metric_names = [metric.name for metric in metrics]
        cols_to_plot = [col for col in eval_results.columns if col.split("/score")[0] in metric_names]

        for col in cols_to_plot: 
            fig, _ = plot_violinplot(eval_results[col], palette="pastel", x_label=col.split("/score")[0], y_label="Score")
            mlflow.log_figure(fig, f"{col}_violinplot.png", save_kwargs={"dpi": 300, "bbox_inches": "tight"})

            fig, _ = plot_boxplot(eval_results[col], palette="pastel", plot_scatter=True, x_label=col.split("/score")[0], y_label="Score")
            mlflow.log_figure(fig, f"{col}_boxplot.png", save_kwargs={"dpi": 300, "bbox_inches": "tight"})
