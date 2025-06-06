"""
Minimal example of using MLflow integration with Lettuce pipelines.
No dependency on EvaluationFramework.
"""
import pandas as pd
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from evaluation.pipelines import LLMPipeline
from evaluation.metrics import UncasedMatch, FuzzyMatchRatio
from evaluation.eval_tests import LLMPipelineTest
from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework
from components.models import local_models
from options.pipeline_options import LLMModel


def main():
    # 1. Load a simple evaluation dataset
    dataloader = SingleInputCSVforLLM("./evaluation/datasets/example.csv")
    
    # 2. Set up a pipeline
    model_name = LLMModel.LLAMA_3_1_8B.value
    llm = Llama(
        hf_hub_download(**local_models[model_name]),
        n_ctx=0,
        n_batch=512,
        model_kwargs={"n_gpu_layers": -1},
        generation_kwargs={"max_tokens": 50, "temperature": 0}
    )
    
    prompt_template_str = """You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

    Examples:

    Informal name: Tylenol
    Response: Acetaminophen

    Informal name: Advil
    Response: Ibuprofen

    Informal name: Motrin
    Response: Ibuprofen

    Informal name: Aleve
    Response: Naproxen

    Task:

    Informal name: {{informal_name}}<|eot_id|>
    Response:"""

    pipeline = LLMPipeline(
        llm=llm,
        prompt_template_str=prompt_template_str,
        template_vars=["informal_name"]
    )

    pipeline_test= LLMPipelineTest(model_name, pipeline, [UncasedMatch(), FuzzyMatchRatio()])
    
    evaluation_framework = EvaluationFramework(
        name="Example MLflow",
        pipeline_tests=[pipeline_test],
        dataset=dataloader, 
        description="Demonstration of running mlflow logging",
        results_path="mlflow_example_output.json",
        use_mlflow=True, 
        experiment_name="lettuce-evaluation-mlflow-example"
    )

    evaluation_framework.run_evaluations()
    

if __name__ == "__main__":
    main()