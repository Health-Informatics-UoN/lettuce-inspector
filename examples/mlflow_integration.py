"""
Minimal example of using MLflow integration with Lettuce pipelines.
No dependency on EvaluationFramework.
"""
import pandas as pd

from components.models import LLMModel 
from components.embeddings import get_embedding_model 
from integrations.mlflow.config import (
    LLMConfig, 
    EmbeddingConfig, 
    RetrievalConfig, 
    DatabaseConfig, 
    RAGPipelineConfig
)
from integrations.mlflow.pipeline_wrapper import MLflowRAGPipeline 
from integrations.mlflow.evaluation_runner import MLflowEvaluationRunner


def main():
    # 1. Load a simple evaluation dataset
    eval_df = pd.read_csv("./evaluation/datasets/example.csv")

    # 2. Define pipeline configuration 
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

    Respond only with the formal name of the medication, without any extra explanation.

    Examples:

    Informal name: Tylenol
    Response: Acetaminophen

    Informal name: Advil
    Response: Ibuprofen

    Informal name: Motrin
    Response: Ibuprofen

    Informal name: Aleve
    Response: Naproxen<|eot_id|><|start_header_id|>user<|end_header_id|>

    Possible related terms:
    {% for result in vec_results %}
    {{result.content}}
    {% endfor %}

    Task:
    Informal name: {{informal_name}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
    config = RAGPipelineConfig(
        llm=LLMConfig(model_name=LLMModel.LLAMA_3_1_8B.value),
        embedding=EmbeddingConfig(model_name=get_embedding_model("BGESMALL").info.path),
        database=DatabaseConfig.from_env(),
        retrieval=RetrievalConfig(vocab_ids=["RxNorm"], standard_concept=True), 
        prompt_template=prompt_template, 
        template_vars = ["informal_name", "vec_results"]
    ) 

    # 3. Initialise the pipeline from the config 
    pipeline = MLflowRAGPipeline(config)

    # 4. Initialise the evaluation runner responsible for logging the experiment data
    runner = MLflowEvaluationRunner(
        experiment_name="mlflow_integration_example", 
        tracking_uri="=http://localhost:5000"
    )

    # 5. Run the evaluation and the start the UI from command line 
    breakpoint()
    

if __name__ == "__main__":
    main()