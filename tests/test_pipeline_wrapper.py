import pandas as pd 
import pytest 
import mlflow 

from integrations.mlflow.pipeline_wrapper import MLflowLLMPipeline, MLflowRAGPipeline
from integrations.mlflow.config import (
    LLMConfig, 
    EmbeddingConfig, 
    RetrievalConfig, 
    DatabaseConfig, 
    LLMPipelineConfig, 
    RAGPipelineConfig
)
from components.models import  LLMModel 
from components.embeddings import get_embedding_model 


@pytest.fixture(scope="session")
def prompt_template_str_simple(): 
    return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

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

Informal name: {{informal_name}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""   


@pytest.fixture(scope="session")
def prompt_template_str_rag():
    return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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

def test_predict_llm_pipeline_wrapper(prompt_template_str_simple):     
    model_input = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "expected_output": ["acetaminophen", "codeine"]
    })
    config = LLMPipelineConfig(
        llm=LLMConfig(model_name=LLMModel.LLAMA_3_1_8B.value),
        prompt_template=prompt_template_str_simple, 
        template_vars = ["informal_name", "vec_results"]
    )
    pipeline = MLflowLLMPipeline(config)
    predictions = pipeline.predict(model_input=model_input)
    assert predictions["predictions"].iloc[0].strip().lower() == "acetaminophen"
    assert predictions["predictions"].iloc[1].strip().lower() == "codeine"


def test_predict_rag_pipeline_wrapper(prompt_template_str_rag): 
    model_input = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "expected_output": ["acetaminophen", "codeine"]
    })
    config = RAGPipelineConfig(
        llm=LLMConfig(model_name=LLMModel.LLAMA_3_1_8B.value),
        embedding=EmbeddingConfig(model_name=get_embedding_model("BGESMALL").info.path),
        database=DatabaseConfig.from_env(),
        retrieval=RetrievalConfig(vocab_ids=["RxNorm"], standard_concept=True), 
        prompt_template=prompt_template_str_rag, 
        template_vars = ["informal_name", "vec_results"]
    ) 
    pipeline = MLflowRAGPipeline(config=config)
    predictions = pipeline.predict(model_input)
    assert predictions["predictions"].iloc[0].strip().lower() == "acetaminophen"
    assert predictions["predictions"].iloc[1].strip().lower() == "codeine"


def test_pyfunc_model_logging(tmp_path): 
    tracking_uri = tmp_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("pyfunc_llm_test")

    with mlflow.start_run() as run:
        # Log the model to the run's artifacts
        config = LLMPipelineConfig(
            llm=LLMConfig(model_name=LLMModel.LLAMA_3_1_8B.value),
            prompt_template=prompt_template_str_simple, 
            template_vars = ["informal_name", "vec_results"]
        )
        pipeline = MLflowLLMPipeline(config)
        mlflow.pyfunc.log_model(
            artifact_path="llm_model",
            python_model=pipeline
        )
        run_id = run.info.run_id 

    model_uri = f"runs:/{run_id}/llm_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    assert loaded_model is not None


def test_llm_pipeline_from_wrong_config_error(): 
    pass 

def test_rag_pipeline_from_wrong_config_error(): 
    pass 

def test_embeddings_pipeline_from_wrong_config_error(): 
    pass 
