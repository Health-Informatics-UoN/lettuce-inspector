import pandas as pd 
import pytest 
import mlflow 
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from integrations.mlflow.pipeline_wrapper import MLflowPipelineWrapper, MLflowRAGPipeline
from integrations.mlflow.config import (
    LLMConfig, 
    EmbeddingConfig, 
    RetrievalConfig, 
    DatabaseConfig, 
    RAGPipelineConfig
)
from evaluation.pipelines import LLMPipeline
from components.models import local_models, LLMModel 
from components.embeddings import get_embedding_model 


@pytest.fixture(scope="session") 
def llama_model(
    model_name: str = LLMModel.TINYLLAMA_1_1B_CHAT.value, 
    n_ctx: int = 512, 
    n_batch: int = 32, 
    max_tokens: int = 128
): 
    return Llama(
        hf_hub_download(**local_models[model_name]),
        model_kwargs={
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_gpu_layers": -1,
            "verbose": True
        },
        generation_kwargs={"max_tokens": max_tokens, "temperature": 0.}
    )


@pytest.fixture(scope="session")
def prompt_template_str(): 
    return """You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

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
    Response: """   


@pytest.fixture(scope="session")
def prompt_template_str_rag(): 
    return """You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

    Respond only with the formal name of the medication, without any extra explanation.

    Examples:

    Informal name: Tylenol
    Response: Acetaminophen

    Informal name: Advil
    Response: Ibuprofen

    Informal name: Motrin
    Response: Ibuprofen

    Informal name: Aleve
    Response: Naproxen

    Possible related terms:
    {% for result in vec_results %}
        {{result.content}}
    {% endfor %}

    Task:

    Informal name: {{informal_name}}
    Response: """


def test_predict_llm_pipeline_wrapper(llama_model, prompt_template_str):     
    pipeline = LLMPipeline(
        llama_model, 
        prompt_template_str=prompt_template_str, 
        template_vars=["informal_name"]
    )
    wrapper = MLflowPipelineWrapper(pipeline, pipeline_type="llm")
    model_input = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "expected_output": ["acetaminophen", "codeine"]
    })
    predictions = wrapper.predict(model_input=model_input)
    assert predictions["predictions"].iloc[0].strip().lower() == "acetaminophen"
    assert predictions["predictions"].iloc[1].strip().lower() == "codeine"


def test_predict_rag_pipeline_wrapper(prompt_template_str_rag): 
    model_input = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "expected_output": ["acetaminophen", "codeine"]
    })
    config = RAGPipelineConfig(
        llm=LLMConfig(model_name=LLMModel.TINYLLAMA_1_1B_CHAT.value),
        embedding=EmbeddingConfig(model_name=get_embedding_model("BGESMALL").info.path),
        database=DatabaseConfig.from_env(),
        retrieval=RetrievalConfig(), 
        prompt_template=prompt_template_str_rag, 
        template_vars = ["informal_name", "vec_results"]
    ) 
    pipeline = MLflowRAGPipeline(config=config)
    result = pipeline.predict(model_input)
    breakpoint()


def test_error_thrown_if_input_data_not_present(): 
    pass 


def test_pyfunc_model_logging(tmp_path, llama_model, prompt_template_str): 
    tracking_uri = tmp_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("pyfunc_llm_test")

    with mlflow.start_run() as run:
        # Log the model to the run's artifacts
        pipeline = LLMPipeline(
            llama_model, 
            prompt_template_str=prompt_template_str, 
            template_vars=["informal_name"]
        )
        wrapper = MLflowPipelineWrapper(pipeline, pipeline_type="llm")
        mlflow.pyfunc.log_model(
            artifact_path="llm_model",
            python_model=wrapper
        )
        run_id = run.info.run_id 

    model_uri = f"runs:/{run_id}/llm_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    assert loaded_model is not None
