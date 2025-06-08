
import pandas as pd 
import pytest 
import mlflow 
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from integrations.mlflow.pipeline_wrapper import MLflowPipelineWrapper
from evaluation.pipelines import LLMPipeline
from components.models import local_models, LLMModel 


@pytest.fixture(scope="session") 
def llama_model(
    model_name: str = LLMModel.LLAMA_3_1_8B.value, 
    n_ctx: int = 512, 
    n_batch: int = 32, 
    max_tokens: int = 128
): 
    return Llama(
        hf_hub_download(**local_models[model_name]),
        n_ctx=n_ctx,
        n_batch=n_batch,
        model_kwargs={"n_gpu_layers": -1},
        generation_kwargs={"max_tokens": max_tokens, "temperature": 0}
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


def test_predict_llm_pipeline_wrapper(llama_model, prompt_template_str):     
    pipeline = LLMPipeline(
        llama_model, 
        prompt_template_str=prompt_template_str, 
        template_vars=["informal_name"]
    )
    wrapper = MLflowPipelineWrapper(pipeline, pipeline_type="llm")
    model_input = pd.DataFrame(
        {"input_data": ["paracetamol", "codeine"], 
         "expected_output": ["acetaminophen", "codeine"]}
    )
    predictions = wrapper.predict(model_input=model_input)
    breakpoint()


def test_error_thrown_if_input_data_not_present(): 
    pass 


def test_predict_rag_pipeline_wrapper(): 
    pass 


def test_pyfunc_model_logging(tmp_path): 
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

    logged_model_path = tmp_path / run_id / "artifacts" / "llm_model"
    assert logged_model_path.exists()
