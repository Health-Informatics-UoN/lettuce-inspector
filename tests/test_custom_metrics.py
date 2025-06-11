import pandas as pd 
import mlflow 
import pytest 
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from components.models import LLMModel, local_models
from evaluation.pipelines import LLMPipeline 
from integrations.mlflow.custom_metrics import exact_match_metric 
from integrations.mlflow.pipeline_wrapper import MLflowPipelineWrapper 


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

    Informal name: {{informal_name}}"""


def test_exact_match_metric_success(): 
    eval_df = pd.DataFrame({
        "predictions": ["hello", "yolo"], 
        "targets": ["hello", "lol"]
    })
    metric_value = exact_match_metric.eval_fn(eval_df["predictions"], eval_df["targets"])
    
    assert isinstance(metric_value.scores, list)
    assert "mean" in metric_value.aggregate_results


def test_exact_match_metric_with_mlflow_evaluate(
    tmp_path, 
    llama_model, 
    prompt_template_str
): 
    eval_df = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "targets": ["acetaminophen", "codeine"]
    })

    tracking_uri = tmp_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("eval_exact_match_test")

    pipeline = LLMPipeline(
        llama_model, 
        prompt_template_str, 
        template_vars=["informal_name"]
    )
    model_path = tmp_path / "my_model"
    mlflow.pyfunc.save_model(
        path=str(model_path),
        python_model=MLflowPipelineWrapper(pipeline, pipeline_type="llm")
    )

    with mlflow.start_run() as run: 
        try: 
            result = mlflow.evaluate(
                model=str(model_path),         
                data=eval_df,                 
                targets="targets",               
                extra_metrics=[exact_match_metric], 
                evaluator_config={
                    "col_mapping": {
                        "predictions": "predictions", 
                        "targets": "targets"          
                    }
                }
            )
            run_id = run.info.run_id
        except Exception as e: 
            print(f"FULL ERROR: {str(e)}")
            print(f"ERROR TYPE: {type(e)}")
            import traceback
            print("FULL TRACEBACK:")
            traceback.print_exc()
            raise  

    print("Evaluation results table:")
    print(result.tables["eval_results_table"].to_string())
    
    assert "exact_match_eval_fn/mean" in result.metrics
    assert "exact_match_eval_fn/variance" in result.metrics
    assert "exact_match_eval_fn/median" in result.metrics