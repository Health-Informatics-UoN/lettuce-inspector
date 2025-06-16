import pandas as pd 
import pytest 
import mlflow 
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from components.models import LLMModel, local_models
from integrations.mlflow.evaluation_runner import (
    MLflowEvaluationRunner
)
from integrations.mlflow.custom_metrics import make_simple_string_comparison_metrics
from evaluation.pipelines import LLMPipeline 


@pytest.fixture(scope="session") 
def llama_model(
    model_name: str = LLMModel.TINYLLAMA_1_1B_CHAT.value, 
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



def test_log_pipeline(tmp_path, llama_model, prompt_template_str): 
    eval_df = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "targets": ["acetaminophen", "codeine"]
    })

    experiment_name = "Pipeline logging test"
    tracking_uri = tmp_path.as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    runner = MLflowEvaluationRunner(
        experiment_name=experiment_name, 
        tracking_uri=tracking_uri
    )

    pipeline = LLMPipeline(
        llama_model, 
        prompt_template_str, 
        template_vars=["informal_name"]
    )
    pipeline_name = "llm_test_pipeline_for_logging"
    pipeline_type = "llm"

    with mlflow.start_run() as run: 
        runner._log_pipeline(
            pipeline, 
            pipeline_name=pipeline_name, 
            pipeline_type=pipeline_type, 
            input_example=eval_df 
        )
    
    # Check experiment and run exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None

    run_id = run.info.run_id
    assert run_id is not None

    # Check pipeline artifacts directory was created 
    artifact_dir = tmp_path / experiment.experiment_id / run_id / "artifacts" 
    metrics_dir = tmp_path / experiment.experiment_id / run_id / "metrics"
    model_dir = artifact_dir / pipeline_name 

    assert artifact_dir.exists()
    assert artifact_dir.is_dir()
    assert metrics_dir.exists()
    assert metrics_dir.is_dir()
    assert model_dir.is_dir() 

    # Check expected files in model_dir 
    expected_files = ["requirements.txt", "python_env.yaml"]
    for fname in expected_files:
        assert (model_dir / fname).exists(), f"{fname} not found in {model_dir}"


def test_log_dataset(tmp_path): 
    eval_df = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "targets": ["acetaminophen", "codeine"]
    })
    eval_df.to_csv(tmp_path / "input_data.csv")

    experiment_name = "Dataset logging test"
    tracking_uri = tmp_path.as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    runner = MLflowEvaluationRunner(
        experiment_name=experiment_name, 
        tracking_uri=tracking_uri
    )
    
    with mlflow.start_run() as run: 
        runner._log_dataset(eval_df)
    
    # Check experiment and run exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None

    run_id = run.info.run_id
    assert run_id is not None

    inputs_dir = tmp_path / experiment.experiment_id / run_id / "inputs" 

    # Check inputs directory exists 
    assert inputs_dir.exists() 
    assert inputs_dir.is_dir() 

    # Check expected files in inputs_dir 
    expected_files = ["meta.yaml"]

    subdirs = [item for item in inputs_dir.iterdir() if item.is_dir()]
    assert len(subdirs) == 1, f"Expected one subdirectory in {inputs_dir}, found: {subdirs}"

    input_subdir = subdirs[0]

    for fname in expected_files:
        file_path = input_subdir / fname
        assert file_path.exists(), f"{fname} not found in {input_subdir}"
      

def test_evaluation_run(tmp_path, llama_model, prompt_template_str):
    eval_df = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "targets": ["acetaminophen", "codeine"]
    })
    eval_df.to_csv(tmp_path / "input_data.csv")

    pipeline = LLMPipeline(
        llama_model, 
        prompt_template_str, 
        template_vars=["informal_name"]
    )
    pipeline_name = "llm_test_pipeline_for_logging"
    pipeline_type = "llm"
    
    tracking_uri = tmp_path.as_uri()
    experiment_name = "test_evaluation_run"
    runner = MLflowEvaluationRunner(
        experiment_name=experiment_name, 
        tracking_uri=tracking_uri
    )

    run = runner.run_evaluation(
        pipeline, 
        pipeline_name, 
        pipeline_type,
        eval_df,
        metrics=make_simple_string_comparison_metrics()
    )

    # Check experiment and run exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None

    run_id = run.info.run_id
    assert run_id is not None

    # Check pipeline artifacts directory was created 
    artifact_dir = tmp_path / experiment.experiment_id / run_id / "artifacts" 
    metrics_dir = tmp_path / experiment.experiment_id / run_id / "metrics"
    model_dir = artifact_dir / pipeline_name 
    inputs_dir = tmp_path / experiment.experiment_id / run_id / "inputs" 

    assert artifact_dir.exists()
    assert artifact_dir.is_dir()
    assert metrics_dir.exists()
    assert metrics_dir.is_dir()
    assert model_dir.is_dir() 
    assert inputs_dir.exists() 
    assert inputs_dir.is_dir() 

    # Check expected files in model_dir 
    expected_files = ["requirements.txt", "python_env.yaml"]
    for fname in expected_files:
        assert (model_dir / fname).exists(), f"{fname} not found in {model_dir}"

    # Check expected files in inputs_dir 
    expected_files = ["meta.yaml"]

    subdirs = [item for item in inputs_dir.iterdir() if item.is_dir()]
    assert len(subdirs) == 1, f"Expected one subdirectory in {inputs_dir}, found: {subdirs}"

    input_subdir = subdirs[0]

    for fname in expected_files:
        file_path = input_subdir / fname
        assert file_path.exists(), f"{fname} not found in {input_subdir}"

    # Check expected files in metrics dir 
    metric_names = ['ExactMatch', 'UncasedMatch', 'FuzzyMatchRatio']
    expected_files = ["mean", "variance", "median"]
    for metric_name in metric_names: 
        for fname in expected_files: 
            file_path = metrics_dir / metric_name / fname
            assert file_path.exists(), f"{fname} not found in {metrics_dir}"

