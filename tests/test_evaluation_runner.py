import numpy as np 
import pandas as pd 
import pytest 
import mlflow 

from components.models import LLMModel
from components.embeddings import get_embedding_model 
from integrations.mlflow.evaluation_runner import (
    MLflowEvaluationRunner, 
    plot_histogram, plot_boxplot, plot_violinplot
)
from integrations.mlflow.pipeline_wrapper import MLflowLLMPipeline, MLflowRAGPipeline
from integrations.mlflow.config import (
    LLMConfig, 
    EmbeddingConfig, 
    RetrievalConfig, 
    DatabaseConfig, 
    LLMPipelineConfig, 
    RAGPipelineConfig
)
from integrations.mlflow.custom_metrics import make_simple_string_comparison_metrics


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



def test_log_pipeline(tmp_path, prompt_template_str_simple): 
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

    config = LLMPipelineConfig(
        llm=LLMConfig(model_name=LLMModel.LLAMA_3_1_8B.value),
        prompt_template=prompt_template_str_simple, 
        template_vars = ["informal_name"]
    )
    pipeline = MLflowLLMPipeline(config)
    pipeline_name = "llm_test_pipeline_for_logging"
    pipeline_type = "llm"

    with mlflow.start_run() as run: 
        runner._log_pipeline(
            pipeline, 
            pipeline_name=pipeline_name, 
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
      

def test_evaluation_run_llm_pipeline(tmp_path, prompt_template_str_simple):
    eval_df = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "targets": ["acetaminophen", "codeine"]
    })
    eval_df.to_csv(tmp_path / "input_data.csv")

    config = LLMPipelineConfig(
        llm=LLMConfig(model_name=LLMModel.LLAMA_3_1_8B.value),
        prompt_template=prompt_template_str_simple, 
        template_vars = ["informal_name"]
    )
    pipeline = MLflowLLMPipeline(config)
    pipeline_name = "llm_test_pipeline_for_logging"
    pipeline_type = "llm"
    
    tracking_uri = tmp_path.as_uri()
    experiment_name = "test_evaluation_run"
    runner = MLflowEvaluationRunner(
        experiment_name=experiment_name, 
        tracking_uri=tracking_uri
    )

    run = runner.run_evaluation(
        pipeline=pipeline,
        pipeline_type=pipeline_type,
        pipeline_name=pipeline_name, 
        metrics=make_simple_string_comparison_metrics(),
        eval_df=eval_df
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


def test_evaluation_run_rag_pipeline(tmp_path, prompt_template_str_rag):
    eval_df = pd.DataFrame({
        "input_data": ["paracetamol", "codeine"], 
        "targets": ["acetaminophen", "codeine"]
    })
    eval_df.to_csv(tmp_path / "input_data.csv")

    config = RAGPipelineConfig(
        llm=LLMConfig(model_name=LLMModel.LLAMA_3_1_8B.value),
        embedding=EmbeddingConfig(model_name=get_embedding_model("BGESMALL").info.path),
        database=DatabaseConfig.from_env(),
        retrieval=RetrievalConfig(vocab_ids=["RxNorm"], standard_concept=True), 
        prompt_template=prompt_template_str_rag, 
        template_vars = ["informal_name", "vec_results"]
    ) 
    pipeline = MLflowRAGPipeline(config=config)
    pipeline_name = "rag_test_pipeline_for_logging"
    pipeline_type = "rag"
    
    tracking_uri = tmp_path.as_uri()
    experiment_name = "test_evaluation_run"
    runner = MLflowEvaluationRunner(
        experiment_name=experiment_name, 
        tracking_uri=tracking_uri
    )

    run = runner.run_evaluation(
        pipeline=pipeline,
        pipeline_type=pipeline_type,
        pipeline_name=pipeline_name, 
        metrics=make_simple_string_comparison_metrics(),
        eval_df=eval_df
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
