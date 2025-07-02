# MLflow Evaluation Integration 
This integration provides a way to track and evaluate `lettuce` pipelines using MLflow. 

## Key Features of MLflow 
- **Experiment tracking**: Log pipeline configurations, metrics, and artifacts. 
- **Model registry**: Version and deploy your pipelines as MLflow models. 
- **Standardised evaluation**: Use custom metrics alongside MLflow's built-in metrics. 
- **Configuration management**: YAML-based configuration for experiments. 
- **Visualisation**: Automatic generation of metric distribution plots.  

## Quickstart 
### Basic example (see `examples/mlflow_integration.py`)
#### 1. Load evaluation data
```python
eval_df = pd.read_csv("evaluation_data.csv")
eval_df = eval_df.rename(columns={"expected_output": "targets"})
```

#### 2. Define configuration (LLM only pipeline)
```python
llm_pipeline_config = LLMPipelineConfig(
    llm=LLMConfig(
        model_name="your_model_name", 
        context_length = 2048, 
        batch_size = 512,
        temperature=0.0,
        max_tokens=256, 
        threads = 4 
    ),
    prompt_template="Your prompt template: {{informal_name}}",
    template_vars=["informal_name"]
)
```

#### 3. Create pipeline
```python
pipeline = MLflowRAGPipeline(rag_pipeline_config)
```

#### 4. Run the evaluation 
```python
runner = MLflowEvaluationRunner(
    experiment_name="my_experiment",
    tracking_uri="./mlruns"
)

runner.run_evaluation(
    pipeline=pipeline,
    pipeline_type="llm",
    pipeline_name="my_llm_pipeline",
    eval_df=eval_df,
    metrics=make_simple_string_comparison_metrics()
)
```

#### 5. View results 
After running the evaluation: 
```bash
mlflow ui --backend-store-uri ./mlruns 
```
Navigate to http://localhost:5000 to view your experiments. 

## Architecture of `integrations/mlflow`
Contains the components for the interface between `lettuce-inspector` and `mlflow`. 

### `config.py`
Dataclasses for managing pipeline configurations. 

### `pipeline_wrapper.py`
MLflow compatible wrappers for different pipeline types. 

### `evaluation_runner.py` 
Main orchestrator for running evaluations. 

### `custom_metrics.py`
Adapter for using Lettuce metrics with MLflow. 

### `plotting.py`
Example plots which can be logged in the `log_metric_distribution_figures`. 

### `utils.py`
Utility functions for the `mlflow` integration. 

## Future enhancements
- Implement all available metrics - use mlflow built-ins where possible 
- Also look at built-in options for metric calculation (F1, Recall, Precision already present in mlflow along with other experimental metrics)
- Future: Change model logging to the logging with code feature in mlflow 
- Fix other examples that use Jinja templates as input into the model pipelines
- Update log_dataset so the raw .csv file is optionally logged - not sure if this is a good idea for larger datasets 
- If time look at the database session init logic (context manager?)
- Have cleanup method for mlflow pipeline wrappers 