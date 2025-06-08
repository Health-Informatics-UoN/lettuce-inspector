import mlflow 
from mlflow.models import infer_signature

from integrations.mlflow.pipeline_wrapper import MLflowPipelineWrapper 


class MLflowLogger(): 
    def __init__(self, experiment_name: str, tracking_uri: str):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri 
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_model(self, model: MLflowPipelineWrapper): 
        mlflow.pyfunc.log_model(
            artifact_path="llm_model",
            python_model=model
        )

    def log_dataset(self): 
        pass 

    def log_metrics(self): 
        pass 

    def log_evaluation_run(self): 
        with mlflow.start_run() as run: 
            model = MLflowPipelineWrapper(pipeline, pipeline_type)
            signature = infer_signature(X_test, model.predict(X_test))

            self.log_dataset()

            self.log_model()

            self.log_metrics()
