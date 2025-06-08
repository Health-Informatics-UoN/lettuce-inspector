"""
Wrapper around mlflow's PythonModel 
"""
import pandas as pd 
import mlflow


class MLflowPipelineWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper to make lettuce pipelines compatible with MLflow's model interface. 
    """

    def __init__(self, pipeline, pipeline_type):
        super().__init__()
        self.pipeline = pipeline 
        self.pipeline_type = pipeline_type 

    def predict(self, model_input: pd.DataFrame, params=None):
        """
        Predict method expected by MLflow. 

        Args:
            model_input: DataFrame with input data 

        Returns:
            DataFrame with predictions 
        """
        predictions = []

        if "input_data" not in model_input.columns: 
            raise ValueError("The column input_data must be present")
        
        for _, row in model_input.iterrows(): 
            input_data = row["input_data"]
            if isinstance(input_data, str): 
                input_data = [input_data]
            else: 
                raise TypeError("Search terms in input_data must be strings!")
            prediction = self.pipeline.run(input_data)
            predictions.append(prediction)

        return pd.DataFrame({"predictions": predictions})
