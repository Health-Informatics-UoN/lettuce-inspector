"""
Wrapper around mlflow's PythonModel 
"""
import pandas as pd 
import mlflow

from evaluation.evaltypes import SingleResultPipeline 


class MLflowPipelineWrapper(mlflow.pyfunc.PythonModel): 
    """
    Wrapper class around PythonModel. 
    """

    def __init__(self, pipeline: SingleResultPipeline):
        super().__init__()
        self.pipeline = pipeline 

    def predict(self, model_input: pd.DataFrame, params = None):
        """
        Override of the predict method. 
        
        Assumes that the model input will be a Pandas dataframe. 

        Args:
            model_input: pd.DataFrame 
                Input data, should have fields 'model_input' and 'expected_output'. 
        """

        predictions = []
        breakpoint()
        for X_, _ in model_input.iterrows():
            y_pred = self.pipeline.run(X_)
            predictions.append(y_pred)
    
        return pd.DataFrame(predictions, columns=["predictions"])
