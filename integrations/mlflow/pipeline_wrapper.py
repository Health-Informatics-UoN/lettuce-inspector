"""
Wrapper around mlflow's PythonModel 
"""
from typing import Optional
import pandas as pd 
import mlflow

from components.models import local_models
from omop.omop_queries import query_vector
from integrations.mlflow.config import RAGPipelineConfig 


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


class MLflowRAGPipeline(mlflow.pyfunc.PythonModel): 
    """RAG Pipeline that loads configuration from YAML"""
    
    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        self.config: Optional[RAGPipelineConfig] = config 
        self.llm = None
        self.embedding_model = None
        self.session = None
        self._initialised = False 

    def _initialise_from_config(self): 
        """Initialize all components from YAML configuration"""   
        if self._initialised:
            return
    
        from jinja2 import Environment
        
        print("Initializing RAG pipeline from YAML configuration...")

        self.jinja_env = jinja_env = Environment()
        self.prompt_template = jinja_env.from_string(self.config.prompt_template)
        self._build_llm()
        self._build_embedding_model()
        self._build_database_connection()
        
        self._initialised = True
        print("RAG pipeline initialization complete!")
    
    def _build_llm(self): 
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        
        self.llm = Llama(
            model_path=hf_hub_download(**local_models[self.config.llm.model_name]),
            n_ctx=self.config.llm.context_length,
            n_batch=self.config.llm.batch_size,
            n_gpu_layers=-1,
            verbose=True
        )

    def _build_embedding_model(self): 
        """Build embedding model from YAML config"""
        from sentence_transformers import SentenceTransformer
        
        emb_config = self.config.embedding
        self.embedding_model = SentenceTransformer(
            emb_config.model_name,
            device=emb_config.device
        )

    def _build_database_connection(self): 
        """Build database connection from YAML config"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.sql import text 
        
        db_config = self.config.database
        connection_string = db_config.get_connection_string()
        
        self.db_engine = create_engine(
            connection_string,
            pool_size=db_config.pool_size,
            pool_recycle=db_config.pool_recycle,
            echo=db_config.echo_sql
        )
        
        SessionLocal = sessionmaker(bind=self.db_engine)
        self.session = SessionLocal()
        
        # Test connection
        self.session.execute(text("SELECT 1")) 

    def predict(self, model_input: pd.DataFrame, params=None): 
        if "input_data" not in model_input.columns: 
            raise ValueError("The column input_data must be present")
        
        self._initialise_from_config()

        predictions = []
        for _, row in model_input.iterrows(): 
            input_data = row["input_data"]
            if not isinstance(input_data, str): 
                raise TypeError("Search terms in input_data must be strings!")
            prediction = self._process_single_input(search_term=input_data)
            predictions.append(prediction)

        return pd.DataFrame({"predictions": predictions})


    def _process_single_input(self, search_term: str): 
        embedding = self.embedding_model.encode(search_term)
        search_query = query_vector(
            embedding,
            embed_vocab=["RxNorm"], 
            standard_concept=True, 
            n=self.config.retrieval.top_k
        )
        retrieved_vecs = self.session.execute(search_query).mappings().all()

        template_context = {
            "informal_name": search_term,
            "vec_results": retrieved_vecs
        }
        prompt = self.prompt_template.render(template_context)
        
        reply = self.llm.create_completion(prompt=prompt)["choices"][0]["text"]
        return reply
