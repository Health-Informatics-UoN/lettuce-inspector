"""
Wrapper around mlflow's PythonModel 
"""
from typing import Optional, List
from abc import abstractmethod
import pandas as pd 
import mlflow

from omop.omop_queries import query_vector
from integrations.mlflow.config import (
    LLMPipelineConfig, 
    EmbeddingPipelineConfig, 
    RAGPipelineConfig 
)


class ComponentBuilder: 
    """Helper class for building pipeline components"""

    @staticmethod
    def build_llm(llm_config):
        """Build LLM from configuration"""
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from components.models import local_models
        
        if llm_config.model_path:
            model_path = llm_config.model_path
        else:
            model_path = hf_hub_download(**local_models[llm_config.model_name])
        
        return Llama(
            model_path=model_path,
            n_ctx=llm_config.context_length,
            n_batch=llm_config.batch_size,
            n_gpu_layers=-1,
            verbose=True
        )
    
    @staticmethod
    def build_embedding_model(embedding_config):
        """Build embedding model from configuration"""
        from sentence_transformers import SentenceTransformer
        
        return SentenceTransformer(
            embedding_config.model_name,
            device=embedding_config.device
        )

    @staticmethod
    def build_database_session(db_config):
        """Build database session from configuration"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.sql import text
        
        connection_string = db_config.get_connection_string()
        
        engine = create_engine(
            connection_string,
            pool_size=db_config.pool_size,
            pool_recycle=db_config.pool_recycle,
            echo=db_config.echo_sql
        )
        
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        
        # Test connection
        session.execute(text("SELECT 1"))
        
        return session, engine

    
class MLflowBasePipeline(mlflow.pyfunc.PythonModel): 
    """Base class for mlflow pipelines"""

    def __init__(self, config): 
        self.config = config 
        self.llm = None
        self.embedding_model = None
        self.session = None
        self._initialised = False 

    @abstractmethod 
    def _initialise_from_config(self): 
        pass 

    @abstractmethod 
    def _process_single_input(self, search_term: str): 
        pass 

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

      
class MLflowLLMPipeline(MLflowBasePipeline): 
    """LLM-only pipeline for MLflow"""

    def __init__(self, config: LLMPipelineConfig):
        super().__init__(config)


    def _initialise_from_config(self): 
        """Initialize all components from YAML configuration"""   
        if self._initialised:
            return
    
        from jinja2 import Environment

        self.jinja_env = Environment()
        self.prompt_template = self.jinja_env.from_string(self.config.prompt_template)
        self.llm = ComponentBuilder.build_llm(self.config.llm)
        
        self._initialised = True

    def _process_single_input(self, search_term: str):
        prompt = self.prompt_template.render({self.config.template_vars[0]: search_term})
        response = self.llm.create_completion(
            prompt=prompt, 
            max_tokens=self.config.llm.max_tokens, 
            temperature=self.config.llm.temperature 
        )
        return response["choices"][0]["text"]


class MLflowEmbeddingPipeline(MLflowBasePipeline):
    """Embedding-only pipeline for MLflow"""
    
    def __init__(self, config: EmbeddingPipelineConfig):
        super().__init__(config)

    def _initialise_components(self):
        """Initialize embedding model"""
        self.embedding_model = ComponentBuilder.build_embedding_model(self.config.embedding)
    
    def _process_single_input(self, search_term: str) -> List[float]:
        """Generate embedding for single input"""
        embedding = self.embedding_model.encode(
            search_term,
            normalize_embeddings=self.config.embedding.normalize_embeddings
        )
        return embedding.tolist()


class MLflowRAGPipeline(MLflowBasePipeline): 
    """RAG Pipeline that loads configuration from YAML"""
    
    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        super().__init__(config)

    def _initialise_from_config(self): 
        """Initialise all components from YAML configuration"""   
        if self._initialised:
            return
    
        from jinja2 import Environment

        self.jinja_env = jinja_env = Environment()
        self.prompt_template = jinja_env.from_string(self.config.prompt_template)

        self.llm = ComponentBuilder.build_llm(self.config.llm)
        self.embedding_model = ComponentBuilder.build_embedding_model(self.config.embedding)

        session, engine = ComponentBuilder.build_database_session(self.config.database)
        self.session = session 
        self.engine = engine 
        
        self._initialised = True

    def _process_single_input(self, search_term: str): 
        embedding = self.embedding_model.encode(search_term)
        search_query = query_vector(
            embedding,
            embed_vocab=self.config.retrieval.vocab_ids, 
            standard_concept=self.config.retrieval.standard_concept, 
            n=self.config.retrieval.top_k
        )
        retrieved_vecs = self.session.execute(search_query).mappings().all()

        template_context = dict(zip(
            self.config.template_vars,
            [search_term, retrieved_vecs]
        ))

        prompt = self.prompt_template.render(template_context)
        
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature
        )
        
        return response["choices"][0]["text"]
