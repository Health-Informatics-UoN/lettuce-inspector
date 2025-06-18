"""
Wrapper around mlflow's PythonModel 
"""
from typing import Optional
from abc import abstractmethod
import pandas as pd 
import mlflow

from components.models import local_models
from omop.omop_queries import query_vector
from integrations.mlflow.config import RAGPipelineConfig 


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
            verbose=True,
            temperature=llm_config.temperature
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

    @abstractmethod 
    def _process_single_input(self, search_term: str): 
        pass 
    

class MLflowLLMPipeline(MLflowBasePipeline): 
    """LLM pipeline that loads configuration from YAML"""

    def _initialise_from_config(self): 
        """Initialize all components from YAML configuration"""   
        if self._initialised:
            return
    
        from jinja2 import Environment

        self.jinja_env = jinja_env = Environment()
        self.prompt_template = jinja_env.from_string(self.config.prompt_template)
        self._build_llm()
        
        self._initialised = True
        print("LLM pipeline initialization complete!")

    def _process_single_input(self, search_term: str):
        prompt = self.prompt_template.render({self.config.template_vars[0]: search_term})
        reply = self.llm.create_completion(prompt=prompt)["choices"][0]["text"]
        return reply

class MLflowRAGPipeline(MLflowBasePipeline): 
    """RAG Pipeline that loads configuration from YAML"""
    
    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        super().__init__(config)

    def _initialise_from_config(self): 
        """Initialize all components from YAML configuration"""   
        if self._initialised:
            return
    
        from jinja2 import Environment

        self.jinja_env = jinja_env = Environment()
        self.prompt_template = jinja_env.from_string(self.config.prompt_template)
        self._build_llm()
        self._build_embedding_model()
        self._build_database_connection()
        
        self._initialised = True
        print("RAG pipeline initialization complete!")

    def _process_single_input(self, search_term: str): 
        embedding = self.embedding_model.encode(search_term)
        search_query = query_vector(
            embedding,
            embed_vocab=self.config.retrieval.vocab_ids, 
            standard_concept=self.config.retrieval.standard_concept, 
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
