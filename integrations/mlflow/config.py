import yaml 
import os 
from dataclasses import dataclass, asdict 
from typing import Dict, Any, Optional, List 
from pathlib import Path 


@dataclass
class LLMConfig:
    """Configuration for LLM initialization"""
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    context_length: int = 2048
    batch_size: int = 512
    temperature: float = 0.0
    max_tokens: int = 256
    threads: int = 4

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if not self.model_path and not self.model_name:
            raise ValueError("Either 'model_path' or 'model_name' must be provided.")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 32
    
    def __post_init__(self):
        """Validate embedding configuration"""
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError("Device must be 'cpu', 'cuda', or 'mps'")
        

@dataclass
class DatabaseConfig:
    """Configuration for database connection"""
    host: str
    port: int
    database: str
    username: str
    password: str 
    schema: str 
    vectable: str 
    vecsize: int 
    pool_size: int = 5
    pool_recycle: int = 3600
    echo_sql: bool = False
    
    @classmethod
    def from_env(cls, prefix: str = "DB_") -> 'DatabaseConfig':
        """Create from environment variables - evaluated at call time"""
        return cls(
            host=os.getenv(f'{prefix}HOST', 'localhost'),
            port=int(os.getenv(f'{prefix}PORT', '5432')),
            database=os.getenv(f'{prefix}NAME', 'omop'),
            username=os.getenv(f'{prefix}USER', 'postgres'),
            password=os.getenv(f'{prefix}PASSWORD', 'password'),
            schema=os.getenv(f'{prefix}SCHEMA', 'cdm'), 
            vectable=os.getenv(f'{prefix}VECTABLE', "embeddings"), 
            vecsize=int(os.getenv(f'{prefix}VECSIZE', '384')) 
        )

    def get_connection_string(self) -> str:
        """Build connection string with password from env if needed"""
        password = self.password or os.getenv('DB_PASSWORD')
        if not password:
            raise ValueError("Database password must be provided via config or DB_PASSWORD env var")
        
        return f"postgresql://{self.username}:{password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    max_distance: float = 1.0
    rerank: bool = False


@dataclass
class RAGPipelineConfig:
    """Complete RAG pipeline configuration"""
    llm: LLMConfig
    embedding: EmbeddingConfig
    database: DatabaseConfig
    retrieval: RetrievalConfig
    prompt_template: str
    template_vars: List[str]
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'RAGPipelineConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            llm=LLMConfig(**data['llm']),
            embedding=EmbeddingConfig(**data['embedding']),
            database=DatabaseConfig(**data['database']),
            retrieval=RetrievalConfig(**data.get('retrieval', {})),
            prompt_template=data['prompt_template'],
            template_vars=data['template_vars'],
            description=data.get('description', '')
        )
    
    def validate(self) -> bool:
        """Validate the entire configuration"""
        try:
            # Validation happens in __post_init__ methods
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
        