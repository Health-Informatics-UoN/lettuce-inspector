from enum import Enum
import os
import duckdb
from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import List
from .handler_type import ConceptIDQueryHandler

class SimilarityFunction(Enum):
    # Changed to use consistent naming of functions and directions
    EUCLIDEAN = ("array_distance", "ASC")  # Lower distance is better
    COSINE_DISTANCE = ("array_cosine_distance", "ASC")  # Lower distance is better
    COSINE_SIMILARITY = ("array_cosine_similarity", "DESC")  # Higher similarity is better
    INNER = ("array_negative_inner_product", "ASC")  # Lower negative product is better


class ParquetFileVectorReader(ConceptIDQueryHandler):
    def __init__(self,
                 file_path: str,  # Changed to str for better type hinting
                 model: SentenceTransformer,
                 similarity_function: SimilarityFunction=SimilarityFunction.COSINE_DISTANCE,  # Changed default
                 top_k: int=10,
                 vector_dimension: int=384,  # Added dimension parameter
                 vector_type: str="DOUBLE"  # Added type parameter (FLOAT vs DOUBLE)
                 ) -> None:
        self._path = os.path.expanduser(file_path)  # Handle ~ in paths
        self._model = model
        self._sim = similarity_function
        self._top_k = top_k
        self._vector_dim = vector_dimension
        self._vector_type = vector_type
        self._db = self.db_config()
    
    def db_config(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.sql(f"""INSTALL vss;
        LOAD vss;

        CREATE TABLE vectors AS
        SELECT *
        FROM '{self._path}'""")
        return con

    def vss_query(self, embedding: Tensor):
        function_name, direction = self._sim.value
        return f"""
    SELECT concept_id, description
    FROM vectors
    ORDER BY {function_name}(embeddings::{self._vector_type}[{self._vector_dim}], 
                            {embedding.tolist()}::{self._vector_type}[{self._vector_dim}]) {direction}
    LIMIT {self._top_k};
    """

    def search(self, queries: List[str] | str) -> List[List[int]]:
        if isinstance(queries, str):
            queries = [queries]
        
        results = []
        for query in queries:
            embedding = self._model.encode(query)
            result = self._db.sql(self.vss_query(embedding)).fetchall()
            concept_ids = [r[0] for r in result]
            results.append(concept_ids)
        
        return results
