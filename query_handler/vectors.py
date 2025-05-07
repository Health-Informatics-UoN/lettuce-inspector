import os
import duckdb
from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import List

from query_handler.duckdb_connectors import connect_to_concept_csv, connect_to_vector_parquet
from query_handler.duckdb_queries import rrf_query, vector_search, SimilarityFunction
from query_handler.handler_type import ConceptIDQueryHandler


class ParquetFileVectorSearcher(ConceptIDQueryHandler):
    def __init__(self,
                 file_path: str,
                 model: SentenceTransformer,
                 similarity_function: SimilarityFunction=SimilarityFunction.COSINE_DISTANCE,
                 top_k: int=10,
                 vector_dimension: int=384,
                 vector_type: str="DOUBLE"
                 ) -> None:
        self._path = os.path.expanduser(file_path)
        self._model = model
        self._sim = similarity_function
        self._top_k = top_k
        self._vector_dim = vector_dimension
        self._vector_type = vector_type
        self._db = self.db_config()
    
    def db_config(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        connect_to_vector_parquet(con, self._path)
        return con

    def search(self, queries: List[str] | str) -> List[List[int]]:
        if isinstance(queries, str):
            queries = [queries]
        
        results = []
        for query in queries:
            embedding = self._model.encode(query)
            result = vector_search(
                    con=self._db,
                    similarity_function=self._sim,
                    embedding=embedding,
                    vector_type=self._vector_type,
                    vector_dim=self._vector_dim,
                    top_k=self._top_k,
                    ).fetchall()
            concept_ids = [r[0] for r in result]
            results.append(concept_ids)
        
        return results

class ReciprocalRankFusionSearcher(ConceptIDQueryHandler):
    def __init__(self,
                 concept_table_file_path: str,
                 vector_file_path: str,
                 model: SentenceTransformer,
                 similarity_function: SimilarityFunction=SimilarityFunction.COSINE_DISTANCE,
                 top_k: int=10,
                 vector_dimension: int=384,
                 vector_type: str="DOUBLE"
                 ) -> None:
        self._concept_path = os.path.expanduser(concept_table_file_path)
        self._vec_path = os.path.expanduser(vector_file_path)
        self._model = model
        self._sim = similarity_function
        self._top_k = top_k
        self._vector_dim = vector_dimension
        self._vector_type = vector_type
        self._db = self.db_config()
    
    def db_config(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        connect_to_vector_parquet(con, self._vec_path)
        connect_to_concept_csv(con, self._concept_path)
        con.sql("""
                CREATE OR REPLACE MACRO rrf(rank, k:=60) AS
                coalesce((1 / (k + rank)), 0)
                """)
        return con

    def search(self, queries: List[str] | str) -> List[List[int]]:
        if isinstance(queries, str):
            queries = [queries]
        
        results = []
        for query in queries:
            embedding = self._model.encode(query)
            result = rrf_query(
                    con=self._db,
                    similarity_function=self._sim,
                    embedding=embedding,
                    vector_type=self._vector_type,
                    vector_dim=self._vector_dim,
                    query=query,
                    rrf_top_k=self._top_k,
                    ).fetchall()
            concept_ids = [r[0] for r in result]
            results.append(concept_ids)
        
        return results
