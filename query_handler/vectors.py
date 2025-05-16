import duckdb
from sentence_transformers import SentenceTransformer
from typing import List

from query_handler.duckdb_queries import rrf_query, vector_search, SimilarityFunction
from query_handler.handler_type import ConceptIDQueryHandler


class ParquetFileVectorSearcher(ConceptIDQueryHandler):
    def __init__(self,
                 db: duckdb.DuckDBPyConnection,
                 model: SentenceTransformer,
                 similarity_function: SimilarityFunction=SimilarityFunction.COSINE_DISTANCE,
                 vocabulary_ids: List[str] | None=None,
                 top_k: int=10,
                 vector_dimension: int=384,
                 vector_type: str="DOUBLE"
                 ) -> None:
        self._model = model
        self._sim = similarity_function
        self._vocabulary_ids = vocabulary_ids
        self._top_k = top_k
        self._vector_dim = vector_dimension
        self._vector_type = vector_type
        self._db = db
    
    def search(self, queries: List[str] | str) -> List[List[int]]:
        if isinstance(queries, str):
            queries = [queries]
        
        results = []
        for query in queries:
            embedding = self._model.encode(query)
            if len(embedding.shape) > 1:
                embedding = embedding[0]
            result = vector_search(
                    con=self._db,
                    similarity_function=self._sim,
                    embedding=embedding,
                    vector_type=self._vector_type,
                    vector_dim=self._vector_dim,
                    vocabulary_ids=self._vocabulary_ids,
                    top_k=self._top_k,
                    ).fetchall()
            concept_ids = [r[0] for r in result]
            results.append(concept_ids)
        
        return results

class ReciprocalRankFusionSearcher(ConceptIDQueryHandler):
    def __init__(self,
                 db: duckdb.DuckDBPyConnection,
                 model: SentenceTransformer,
                 similarity_function: SimilarityFunction=SimilarityFunction.COSINE_DISTANCE,
                 vocabulary_ids: List[str] | None=None,
                 top_k: int=10,
                 vector_dimension: int=384,
                 vector_type: str="DOUBLE"
                 ) -> None:
        self._model = model
        self._sim = similarity_function
        self._vocabulary_ids = vocabulary_ids
        self._top_k = top_k
        self._vector_dim = vector_dimension
        self._vector_type = vector_type
        self._db = db
        self.db_config()
    
    def db_config(self) -> None:
        self._db.sql("""
                CREATE OR REPLACE MACRO rrf(rank, k:=60) AS
                coalesce((1 / (k + rank)), 0)
                """)

    def search(self, queries: List[str] | str) -> List[List[int]]:
        if isinstance(queries, str):
            queries = [queries]
        
        results = []
        for query in queries:
            embedding = self._model.encode(query)
            if len(embedding.shape) > 1:
                embedding = embedding[0]
            result = rrf_query(
                    con=self._db,
                    similarity_function=self._sim,
                    embedding=embedding,
                    vocabulary_ids=self._vocabulary_ids,
                    vector_type=self._vector_type,
                    vector_dim=self._vector_dim,
                    query=query,
                    rrf_top_k=self._top_k,
                    ).fetchall()
            concept_ids = [r[0] for r in result]
            results.append(concept_ids)
        
        return results
