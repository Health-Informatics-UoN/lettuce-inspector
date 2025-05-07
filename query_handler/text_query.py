import os
import duckdb
from typing import List
from query_handler.duckdb_queries import bm25_query
from query_handler.handler_type import ConceptIDQueryHandler
from query_handler.duckdb_connectors import connect_to_concept_csv

# You might be wondering why I'm doing this in duckdb when I have an omop database already
# It's quite simple - postgres doesn't have an in-built BM25 function
class BmFileSearcher(ConceptIDQueryHandler):
    def __init__(self, file_path: str, top_k: int=10) -> None:
        self._path = os.path.expanduser(file_path)
        self._db = self.db_config()
        self._top_k = top_k

    def db_config(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        connect_to_concept_csv(con, self._path)
        return con

    
    def search(self, queries: List[str] | str) -> List[List[int]]:
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for query in queries:
            result = bm25_query(self._db, query, self._top_k).fetchall()
            concept_ids = [res[0] for res in result]
            results.append(concept_ids)
        
        return results
        
