import duckdb
from typing import List
from query_handler.duckdb_queries import bm25_query
from query_handler.handler_type import ConceptIDQueryHandler

# You might be wondering why I'm doing this in duckdb when I have an omop database already
# It's quite simple - postgres doesn't have an in-built BM25 function
class BmFileSearcher(ConceptIDQueryHandler):
    def __init__(self, db: duckdb.DuckDBPyConnection, top_k: int=10) -> None:
        self._db = db
        self._top_k = top_k

    def search(self, queries: List[str] | str) -> List[List[int]]:
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for query in queries:
            result = bm25_query(self._db, query, self._top_k).fetchall()
            concept_ids = [res[0] for res in result]
            results.append(concept_ids)
        
        return results
        
