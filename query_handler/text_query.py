import os
import duckdb
from typing import List
from .handler_type import ConceptIDQueryHandler

class BmFileSearcher(ConceptIDQueryHandler):
    def __init__(self, file_path: str, top_k: int=10) -> None:
        self._path = os.path.expanduser(file_path)
        self._db = self.db_config()
        self._top_k = top_k
        
    def db_config(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.sql(f"""
        INSTALL fts;
        LOAD fts;
        
        CREATE TABLE concepts AS
        SELECT *
        FROM '{self._path}';
        """
        )
        con.sql("""
        PRAGMA create_fts_index('concepts', 'concept_id', 'concept_name');
        """)
        return con

    def bm25_query(self, query):
       return f"""
    SELECT concept_id, concept_name, score
    FROM (
        SELECT *, fts_main_concepts.match_bm25(
            concept_id,
            '{query}',
            fields := 'concept_name'
        ) AS score
        FROM concepts
    ) sq
    WHERE score IS NOT NULL
    ORDER BY score DESC
    LIMIT {self._top_k};
    """

    def search(self, queries: List[str] | str) -> List[List[int]]:
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for query in queries:
            result = self._db.sql(self.bm25_query(query)).fetchall()
            concept_ids = [res[0] for res in result]
            results.append(concept_ids)
        
        return results
        
