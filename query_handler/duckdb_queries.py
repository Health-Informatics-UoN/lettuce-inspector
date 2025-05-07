import duckdb
from enum import Enum
from torch import Tensor

class SimilarityFunction(Enum):
    EUCLIDEAN = ("array_distance", "ASC")  # Lower distance is better
    COSINE_DISTANCE = ("array_cosine_distance", "ASC")  # Lower distance is better
    COSINE_SIMILARITY = ("array_cosine_similarity", "DESC")  # Higher similarity is better
    INNER = ("array_negative_inner_product", "ASC")  # Lower negative product is betterrom query_handler.vectors import SimilarityFunction

def vector_search(
        con: duckdb.DuckDBPyConnection,
        similarity_function: SimilarityFunction,
        embedding: Tensor,
        vector_type: str,
        vector_dim: int,
        top_k: int,
        ) -> duckdb.DuckDBPyRelation:
    function_name, direction = similarity_function.value
    return con.sql(f"""
                   SELECT concept_id,
                          description,
                          {function_name}(embeddings::{vector_type}[{vector_dim}], 
                                           $embedding::{vector_type}[{vector_dim}]) as score
                   FROM vectors
                   ORDER BY score {direction}
                   LIMIT $top_k;
                   """,
                   params={
                        "embedding": embedding.tolist(),
                        "top_k": top_k,
                       }
                   )

def bm25_query(con: duckdb.DuckDBPyConnection, query: str, top_k: int,) -> duckdb.DuckDBPyRelation:
    return con.sql("""
                   SELECT concept_id, concept_name, score
                   FROM (
                       SELECT *, fts_main_concepts.match_bm25(
                           concept_id,
                           $query,
                           fields := 'concept_name'
                       ) AS score
                       FROM concepts
                   ) sq
                   WHERE score IS NOT NULL
                   ORDER BY score DESC
                   LIMIT $top_k;
                   """,
                   params={
                       "query": query,
                       "top_k": top_k
                       }
                   )

def rrf_query(
        con: duckdb.DuckDBPyConnection,
        similarity_function: SimilarityFunction,
        embedding: Tensor,
        vector_type: str,
        vector_dim: int,
        query: str,
        rrf_top_k: int,
        ) -> duckdb.DuckDBPyRelation:
    function_name, _ = similarity_function.value
    return con.sql(f"""
                    WITH fts AS (
                        SELECT concept_id, fts_main_concepts.match_bm25(
                            concept_id,
                            $query,
                            fields := 'concept_name'
                        ) AS score
                        FROM concepts
                    ),
                    vs AS (
                        SELECT concept_id,
                        {function_name}(
                            embeddings::{vector_type}[{vector_dim}],
                            $embedding::{vector_type}[{vector_dim}]
                            ) AS score
                            FROM vectors
                    )
                    SELECT
                        fts.concept_id,
                        fts.score AS fts_score,
                        vs.score AS vs_score,
                        rrf(fts.score) + rrf(vs.score) AS rrf_score
                    FROM fts
                    INNER JOIN vs ON fts.concept_id = vs.concept_id
                    ORDER BY rrf_score DESC
                    LIMIT $rrf_top_k
                    """,
                   params={
                       "query": query,
                       "embedding": embedding,
                       "rrf_top_k": rrf_top_k
                       }
                   )
# Can try convex normalisation as https://motherduck.com/blog/search-using-duckdb-part-3/
