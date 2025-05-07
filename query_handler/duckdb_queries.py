import duckdb
from enum import Enum
from torch import Tensor

class SimilarityFunction(Enum):
    EUCLIDEAN = ("array_distance", "ASC")  # Lower distance is better
    COSINE_DISTANCE = ("array_cosine_distance", "ASC")  # Lower distance is better
    COSINE_SIMILARITY = ("array_cosine_similarity", "DESC")  # Higher similarity is better
    INNER = ("array_negative_inner_product", "ASC")  # Lower negative product is better

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
    function_name, direction = similarity_function.value
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
                    ),
                    ranks AS (
                        SELECT
                            fts.concept_id,
                            rank() OVER (ORDER BY fts.score DESC) as fts_rank,
                            rank() OVER (ORDER BY vs.score {direction}) as vs_rank
                        FROM fts
                        FULL OUTER JOIN vs ON fts.concept_id = vs.concept_id
                    )
                    SELECT
                        concept_id,
                        rrf(fts_rank) + rrf(vs_rank) AS rrf_score
                    FROM ranks
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
