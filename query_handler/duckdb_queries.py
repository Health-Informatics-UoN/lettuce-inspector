from typing import List
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
        vocabulary_ids: List[str] | None,
        top_k: int,
        ) -> duckdb.DuckDBPyRelation:
    if len(embedding) != vector_dim:
        raise ValueError(f"Expected embedding dimension {vector_dim}, got {len(embedding)}")
    function_name, direction = similarity_function.value

    if vocabulary_ids is not None:
        vocab_filter = """
        JOIN concepts ON vectors.concept_id = concepts.concept_id
        WHERE concepts.vocabulary_id IN $vocabulary_ids
        """
        query_params = {
                "embedding": embedding.tolist(),
                "top_k": top_k,
                "vocabulary_ids": vocabulary_ids
                }
    else:
        vocab_filter = ""
        query_params = {
                "embedding": embedding.tolist(),
                "top_k": top_k,
                }

    query = f"""
    SELECT vectors.concept_id,
           vectors.description,
           {function_name}(vectors.embeddings::{vector_type}[{vector_dim}], 
                            $embedding::{vector_type}[{vector_dim}]) as score
    FROM vectors
    {vocab_filter}
    ORDER BY score {direction}
    LIMIT $top_k;
    """

    return con.sql(query, params=query_params)

def bm25_query(
        con: duckdb.DuckDBPyConnection,
        query: str,
        vocabulary_ids: List[str] | None,
        top_k: int,
        ) -> duckdb.DuckDBPyRelation:
    if vocabulary_ids is not None:
        vocab_filter = "AND vocabulary_id IN $vocabulary_ids"
        query_params = {
                "query": query,
                "top_k": top_k,
                "vocabulary_ids": vocabulary_ids,
                }

    else:
        vocab_filter = ""
        query_params = {
                "query": query,
                "top_k": top_k
                }

    query = f"""
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
    {vocab_filter}
    ORDER BY score DESC
    LIMIT $top_k;
    """
    return con.sql(query, params=query_params)

def rrf_query(
        con: duckdb.DuckDBPyConnection,
        similarity_function: SimilarityFunction,
        embedding: Tensor,
        vector_type: str,
        vector_dim: int,
        query: str,
        vocabulary_ids: List[str] | None,
        rrf_top_k: int,
        ) -> duckdb.DuckDBPyRelation:
    function_name, direction = similarity_function.value
    
    if vocabulary_ids is not None:
        fts_vocab_filter = "WHERE vocabulary_id IN $vocabulary_ids"
        vs_vocab_filter = """
        JOIN concepts ON concepts.concept_id = vectors.concept_id
        WHERE concepts.vocabulary_id IN $vocabulary_ids
        """
        query_params = {
                "query": query,
                "embedding": embedding,
                "rrf_top_k": rrf_top_k,
                "vocabulary_ids": vocabulary_ids,
                }

    else:
        fts_vocab_filter = ""
        vs_vocab_filter = ""
        query_params = {
                "query": query,
                "embedding": embedding,
                "rrf_top_k": rrf_top_k,
                }


    query = f"""
    WITH fts AS (
        SELECT concept_id, fts_main_concepts.match_bm25(
            concept_id,
            $query,
            fields := 'concept_name'
        ) AS score
        FROM concepts
        {fts_vocab_filter}
    ),
    vs AS (
        SELECT vectors.concept_id,
        {function_name}(
            embeddings::{vector_type}[{vector_dim}],
            $embedding::{vector_type}[{vector_dim}]
            ) AS score
        FROM vectors
        {vs_vocab_filter}
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
        ranks.concept_id,
        rrf(ranks.fts_rank) + rrf(ranks.vs_rank) AS rrf_score
    FROM ranks
    ORDER BY rrf_score DESC
    LIMIT $rrf_top_k
    """
    return con.sql(query, params=query_params)
# Can try convex normalisation as https://motherduck.com/blog/search-using-duckdb-part-3/
