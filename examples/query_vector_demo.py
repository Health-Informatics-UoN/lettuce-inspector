import duckdb
from sentence_transformers import SentenceTransformer
from time import time

from omop.db_manager import db_session
from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework, InformationRetrievalPipelineTest
from evaluation.metrics import IDMatchInList, RelatedIDPrecision, RelatedIDRecall
from evaluation.pipelines import DBRankRetrievalPipeline, EmbeddingsRetrievalPipeline
from query_handler.duckdb_connectors import connect_to_concept_csv, connect_to_vector_parquet
from query_handler.vectors import ParquetFileVectorSearcher, ReciprocalRankFusionSearcher
from query_handler.text_query import BmFileSearcher

description = """
Testing different types of retrievers at k=10

Dataset: NUH drugs
Vector search:
    model: neuml/pubmedbert-base-embeddings
    available vocabularies: RxNorm, RxNorm Extension
    string representation: longer

Pipelines:
    Vector search: Cosine Similarity Search
    Reciprocal rank fusion: RRF with BM25 and Vector search scores
"""

dataset = SingleInputCSVforLLM("path/to/data.csv")

vocabularies = ["RxNorm", "RxNorm Extension"]

print("Connecting to evaluation database...")
eval_conn = db_session()

print("Initialising model...")
model_init = time()
bge_small = SentenceTransformer("BAAI/bge-small-en-v1.5")
print(f"Loaded model in {(time() - model_init):.2f} seconds")

print("Initialising retrieval database...")
db_init = time()
db = duckdb.connect()

connect_to_concept_csv(db, "path/to/CONCEPT.csv")
connect_to_vector_parquet(db, "path/to/bge_embeddings.parquet")
print(f"Data loaded in {(time() - db_init):.2f} seconds")

print("Loading retrievers...")
pq = ParquetFileVectorSearcher(
        db=db,
        model=bge_small,
        vocabulary_ids=vocabularies,
        vector_dimension= 384,
        )

# bm25 = BmFileSearcher(db=db, vocabulary_ids=vocabularies)

rrf = ReciprocalRankFusionSearcher(
        db=db,
        model=bge_small,
        vocabulary_ids=vocabularies,
        vector_dimension= 384,
        )
print("Retrievers loaded")

vs_pipeline = EmbeddingsRetrievalPipeline(
        embedding_model=bge_small,
        retriever=pq
        )
print("Vector search pipeline initialised")

# bm25_pipeline = DBRankRetrievalPipeline(retriever=bm25)
# print("BM25 search pipeline initialised")

rrf_pipeline = EmbeddingsRetrievalPipeline(
        embedding_model=bge_small,
        retriever=rrf
        )
print("RRF search pipeline initialised")

metrics = [
        RelatedIDPrecision(connection = eval_conn, vocabulary_ids=vocabularies),
        RelatedIDRecall(connection=eval_conn, vocabulary_ids=vocabularies),
        IDMatchInList(connection = eval_conn, vocabulary_ids=vocabularies)
        ]

tests = [
        # InformationRetrievalPipelineTest(
        #     name="BM25",
        #     pipeline=bm25_pipeline,
        #     metrics=metrics,
        #     ),
        InformationRetrievalPipelineTest(
            name="Embeddings (longer embeddings, bge-small-en-v1.5)",
            pipeline=vs_pipeline,
            metrics=metrics,
            ),
        InformationRetrievalPipelineTest(
            name="Reciprocal Rank Fusion (longer embeddings, bge-small-en-v1.5)",
            pipeline=rrf_pipeline,
            metrics=metrics,
            )
        ]

framework = EvaluationFramework(
        name="Retriever comparison - k=10",
        pipeline_tests=tests,
        dataset=dataset,
        description=description,
        )

if __name__ == "__main__":
    framework.run_evaluations()
