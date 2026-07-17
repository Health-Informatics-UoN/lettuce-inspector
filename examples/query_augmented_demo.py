from sentence_transformers import SentenceTransformer
from jinja2 import Environment
from time import time
import duckdb

from evaluation.evaltypes import EvaluationFramework
from options.pipeline_options import LLMModel
from components.models import local_models
from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.pipelines import AugmentedQueryPipeline
from evaluation.metrics import RelatedIDPrecision, RelatedIDRecall, IDMatchInList
from evaluation.evaltypes import InformationRetrievalPipelineTest
from query_handler.duckdb_connectors import connect_to_concept_csv, connect_to_vector_parquet
from query_handler.text_query import BmFileSearcher
from query_handler.vectors import ParquetFileVectorSearcher, ReciprocalRankFusionSearcher
from omop.db_manager import db_session
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

description = """
Testing different types of retrievers at k=10
Each pipeline takes the response of an LLM and searches for that

Dataset: First 400 HELIOS
Vector search:
    model: BAAI/bge-small-en-v1.5
    available vocabularies: RxNorm, RxNorm Extension
    string representation: short

Pipelines:
    BM25: Okapi BM25
"""

"""
    Vector search: Cosine Similarity Search
    Reciprocal rank fusion: RRF with BM25 and Vector search scores
"""

dataset = SingleInputCSVforLLM("evaluation/datasets/EU_test_set.csv")

vocabularies = ["RxNorm", "RxNorm Extension"]

print("Connecting to evaluation database...")
eval_conn = db_session()

llm_details = LLMModel.LLAMA_3_1_8B
llm = Llama(
    hf_hub_download(**local_models[llm_details.value]),
    n_ctx=0,
    n_batch=512,
    model_kwargs={"n_gpu_layers": -1, "verbose": True},
    generation_kwargs={"max_tokens": 128, "temperature": 0},
)
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
template_env = Environment()

llm_prompt_template = template_env.from_string(
    """The informal name is '{{informal_name}}'"""
)

template_vars = ["informal_name"]

print("Initialising retrieval database...")
db_init = time()
db = duckdb.connect()

connect_to_concept_csv(db, "~/Documents/GitHub/omop-lite/data/CONCEPT.csv")
# connect_to_vector_parquet(db, "~/OneDrive - The University of Nottingham/results/bge_embeddings.parquet")
print(f"Data loaded in {(time() - db_init):.2f} seconds")

print("Initialising model...")
model_init = time()
# bge_small = SentenceTransformer("BAAI/bge-small-en-v1.5")
# print(f"Loaded model in {(time() - model_init):.2f} seconds")


bm25_retriever = BmFileSearcher(db, vocabulary_ids=vocabularies)
# vector_retriever = ParquetFileVectorSearcher(db, model=bge_small, vocabulary_ids=vocabularies,)
# rrf = ReciprocalRankFusionSearcher(
#         db=db,
#         model=bge_small,
#         vocabulary_ids=vocabularies,
#         )

bm25_qar = AugmentedQueryPipeline(
        prompt_template=llm_prompt_template,
        llm=llm,
        template_vars=template_vars,
        retriever=bm25_retriever,
        )
# vector_qar = AugmentedQueryPipeline(
#         prompt_template=llm_prompt_template,
#         llm=llm,
#         template_vars=template_vars,
#         retriever=vector_retriever,
#         )
#
# rrf_qar = AugmentedQueryPipeline(
#         prompt_template=llm_prompt_template,
#         llm=llm,
#         template_vars=template_vars,
#         retriever=rrf,
#         )

metrics = [
        RelatedIDPrecision(connection = eval_conn, vocabulary_ids=vocabularies),
        RelatedIDRecall(connection=eval_conn, vocabulary_ids=vocabularies),
        IDMatchInList(connection = eval_conn, vocabulary_ids=vocabularies)
        ]

tests = [
        InformationRetrievalPipelineTest(
            name="BM25",
            pipeline=bm25_qar,
            metrics=metrics,
            ),
        InformationRetrievalPipelineTest(
            name="Embeddings (longer embeddings, bge-small-en-v1.5)",
            pipeline=vector_qar,
            metrics=metrics,
            ),
        InformationRetrievalPipelineTest(
            name="Reciprocal Rank Fusion (longer embeddings, bge-small-en-v1.5)",
            pipeline=rrf_qar,
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
