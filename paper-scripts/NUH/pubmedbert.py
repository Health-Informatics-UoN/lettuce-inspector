import sys
import logging
import duckdb
from jinja2 import Environment
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import torch

from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework
from evaluation.metrics import IDMatchInList, UncasedMatch, RelatedNameUncasedMatch
from evaluation.pipelines import EmbeddingsRetrievalPipeline, DuckdbRAGChatPipeline
from evaluation.eval_tests import RAGPipelineTest
from evaluation.evaltypes import InformationRetrievalPipelineTest
from omop.db_manager import db_session
from options.pipeline_options import LLMModel
from query_handler.duckdb_connectors import connect_to_concept_csv, connect_to_vector_parquet
from components.models import local_models
from query_handler.vectors import ParquetFileVectorSearcher

logger = logging.Logger("BGE-small-test")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

description = """
Testing pubmedbert for the Lettuce paper

Dataset: NUH data

## Pipelines
Retrieval: pubmedbert with short embedding strings, top 10
Generation: pubmedbert for RAG
    - llama 3.2 3B
    - llama 3.1 8B
    - phi-4
    - Qwen2.5 14b
    - Gemma 3 12b

## Metrics
Retrieval: IDMatchInList
Generation: UncasedMatch, RelatedNameUncasedMatch
"""

logger.info("Connecting to evaluation dataset")
dataset = SingleInputCSVforLLM("evaluation/datasets/nuh-data.csv")

logger.info("Connecting to evaluation database")
eval_conn = db_session()

logger.info("Initialising retrieval database")
db = duckdb.connect()
connect_to_vector_parquet(db, "~/embeddings-storage/pubmed.parquet")
connect_to_concept_csv(db, "~/vocabs/CONCEPT.csv")

logger.info("Initialising embedding model")
pubmedbert = SentenceTransformer("neuml/pubmedbert-base-embeddings", device="cuda")

logger.info("Loading retriever")
pq = ParquetFileVectorSearcher(
        db=db,
        model=pubmedbert,
        vocabulary_ids=None,
        vector_dimension=pubmedbert.get_sentence_embedding_dimension()
        )
pubmedbert_retriever = EmbeddingsRetrievalPipeline(
        embedding_model=pubmedbert,
        retriever=pq
        )

device = -1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0

llm_details = {
        "llama_3.2": LLMModel.LLAMA_3_2_3B,
        "llama_3.1": LLMModel.LLAMA_3_1_8B,
        "phi-4": LLMModel.PHI_4_IQ4_XS,
        "Qwen2.5 14b": LLMModel.QWEN2_5_14B_INSTRUCT,
        "Gemma": LLMModel.GEMMA_12B,
        }

logger.info("Initialising LLMs...")
llms = {
        key: Llama(
            hf_hub_download(**local_models[llm.value]),
            n_ctx=1024,
            n_batch=512,
            n_gpu_layers=-1,
            generation_kwargs={"max_tokens": 128, "temperature": 0},
            ) for key, llm in llm_details.items()
        }


rag_system_prompt = """You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

 Respond only with the formal name of the medication, without any extra explanation.

Examples:

Informal name: Tylenol
Response: Acetaminophen

Informal name: Advil
Response: Ibuprofen

Informal name: Motrin
Response: Ibuprofen

Informal name: Aleve
Response: Naproxen
"""

template_env = Environment()

prompt_template_name_only = template_env.from_string(
    """Possible related terms:
{% for result in search_results %}
    {{result.concept_id}}
{% endfor %}

Informal name: {{informal_name}}"""
        )

prompt_template_name_and_score = template_env.from_string(
    """Possible related terms:
{% for result in search_results %}
    {{result.concept_id}} (score = {{result.score|round(3)}})
{% endfor %}

Informal name: {{informal_name}}"""
        )

prompt_template_csv = template_env.from_string(
    """Possible related terms:
concepts[5]{concept_name, domain, vocabulary, concept_class, cosine_distance}
{%- for result in search_results %}
{{result.concept_id}}, {{result.domain}}, {{result.vocabulary}}, {{result.concept_class}}, {{result.score|round(3)}}
{%- endfor %}

Informal name: {{informal_name}}"""
        )

template_vars=["informal_name", "search_results"]

name_only_pipelines = {
        name: DuckdbRAGChatPipeline(
            llm=llm,
            system_prompt=rag_system_prompt,
            prompt_template=prompt_template_name_only,
            template_vars=template_vars,
            embedding_model=pubmedbert,
            db=db,
            vocabulary_ids=None,
            top_k=10,
            vector_dimension=pubmedbert.get_sentence_embedding_dimension(),
            verbose=True
            )
        for name, llm in llms.items()
        }

name_and_score_pipelines = {
        name: DuckdbRAGChatPipeline(
            llm=llm,
            system_prompt=rag_system_prompt,
            prompt_template=prompt_template_name_and_score,
            template_vars=template_vars,
            embedding_model=pubmedbert,
            db=db,
            vocabulary_ids=None,
            top_k=10,
            vector_dimension=pubmedbert.get_sentence_embedding_dimension(),
            verbose=True
            )
        for name, llm in llms.items()
        }

csv_pipelines = {
        name: DuckdbRAGChatPipeline(
            llm=llm,
            system_prompt=rag_system_prompt,
            prompt_template=prompt_template_csv,
            template_vars=template_vars,
            embedding_model=pubmedbert,
            db=db,
            vocabulary_ids=None,
            top_k=10,
            vector_dimension=pubmedbert.get_sentence_embedding_dimension(),
            verbose=True
            )
        for name, llm in llms.items()
        }

retrieval_metrics = [IDMatchInList(connection=eval_conn, vocabulary_ids=["RxNorm", "RxNorm Extension", "SNOMED"])]
generation_metrics = [UncasedMatch(), RelatedNameUncasedMatch(eval_conn)]

tests = [
        InformationRetrievalPipelineTest("pubmedbert retrieval", pubmedbert_retriever, retrieval_metrics),
        *[
            RAGPipelineTest(
                f"{name} RAG with pubmedbert(concept name only in prompt)",
                pipeline,
                generation_metrics,
                )
            for name, pipeline in name_only_pipelines.items()
            ],
        *[
            RAGPipelineTest(
                f"{name} RAG with pubmedbert (concept name and score in prompt)",
                pipeline,
                generation_metrics,
                )
            for name, pipeline in name_and_score_pipelines.items()
            ],
        *[
            RAGPipelineTest(
                f"{name} RAG with pubmedbert (concept attributes in prompt)",
                pipeline,
                generation_metrics,
                )
            for name, pipeline in csv_pipelines.items()
            ]
        ]

framework = EvaluationFramework(
        name = "pubmedbert test NUH data",
        pipeline_tests=tests,
        dataset=dataset,
        description=description,
        results_path="paper-scripts/results.json"
        )

if __name__ == "__main__":
    framework.run_evaluations()
