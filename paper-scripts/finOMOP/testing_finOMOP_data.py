import sys
import logging
import duckdb
from jinja2 import Environment
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import torch

from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework
from evaluation.metrics import IDMatchInList, UncasedMatch, RelatedNameUncasedMatch
from evaluation.pipelines import BM25RAGChatPipeline, DBRankRetrievalPipeline
from evaluation.eval_tests import RAGPipelineTest
from evaluation.evaltypes import InformationRetrievalPipelineTest
from omop.db_manager import db_session
from options.pipeline_options import LLMModel
from query_handler.duckdb_connectors import connect_to_concept_csv
from components.models import local_models
from query_handler.text_query import BmFileSearcher

logger = logging.Logger("BM25-test")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

description = """
Test run on first 10 finOMOP mappings

Dataset: First 10 FinOMOP mappings

## Pipelines
Retrieval: BM25, top 10
Generation: BM25 for RAG
    - Gemma 3 12b


## Metrics
Retrieval: IDMatchInList
Generation: UncasedMatch, RelatedNameUncasedMatch
"""

logger.info("Connecting to evaluation dataset")
dataset = SingleInputCSVforLLM(
        "~/clean-finOMOP-for-eval/matched_concepts_first_10.csv",
        input_column="sourceName",
        expected_output_column="conceptName"
        )

logger.info("Connecting to evaluation database")
eval_conn = db_session()

logger.info("Initialising retrieval database")
db = duckdb.connect()
connect_to_concept_csv(db, "~/omop-lite/data/CONCEPT.csv")

logger.info("Loading retriever")
bm25 = BmFileSearcher(db=db, vocabulary_ids=None)
bm25_pipeline = DBRankRetrievalPipeline(retriever=bm25)
logger.info("Retriever initialised")


device = -1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0
logger.info(f"Using {device} gpu layers")

llm_details = {
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

rag_system_prompt = """You are an assistant that suggests standard OMOP concepts for a source term. You will be given a source term, along with some possibly related concepts. If you do not think these concepts are related, ignore them when making your suggestion.

 Respond only with the OMOP concept, without any extra explanation.

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
    {{result[1]}}
{% endfor %}

Informal name: {{informal_name}}"""
        )

prompt_template_name_and_score = template_env.from_string(
    """Possible related terms:
{% for result in search_results %}
    {{result[1]}} (score = {{result[2]|round(3)}})
{% endfor %}

Informal name: {{informal_name}}"""
        )

prompt_template_csv = template_env.from_string(
    """Possible related terms:
concepts[5]{concept_name, domain, vocabulary, concept_class, BM25_score}
{%- for result in search_results %}
{{result[1]}}, {{result[3]}}, {{result[4]}}, {{result[5]}}, {{result[2]|round(3)}}
{%- endfor %}

Informal name: {{informal_name}}"""
        )

template_vars=["informal_name", "search_results"]

name_only_pipelines = {
        name: BM25RAGChatPipeline(
            llm=llm,
            system_prompt=rag_system_prompt,
            prompt_template=prompt_template_name_and_score,
            template_vars=template_vars,
            connection=db,
            verbose=True,
            )
        for name, llm in llms.items()
        }

name_and_score_pipelines = {
        name: BM25RAGChatPipeline(
            llm=llm,
            system_prompt=rag_system_prompt,
            prompt_template=prompt_template_name_and_score,
            template_vars=template_vars,
            connection=db,
            verbose=True,
            )
        for name, llm in llms.items()
        }

csv_pipelines = {
        name: BM25RAGChatPipeline(
            llm=llm,
            system_prompt=rag_system_prompt,
            prompt_template=prompt_template_name_and_score,
            template_vars=template_vars,
            connection=db,
            verbose=True,
            )
        for name, llm in llms.items()
        }
retrieval_metrics = [IDMatchInList(connection=eval_conn, vocabulary_ids=["RxNorm", "RxNorm Extension", "SNOMED"])]
generation_metrics = [UncasedMatch(), RelatedNameUncasedMatch(eval_conn)]

tests = [
        InformationRetrievalPipelineTest("BM25 retrieval", bm25_pipeline, retrieval_metrics),
        *[
            RAGPipelineTest(
                f"{name} RAG with BM25 (name only in prompt)",
                pipeline,
                generation_metrics,
                )
            for name, pipeline in name_only_pipelines.items()
            ],
        *[
            RAGPipelineTest(
                f"{name} RAG with BM25 (name and score in prompt)",
                pipeline,
                generation_metrics
                )
            for name, pipeline in name_and_score_pipelines.items()
            ],
        *[
            RAGPipelineTest(
                f"{name} RAG with BM25 (concept details in prompt)",
                pipeline,
                generation_metrics
                )
            for name, pipeline in csv_pipelines.items()
            ]
        ]

framework = EvaluationFramework(
        name="BM25 test",
        pipeline_tests=tests,
        dataset=dataset,
        description=description,
        results_path="paper-scripts/finOMOP/results.json"
        )

if __name__ == "__main__":
    framework.run_evaluations()
