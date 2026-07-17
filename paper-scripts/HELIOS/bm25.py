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
Testing BM25 for the Lettuce paper

Dataset: 400 HELIOS self-reported medications

## Pipelines
Retrieval: BM25, top 10
Generation: BM25 for RAG
    - llama 3.2 3B
    - phi-4
    - Qwen2.5 14b

## Metrics
Retrieval: IDMatchInList
Generation: UncasedMatch
"""

logger.info("Connecting to evaluation dataset")
dataset = SingleInputCSVforLLM("evaluation/datasets/helios-eval-set.csv")

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
        "llama_3.2": LLMModel.LLAMA_3_2_3B,
        "Phi-4": LLMModel.PHI_4_IQ4_XS,
        "Qwen2.5 14b": LLMModel.QWEN2_5_14B_INSTRUCT
        }

logger.info("Initialising LLMs...")
llms= {
        key: Llama(
            hf_hub_download(**local_models[llm.value]),
            n_ctx=0,
            n_batch=512,
            n_gpu_layers=device,
            model_kwargs={
                "n_ctx": 1024,
                "n_batch": 32,
                "n_gpu_layers": device,
            }, 
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

rag_prompt_template = template_env.from_string(
    """Possible related terms:
{% for result in search_results %}
    {{result[1]}} (score = {{result[2]|round(3)}})
{% endfor %}

Informal name: {{informal_name}}"""
        )

template_vars=["informal_name", "search_results"]

llama_pipeline = BM25RAGChatPipeline(
        llm=llms["llama_3.2"],
        system_prompt=rag_system_prompt,
        prompt_template=rag_prompt_template,
        template_vars=template_vars,
        connection=db,
        verbose=True,
        )
phi_pipeline = BM25RAGChatPipeline(
        llm=llms["Phi-4"],
        system_prompt=rag_system_prompt,
        prompt_template=rag_prompt_template,
        template_vars=template_vars,
        connection=db,
        verbose=True,
        )
qwen_pipeline = BM25RAGChatPipeline(
        llm=llms["Qwen2.5 14b"],
        system_prompt=rag_system_prompt,
        prompt_template=rag_prompt_template,
        template_vars=template_vars,
        connection=db,
        verbose=True,
        )

retrieval_metrics = [IDMatchInList(connection=eval_conn, vocabulary_ids=["RxNorm", "RxNorm Extension", "SNOMED"])]
generation_metrics = [UncasedMatch(), RelatedNameUncasedMatch(eval_conn)]

tests = [
        InformationRetrievalPipelineTest("BM25 retrieval", bm25_pipeline, retrieval_metrics),
        RAGPipelineTest(
            "llama 3.2 3b with BM25 RAG",
            llama_pipeline,
            generation_metrics
            ),
        RAGPipelineTest(
            "phi-4 with BM25 RAG",
            phi_pipeline,
            generation_metrics,
            ),
        RAGPipelineTest(
            "Qwen2.5 14b with BM25 RAG",
            qwen_pipeline,
            generation_metrics,
            )
        ]

framework = EvaluationFramework(
        name="BM25 test",
        pipeline_tests=tests,
        dataset=dataset,
        description=description,
        results_path="paper-scripts/results.json"
        )

if __name__ == "__main__":
    framework.run_evaluations()
