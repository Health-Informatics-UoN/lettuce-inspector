from sentence_transformers import SentenceTransformer
from jinja2 import Environment
import torch
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from evaluation.evaltypes import EvaluationFramework
from omop.db_manager import db_session
from evaluation.metrics import (
    AncestorNameUncasedMatch,
    FuzzyMatchRatio,
    RelatedNameUncasedMatch,
    UncasedMatch,
)
from options.pipeline_options import LLMModel
from components.models import local_models
from evaluation.pipelines import RAGChatPipeline, RAGPipeline
from evaluation.eval_tests import RAGPipelineTest
from evaluation.eval_data_loaders import SingleInputCSVforLLM

dataset = SingleInputCSVforLLM("evaluation/datasets/EU_test_set.csv")

db_connection = db_session()

description = """
A test of increasing the number of suggestions included in a RAG prompt.

Dataset: The first 400 HELIOS self-reported medications
LLMs:
    - Gemma-3 12b, quantised to 4 bit
Vector Search:
    Model: BAAI/bge-small-en-v1.5
    Available vocabularies: RxNorm

5 matches+scores: Appends the first 10 matches, including similarity scores
20 matches+scores: Appends the first 10 matches, including similarity scores
"""

device = -1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0

llm_details = LLMModel.GEMMA_12B

llm = Llama(
        hf_hub_download(**local_models[llm_details.value]),
        n_ctx=0,
        n_batch=512,
        model_kwargs={
            "n_ctx": 1024,
            "n_batch": 32,
            "n_gpu_layers": device,
        }, 
        generation_kwargs={"max_tokens": 128, "temperature": 0},
        )

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

template_env = Environment()

gemma_rag_prompt_template = template_env.from_string(
        """<start_of_turn>user
You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

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

Possible related terms:
{% for result in vec_results %}
    {{result.content}} (score = {{1-result.score|round(3)}})
{% endfor %}

Informal name: {{informal_name}}<end_of_turn>
<start_of_turn>model\n
"""
        )

template_vars = ["informal_name", "vec_results"]

rag_5_pipeline = RAGPipeline(
        llm=llm,
        prompt_template=gemma_rag_prompt_template,
        template_vars=template_vars,
        embedding_model=embedding_model,
        session=db_connection,
        embed_vocab=["RxNorm"],
        top_k=5,
        verbose=True
        )

rag_20_pipeline = RAGPipeline(
        llm=llm,
        prompt_template=gemma_rag_prompt_template,
        template_vars=template_vars,
        embedding_model=embedding_model,
        session=db_connection,
        embed_vocab=["RxNorm"],
        top_k=20,
        verbose=True
        )
metrics = [
    UncasedMatch(),
    FuzzyMatchRatio(),
    RelatedNameUncasedMatch(db_connection),
    AncestorNameUncasedMatch(db_connection),
]

tests = [
    RAGPipelineTest("top 5 RAG", rag_5_pipeline, metrics),
    RAGPipelineTest("top 20 RAG", rag_20_pipeline, metrics)
]

framework = EvaluationFramework(
    "Comparing Gemma-3 with different numbers for RAG",
    tests,
    dataset,
    description,
)

if __name__ == "__main__":
    framework.run_evaluations()
