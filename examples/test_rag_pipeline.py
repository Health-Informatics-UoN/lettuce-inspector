from sentence_transformers import SentenceTransformer
from jinja2 import Environment

from evaluation.evaltypes import EvaluationFramework
from evaluation.metrics import (
    AncestorNameUncasedMatch,
    FuzzyMatchRatio,
    RelatedNameUncasedMatch,
    UncasedMatch,
)
from options.pipeline_options import LLMModel
from components.models import local_models
from evaluation.pipelines import LLMPipeline, RAGPipeline
from evaluation.eval_tests import LLMPipelineTest, RAGPipelineTest
from evaluation.eval_data_loaders import SingleInputCSVforLLM
from omop.db_manager import db_session
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

dataset = SingleInputCSVforLLM("evaluation/datasets/example.csv")

db_connection = db_session()

description = """
A test of increasing the number of suggestions included in a RAG prompt.

Dataset: The first 39 HELIOS self-reported medications
LLM: Llama 3.1 8b, quantized to 4 bit precision
Vector Search:
    Model: BAAI/bge-small-en-v1.5
    Available vocabularies: RxNorm

Pipelines:
    LLM only: No vector search
    5 matches: Appends the first 5 matches
    5 matches+scores: Appends the first 5 matches, including similarity scores
    10 matches: Appends the first 10 matches
    10 matches+scores: Appends the first 10 matches, including similarity scores
    20 matches: Appends the first 20 matches
    20 matches+scores: Appends the first 20 matches, including similarity scores
"""

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
    """You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

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

Task:

Informal name: {{informal_name}}<|eot_id|>
Response:
"""
)

rag_prompt_template = template_env.from_string(
    """You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

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
    {{result.content}}
{% endfor %}

Task:

Informal name: {{informal_name}}<|eot_id|>
Response:"""
)

rag_prompt_template_with_scores = template_env.from_string(
    """You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

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
    {{result.content}} (cosine similarity = {{result.score}})
{% endfor %}

Task:

Informal name: {{informal_name}}<|eot_id|>
Response:"""
)


template_vars = ["informal_name", "vec_results"]

llm_only = LLMPipeline(llm, llm_prompt_template, ["informal_name"])
rag_5 = RAGPipeline(
    llm=llm,
    prompt_template=rag_prompt_template,
    template_vars=template_vars,
    embedding_model=embedding_model,
    session=db_session(),
)

rag_5_with_score = RAGPipeline(
    llm=llm,
    prompt_template=rag_prompt_template_with_scores,
    template_vars=template_vars,
    embedding_model=embedding_model,
    session=db_session(),
)

rag_10 = RAGPipeline(
    llm=llm,
    prompt_template=rag_prompt_template,
    template_vars=template_vars,
    embedding_model=embedding_model,
    session=db_session(),
    top_k=10,
)

rag_10_with_score = RAGPipeline(
    llm=llm,
    prompt_template=rag_prompt_template_with_scores,
    template_vars=template_vars,
    embedding_model=embedding_model,
    session=db_session(),
    top_k=10,
)

rag_20 = RAGPipeline(
    llm=llm,
    prompt_template=rag_prompt_template,
    template_vars=template_vars,
    embedding_model=embedding_model,
    session=db_session(),
    top_k=20,
)

rag_20_with_score = RAGPipeline(
    llm=llm,
    prompt_template=rag_prompt_template_with_scores,
    template_vars=template_vars,
    embedding_model=embedding_model,
    session=db_session(),
    top_k=20,
)

metrics = [
    UncasedMatch(),
    FuzzyMatchRatio(),
    RelatedNameUncasedMatch(db_connection),
    AncestorNameUncasedMatch(db_connection),
]

tests = [
    LLMPipelineTest("LLM only", llm_only, metrics),
    RAGPipelineTest("5 matches", rag_5, metrics),
    RAGPipelineTest("5 matches+scores", rag_5_with_score, metrics),
    RAGPipelineTest("10 matches", rag_10, metrics),
    RAGPipelineTest("10 matches+scores", rag_10_with_score, metrics),
    RAGPipelineTest("20 matches", rag_20, metrics),
    RAGPipelineTest("20 matches+scores", rag_20_with_score, metrics),
]

framework = EvaluationFramework(
    "Comparing different RAG numbers with an LLM alone",
    tests,
    dataset,
    description,
)

if __name__ == "__main__":
    framework.run_evaluations()
