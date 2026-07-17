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
    - Phi-4, quantized IQ4 XS
    - Qwen2.5 14B, quantized
Vector Search:
    Model: BAAI/bge-small-en-v1.5
    Available vocabularies: RxNorm

10 matches+scores: Appends the first 10 matches, including similarity scores
"""

device = -1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0

llm_details = {
        # "llama_3.2": LLMModel.LLAMA_3_2_3B,
        # "med_llama": LLMModel.MED_LLAMA_3_8B_V4,
        # "gemma": LLMModel.GEMMA_12B
        "Phi-4": LLMModel.PHI_4_IQ4_XS,
        "Qwen2.5 14b": LLMModel.QWEN2_5_3B_INSTRUCT
        }
llms= {
        key: Llama(
            hf_hub_download(**local_models[llm.value]),
            n_ctx=0,
            n_batch=512,
            model_kwargs={
                "n_ctx": 1024,
                "n_batch": 32,
                "n_gpu_layers": device,
            }, 
            generation_kwargs={"max_tokens": 128, "temperature": 0},
            ) for key, llm in llm_details.items()
        }

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

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
{% for result in vec_results %}
    {{result.content}} (score = {{1-result.score|round(3)}})
{% endfor %}

Informal name: {{informal_name}}"""
)

# gemma_rag_prompt_template = template_env.from_string(
#         """<start_of_turn>user
# You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.
#
# Respond only with the formal name of the medication, without any extra explanation.
#
# Examples:
#
# Informal name: Tylenol
# Response: Acetaminophen
#
# Informal name: Advil
# Response: Ibuprofen
#
# Informal name: Motrin
# Response: Ibuprofen
#
# Informal name: Aleve
# Response: Naproxen
#
# Possible related terms:
# {% for result in vec_results %}
#     {{result.content}} (score = {{1-result.score|round(3)}})
# {% endfor %}
#
# Informal name: {{informal_name}}<end_of_turn>
# <start_of_turn>model\n
# """
#         )


template_vars = ["informal_name", "vec_results"]

# llama_pipeline = RAGChatPipeline(
#         llm=llms["llama_3.2"],
#         system_prompt=rag_system_prompt,
#         prompt_template=llama_rag_prompt_template,
#         template_vars=template_vars,
#         embedding_model=embedding_model,
#         session=db_connection,
#         embed_vocab=["RxNorm"],
#         top_k=10,
#         verbose=True
#         )
#
# gemma_pipeline = RAGPipeline(
#         llm=llms["gemma"],
#         prompt_template=gemma_rag_prompt_template,
#         template_vars=template_vars,
#         embedding_model=embedding_model,
#         session=db_connection,
#         embed_vocab=["RxNorm"],
#         top_k=10,
#         verbose=True
#         )

phi4_pipeline = RAGChatPipeline(
        llm=llms["Phi-4"],
        system_prompt=rag_system_prompt,
        prompt_template=rag_prompt_template,
        template_vars=template_vars,
        embedding_model=embedding_model,
        session=db_connection,
        embed_vocab=["RxNorm"],
        top_k=10,
        verbose=True
        )

qwen2_5_pipeline = RAGChatPipeline(
        llm=llms["Qwen2.5 14b"],
        system_prompt=rag_system_prompt,
        prompt_template=rag_prompt_template,
        template_vars=template_vars,
        embedding_model=embedding_model,
        session=db_connection,
        embed_vocab=["RxNorm"],
        top_k=10,
        verbose=True
        )

metrics = [
    UncasedMatch(),
    FuzzyMatchRatio(),
    RelatedNameUncasedMatch(db_connection),
    AncestorNameUncasedMatch(db_connection),
]

tests = [
    # RAGPipelineTest("Llama 3.2 3b", llama_pipeline, metrics),
    # RAGPipelineTest("gemma", gemma_pipeline, metrics),
    RAGPipelineTest("phi-4", phi4_pipeline, metrics),
    RAGPipelineTest("qwen2.5", qwen2_5_pipeline, metrics)
]

framework = EvaluationFramework(
    "Comparing different models with RAG, top k = 10",
    tests,
    dataset,
    description,
)

if __name__ == "__main__":
    framework.run_evaluations()
