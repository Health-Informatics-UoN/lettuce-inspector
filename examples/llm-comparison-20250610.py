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
from evaluation.pipelines import LLMChatPipeline, LLMPipeline
from evaluation.eval_tests import LLMPipelineTest
from evaluation.eval_data_loaders import SingleInputCSVforLLM

dataset = SingleInputCSVforLLM("evaluation/datasets/EU_test_set.csv")

db_connection = db_session()

description = """
A test of different LLMs with no RAG

Dataset: The first 400 HELIOS self-reported medications
LLMs:
    - Gemma 3 12b
    - Phi-4
    - Qwen2.5 14b
"""

device = -1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0

llm_details = {
        # "llama_3.2": LLMModel.LLAMA_3_2_3B,
        "gemma": LLMModel.GEMMA_12B,
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
system_prompt = """You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication for which you will supply the formal name.

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

user_prompt = template_env.from_string("Informal name: {{informal_name}}")

gemma_combined_prompt = template_env.from_string("""<start_of_turn>user
You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication for which you will supply the formal name.

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

Informal name:{{informal_name}}<end_of_turn>
<start_of_turn>model\n
""")

template_vars = ["informal_name"]

# llama_pipeline = LLMChatPipeline(
#         llm=llms["llama_3.2"],
#         system_prompt=system_prompt,
#         prompt_template=user_prompt,
#         template_vars=template_vars,
#         verbose=True
#         )
gemma_pipeline = LLMPipeline(
        llm=llms["gemma"],
        prompt_template=gemma_combined_prompt,
        template_vars=template_vars,
        )
phi_pipeline = LLMChatPipeline(
        llm=llms["Phi-4"],
        system_prompt=system_prompt,
        prompt_template=user_prompt,
        template_vars=template_vars,
        )
qwen_pipeline = LLMChatPipeline(
        llm=llms["Qwen2.5 14b"],
        system_prompt=system_prompt,
        prompt_template=user_prompt,
        template_vars=template_vars,
        )

metrics = [
    UncasedMatch(),
    FuzzyMatchRatio(),
    RelatedNameUncasedMatch(db_connection),
    AncestorNameUncasedMatch(db_connection),
]

tests = [
        # LLMPipelineTest("llama 3.2 3b", llama_pipeline, metrics),
        LLMPipelineTest("gemma 3 12b", gemma_pipeline, metrics),
        LLMPipelineTest("Phi-4", phi_pipeline, metrics),
        LLMPipelineTest("Qwen2.5", qwen_pipeline, metrics),
        ]

framework = EvaluationFramework(
    "Comparing different models without RAG",
    tests,
    dataset,
    description,
)

if __name__ == "__main__":
    framework.run_evaluations()
