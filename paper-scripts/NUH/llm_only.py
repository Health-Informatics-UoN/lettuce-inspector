import sys
import logging
from jinja2 import Environment
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import torch

from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework
from evaluation.metrics import UncasedMatch, RelatedNameUncasedMatch
from evaluation.pipelines import LLMChatPipeline
from evaluation.eval_tests import LLMPipelineTest
from omop.db_manager import db_session
from options.pipeline_options import LLMModel
from components.models import local_models

logger = logging.Logger("LLM-test")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

description = """
Testing LLMs for the Lettuce paper

Dataset: NUH data
## Pipelines
Generation:
    - llama 3.2 3B
    - llama 3.1 8B
    - phi-4
    - Qwen2.5 14b
    - Gemma 3 12b

## Metrics
Generation: UncasedMatch, RelatedNameUncasedMatch
"""

logger.info("Connecting to evaluation dataset")
dataset = SingleInputCSVforLLM("evaluation/datasets/nuh-data.csv")

logger.info("Connecting to evaluation database")
eval_conn = db_session()

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

system_prompt = """You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

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

prompt_template = template_env.from_string("Informal name: {{informal_name}}")

template_vars = ["informal_name"]

pipelines = [
        (
            name,
            LLMChatPipeline(
                llm=llm, system_prompt=system_prompt, prompt_template=prompt_template, template_vars=template_vars
                )
            )
        for name, llm in llms.items()
        ]

metrics = [UncasedMatch(), RelatedNameUncasedMatch(eval_conn)]

pipeline_tests = [
        LLMPipelineTest(name, pipeline, metrics)
        for name, pipeline in pipelines
        ]

evaluation = EvaluationFramework(
        name="LLM only test",
        description=description,
        dataset=dataset,
        pipeline_tests=pipeline_tests,
        results_path="paper-scripts/results.json"
        )

if __name__ == "__main__":
    evaluation.run_evaluations()
