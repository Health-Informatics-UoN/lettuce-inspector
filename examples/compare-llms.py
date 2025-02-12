from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework
from evaluation.metrics import UncasedMatch, FuzzyMatchRatio
from evaluation.pipelines import LLMPipeline
from options.pipeline_options import LLMModel
from evaluation.eval_tests import LLMPipelineTest
from components.models import local_models
from jinja2 import Environment
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


dataloader = SingleInputCSVforLLM("./evaluation/datasets/example.csv")

jinja_env = Environment()
prompt_template = jinja_env.from_string(
    """You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

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
Response:"""
)

template_vars = ["informal_name"]

model_names = [
    model.value
    for model in [
        LLMModel.LLAMA_3_1_8B,
        LLMModel.LLAMA_3_2_3B,
        LLMModel.LLAMA_3_8B,
    ]
]

llms = [
    (
        name,
        Llama(
            hf_hub_download(**local_models[name]),
            n_ctx=0,
            n_batch=512,
            model_kwargs={"n_gpu_layers": -1, "verbose": True},
            generation_kwargs={"max_tokens": 128, "temperature": 0},
        ),
    )
    for name in model_names
]

pipelines = [
    (
        name,
        LLMPipeline(
            llm=llm, prompt_template=prompt_template, template_vars=template_vars
        ),
    )
    for name, llm in llms
]


pipeline_tests = [
    LLMPipelineTest(name, pipeline, [UncasedMatch(), FuzzyMatchRatio()])
    for name, pipeline in pipelines
]

if __name__ == "__main__":
    evaluation = EvaluationFramework(
        name="Example benchmarking",
        pipeline_tests=pipeline_tests,
        dataset=dataloader,
        description="Demonstration of running LLMs against a benchmark, applying two metrics. The example data is the first 40 entries from the HELIOS self-reported medications dataset",
        results_path="example_output.json",
    )

    evaluation.run_evaluations()
