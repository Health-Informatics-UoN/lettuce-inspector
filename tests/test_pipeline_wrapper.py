
import pandas as pd 
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from integrations.mlflow.pipeline_wrapper import MLflowPipelineWrapper
from evaluation.pipelines import LLMPipeline
from components.models import local_models, LLMModel 


def test_lettuce_llm_pipeline_wrapper(): 
    model_name = LLMModel.LLAMA_3_1_8B.value
    llm = Llama(
        hf_hub_download(**local_models[model_name]),
        n_ctx=0,
        n_batch=512,
        model_kwargs={"n_gpu_layers": -1},
        generation_kwargs={"max_tokens": 50, "temperature": 0}
    )
    
    prompt_template_str = """You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

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
    
    pipeline = LLMPipeline(
        llm, 
        prompt_template_str=prompt_template_str, 
        template_vars=["informal_names"]
    )
    
    wrapper = MLflowPipelineWrapper(pipeline)

    model_input = pd.DataFrame(
        {"model_input": ["paracetamol"], "expected_output": ["acetaminophen"]}
    )

    predictions = wrapper.predict(model_input=model_input)

    breakpoint()