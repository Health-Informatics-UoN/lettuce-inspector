import logging
from options.pipeline_options import LLMModel
import torch


local_models = {
    "llama-2-7b-chat": {
        "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "filename": "llama-2-7b-chat.Q4_0.gguf",
    },
    "llama-3-8b": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-GGUF-v2",
        "filename": "Meta-Llama-3-8B.Q4_K_M.gguf",
    },
    "llama-3-70b": {
        "repo_id": "QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2",
        "filename": "Meta-Llama-3-70B-Instruct-v2.Q4_K_M.gguf",
    },
    "gemma-7b": {
        "repo_id": "MaziyarPanahi/gemma-7b-GGUF",
        "filename": "gemma-7b.Q4_K_M.gguf",
    },
    "llama-3.1-8b": {
        "repo_id": "MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    },
    "llama-3.2-3b": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q6_K.gguf",
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-GGUF",
        "filename": "mistral-7b.Q4_K_M.gguf",
    },
    "kuchiki-l2-7b": {
        "repo_id": "TheBloke/Kuchiki-L2-7B-GGUF",
        "filename": "kuchiki-l2-7b.Q4_K_M.gguf",
    },
    "tinyllama-1.1b-chat": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF",
        "filename": "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
    },
    "biomistral-7b": {
        "repo_id": "MaziyarPanahi/BioMistral-7B-GGUF",
        "filename": "BioMistral-7B.Q4_K_M.gguf",
    },
    "qwen2.5-3b-instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q5_k_m.gguf",
    },
    "airoboros-3b": {
        "repo_id": "afrideva/airoboros-3b-3p0-GGUF",
        "filename": "airoboros-3b-3p0.q4_k_m.gguf",
    },
    "medicine-chat": {
        "repo_id": "TheBloke/medicine-chat-GGUF",
        "filename": "medicine-chat.Q4_K_M.gguf",
    },
    "medicine-llm-13b": {
        "repo_id": "TheBloke/medicine-LLM-13B-GGUF",
        "filename": "medicine-llm-13b.Q3_K_S.gguf",
    },
    "med-llama-3-8b-v1": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q5_K_S.gguf",
    },
    "med-llama-3-8b-v2": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q4_K_M.gguf",
    },
    "med-llama-3-8b-v3": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q3_K_M.gguf",
    },
    "med-llama-3-8b-v4": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-IQ3_M.gguf",
    },
}
