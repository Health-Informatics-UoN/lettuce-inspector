from enum import Enum
from pydantic import BaseModel
from components.embeddings import EmbeddingModelName


class LLMModel(str, Enum):
    """
    This enum holds the names of the different models the assistant can use
    """

    GPT_3_5_TURBO = "gpt-3.5-turbo-0125"
    GPT_4 = "gpt-4"
    LLAMA_2_7B = "llama-2-7b-chat"
    LLAMA_3_8B = "llama-3-8b"
    LLAMA_3_70B = "llama-3-70b"
    GEMMA_7B = "gemma-7b"
    LLAMA_3_1_8B = "llama-3.1-8b"
    LLAMA_3_2_3B = "llama-3.2-3b"
    MISTRAL_7B = "mistral-7b"
    KUCHIKI_L2_7B = "kuchiki-l2-7b"
    TINYLLAMA_1_1B_CHAT = "tinyllama-1.1b-chat"
    BIOMISTRAL_7B = "biomistral-7b"
    QWEN2_5_3B_INSTRUCT = "qwen2.5-3b-instruct"
    AIROBOROS_3B = "airoboros-3b"
    MEDICINE_CHAT = "medicine-chat"
    MEDICINE_LLM_13B = "medicine-llm-13b"
    MED_LLAMA_3_8B_V1 = "med-llama-3-8b-v1"
    MED_LLAMA_3_8B_V2 = "med-llama-3-8b-v2"
    MED_LLAMA_3_8B_V3 = "med-llama-3-8b-v3"
    MED_LLAMA_3_8B_V4 = "med-llama-3-8b-v4"

    def get_eot_token(self) -> str:
        if self.value in [
            "llama-3.1-8b",
        ]:
            return "<|eot_id|>"
        return ""
