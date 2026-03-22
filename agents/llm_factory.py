# All agents import from here so the endpoint URL is in one place only.

import os
from langchain_huggingface import HuggingFaceEndpoint

_HF_ROUTER = "https://router.huggingface.co/hf-inference/models"

def make_llm(model_id: str, max_new_tokens: int = 512, temperature: float = 0.7) -> HuggingFaceEndpoint:
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Add your HuggingFace Read token in Space secrets or .env.")
    return HuggingFaceEndpoint(
        endpoint_url=f"{_HF_ROUTER}/{model_id}",
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=token,
        timeout=90,
    )