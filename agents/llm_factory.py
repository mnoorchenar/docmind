# Replaced HuggingFaceEndpoint (broken) with InferenceClient.chat_completion
# which is the officially supported method as of 2026.

import os
from huggingface_hub import InferenceClient

# All models below are available through the HF free monthly credits
# via the router.huggingface.co inference providers (Sambanova, Together, etc.)
AVAILABLE_MODELS = {
    "llama3-8b": {
        "id":          "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "label":       "Llama 3.1 8B (Meta)",
        "description": "Meta's Llama 3.1 8B — best balance of quality and speed. Most popular free-tier model in 2026.",
        "speed":       "fast",
        "params":      "8B",
    },
    "qwen2-7b": {
        "id":          "Qwen/Qwen2.5-7B-Instruct",
        "label":       "Qwen 2.5 7B (Alibaba)",
        "description": "Strong multilingual reasoning. Excellent for structured output and document analysis.",
        "speed":       "fast",
        "params":      "7B",
    },
    "phi35-mini": {
        "id":          "microsoft/Phi-3.5-mini-instruct",
        "label":       "Phi-3.5 Mini (Microsoft)",
        "description": "3.8B params — fastest option on the free tier. Good for simple Q&A and summarization.",
        "speed":       "fast",
        "params":      "3.8B",
    },
    "mistral-7b": {
        "id":          "mistralai/Mistral-7B-Instruct-v0.3",
        "label":       "Mistral 7B v0.3",
        "description": "Strong instruction following. Available via Sambanova provider on free credits.",
        "speed":       "medium",
        "params":      "7B",
    },
    "gemma2-9b": {
        "id":          "google/gemma-2-9b-it",
        "label":       "Gemma 2 9B (Google)",
        "description": "Google's Gemma 2 instruction-tuned — very strong factual grounding.",
        "speed":       "medium",
        "params":      "9B",
    },
}

_current_model_key = "llama3-8b"


def get_current_model_key() -> str:
    return _current_model_key


def set_current_model(key: str):
    global _current_model_key
    if key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model key '{key}'. Valid: {list(AVAILABLE_MODELS)}")
    _current_model_key = key


def get_current_model_id() -> str:
    return AVAILABLE_MODELS[_current_model_key]["id"]


def call_llm(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Single unified LLM call using InferenceClient.chat_completion."""
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Add your HuggingFace Read token in Space secrets or .env.")
    client   = InferenceClient(api_key=token)
    model_id = get_current_model_id()
    response = client.chat_completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=max(temperature, 0.01),   # 0.0 is not accepted by all providers
    )
    return response.choices[0].message.content.strip()
