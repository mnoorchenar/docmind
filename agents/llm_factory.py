import os
from openai import OpenAI

# HF router OpenAI-compatible endpoint — officially documented
_HF_BASE_URL = "https://router.huggingface.co/v1"

AVAILABLE_MODELS = {
    "llama3-8b": {
        "id":          "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "label":       "Llama 3.1 8B (Meta)",
        "description": "Best balance of quality and speed. Most widely available on free-tier providers.",
        "speed":       "fast",
        "params":      "8B",
    },
    "qwen25-7b": {
        "id":          "Qwen/Qwen2.5-7B-Instruct",
        "label":       "Qwen 2.5 7B (Alibaba)",
        "description": "Strong multilingual reasoning. Excellent for structured output and document analysis.",
        "speed":       "fast",
        "params":      "7B",
    },
    "phi35-mini": {
        "id":          "microsoft/Phi-3.5-mini-instruct",
        "label":       "Phi-3.5 Mini (Microsoft)",
        "description": "3.8B params — fastest option. Good for simple Q&A and quick demos.",
        "speed":       "fast",
        "params":      "3.8B",
    },
    "mistral-nemo": {
        "id":          "mistralai/Mistral-Nemo-Instruct-2407",
        "label":       "Mistral Nemo 12B",
        "description": "Mistral's Nemo model — strong instruction following and reasoning. Available via HF router.",
        "speed":       "medium",
        "params":      "12B",
    },
    "gemma2-9b": {
        "id":          "google/gemma-2-9b-it",
        "label":       "Gemma 2 9B (Google)",
        "description": "Google's Gemma 2 instruction-tuned — strong factual grounding and reasoning.",
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
    """Call the HF router using OpenAI-compatible API — the official 2026 method."""
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Add your HuggingFace Read token in Space secrets or .env.")

    client   = OpenAI(base_url=_HF_BASE_URL, api_key=token)
    model_id = get_current_model_id()

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=max(temperature, 0.01),
    )
    return response.choices[0].message.content.strip()
