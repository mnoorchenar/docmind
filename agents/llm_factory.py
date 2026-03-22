
import os
from langchain_huggingface import HuggingFaceEndpoint

_HF_ROUTER = "https://router.huggingface.co/hf-inference/models"

# ── Available free-tier models ────────────────────────────────────────────
# All confirmed available on the HF free Inference router as of March 2026.
# "speed" is relative: fast < medium < slow (affects free-tier cold starts).
AVAILABLE_MODELS = {
    "zephyr-7b": {
        "id":          "HuggingFaceH4/zephyr-7b-beta",
        "label":       "Zephyr-7B-β",
        "description": "Fine-tuned for instruction following and evaluation. Confirmed free tier. Best overall balance.",
        "speed":       "medium",
        "params":      "7B",
    },
    "phi3-mini": {
        "id":          "microsoft/Phi-3-mini-4k-instruct",
        "label":       "Phi-3 Mini (Microsoft)",
        "description": "3.8B params — very fast responses. Good for simple Q&A and summarization on free tier.",
        "speed":       "fast",
        "params":      "3.8B",
    },
    "gemma2-2b": {
        "id":          "google/gemma-2-2b-it",
        "label":       "Gemma-2 2B (Google)",
        "description": "Google's smallest instruction-tuned model. Fastest option — ideal for quick demos.",
        "speed":       "fast",
        "params":      "2B",
    },
    "falcon-7b": {
        "id":          "tiiuae/falcon-7b-instruct",
        "label":       "Falcon-7B (TII)",
        "description": "Classic open-source 7B instruct model from UAE's TII. Good factual recall.",
        "speed":       "medium",
        "params":      "7B",
    },
    "mistral-v2": {
        "id":          "mistralai/Mistral-7B-Instruct-v0.2",
        "label":       "Mistral-7B v0.2",
        "description": "Mistral's v0.2 instruct — stronger reasoning than v0.1. May occasionally be unavailable on free tier.",
        "speed":       "medium",
        "params":      "7B",
    },
}

# Default model key — must be a key in AVAILABLE_MODELS
_current_model_key = "zephyr-7b"


def get_current_model_key() -> str:
    return _current_model_key


def set_current_model(key: str):
    global _current_model_key
    if key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model key '{key}'. Valid keys: {list(AVAILABLE_MODELS)}")
    _current_model_key = key


def get_current_model_id() -> str:
    return AVAILABLE_MODELS[_current_model_key]["id"]


def make_llm(max_new_tokens: int = 512, temperature: float = 0.7) -> HuggingFaceEndpoint:
    """Build an LLM using whatever model the user currently has selected."""
    token    = os.getenv("HF_TOKEN", "")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Add your HuggingFace Read token in Space secrets or .env.")
    model_id = get_current_model_id()
    return HuggingFaceEndpoint(
        endpoint_url=f"{_HF_ROUTER}/{model_id}",
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=token,
        timeout=90,
    )