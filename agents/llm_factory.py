"""
LLM factory — returns a LangChain ChatOpenAI instance wired to the
HuggingFace Router (OpenAI-compatible endpoint).

AVAILABLE_MODELS lists every model available to the UI picker.
set_model() / get_current_model() let the Flask layer switch models
without restarting the server.
"""
import os
from langchain_openai import ChatOpenAI

_BASE_URL = "https://router.huggingface.co/v1"

AVAILABLE_MODELS: dict[str, dict] = {
    "qwen-7b": {
        "id":     "Qwen/Qwen2.5-7B-Instruct",
        "label":  "Qwen 2.5 · 7B",
        "desc":   "Default · fast & free · best for most queries",
        "speed":  3,
        "params": "7B",
        "color":  "#5b8ff9",
    },
    "mistral-nemo": {
        "id":     "mistralai/Mistral-Nemo-Instruct-2407",
        "label":  "Mistral Nemo · 12B",
        "desc":   "Stronger reasoning · slightly slower",
        "speed":  2,
        "params": "12B",
        "color":  "#a78bfa",
    },
    "phi-3-mini": {
        "id":     "microsoft/Phi-3.5-mini-instruct",
        "label":  "Phi-3.5 Mini · 3.8B",
        "desc":   "Ultra-fast · great for focused questions",
        "speed":  3,
        "params": "3.8B",
        "color":  "#22d47a",
    },
}

_current_model: str = "qwen-7b"


def set_model(key: str) -> None:
    """Switch the active model for all subsequent LLM calls."""
    global _current_model
    if key in AVAILABLE_MODELS:
        _current_model = key


def get_current_model() -> dict:
    """Return the full metadata dict for the currently selected model."""
    return {"key": _current_model, **AVAILABLE_MODELS[_current_model]}


def get_llm(temperature: float = 0.7, max_tokens: int = 512) -> ChatOpenAI:
    """Return a LangChain ChatOpenAI using the currently selected model.

    Always reads _current_model at call time so model switching takes effect
    immediately — no stale cached chains.

    .with_retry() handles transient 429/503 errors from the free HF tier.
    """
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add your HuggingFace token in Space secrets."
        )
    model_id = AVAILABLE_MODELS[_current_model]["id"]
    return ChatOpenAI(
        base_url=_BASE_URL,
        api_key=token,
        model=model_id,
        temperature=max(temperature, 0.01),
        max_tokens=max_tokens,
    ).with_retry(stop_after_attempt=2)
