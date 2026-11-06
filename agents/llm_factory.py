"""
LLM factory — returns a LangChain ChatOpenAI instance wired to an
OpenAI-compatible endpoint.

Two providers are supported. Groq is used when GROQ_API_KEY is set, because its
free tier is fast; otherwise the Hugging Face router is used with HF_TOKEN.

AVAILABLE_MODELS lists every model the UI picker offers for the active provider.
set_model() / get_current_model() let the Flask layer switch models without
restarting the server.
"""
import os
from langchain_openai import ChatOpenAI

PROVIDERS: dict[str, dict] = {
    "groq": {
        "label":    "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env":  "GROQ_API_KEY",
        "default":  "gpt-oss-20b",
        "models": {
            "gpt-oss-20b": {
                "id":     "openai/gpt-oss-20b",
                "label":  "GPT-OSS · 20B",
                "desc":   "Default · fast · best for most queries",
                "speed":  3,
                "params": "20B",
                "color":  "#4a7cf7",
                "reasoning": True,
            },
            "gpt-oss-120b": {
                "id":     "openai/gpt-oss-120b",
                "label":  "GPT-OSS · 120B",
                "desc":   "Stronger reasoning · slightly slower",
                "speed":  2,
                "params": "120B",
                "color":  "#6d28d9",
                "reasoning": True,
            },
            "qwen-27b": {
                "id":     "qwen/qwen3.8-27b",
                "label":  "Qwen 3.8 · 27B",
                "desc":   "Fast · concise answers",
                "speed":  3,
                "params": "27B",
                "color":  "#15924f",
                "reasoning": False,
            },
        },
    },
    "huggingface": {
        "label":    "Hugging Face",
        "base_url": "https://router.huggingface.co/v1",
        "key_env":  "HF_TOKEN",
        "default":  "qwen-7b",
        "models": {
            "qwen-7b": {
                "id":     "Qwen/Qwen2.5-7B-Instruct",
                "label":  "Qwen 2.5 · 7B",
                "desc":   "Default · fast & free · best for most queries",
                "speed":  3,
                "params": "7B",
                "color":  "#4a7cf7",
                "reasoning": False,
            },
            "mistral-nemo": {
                "id":     "mistralai/Mistral-Nemo-Instruct-2407",
                "label":  "Mistral Nemo · 12B",
                "desc":   "Stronger reasoning · slightly slower",
                "speed":  2,
                "params": "12B",
                "color":  "#6d28d9",
                "reasoning": False,
            },
            "phi-3-mini": {
                "id":     "microsoft/Phi-3.5-mini-instruct",
                "label":  "Phi-3.5 Mini · 3.8B",
                "desc":   "Ultra-fast · great for focused questions",
                "speed":  3,
                "params": "3.8B",
                "color":  "#15924f",
                "reasoning": False,
            },
        },
    },
}

# Extra completion tokens for models that spend part of the budget on hidden reasoning.
_REASONING_HEADROOM = 600


def active_provider() -> str:
    """Groq when its key is set, otherwise the Hugging Face router."""
    return "groq" if os.getenv("GROQ_API_KEY") else "huggingface"


_PROVIDER = PROVIDERS[active_provider()]
AVAILABLE_MODELS: dict[str, dict] = _PROVIDER["models"]
_current_model: str = _PROVIDER["default"]


def set_model(key: str) -> None:
    """Switch the active model for all subsequent LLM calls."""
    global _current_model
    if key in AVAILABLE_MODELS:
        _current_model = key


def get_current_model() -> dict:
    """Return the full metadata dict for the currently selected model."""
    return {"key": _current_model, "provider": _PROVIDER["label"], **AVAILABLE_MODELS[_current_model]}


def has_credentials() -> bool:
    return bool(os.getenv(_PROVIDER["key_env"]))


def get_llm(temperature: float = 0.7, max_tokens: int = 512) -> ChatOpenAI:
    """Return a LangChain ChatOpenAI using the currently selected model.

    Always reads _current_model at call time so model switching takes effect
    immediately — no stale cached chains.

    .with_retry() handles transient 429/503 errors from the free tiers.
    """
    key_env = _PROVIDER["key_env"]
    token = os.getenv(key_env, "")
    if not token:
        raise EnvironmentError(
            f"{key_env} is not set. Add it to your .env file or the Space secrets."
        )
    model = AVAILABLE_MODELS[_current_model]
    kwargs = {}
    if model.get("reasoning"):
        # gpt-oss models think before answering; keep that short and leave room for the answer.
        max_tokens += _REASONING_HEADROOM
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    return ChatOpenAI(
        base_url=_PROVIDER["base_url"],
        api_key=token,
        model=model["id"],
        temperature=max(temperature, 0.01),
        max_tokens=max_tokens,
        **kwargs,
    ).with_retry(stop_after_attempt=2)
