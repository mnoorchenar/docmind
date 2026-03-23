"""
LLM factory — returns a LangChain ChatOpenAI instance wired to the
HuggingFace Router (OpenAI-compatible endpoint).

Each LLM agent (planner, generator, critic) calls get_llm() and builds its
own LCEL chain, keeping temperature and token budgets tuned per role.
"""
import os
from langchain_openai import ChatOpenAI

_BASE_URL = "https://router.huggingface.co/v1"
_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"


def get_llm(temperature: float = 0.7, max_tokens: int = 512) -> ChatOpenAI:
    """Return a LangChain ChatOpenAI wired to the HuggingFace Router.

    .with_retry() wraps the call with up to 2 attempts, which gracefully
    handles transient 429 / 503 errors from the free inference tier.
    """
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add your HuggingFace token in Space secrets."
        )
    return ChatOpenAI(
        base_url=_BASE_URL,
        api_key=token,
        model=_MODEL_ID,
        temperature=max(temperature, 0.01),
        max_tokens=max_tokens,
    ).with_retry(stop_after_attempt=2)
