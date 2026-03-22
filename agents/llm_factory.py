import os
from openai import OpenAI

_BASE_URL = "https://router.huggingface.co/v1"
_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"


def call_llm(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Add your HuggingFace token in Space secrets.")
    client = OpenAI(base_url=_BASE_URL, api_key=token)
    response = client.chat.completions.create(
        model=_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=max(temperature, 0.01),
    )
    return response.choices[0].message.content.strip()
