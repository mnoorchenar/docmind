"""
Embedding layer — LangChain HuggingFaceEmbeddings.

Wraps BAAI/bge-small-en-v1.5 as a LangChain-native embeddings object so it
can be swapped for any other LangChain-compatible embeddings (OpenAI,
Cohere, etc.) in one line.

The model is lazy-loaded on first call and cached as a module-level singleton
(SentenceTransformer load costs ~1–2 s; subsequent calls are instant).
"""
import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress tokenizer parallelism warning in multi-threaded Flask context
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def embed(texts: list) -> np.ndarray:
    """Embed a list of texts. Returns a float32 ndarray of shape (N, dim)."""
    vecs = _get_embeddings().embed_documents(texts)
    return np.array(vecs, dtype="float32")
