import numpy as np
from sentence_transformers import SentenceTransformer

_model = None   # lazy-loaded singleton

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def embed(texts: list) -> np.ndarray:
    """Returns float32 numpy array of shape (N, dim)."""
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")