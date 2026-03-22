def run_retriever(question: str, vector_store, k: int = 5) -> list:
    """Returns list of dicts with keys: page_content, source, page, score."""
    return vector_store.hybrid_search(question, k=k)