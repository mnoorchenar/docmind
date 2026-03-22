def run_retriever(question: str, vector_store, k: int = 5) -> list:
    return vector_store.hybrid_search(question, k=k)
