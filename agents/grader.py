# Best possible RRF score (rank 0 in both FAISS and BM25, k=60): 2 / 60.
_MAX_RRF = 2 / 60


def run_grader(question: str, documents: list) -> list:
    """
    Score-based grader — no LLM call needed.
    The hybrid search score already reflects relevance; normalise to 0-1.
    """
    q_words = set(question.lower().split())
    graded  = []
    for doc in documents:
        base  = min(doc.get("score", 0.0) / _MAX_RRF, 1.0)   # hybrid-search score → 0-1
        words = set(doc.get("page_content", "").lower().split())
        overlap = len(q_words & words) / max(len(q_words), 1)  # keyword overlap boost
        grade = min(base * 0.7 + overlap * 0.3, 1.0)
        graded.append({**doc, "grade": round(grade, 3)})
    return graded
