import re
from agents.llm_factory import call_llm

_TEMPLATE = """You are a document relevance grader. Rate how relevant this document is to the question.
Respond with ONLY a decimal number between 0.0 (irrelevant) and 1.0 (highly relevant). Nothing else.

Question: {question}
Document excerpt: {document}

Relevance score:"""

def grade_document(question: str, document: str) -> float:
    prompt = _TEMPLATE.format(question=question, document=document[:800])
    raw    = call_llm(prompt, max_new_tokens=10, temperature=0.05)
    nums   = re.findall(r"[0-9]+\.?[0-9]*", raw)
    return min(float(nums[0]), 1.0) if nums else 0.5

def run_grader(question: str, documents: list) -> list:
    return [{**doc, "grade": grade_document(question, doc["page_content"])} for doc in documents]
