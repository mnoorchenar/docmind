import os, re
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

_TEMPLATE = """You are a document relevance grader. Rate how relevant this document is to the question.
Respond with ONLY a decimal number between 0.0 (irrelevant) and 1.0 (highly relevant). Nothing else.

Question: {question}
Document excerpt: {document}

Relevance score:"""

def grade_document(question: str, document: str) -> float:
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=10,
        temperature=0.05,
        huggingfacehub_api_token=os.getenv("HF_TOKEN", ""),
        timeout=45,
    )
    chain  = PromptTemplate(input_variables=["question", "document"], template=_TEMPLATE) | llm
    result = chain.invoke({"question": question, "document": document[:800]})
    raw    = result.strip() if isinstance(result, str) else str(result).strip()
    nums   = re.findall(r"[0-9]+\.?[0-9]*", raw)
    return min(float(nums[0]), 1.0) if nums else 0.5

def run_grader(question: str, documents: list) -> list:
    """Returns same list with 'grade' float added to each doc dict."""
    graded = []
    for doc in documents:
        score = grade_document(question, doc["page_content"])
        graded.append({**doc, "grade": score})
    return graded