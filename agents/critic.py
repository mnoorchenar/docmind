import os, re
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

_TEMPLATE = """You are a strict quality-control critic. Evaluate this answer for accuracy and grounding.
Output EXACTLY one of these two lines first, then a one-sentence explanation:
VERDICT: APPROVED
VERDICT: NEEDS_REVIEW

Criteria for NEEDS_REVIEW: answer contains claims not in the context, is incomplete, or is incoherent.

Question: {question}
Context (first 1500 chars): {context}
Answer: {answer}

Evaluation:"""

def run_critic(question: str, answer: str, documents: list) -> dict:
    context = " ".join(d["page_content"] for d in documents)[:1500]
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=150,
        temperature=0.1,
        huggingfacehub_api_token=os.getenv("HF_TOKEN", ""),
        timeout=60,
    )
    chain  = PromptTemplate(input_variables=["question", "context", "answer"], template=_TEMPLATE) | llm
    result = chain.invoke({"question": question, "context": context, "answer": answer})
    raw    = result.strip() if isinstance(result, str) else str(result).strip()

    verdict = "APPROVED"
    if re.search(r"NEEDS_REVIEW", raw, re.IGNORECASE):
        verdict = "NEEDS_REVIEW"
    elif re.search(r"APPROVED", raw, re.IGNORECASE):
        verdict = "APPROVED"

    explanation = raw.split("\n", 1)[-1].strip() if "\n" in raw else raw
    return {"verdict": verdict, "explanation": explanation[:300]}