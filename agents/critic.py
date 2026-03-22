import re
from langchain.prompts import PromptTemplate
from agents.llm_factory import make_llm

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
    chain   = PromptTemplate(input_variables=["question","context","answer"], template=_TEMPLATE) | make_llm(max_new_tokens=150, temperature=0.1)
    result  = chain.invoke({"question": question, "context": context, "answer": answer})
    raw     = result.strip() if isinstance(result, str) else str(result).strip()
    verdict = "NEEDS_REVIEW" if re.search(r"NEEDS_REVIEW", raw, re.IGNORECASE) else "APPROVED"
    explanation = raw.split("\n",1)[-1].strip() if "\n" in raw else raw
    return {"verdict": verdict, "explanation": explanation[:300]}

