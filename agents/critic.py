import re
from agents.llm_factory import call_llm

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
    prompt  = _TEMPLATE.format(question=question, context=context, answer=answer)
    raw     = call_llm(prompt, max_new_tokens=150, temperature=0.1)
    verdict = "NEEDS_REVIEW" if re.search(r"NEEDS_REVIEW", raw, re.IGNORECASE) else "APPROVED"
    explanation = raw.split("\n", 1)[-1].strip() if "\n" in raw else raw
    return {"verdict": verdict, "explanation": explanation[:300]}
