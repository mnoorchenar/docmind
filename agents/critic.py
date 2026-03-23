"""
Critic agent — LangChain LCEL chain.

Evaluates the generated answer for hallucinations and completeness against
the source context. Returns a verdict dict consumed by the LangGraph node.

Chain:  ChatPromptTemplate | ChatOpenAI (Qwen 2.5-7B, temp 0.1) | StrOutputParser
"""
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.llm_factory import get_llm

_SYSTEM = """You are a strict quality-control critic. Evaluate the answer for accuracy and grounding.
Output EXACTLY one of these two verdict lines first, then a one-sentence explanation:
VERDICT: APPROVED
VERDICT: NEEDS_REVIEW

Criteria for NEEDS_REVIEW: answer contains claims not supported by the context, \
is incomplete, or is incoherent."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Question: {question}\nContext (first 1500 chars): {context}\nAnswer: {answer}"),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        # Low temperature → near-deterministic evaluation
        _chain = _prompt | get_llm(temperature=0.1, max_tokens=150) | StrOutputParser()
    return _chain


def run_critic(question: str, answer: str, documents: list) -> dict:
    context     = " ".join(d["page_content"] for d in documents)[:1500]
    raw         = _get_chain().invoke({"question": question, "context": context, "answer": answer})
    verdict     = "NEEDS_REVIEW" if re.search(r"NEEDS_REVIEW", raw, re.IGNORECASE) else "APPROVED"
    explanation = raw.split("\n", 1)[-1].strip() if "\n" in raw else raw
    return {"verdict": verdict, "explanation": explanation[:300]}
