"""
Planner agent — LangChain LCEL chain.

Decomposes the user question into a brief research plan that guides
downstream retrieval and generation steps.

Chain:  ChatPromptTemplate | ChatOpenAI (Qwen 2.5-7B) | StrOutputParser
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.llm_factory import get_llm

_SYSTEM = (
    "You are a research planning agent. Given the user's question, produce a "
    "brief research plan describing which aspects of the uploaded document are "
    "most relevant to answer it. Output 2–3 concise sentences. Start with 'PLAN:'."
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "{question}"),
])

# Lazy-initialised so HF_TOKEN is not required at import time
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = _prompt | get_llm(temperature=0.3, max_tokens=200) | StrOutputParser()
    return _chain


def run_planner(question: str) -> str:
    return _get_chain().invoke({"question": question})
