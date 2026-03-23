"""
Generator agent — LangChain LCEL chain.

Synthesises a cited answer from the top-graded context chunks passed in by
the LangGraph orchestrator. Sources are formatted as [Source: name, p.N].

Chain:  ChatPromptTemplate | ChatOpenAI (Qwen 2.5-7B) | StrOutputParser
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.llm_factory import get_llm

_SYSTEM = (
    "You are an expert research analyst. Answer the question using ONLY the "
    "context provided below. Cite sources inline as [Source: filename, p.N]. "
    "If the context lacks sufficient information, state that clearly."
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = _prompt | get_llm(temperature=0.4, max_tokens=512) | StrOutputParser()
    return _chain


def _format_context(documents: list) -> str:
    parts = [
        f"[Source: {d.get('source', 'unknown')}, p.{d.get('page', '?')}]\n{d['page_content']}"
        for d in documents
    ]
    return "\n\n".join(parts) if parts else "No context available."


def run_generator(question: str, documents: list) -> str:
    context = _format_context(documents)
    return _get_chain().invoke({"context": context, "question": question})
