"""
Generator agent — LangChain LCEL chain.

Chain:  ChatPromptTemplate | ChatOpenAI (current model, temp 0.4) | StrOutputParser
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


def _format_context(documents: list) -> str:
    parts = [
        f"[Source: {d.get('source', 'unknown')}, p.{d.get('page', '?')}]\n{d['page_content']}"
        for d in documents
    ]
    return "\n\n".join(parts) if parts else "No context available."


def run_generator(question: str, documents: list) -> str:
    chain   = _prompt | get_llm(temperature=0.4, max_tokens=512) | StrOutputParser()
    context = _format_context(documents)
    return chain.invoke({"context": context, "question": question})
