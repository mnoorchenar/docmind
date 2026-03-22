from langchain.prompts import PromptTemplate
from agents.llm_factory import make_llm

_TEMPLATE = """You are an expert research analyst. Answer the question using ONLY the context below.
Cite sources as [Source: filename, p.N] inline. If the context lacks enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

def run_generator(question: str, documents: list) -> str:
    context_parts = [
        f"[Source: {d.get('source','unknown')}, p.{d.get('page','?')}]\n{d['page_content']}"
        for d in documents
    ]
    context = "\n\n".join(context_parts) if context_parts else "No context available."
    chain   = PromptTemplate(input_variables=["question","context"], template=_TEMPLATE) | make_llm(max_new_tokens=512, temperature=0.4)
    result  = chain.invoke({"question": question, "context": context})
    return result.strip() if isinstance(result, str) else str(result).strip()


