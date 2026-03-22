from agents.llm_factory import call_llm

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
    prompt  = _TEMPLATE.format(question=question, context=context)
    return call_llm(prompt, max_new_tokens=512, temperature=0.4)
