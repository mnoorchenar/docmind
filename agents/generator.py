import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

_TEMPLATE = """You are an expert research analyst. Answer the question using ONLY the context below.
Cite sources as [Source: filename, p.N] inline. If the context lacks enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

def run_generator(question: str, documents: list) -> str:
    context_parts = []
    for d in documents:
        src  = d.get("source", "unknown")
        page = d.get("page", "?")
        context_parts.append(f"[Source: {src}, p.{page}]\n{d['page_content']}")
    context = "\n\n".join(context_parts) if context_parts else "No context available."

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.4,
        huggingfacehub_api_token=os.getenv("HF_TOKEN", ""),
        timeout=90,
    )
    chain  = PromptTemplate(input_variables=["question", "context"], template=_TEMPLATE) | llm
    result = chain.invoke({"question": question, "context": context})
    return result.strip() if isinstance(result, str) else str(result).strip()