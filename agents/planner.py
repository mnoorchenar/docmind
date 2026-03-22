import os, time
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

_TEMPLATE = """You are a research planning agent. Given the user's question, produce a brief research plan.
Decide: should the answer be grounded in uploaded documents, web search, or both?
Output your plan in 2-3 concise sentences. Start with "PLAN:".

Question: {question}

Plan:"""

def run_planner(question: str) -> str:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=200,
        temperature=0.3,
        huggingfacehub_api_token=os.getenv("HF_TOKEN", ""),
        timeout=60,
    )
    chain  = PromptTemplate(input_variables=["question"], template=_TEMPLATE) | llm
    result = chain.invoke({"question": question})
    return result.strip() if isinstance(result, str) else str(result).strip()