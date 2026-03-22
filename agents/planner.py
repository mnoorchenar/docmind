from langchain.prompts import PromptTemplate
from agents.llm_factory import make_llm

_TEMPLATE = """You are a research planning agent. Given the user's question, produce a brief research plan.
Decide: should the answer be grounded in uploaded documents, web search, or both?
Output your plan in 2-3 concise sentences. Start with "PLAN:".

Question: {question}

Plan:"""

def run_planner(question: str) -> str:
    chain  = PromptTemplate(input_variables=["question"], template=_TEMPLATE) | make_llm(max_new_tokens=200, temperature=0.3)
    result = chain.invoke({"question": question})
    return result.strip() if isinstance(result, str) else str(result).strip()

