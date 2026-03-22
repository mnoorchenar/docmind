from agents.llm_factory import call_llm

_TEMPLATE = """You are a research planning agent. Given the user's question, produce a brief research plan.
Describe which aspects of the uploaded document are most relevant to answer the question.
Output your plan in 2-3 concise sentences. Start with "PLAN:".

Question: {question}

Plan:"""

def run_planner(question: str) -> str:
    prompt = _TEMPLATE.format(question=question)
    return call_llm(prompt, max_new_tokens=200, temperature=0.3)
