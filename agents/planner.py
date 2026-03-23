"""
Planner agent — LangChain LCEL chain.

Chain:  ChatPromptTemplate | ChatOpenAI (current model, temp 0.3) | StrOutputParser
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


def run_planner(question: str) -> str:
    chain = _prompt | get_llm(temperature=0.3, max_tokens=200) | StrOutputParser()
    return chain.invoke({"question": question})
