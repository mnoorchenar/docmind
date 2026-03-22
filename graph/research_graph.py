import time
from datetime import datetime
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

from agents.planner   import run_planner
from agents.retriever import run_retriever
from agents.grader    import run_grader
from agents.generator import run_generator
from agents.critic    import run_critic


class GraphState(TypedDict):
    question:    str
    query_id:    str
    plan:        str
    documents:   List[Any]
    graded_docs: List[Any]
    generation:  str
    critique:    str
    verdict:     str
    timestamp:   str


class ResearchGraph:
    def __init__(self, vector_store, tracer):
        self.vs     = vector_store
        self.tracer = tracer
        self.graph  = self._build()

    def _planner_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "planner", "Planning research approach…", "running", 0)
        plan = run_planner(state["question"])
        ms   = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "planner", plan[:200], "complete", ms)
        return {"plan": plan}

    def _retriever_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "retriever", "Searching knowledge base (FAISS + BM25)…", "running", 0)
        docs = run_retriever(state["question"], self.vs, k=5)
        ms   = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "retriever", f"Retrieved {len(docs)} chunks.", "complete", ms)
        return {"documents": docs}

    def _grader_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "grader", f"Grading {len(state['documents'])} chunks for relevance…", "running", 0)
        graded = run_grader(state["question"], state["documents"])
        avg    = sum(d["grade"] for d in graded) / len(graded) if graded else 0.0
        ms     = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "grader", f"Avg relevance: {avg:.2f}", "complete", ms)
        return {"graded_docs": graded}

    def _generator_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "generator", "Generating answer from relevant context…", "running", 0)
        good_docs = [d for d in state["graded_docs"] if d.get("grade", 0) >= 0.35] or state["graded_docs"]
        gen  = run_generator(state["question"], good_docs[:4])
        ms   = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "generator", f"Answer generated ({len(gen)} chars).", "complete", ms)
        return {"generation": gen}

    def _critic_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "critic", "Evaluating answer quality…", "running", 0)
        result = run_critic(state["question"], state["generation"], state["graded_docs"])
        ms     = int((time.time() - t0) * 1000)
        label  = "✅ High confidence." if result["verdict"] == "APPROVED" else "⚠️ Low confidence — verify with source."
        self.tracer.add(state["query_id"], "critic", f"{label} {result['explanation'][:160]}", "complete", ms)
        return {"critique": result["explanation"], "verdict": result["verdict"]}

    def _build(self):
        wf = StateGraph(GraphState)
        wf.add_node("planner",   self._planner_node)
        wf.add_node("retriever", self._retriever_node)
        wf.add_node("grader",    self._grader_node)
        wf.add_node("generator", self._generator_node)
        wf.add_node("critic",    self._critic_node)
        wf.set_entry_point("planner")
        wf.add_edge("planner",   "retriever")
        wf.add_edge("retriever", "grader")
        wf.add_edge("grader",    "generator")
        wf.add_edge("generator", "critic")
        wf.add_edge("critic",    END)
        return wf.compile()

    def run(self, question: str, query_id: str) -> dict:
        init = GraphState(
            question=question, query_id=query_id, plan="",
            documents=[], graded_docs=[], generation="",
            critique="", verdict="", timestamp=datetime.utcnow().isoformat(),
        )
        return dict(self.graph.invoke(init))
