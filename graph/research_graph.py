import time
from datetime import datetime
from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END

from agents.planner   import run_planner
from agents.retriever import run_retriever
from agents.grader    import run_grader
from agents.generator import run_generator
from agents.critic    import run_critic


class GraphState(TypedDict):
    question:          str
    query_id:          str
    plan:              str
    documents:         List[Any]
    graded_docs:       List[Any]
    generation:        str
    critique:          str
    verdict:           str
    needs_human_review:bool
    iteration:         int
    timestamp:         str


class ResearchGraph:
    def __init__(self, vector_store, tracer):
        self.vs     = vector_store
        self.tracer = tracer
        self.graph  = self._build()

    # ── NODE FUNCTIONS ─────────────────────────────────────────────────────

    def _planner_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "planner", "Planning research approach…", "running", 0)
        plan = run_planner(state["question"])
        ms   = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "planner", plan[:200], "complete", ms)
        return {"plan": plan}

    def _retriever_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "retriever", "Running hybrid search (FAISS + BM25)…", "running", 0)
        docs = run_retriever(state["question"], self.vs, k=5)
        ms   = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "retriever", f"Retrieved {len(docs)} chunks via hybrid search.", "complete", ms)
        return {"documents": docs}

    def _grader_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "grader", f"Grading {len(state['documents'])} retrieved chunks…", "running", 0)
        graded = run_grader(state["question"], state["documents"])
        avg    = sum(d["grade"] for d in graded) / len(graded) if graded else 0.0
        ms     = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "grader", f"Avg relevance score: {avg:.2f} across {len(graded)} chunks.", "complete", ms)
        return {"graded_docs": graded}

    def _rewriter_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "rewriter", "Low relevance scores — rewriting query for better retrieval…", "running", 0)
        # Simple heuristic rewrite: add "explain in detail" framing
        new_q = f"Provide a detailed explanation about: {state['question']}"
        ms    = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "rewriter", f"Rewritten query: {new_q[:120]}", "complete", ms)
        return {"question": new_q, "iteration": state.get("iteration", 0) + 1}

    def _generator_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "generator", "Generating answer from graded context…", "running", 0)
        good_docs = [d for d in state["graded_docs"] if d.get("grade", 0) >= 0.35] or state["graded_docs"]
        gen  = run_generator(state["question"], good_docs[:4])
        ms   = int((time.time() - t0) * 1000)
        self.tracer.add(state["query_id"], "generator", f"Answer generated ({len(gen)} chars).", "complete", ms)
        return {"generation": gen}

    def _critic_node(self, state: GraphState) -> dict:
        t0 = time.time()
        self.tracer.add(state["query_id"], "critic", "Evaluating answer quality and hallucination risk…", "running", 0)
        result = run_critic(state["question"], state["generation"], state["graded_docs"])
        ms     = int((time.time() - t0) * 1000)
        needs_review = result["verdict"] == "NEEDS_REVIEW"
        label  = "⚠️ Flagged for human review." if needs_review else "✅ Answer approved."
        self.tracer.add(state["query_id"], "critic", f"{label} {result['explanation'][:160]}", "complete", ms)
        return {
            "critique":          result["explanation"],
            "verdict":           result["verdict"],
            "needs_human_review":needs_review,
        }

    # ── CONDITIONAL EDGE FUNCTIONS ─────────────────────────────────────────

    def _after_grader(self, state: GraphState) -> str:
        graded = state.get("graded_docs", [])
        avg    = sum(d.get("grade", 0) for d in graded) / len(graded) if graded else 0.0
        itr    = state.get("iteration", 0)
        if avg < 0.45 and itr < 2:
            return "rewrite"
        return "generate"

    def _after_critic(self, state: GraphState) -> str:
        return "end"   # always end — human review is handled outside graph via Flask

    # ── BUILD ──────────────────────────────────────────────────────────────

    def _build(self):
        wf = StateGraph(GraphState)
        wf.add_node("planner",   self._planner_node)
        wf.add_node("retriever", self._retriever_node)
        wf.add_node("grader",    self._grader_node)
        wf.add_node("rewriter",  self._rewriter_node)
        wf.add_node("generator", self._generator_node)
        wf.add_node("critic",    self._critic_node)

        wf.set_entry_point("planner")
        wf.add_edge("planner",   "retriever")
        wf.add_edge("retriever", "grader")
        wf.add_conditional_edges("grader", self._after_grader, {"rewrite": "rewriter", "generate": "generator"})
        wf.add_edge("rewriter",  "retriever")
        wf.add_edge("generator", "critic")
        wf.add_conditional_edges("critic", self._after_critic, {"end": END})
        return wf.compile()

    # ── PUBLIC RUN ─────────────────────────────────────────────────────────

    def run(self, question: str, query_id: str) -> dict:
        init_state = GraphState(
            question=question, query_id=query_id, plan="",
            documents=[], graded_docs=[], generation="",
            critique="", verdict="", needs_human_review=False,
            iteration=0, timestamp=datetime.utcnow().isoformat(),
        )
        final = self.graph.invoke(init_state)
        return dict(final)