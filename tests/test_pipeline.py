"""Fast tests: no network, no model downloads, no LLM calls."""

import zlib

import numpy as np
import pytest

from agents.grader import run_grader
from graph.research_graph import _select_docs
from tracing.tracer import Tracer


# ── Grader ───────────────────────────────────────────────────────────────────

def test_grader_scores_the_best_possible_retrieval_near_one():
    docs = [{"page_content": "retrieval augmented generation", "score": 2 / 60}]
    graded = run_grader("retrieval augmented generation", docs)
    assert graded[0]["grade"] >= 0.95


def test_grader_ranks_relevant_chunks_above_weak_ones():
    docs = [
        {"page_content": "hybrid search fuses keyword and vector results", "score": 2 / 60},
        {"page_content": "completely unrelated text about cooking", "score": 1 / 70},
    ]
    strong, weak = run_grader("how does hybrid search work", docs)
    assert strong["grade"] > weak["grade"]
    assert 0 <= weak["grade"] <= 1


def test_grader_handles_empty_input():
    assert run_grader("anything", []) == []


# ── Chunk selection shared by generator and critic ──────────────────────────

def test_select_docs_filters_by_grade_and_limits():
    docs = [{"grade": g} for g in (0.9, 0.8, 0.7, 0.6, 0.5, 0.1)]
    chosen = _select_docs(docs)
    assert [d["grade"] for d in chosen] == [0.9, 0.8, 0.7, 0.6]


def test_select_docs_falls_back_when_nothing_passes():
    docs = [{"grade": 0.1}, {"grade": 0.2}]
    assert _select_docs(docs) == docs


# ── Tracer ───────────────────────────────────────────────────────────────────

def test_tracer_keeps_memory_bounded():
    tracer = Tracer()
    for i in range(250):
        tracer.add(f"q{i}", "planner", "step", "complete", 1)
    assert tracer.get("q0") == []          # oldest trace dropped
    assert len(tracer.get("q249")) == 1    # newest kept


# ── Hybrid search ────────────────────────────────────────────────────────────

def _fake_embed(texts):
    """Deterministic bag-of-words embedding, so tests need no model download."""
    out = np.zeros((len(texts), 64), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().split():
            out[row, zlib.crc32(word.encode()) % 64] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


@pytest.fixture
def store(monkeypatch):
    from rag import vector_store as vs

    monkeypatch.setattr(vs, "embed", _fake_embed)
    s = vs.HybridVectorStore()
    s.add_documents(
        [
            {"page_content": "Gradient descent minimises a loss function step by step", "source": "a.pdf", "page": 1},
            {"page_content": "The mitochondria is the powerhouse of the cell", "source": "a.pdf", "page": 2},
            {"page_content": "Reciprocal rank fusion merges two ranked lists", "source": "a.pdf", "page": 3},
        ],
        source_label="a.pdf",
    )
    return s


def test_hybrid_search_finds_the_matching_chunk(store):
    top = store.hybrid_search("what does reciprocal rank fusion do", k=2)
    assert top[0]["page"] == 3
    assert 0 < top[0]["score"] <= 2 / 60


def test_clear_empties_the_store(store):
    store.clear()
    assert store.chunk_count == 0
    assert store.hybrid_search("anything") == []


# ── SSRF guard ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://localhost:7860/",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/",
    "http://192.168.1.1/",
    "http://[::1]/",
])
def test_private_addresses_are_refused(url):
    from rag.ingestor import _assert_public_host

    with pytest.raises(ValueError):
        _assert_public_host(url)


# ── Flask routes ─────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    import app as app_module

    app_module.vector_store.clear()
    app_module.queries.clear()
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), app_module


def test_health_reports_ok(client):
    c, _ = client
    body = c.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["chunks_stored"] == 0


def test_research_requires_a_source(client):
    c, _ = client
    r = c.post("/api/research", json={"question": "hello?"})
    assert r.status_code == 400
    assert "No source" in r.get_json()["error"]


def test_ingest_url_refuses_private_addresses(client):
    c, _ = client
    r = c.post("/api/ingest_url", json={"url": "http://127.0.0.1:7860/health"})
    assert r.status_code == 400
    assert "public" in r.get_json()["error"].lower()


def test_unknown_model_is_rejected(client):
    c, _ = client
    assert c.post("/api/set_model", json={"model": "nope"}).status_code == 400


def _key_env():
    from agents import llm_factory

    return llm_factory._PROVIDER["key_env"]


@pytest.mark.parametrize("raw, expected", [
    ("Error code: 401 - {'error': 'Invalid API Key'}", "rejected"),
    ("Error code: 401 - {'error': 'Invalid username or password.'}", "rejected"),
    ("Error code: 403 - forbidden", "permission"),
    ("Error code: 429 - slow down", "rate limited"),
])
def test_provider_errors_become_actionable_messages(client, raw, expected):
    _, app_module = client
    assert expected in app_module._friendly_error(Exception(raw))


def test_missing_key_error_names_the_secret_to_set(client):
    _, app_module = client
    key = _key_env()
    message = app_module._friendly_error(EnvironmentError(f"{key} is not set. Add it."))
    assert f"No {key} secret" in message


# ── LLM factory ──────────────────────────────────────────────────────────────

def test_groq_is_chosen_only_when_its_key_is_set(monkeypatch):
    from agents import llm_factory

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert llm_factory.active_provider() == "groq"
    monkeypatch.delenv("GROQ_API_KEY")
    assert llm_factory.active_provider() == "huggingface"


def test_every_model_has_what_the_picker_needs():
    from agents import llm_factory

    for provider in llm_factory.PROVIDERS.values():
        assert provider["default"] in provider["models"]
        for model in provider["models"].values():
            assert {"id", "label", "desc", "speed", "params", "color", "reasoning"} <= model.keys()


def test_reasoning_models_get_extra_token_headroom(monkeypatch):
    from agents import llm_factory

    monkeypatch.setenv(llm_factory._PROVIDER["key_env"], "test-key")
    reasoning_key = next(
        (k for k, m in llm_factory.AVAILABLE_MODELS.items() if m["reasoning"]), None
    )
    plain_key = next((k for k, m in llm_factory.AVAILABLE_MODELS.items() if not m["reasoning"]), None)
    original = llm_factory._current_model
    try:
        if reasoning_key:
            llm_factory.set_model(reasoning_key)
            assert llm_factory.get_llm(max_tokens=100).bound.max_tokens > 100
        if plain_key:
            llm_factory.set_model(plain_key)
            assert llm_factory.get_llm(max_tokens=100).bound.max_tokens == 100
    finally:
        llm_factory.set_model(original)


def test_get_llm_without_a_key_explains_what_to_set(monkeypatch):
    from agents import llm_factory

    monkeypatch.delenv(llm_factory._PROVIDER["key_env"], raising=False)
    with pytest.raises(EnvironmentError, match=llm_factory._PROVIDER["key_env"]):
        llm_factory.get_llm()


def test_failed_research_run_surfaces_a_friendly_error(client, monkeypatch):
    import time

    c, app_module = client
    # Indexing needs the embedding model, so pretend one document is loaded.
    monkeypatch.setattr(type(app_module.vector_store), "doc_count", property(lambda self: 1))

    def boom(question, qid):
        raise RuntimeError("Error code: 401 - {'error': 'Invalid API Key'}")

    monkeypatch.setattr(app_module.graph, "run", boom)
    qid = c.post("/api/research", json={"question": "why?"}).get_json()["query_id"]
    for _ in range(50):
        body = c.get(f"/api/trace/{qid}").get_json()
        if body["status"] != "running":
            break
        time.sleep(0.05)
    assert body["status"] == "error"
    assert _key_env() in body["result"]["error"]


def test_query_history_is_bounded(client, monkeypatch):
    c, app_module = client
    monkeypatch.setattr(type(app_module.vector_store), "doc_count", property(lambda self: 1))
    monkeypatch.setattr(app_module.graph, "run", lambda q, i: {"ok": True})
    for _ in range(app_module.MAX_TRACKED_QUERIES + 20):
        assert c.post("/api/research", json={"question": "x?"}).status_code == 200
    assert len(app_module.queries) <= app_module.MAX_TRACKED_QUERIES
