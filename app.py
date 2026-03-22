import os, uuid, threading
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from rag.vector_store import HybridVectorStore
from rag.ingestor import PDFIngestor
from graph.research_graph import ResearchGraph
from tracing.tracer import Tracer
from tools.web_search import web_search
from tools.calculator import calculate
from tools.code_tool import run_code

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

UPLOAD_FOLDER = "/tmp/docmind_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Global singletons (in-memory, scoped to container lifetime) ───────────
vector_store = HybridVectorStore()
tracer       = Tracer()
graph        = ResearchGraph(vector_store, tracer)
queries      = {}   # query_id → {status, result}


# ── ROUTES ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "docs_indexed": vector_store.doc_count,
        "chunks_stored": vector_store.chunk_count,
        "token_set": bool(os.getenv("HF_TOKEN")),
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file attached."}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400
    path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(path)
    try:
        chunks = PDFIngestor().ingest(path)
        vector_store.add_documents(chunks)
        return jsonify({
            "success": True,
            "filename": f.filename,
            "chunks": len(chunks),
            "total_chunks": vector_store.chunk_count,
            "total_docs": vector_store.doc_count,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/research", methods=["POST"])
def research():
    data     = request.json or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400
    if vector_store.doc_count == 0:
        return jsonify({"error": "No documents indexed yet — please upload a PDF first."}), 400

    qid = str(uuid.uuid4())
    queries[qid] = {"status": "running", "result": None}

    def _run():
        try:
            result = graph.run(question, qid)
            queries[qid]["result"]  = result
            queries[qid]["status"]  = "pending_review" if result.get("needs_human_review") else "complete"
        except Exception as exc:
            queries[qid]["status"] = "error"
            queries[qid]["result"] = {"error": str(exc)}

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"query_id": qid})


@app.route("/api/trace/<qid>")
def get_trace(qid):
    q = queries.get(qid)
    if not q:
        return jsonify({"error": "Query not found."}), 404
    return jsonify({"status": q["status"], "trace": tracer.get(qid), "result": q["result"]})


@app.route("/api/review")
def review_queue():
    pending = [
        {"query_id": qid,
         "question":   q["result"].get("question", "") if q["result"] else "",
         "generation": q["result"].get("generation", "") if q["result"] else "",
         "critique":   q["result"].get("critique", "") if q["result"] else ""}
        for qid, q in queries.items()
        if q["status"] == "pending_review" and q["result"]
    ]
    return jsonify({"pending": pending})


@app.route("/api/review/<qid>", methods=["POST"])
def review_action(qid):
    data   = request.json or {}
    action = data.get("action")
    if qid not in queries:
        return jsonify({"error": "Query not found."}), 404
    if action not in ("approve", "reject"):
        return jsonify({"error": "Action must be 'approve' or 'reject'."}), 400
    queries[qid]["status"] = "complete" if action == "approve" else "rejected"
    if queries[qid]["result"]:
        queries[qid]["result"]["human_approved"] = action == "approve"
    tracer.add(qid, "human_review", f"Reviewer {action}d this answer.", "complete", 0)
    return jsonify({"success": True})


@app.route("/api/observability")
def observability():
    return jsonify(tracer.stats())


@app.route("/api/tool/<name>", methods=["POST"])
def tool_run(name):
    inp = ((request.json or {}).get("input") or "").strip()
    if not inp:
        return jsonify({"error": "Input is required."}), 400
    try:
        result = {"web_search": web_search, "calculator": calculate, "code": run_code}.get(name, lambda _: None)(inp)
        if result is None:
            return jsonify({"error": f"Unknown tool '{name}'."}), 400
        return jsonify({"result": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stats")
def stats():
    return jsonify({
        "docs_indexed":    vector_store.doc_count,
        "chunks_stored":   vector_store.chunk_count,
        "queries_run":     len(queries),
        "queries_complete":sum(1 for q in queries.values() if q["status"] == "complete"),
        "pending_review":  sum(1 for q in queries.values() if q["status"] == "pending_review"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)