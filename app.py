import os, uuid, threading
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from rag.vector_store import HybridVectorStore
from rag.ingestor     import PDFIngestor, URLIngestor, SearchIngestor, MAX_PDF_BYTES
from graph.research_graph import ResearchGraph
from tracing.tracer import Tracer
from tools.web_search import web_search
from tools.calculator import calculate
from tools.code_tool  import run_code

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

UPLOAD_FOLDER = "/tmp/docmind_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vector_store = HybridVectorStore()
tracer       = Tracer()
graph        = ResearchGraph(vector_store, tracer)
queries      = {}   # query_id → {status, result}


# ── HELPERS ───────────────────────────────────────────────────────────────

def _clear_uploads():
    """Remove previously uploaded PDFs to free /tmp space."""
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except Exception:
            pass


# ── ROUTES ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "docs_indexed":  vector_store.doc_count,
        "chunks_stored": vector_store.chunk_count,
        "source":        vector_store.source_label,
        "token_set":     bool(os.getenv("HF_TOKEN")),
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file attached."}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # Check size before saving
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    if size > MAX_PDF_BYTES:
        return jsonify({"error": f"File exceeds 10 MB limit ({size/1024/1024:.1f} MB)."}), 400

    _clear_uploads()
    path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(path)

    try:
        chunks = PDFIngestor().ingest(path)
        vector_store.clear()
        vector_store.add_documents(chunks, source_label=f.filename)
        return jsonify({
            "success":  True,
            "filename": f.filename,
            "chunks":   len(chunks),
            "total_chunks": vector_store.chunk_count,
            "total_docs":   vector_store.doc_count,
            "source":       f.filename,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ingest_url", methods=["POST"])
def ingest_url():
    data = request.json or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required."}), 400
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        chunks = URLIngestor().ingest(url)
        vector_store.clear()
        _clear_uploads()
        vector_store.add_documents(chunks, source_label=url)
        return jsonify({
            "success": True,
            "url":     url,
            "chunks":  len(chunks),
            "total_chunks": vector_store.chunk_count,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/search_ingest", methods=["POST"])
def search_ingest():
    data  = request.json or {}
    query = (data.get("query") or "").strip()
    site  = (data.get("site")  or "").strip()
    if not query:
        return jsonify({"error": "Search query is required."}), 400
    try:
        result = SearchIngestor().search_and_ingest(query, site)
        chunks = result["chunks"]
        vector_store.clear()
        _clear_uploads()
        vector_store.add_documents(chunks, source_label=result["url"])
        return jsonify({
            "success": True,
            "url":     result["url"],
            "title":   result["title"],
            "chunks":  len(chunks),
            "total_chunks": vector_store.chunk_count,
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
        return jsonify({"error": "No knowledge base loaded — please fetch a URL, search, or upload a PDF first."}), 400

    qid = str(uuid.uuid4())
    queries[qid] = {"status": "running", "result": None}

    def _run():
        try:
            result = graph.run(question, qid)
            queries[qid]["result"] = result
            queries[qid]["status"] = "pending_review" if result.get("needs_human_review") else "complete"
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
         "question":   q["result"].get("question","")   if q["result"] else "",
         "generation": q["result"].get("generation","") if q["result"] else "",
         "critique":   q["result"].get("critique","")   if q["result"] else ""}
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
        "source":          vector_store.source_label,
        "queries_run":     len(queries),
        "queries_complete":sum(1 for q in queries.values() if q["status"] == "complete"),
        "pending_review":  sum(1 for q in queries.values() if q["status"] == "pending_review"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
