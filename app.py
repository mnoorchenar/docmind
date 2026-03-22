import os, uuid, threading
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from rag.vector_store     import HybridVectorStore
from rag.ingestor         import PDFIngestor, URLIngestor, MAX_PDF_BYTES
from graph.research_graph import ResearchGraph
from tracing.tracer       import Tracer

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

UPLOAD_FOLDER = "/tmp/docmind_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vector_store = HybridVectorStore()
tracer       = Tracer()
graph        = ResearchGraph(vector_store, tracer)
queries      = {}


def _clear_uploads():
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except Exception:
            pass


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
    f.seek(0, 2); size = f.tell(); f.seek(0)
    if size > MAX_PDF_BYTES:
        return jsonify({"error": f"File exceeds 10 MB limit ({size/1024/1024:.1f} MB)."}), 400
    _clear_uploads()
    path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(path)
    try:
        chunks = PDFIngestor().ingest(path)
        vector_store.clear()
        vector_store.add_documents(chunks, source_label=f.filename)
        return jsonify({"success": True, "filename": f.filename, "chunks": len(chunks)})
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
        return jsonify({"success": True, "url": url, "chunks": len(chunks)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/research", methods=["POST"])
def research():
    data     = request.json or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400
    if vector_store.doc_count == 0:
        return jsonify({"error": "No source loaded — please upload a PDF or fetch a URL first."}), 400
    qid = str(uuid.uuid4())
    queries[qid] = {"status": "running", "result": None}

    def _run():
        try:
            result = graph.run(question, qid)
            queries[qid].update({"status": "complete", "result": result})
        except Exception as exc:
            queries[qid].update({"status": "error", "result": {"error": str(exc)}})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"query_id": qid})


@app.route("/api/trace/<qid>")
def get_trace(qid):
    q = queries.get(qid)
    if not q:
        return jsonify({"error": "Query not found."}), 404
    return jsonify({"status": q["status"], "trace": tracer.get(qid), "result": q["result"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
