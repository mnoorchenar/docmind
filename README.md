---
title: DocMind-Agentic-Research
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🧠 DocMind — Agentic Research Platform</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=4f8ef7&center=true&vCenter=true&width=700&lines=LangGraph+%C2%B7+5+Agents+%C2%B7+Hybrid+RAG;Qwen+2.5-7B+%C2%B7+3+LLM+Calls+per+Query;Deployed+Free+on+HuggingFace+Spaces" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-06b6d4?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-4f46e5?style=for-the-badge)](https://langchain.com/)
[![Flask](https://img.shields.io/badge/Flask-3.1-3b82f6?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🧠 DocMind** — A clean, minimal agentic document research platform. Five specialized LangGraph agents plan, retrieve, grade, generate, and critique answers from uploaded PDFs and web pages using hybrid search and Qwen 2.5-7B — all running free on HuggingFace Spaces.

<br/>

---

</div>

## Table of Contents
- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr><td>🧠 <b>LangGraph State Machine</b></td><td>Five agents wired into a linear StateGraph — Planner → Retriever → Grader → Generator → Critic.</td></tr>
  <tr><td>🔍 <b>Hybrid RAG (FAISS + BM25)</b></td><td>Semantic vector search combined with BM25 keyword search, fused via Reciprocal Rank Fusion for precision retrieval.</td></tr>
  <tr><td>🤖 <b>Multi-Agent Orchestration</b></td><td>Planner, Retriever, Grader, Generator, and Critic agents each with specialized roles — only 3 LLM calls per query.</td></tr>
  <tr><td>⚡ <b>Score-Based Grading</b></td><td>Grader uses hybrid search scores + keyword overlap — no LLM call needed, instant and deterministic relevance scoring.</td></tr>
  <tr><td>📄 <b>PDF &amp; URL Ingestion</b></td><td>Upload PDF files up to 10 MB or paste any public URL — both are chunked, embedded, and indexed automatically.</td></tr>
  <tr><td>🔒 <b>Secure by Design</b></td><td>Stateless REST backend, no user data persisted, HF token kept server-side only.</td></tr>
  <tr><td>🐳 <b>Containerized Deployment</b></td><td>Docker-first with Gunicorn, embedding model pre-downloaded at build time for fast cold starts.</td></tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   DocMind — LangGraph Flow                    │
│                                                              │
│  PDF / URL ──▶ Ingestor ──▶ FAISS+BM25 Hybrid Vector Store  │
│                                    │                         │
│  User Query ──▶ [PLANNER Agent]    │   (Qwen 2.5-7B, 0.3)   │
│                      │             │                         │
│                 [RETRIEVER] ◀──────┘  (FAISS+BM25+RRF)      │
│                      │                                       │
│                 [GRADER]  (score-based, no LLM call)         │
│                      │                                       │
│                 [GENERATOR]         (Qwen 2.5-7B, 0.4)       │
│                      │                                       │
│                  [CRITIC]           (Qwen 2.5-7B, 0.1)       │
│                      │                                       │
│                  [OUTPUT]  Flask API + Single-Page UI         │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ · Docker · Git · Free HuggingFace account

### Local Installation

```bash
git clone https://github.com/mnoorchenar/docmind.git
cd docmind

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set HF_TOKEN to your free HuggingFace Read token

python app.py
```

Open `http://localhost:7860` 🎉

### Getting your free HuggingFace token
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens → New Token → Role: **Read**
3. Copy the token and set it as `HF_TOKEN` in your `.env` file or Space secrets

---

## 🐳 Docker Deployment

```bash
docker build -t docmind .
docker run -p 7860:7860 -e HF_TOKEN=hf_your_token_here docmind
```

---

## 📊 App Modules

| Module | Description | Status |
|--------|-------------|--------|
| 📤 Upload & Index | PDF / URL ingest, chunk, embed (local BAAI model), FAISS+BM25 index | ✅ Live |
| 🔍 Research Query | LangGraph 5-agent pipeline with real-time trace log | ✅ Live |

---

## 🧠 ML Models

```python
stack = {
    # ── LLM (LangChain LCEL chains) ──────────────────────────────────────────
    "llm":             "Qwen/Qwen2.5-7B-Instruct",         # via HF Router
    "lcel_chain":      "ChatPromptTemplate | ChatOpenAI | StrOutputParser",
    "retry":           "ChatOpenAI.with_retry(stop_after_attempt=2)",

    # ── RAG (LangChain + custom hybrid) ──────────────────────────────────────
    "splitter":        "RecursiveCharacterTextSplitter (langchain-text-splitters)",
    "documents":       "langchain_core.documents.Document",
    "embeddings":      "HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5, local)",
    "vector_index":    "FAISS IndexFlatIP (cosine)",
    "keyword_index":   "BM25Okapi (rank-bm25)",
    "fusion":          "Reciprocal Rank Fusion (RRF k=60)",
    "grader":          "score-based (hybrid score × 0.7 + keyword overlap × 0.3)",

    # ── Orchestration (LangGraph) ─────────────────────────────────────────────
    "graph":           "LangGraph 0.2 StateGraph — 5 nodes, linear pipeline",
}
```

---

## 📁 Project Structure

```
docmind/
├── 📄 app.py                     # Flask entry point, 5 REST routes
├── 📄 requirements.txt
├── 📄 Dockerfile                 # Port 7860, embedding model pre-downloaded
├── 📄 .env.example
├── 📂 agents/
│   ├── 📄 llm_factory.py         # get_llm() → LangChain ChatOpenAI (HF Router)
│   ├── 📄 planner.py             # LCEL: ChatPromptTemplate | ChatOpenAI | StrOutputParser
│   ├── 📄 retriever.py           # Hybrid FAISS+BM25 search wrapper
│   ├── 📄 grader.py              # Score-based relevance grading (no LLM call)
│   ├── 📄 generator.py           # LCEL chain — cited answer generation
│   └── 📄 critic.py              # LCEL chain — hallucination detection
├── 📂 graph/
│   └── 📄 research_graph.py      # LangGraph StateGraph (5 nodes, linear pipeline)
├── 📂 rag/
│   ├── 📄 ingestor.py            # RecursiveCharacterTextSplitter + Document objects
│   ├── 📄 vector_store.py        # FAISS + BM25 + RRF, accepts Document or dict
│   └── 📄 embeddings.py          # LangChain HuggingFaceEmbeddings (bge-small-en-v1.5)
├── 📂 tracing/
│   └── 📄 tracer.py              # Thread-safe in-memory trace store
├── 📂 templates/
│   └── 📄 index.html             # Dark-mode single-page UI
└── 📂 docs/
    └── 📄 project-template.html  # Portfolio showcase page
```

---

## 👨‍💻 Author

<div align="center">
<table><tr><td align="center" width="100%">
<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%;border:3px solid #4f46e5" alt="Mohammad Noorchenarboo"/>
<h3>Mohammad Noorchenarboo</h3>
<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>
📍 Ontario, Canada &nbsp;&nbsp; 📧 mohammadnoorchenarboo@gmail.com

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)
</td></tr></table>
</div>

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes. All LLM outputs are AI-generated and may contain inaccuracies. No real user data is stored. Provided "as is" without warranty of any kind.</span>

---

## 📜 License

Distributed under the **MIT License**.

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>
</div>