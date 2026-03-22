---
title: DocMind-Agentic-Research
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🧠 DocMind — Agentic Research Platform</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=4f8ef7&center=true&vCenter=true&width=700&lines=LangGraph+%C2%B7+5+Agents+%C2%B7+Corrective+RAG;Multi-Agent+Orchestration+%C2%B7+Human-in-the-Loop;Deployed+Free+on+HuggingFace+Spaces" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-06b6d4?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-4f46e5?style=for-the-badge)](https://langchain.com/)
[![Flask](https://img.shields.io/badge/Flask-3.1-3b82f6?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🧠 DocMind** — A production-grade agentic document research platform. Five specialized LangGraph agents plan, retrieve, grade, generate, and critique answers from uploaded PDFs using Corrective RAG, hybrid search, human-in-the-loop review, and LangSmith-style observability — all running free on HuggingFace Spaces.

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
  <tr><td>🧠 <b>LangGraph State Machine</b></td><td>Five agents wired into a cyclic StateGraph with conditional edges and Corrective RAG rewrite loops.</td></tr>
  <tr><td>🔍 <b>Hybrid RAG (FAISS + BM25)</b></td><td>Semantic vector search combined with BM25 keyword search, fused via Reciprocal Rank Fusion for precision retrieval.</td></tr>
  <tr><td>🤖 <b>Multi-Agent Orchestration</b></td><td>Planner, Retriever, Grader, Generator, and Critic agents each with specialized roles and distinct LLM temperature settings.</td></tr>
  <tr><td>👁️ <b>Human-in-the-Loop</b></td><td>Answers failing the Critic agent's quality threshold are routed to a human review queue before delivery.</td></tr>
  <tr><td>📊 <b>Observability Dashboard</b></td><td>Per-agent call counts, average latency, and Chart.js visualizations — LangSmith-style tracing without the paid tier.</td></tr>
  <tr><td>🔧 <b>Tool Use / Function Calling</b></td><td>Three real tools: DuckDuckGo web search, safe AST calculator, and sandboxed Python code execution.</td></tr>
  <tr><td>🔒 <b>Secure by Design</b></td><td>Stateless REST backend, no user data persisted, sandboxed code tool with restricted builtins only.</td></tr>
  <tr><td>🐳 <b>Containerized Deployment</b></td><td>Docker-first with Gunicorn, embedding model pre-downloaded at build time for fast cold starts.</td></tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   DocMind — LangGraph Flow                    │
│                                                              │
│  PDF Upload ──▶ Ingestor ──▶ FAISS+BM25 Hybrid Vector Store │
│                                    │                         │
│  User Query ──▶ [PLANNER Agent]    │                         │
│                      │             │                         │
│                 [RETRIEVER] ◀──────┘  (hybrid search)        │
│                      │                                       │
│                 [GRADER] ──▶ low score? ──▶ [REWRITER] ──┐  │
│                      │                                   │  │
│                      └──▶ [GENERATOR] ◀──────────────────┘  │
│                                │                             │
│                           [CRITIC] ──▶ flag? ──▶ [REVIEW]   │
│                                │                             │
│                            [OUTPUT]  Flask API + SPA UI      │
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

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 📤 Upload & Index | PDF ingest, chunk, embed (local), FAISS+BM25 index | ✅ Live |
| 🔍 Research Query | LangGraph 5-agent pipeline with real-time trace | ✅ Live |
| 👁️ Human Review | Critic escalation queue with approve/reject | ✅ Live |
| 📊 Observability | Per-agent latency, call counts, Chart.js dashboard | ✅ Live |
| 🔧 Tool Playground | Web search, calculator, code runner | ✅ Live |

---

## 🧠 ML Models

```python
models = {
    "planner_generator": "mistralai/Mistral-7B-Instruct-v0.3",
    "grader_critic":     "HuggingFaceH4/zephyr-7b-beta",
    "embeddings":        "BAAI/bge-small-en-v1.5",
    "vector_index":      "FAISS (faiss-cpu, local)",
    "keyword_index":     "BM25 (rank-bm25, local)",
    "fusion_strategy":   "Reciprocal Rank Fusion (RRF k=60)",
    "graph_framework":   "LangGraph 0.2 StateGraph",
    "chain_syntax":      "LangChain LCEL (prompt | llm)",
}
```

---

## 📁 Project Structure

```
docmind/
├── 📄 app.py                     # Flask entry point, 10 REST routes
├── 📄 requirements.txt
├── 📄 Dockerfile                 # Port 7860, embedding model pre-downloaded
├── 📄 .env.example
├── 📂 agents/
│   ├── 📄 planner.py             # Mistral-7B — task decomposition
│   ├── 📄 retriever.py           # Hybrid FAISS+BM25 search wrapper
│   ├── 📄 grader.py              # Zephyr-7B — 0.0–1.0 relevance scoring
│   ├── 📄 generator.py           # Mistral-7B — cited answer generation
│   └── 📄 critic.py              # Zephyr-7B — hallucination detection
├── 📂 graph/
│   └── 📄 research_graph.py      # LangGraph StateGraph (5 nodes + conditional edges)
├── 📂 rag/
│   ├── 📄 ingestor.py            # PyPDF + overlapping chunker
│   ├── 📄 vector_store.py        # FAISS + BM25 + RRF fusion
│   └── 📄 embeddings.py          # sentence-transformers local wrapper
├── 📂 tools/
│   ├── 📄 web_search.py          # DuckDuckGo free search
│   ├── 📄 calculator.py          # AST-safe math evaluator
│   └── 📄 code_tool.py           # Sandboxed Python exec
├── 📂 tracing/
│   └── 📄 tracer.py              # Thread-safe in-memory trace store
├── 📂 templates/
│   └── 📄 index.html             # Dark-mode 5-page SPA
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