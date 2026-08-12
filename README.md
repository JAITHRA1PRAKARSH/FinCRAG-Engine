# 📈 FinCRAG: Corrective RAG Financial Agent for SEC 10-K Filings

An enterprise-grade, self-correcting financial analysis engine built with **LangGraph**, **Qdrant**, and **Groq (Llama 3)**. 

FinCRAG ingests multi-hundred-page SEC 10-K filings (Apple Inc. & Microsoft Corp.), preserves complex financial tables with section-aware metadata injection, and utilizes an autonomous self-reflection graph to grade context relevance and eliminate hallucinations.

---

## 🏛️ System Architecture

[ User Query ] ──► [ Streamlit UI / FastAPI Gateway ]
│
▼
[ LangGraph Orchestrator ]
│
┌───────────────────────┴───────────────────────┐
▼                                               ▼
[ Hybrid Retrieval Engine ]               [ Autonomous Self-Correction Loop ]
├── Dense Vector Search (Qdrant + FastEmbed)       ├── Grader Node (Relevance Audit)
└── Sparse Keyword Search (BM25Okapi)              └── Query Rewriter (Intent Targeting)
│                                               ▲
▼                                               │
[ Reciprocal Rank Fusion (RRF) ] ───────────────────┘
│
▼
[ Llama 3 Synthesis with Inline Citations ] ──► [ Verifiable Answer ]

---

## 🌟 Key Engineering Highlights

* **Table-Aware Context Injection:** Ingests SEC filings via `MarkdownHeaderTextSplitter` and prepends explicit metadata headers (`[Company: Apple Inc. (AAPL) | Filing: Form 10-K]`) to every chunk to eliminate cross-company context leakage.
* **Hybrid Retrieval (Dense + Sparse):** Combines semantic embeddings (`BAAI/bge-small-en-v1.5`) with exact keyword matching (`BM25Okapi`), merged via **Reciprocal Rank Fusion (RRF)**.
* **Autonomous Reflection & Correction (CRAG):** Uses LangGraph to orchestrate a stateful self-correction cycle. If the internal Grader Node detects incomplete context, it executes an autonomous query rewriting cycle and expands search breadth ($k=6 \rightarrow k=10$).
* **Tiered LLM Execution:** Employs `llama-3.1-8b-instant` for low-latency background grading and query rewriting, reserving `llama-3.3-70b-versatile` for final citation-backed answer synthesis.
* **RAGAS Benchmark Evaluated:** Scientifically evaluated with the RAGAS framework, achieving a **96.15% Faithfulness score**.

---

## 📊 Evaluation & Benchmarks

| Metric | Score | Target | Description |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | **96.15%** | > 85.0% | Measures factual grounding against retrieved SEC disclosures. |
| **Context Precision** | **63.8%** | > 50.0% | Ratio of relevant context to total retrieved chunks. |

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph / LangChain Core
* **LLMs:** Llama 3.3 70B & Llama 3.1 8B (via Groq LPU)
* **Vector Store:** Qdrant
* **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`)
* **Sparse Search:** Rank-BM25
* **Frontend:** Streamlit
* **Evaluation:** RAGAS & Pytest

---

## 🚀 Quickstart

### 1. Clone & Setup Environment
```bash
git clone [https://github.com/JAITHRA1PRAKARSH/FinCRAG-Engine.git](https://github.com/JAITHRA1PRAKARSH/FinCRAG-Engine.git)
cd FinCRAG-Engine

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
