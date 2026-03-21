---
title: Med RAG Bot
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
hardware: t4-small
---

# Med RAG Bot

> **Production-grade Retrieval-Augmented Generation for Indian Public Health.**
> Built from scratch — no LangChain wrappers. Every layer documented and explainable.
> Drop your own PDFs to build a grounded QA system for any domain in under 30 minutes.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/LLM-Qwen3--4B-orange)
![Retrieval](https://img.shields.io/badge/Retrieval-BM25%20%2B%20FAISS%20%2B%20RRF-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)

---

## What is Med RAG Bot?

Med RAG Bot is an end-to-end RAG (Retrieval-Augmented Generation) system built for **Indian public health**. It answers medical and public health questions by retrieving evidence from real government documents — ICMR guidelines, NCDC clinical manuals, WHO reports, and MOHFW protocols — and generating grounded, cited responses.

**Started:** March 16, 2026 · **Corpus:** ICMR · NCDC · WHO · MOHFW

The system is **domain-agnostic by design**: replace the corpus with legal documents, pharma guidelines, HR policies, or any PDF collection and get a production-grade QA system for your domain.

---

## Why Build From Scratch?

Most RAG tutorials wrap LangChain in 50 lines and call it done. This project takes the opposite approach — every component is implemented and documented so you understand **why** each architectural choice was made.

| What most RAG repos do | What this project does |
|---|---|
| `LangChain.VectorStore.similarity_search()` | BM25 + FAISS + RRF fusion, explained and tunable |
| Single vector search | Hybrid retrieval: keyword + semantic, combined with Reciprocal Rank Fusion |
| No re-ranking | Cross-encoder re-ranker for second-pass relevance scoring |
| Print the LLM output | Response formatter: query-type routing, citation parsing, confidence tiers |
| No observability | Full session logging → Streamlit dashboard → cost estimator (USD + INR) |
| No citation tracking | Every answer grounded with source URL, page number, document type |

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │           PDF Corpus Pipeline           │
                          │                                         │
  ICMR / NCDC / WHO ────► │  Download → Language Filter → Extract   │
  MOHFW documents         │  → Chunk (atomic + parent) → Dedupe     │
                          └──────────────┬──────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────────┐
                          │              Index Layer                │
                          │                                         │
                          │  BM25 Index (keyword)                   │
                          │  FAISS IndexFlatIP (semantic vectors)   │
                          │  Embedding: BAAI/bge-base-en-v1.5       │
                          └──────────────┬──────────────────────────┘
                                         │
              User Query ───────────────►│
                                         │
                          ┌──────────────▼──────────────────────────┐
                          │           Hybrid Retriever              │
                          │                                         │
                          │  BM25 top-20  +  FAISS top-20           │
                          │         ↓ RRF fusion (k=60)             │
                          │  Cross-encoder re-ranker (top-5)        │
                          └──────────────┬──────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────────┐
                          │           Generation Layer              │
                          │                                         │
                          │  Qwen3-4B (4-bit NF4 quantized)         │
                          │  Query-type routing (7 types)           │
                          │  Thinking budget control                │
                          │  Conversation memory (last 3 turns)     │
                          └──────────────┬──────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────────┐
                          │         Response Formatter              │
                          │                                         │
                          │  Citation parsing + URL grounding       │
                          │  Confidence tiers (HIGH / MED / LOW)    │
                          │  Follow-up question extraction          │
                          │  Template artifact stripping            │
                          └──────────────┬──────────────────────────┘
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   │                     │                      │
          ┌────────▼──────┐   ┌──────────▼───────┐   ┌────────▼──────────┐
          │   CLI (Rich)  │   │  Streamlit Dash  │   │  Gradio UI (WIP)  │
          │  chat · index │   │  Observability   │   │  HF Spaces        │
          └───────────────┘   └──────────────────┘   └───────────────────┘
                                         │
                          ┌──────────────▼──────────────────────────┐
                          │          Observability Layer            │
                          │                                         │
                          │  JSONL session logging (local)          │
                          │  Streamlit dashboard (5 pages)          │
                          │  Cost estimator USD + INR               │
                          │  MongoDB upload (when ready)            │
                          └─────────────────────────────────────────┘
```

---

## Key Features

### Retrieval
- **Hybrid BM25 + FAISS**: keyword search covers exact medical terms (drug names, ICD codes), semantic search covers conceptual queries — combined via Reciprocal Rank Fusion
- **Cross-encoder re-ranking**: second pass scores each (query, document) pair jointly for much higher precision
- **Atomic + parent chunks**: atomic chunks (220 words) for precise retrieval, parent chunks (700 words) for context
- Configurable `top_k` and `fetch_k` at runtime

### Generation
- **Qwen3-4B with thinking mode**: extended reasoning before answering, with configurable token budget
- **7 query types detected**: FACTUAL, GUIDELINE, PROCEDURE, STATISTICS, COMPARISON, DEFINITION, INSUFFICIENT
- **Grounded responses only**: model instructed to cite `[1][2]` inline with source URL + page number
- **Conversation memory**: last 3 turns passed as context

### Observability
- Every session and message logged to local JSONL files
- 5-page Streamlit dashboard: Overview, Messages, Retrieval, Performance, Feedback
- Cloud cost estimator: shows what each query would cost on GPT-4o, Claude Sonnet, Gemini Flash in USD and INR
- One-command MongoDB upload when ready for production

### CLI UX
- Rich terminal interface with panels, confidence badges, source citations
- Live slash commands: `/think`, `/show-think`, `/budget <N>`, `/chunks`, `/verbose`, `/feedback`, `/export`
- Session export to markdown

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Qwen3-4B (4-bit NF4) | Best open-source reasoning model at 4B params; thinking mode for complex queries |
| Embeddings | BAAI/bge-base-en-v1.5 | Strong English semantic understanding, fast, 768-dim |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Lightweight cross-encoder, ~22MB, joint (query, doc) scoring |
| Vector index | FAISS IndexFlatIP | Exact inner-product search, no approximation error at corpus scale |
| Keyword index | BM25 (rank-bm25) | Proven for medical term matching; complementary to semantic |
| Rank fusion | RRF (k=60) | Parameter-free, robust combination of heterogeneous rankers |
| PDF extraction | PyMuPDF (fitz) | Fast, reliable text extraction with page-level metadata |
| Quantization | bitsandbytes NF4 | 4-bit quantization with double quant; fits Qwen3-4B on 6GB VRAM |
| CLI | Typer + Rich | Clean, declarative CLI with beautiful terminal output |
| Dashboard | Streamlit | Rapid observability UI; no frontend skills required |
| Serialization | orjson | 5-10x faster than stdlib json for JSONL logging |
| Database | MongoDB (optional) | Session persistence; JSONL local logging works without it |

---

## Project Structure

```
med_rag_bot/
├── app/
│   ├── cli.py                    # Main CLI entry point (chat, index, upload-logs)
│   ├── settings.py               # Centralised path + config management
│   ├── console.py                # Rich console helpers
│   ├── extraction/               # PDF download → text extraction → chunking
│   │   ├── corpus_filter.py      # Language detection, doc-type tagging, deduplication
│   │   ├── pdf_extractor.py      # PyMuPDF text extraction per page
│   │   └── pdf_chunker.py        # Atomic + parent chunk generation with metadata
│   ├── retrieval/
│   │   ├── bm25_index.py         # BM25 index build + search
│   │   ├── vector_index.py       # FAISS build + embedding model wrapper
│   │   ├── hybrid_retriever.py   # RRF fusion of BM25 + FAISS results
│   │   └── reranker.py           # Cross-encoder second-pass re-ranking
│   ├── generation/
│   │   ├── llm_client.py         # Qwen3-4B loader, generate(), thinking budget
│   │   ├── prompt_builder.py     # System prompt, query-type templates, history injection
│   │   └── response_formatter.py # Citation parsing, confidence, follow-ups, artifact stripping
│   ├── monitoring/
│   │   ├── logger.py             # JSONL session/message/retrieval/feedback logging
│   │   └── db_client.py          # MongoDB client (graceful offline fallback)
│   └── dashboard/
│       └── streamlit_app.py      # 5-page observability dashboard
├── scripts/                      # One-time corpus pipeline scripts
│   ├── pdf_downloader.py         # Scrape + download PDFs from ICMR/NCDC/WHO/MOHFW
│   ├── filter_language_and_tag_docs.py
│   ├── second_pass.py
│   ├── investigate_pdfs.py
│   └── who.py
├── configs/                      # YAML configuration files
├── data/
│   ├── indexes/                  # BM25 + FAISS indexes (hosted on HF Hub)
│   ├── processed/                # Chunked JSONL (regenerated by pipeline)
│   └── logs/                     # JSONL observability logs (local, gitignored)
├── tests/
├── notebooks/
└── misc/                         # Experimental prototypes (OpenCV region parsers etc.)
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA GPU with 6GB+ VRAM (or CPU with patience)
- Conda recommended

### 1. Clone and install
```bash
git clone https://github.com/KPrashanth/med_rag_bot.git
cd med_rag_bot
conda create -n medrag python=3.10 -y
conda activate medrag
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set MONGODB_URI if you want MongoDB logging (optional)
```

### 3. Download pre-built indexes
```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, pathlib

files = [
    ("bm25/corpus_bm25_index.pkl",   "data/indexes/bm25/corpus_bm25_index.pkl"),
    ("vector/faiss_index.bin",        "data/indexes/vector/faiss_index.bin"),
    ("vector/vector_payload.pkl",     "data/indexes/vector/vector_payload.pkl"),
]
for repo_path, local_path in files:
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download("KPrashanth/med-rag-bot-indexes", repo_path, repo_type="dataset")
    shutil.copy(src, local_path)
    print(f"Downloaded {local_path}")
EOF
```

### 4. Chat
```bash
# Basic chat
python -m app.cli chat

# With extended reasoning (slower but better on complex queries)
python -m app.cli chat --thinking

# With re-ranking (requires ~22MB model download on first run)
python -m app.cli chat --reranker

# All features
python -m app.cli chat --thinking --reranker
```

### 5. Observability dashboard
```bash
streamlit run app/dashboard/streamlit_app.py
```

---

## Chat Interface

Once inside the chat session, type any medical/public health question. Available commands:

| Command | Description |
|---|---|
| `/think` | Toggle thinking computation on/off |
| `/show-think` | Toggle display of the thinking block |
| `/budget <N>` | Set thinking token budget (0 = unlimited) |
| `/chunks` | Toggle showing retrieved source chunks |
| `/verbose` | Toggle full token/latency stats |
| `/feedback <1-5> [comment]` | Rate the last answer |
| `/export` | Save session as markdown |
| `/help` | Show all commands |
| `exit` / `q` | End session |

---

## Bring Your Own Domain

Med RAG Bot is domain-agnostic. To build a grounded QA system for your own PDFs:

**1. Replace the corpus**
```bash
# Drop your PDFs into:
data/raw_corpus/your_domain/
```

**2. Set your domain in settings**
```python
# app/settings.py
domain_name: str = "your domain name"          # e.g. "pharmaceutical regulatory"
trusted_sources: list[str] = ["Source A", "Source B"]
```

**3. Run the pipeline**
```bash
python -m app.cli extract      # Extract text from PDFs
python -m app.cli chunk        # Generate atomic + parent chunks
python -m app.cli build-index  # Build BM25 + FAISS indexes
```

**4. Chat**
```bash
python -m app.cli chat
```

Examples of domains this pattern works for:
- Legal — contract review, case law QA
- Pharma — drug formulary, clinical trial protocols
- HR — policy manuals, compliance documents
- Finance — regulatory filings, product documentation
- Education — textbook QA, curriculum support

---

## Hardware Requirements

| Setup | Min VRAM | Est. Speed | Notes |
|---|---|---|---|
| CUDA GPU (recommended) | 6 GB | 15-30 tok/s | RTX 3060+ |
| CUDA GPU + thinking | 6 GB | 10-20 tok/s | Budget controls speed |
| CPU only | — | 1-3 tok/s | Usable, slow |
| HF Spaces T4 (coming soon) | 16 GB | 25-40 tok/s | No WSL2 overhead |

Models loaded at runtime (auto-downloaded to `~/.cache/huggingface/`):
- `Qwen/Qwen3-4B` (~2.5 GB VRAM in 4-bit)
- `BAAI/bge-base-en-v1.5` (~420 MB VRAM)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22 MB, CPU, optional)

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| Corpus pipeline | ✅ Done | PDF download, language filter, extraction, chunking |
| Hybrid retrieval | ✅ Done | BM25 + FAISS + RRF fusion |
| Generation + CLI | ✅ Done | Qwen3-4B, query routing, citations, conversation memory |
| Observability | ✅ Done | JSONL logging, Streamlit dashboard, cost estimator |
| Re-ranking | ✅ Done | Cross-encoder second-pass re-ranking |
| ChatEngine class | 🔄 In progress | Extract chat logic into reusable class |
| Gradio UI | 🔄 In progress | Browser-based chat; runs locally + HF Spaces |
| HF Spaces deployment | 🔄 In progress | Public demo on T4 GPU |
| Web search fallback | 📋 Planned | DuckDuckGo when corpus is insufficient |
| Table extraction | 📋 Planned | pdfplumber for clinical tables and formularies |
| Evaluation baseline | 📋 Planned | 50-100 queries, Precision@5, MRR, RAGAS |
| FastAPI wrapper | 📋 Planned | HTTP API for mobile apps, n8n, third-party integrations |
| LangSmith integration | 📋 Planned | Production tracing and evaluation |
| MongoDB persistence | 📋 Planned | Long-term session storage and analytics |

---

## Corpus

Current corpus covers **Indian public health** from four trusted sources:

| Source | Type | Coverage |
|---|---|---|
| ICMR | Research newsletters, surveillance reports, training materials | Malaria, TB, NCD, research |
| NCDC | Clinical manuals, standard treatment guidelines | Snake bite, vector-borne diseases |
| WHO SEARO | Regional reports, epidemiological bulletins | South-East Asia public health |
| MOHFW | Health policies, national programs | India-wide health governance |

All documents are publicly available. URLs are preserved in chunk metadata and surfaced in every citation.

---

## Philosophy

> "If you can't explain how your retrieval works, you don't understand your RAG system."

This project was built to be **educational first**. Every component answers the question *why*:

- **Why hybrid retrieval?** BM25 catches exact drug names and ICD codes that semantic search misses. Semantic search catches conceptual queries that BM25 misses. Together they're stronger than either alone.
- **Why RRF over weighted sum?** RRF is parameter-free, robust to score scale differences between BM25 and cosine similarity, and consistently outperforms hand-tuned weighted combinations.
- **Why a cross-encoder re-ranker?** Bi-encoders (FAISS) encode query and document separately. Cross-encoders see both together — dramatically more accurate, but too slow for full corpus search. Running it on 20 candidates takes ~200ms.
- **Why Qwen3-4B?** Best open-source reasoning at the 4B parameter range. Thinking mode produces measurably better answers on complex medical queries without the cost of larger models.
- **Why local first?** Zero API cost. Full control. Educational — you see the actual model weights loading, the actual token counts, the actual latency. No black boxes.

---

## Contributing

This project is actively developed. Contributions welcome — especially:
- Domain-specific corpus extensions
- Evaluation datasets
- Non-English support
- Docker setup

Open an issue to discuss before submitting large PRs.

---

## License

MIT — use it, learn from it, build on it.

---

*Built in India, for India — and anyone else who wants production-grade RAG without the black boxes.*
