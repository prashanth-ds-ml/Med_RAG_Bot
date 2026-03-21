# Med RAG Bot — Commands Reference

> A complete reference of every command used in this project — environment setup,
> pipeline operations, chat interface, observability, git workflow, and HuggingFace Hub.
> Useful as a learning reference and onboarding guide.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Pipeline Commands](#2-pipeline-commands)
3. [Chat Interface](#3-chat-interface)
4. [Chat Slash Commands](#4-chat-slash-commands)
5. [Observability Dashboard](#5-observability-dashboard)
6. [Git Workflow](#6-git-workflow)
7. [HuggingFace Hub](#7-huggingface-hub)
8. [MongoDB](#8-mongodb)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Environment Setup

### Create and activate the conda environment
```bash
conda create -n medrag_all python=3.10 -y
conda activate medrag_all
```

### Install all dependencies
```bash
pip install -r requirements.txt
```

### Deactivate environment
```bash
conda deactivate
```

### List all conda environments
```bash
conda env list
```

### Check which Python / which pip you're using
```bash
which python
which pip
python --version
```

### Install a single missing package
```bash
pip install <package-name>
# Example:
pip install python-dotenv
pip install orjson
pip install bitsandbytes
```

### Set up environment variables
```bash
cp .env.example .env
# Then open .env in your editor and fill in values
```

---

## 2. Pipeline Commands

These are run once (or when rebuilding the corpus). They live in `scripts/` and `app/cli.py`.

### Build BM25 + FAISS indexes from processed chunks
```bash
python -m app.cli build-index
```
What it does: reads `data/processed/corpus_atomic_chunks.jsonl`, builds `data/indexes/bm25/corpus_bm25_index.pkl` and `data/indexes/vector/faiss_index.bin`.

### Extract text from PDFs
```bash
python -m app.cli extract
```
What it does: runs PyMuPDF on every PDF in `data/raw_corpus/`, outputs one JSON per document to `data/corpus_pipeline/extracted/`.

### Chunk extracted text
```bash
python -m app.cli chunk
```
What it does: splits each extracted document into atomic chunks (~220 words) and parent chunks (~700 words), writes `data/processed/corpus_atomic_chunks.jsonl` and `data/processed/corpus_parent_chunks.jsonl`.

### Upload JSONL logs to MongoDB
```bash
python -m app.cli upload-logs

# Dry run — shows what would be uploaded without writing
python -m app.cli upload-logs --dry-run
```

### See all available CLI commands
```bash
python -m app.cli --help
```

### See help for a specific command
```bash
python -m app.cli chat --help
python -m app.cli build-index --help
```

---

## 3. Chat Interface

### Basic chat (no thinking, no reranker)
```bash
python -m app.cli chat
```

### Chat with thinking mode enabled
```bash
python -m app.cli chat --thinking
```
Thinking mode makes Qwen3-4B reason internally before answering. Produces better answers on complex queries but takes longer (~2-5x).

### Chat with cross-encoder re-ranking
```bash
python -m app.cli chat --reranker
```
Downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` on first run (~22MB). Cached forever after. Re-ranking improves answer quality by scoring each (query, document) pair jointly.

### Chat with all features
```bash
python -m app.cli chat --thinking --reranker
```

### Chat with custom number of retrieved chunks
```bash
python -m app.cli chat --top-k 10
# Default is 5. Higher = more context for the model, slower generation.
```

### Chat without thinking (override default)
```bash
python -m app.cli chat --no-thinking
```

### Chat without reranker (override default)
```bash
python -m app.cli chat --no-reranker
```

### Chat with a custom thinking token budget
```bash
python -m app.cli chat --thinking --thinking-budget 256
# Default is 512. Lower = faster thinking. 0 = unlimited.
```

### Run on a different project root (advanced)
```bash
python -m app.cli chat --project-root /path/to/project
```

---

## 4. Chat Slash Commands

These are typed inside an active chat session (after `You:` prompt).

| Command | What it does |
|---|---|
| `/think` | Toggle thinking computation on/off mid-session |
| `/show-think` | Toggle display of the thinking block (computation stays on/off separately) |
| `/budget 256` | Change thinking token budget to 256 for remaining turns |
| `/budget 0` | Remove token budget — unlimited thinking |
| `/chunks` | Toggle showing the raw retrieved chunks after each answer |
| `/verbose` | Toggle full stats (tokens, latency, message ID) |
| `/feedback 4` | Rate the last answer 4 out of 5 |
| `/feedback 2 answer was too vague` | Rate with a comment |
| `/export` | Save the full session as a markdown file in `data/exports/` |
| `/help` | Show all available commands |
| `exit` | End session gracefully (logs session_end) |
| `quit` | Same as exit |
| `q` | Same as exit |

### How the stats line works
```
grounded ✓  |  HIGH  |  3667 tokens  |  386.8s  |  msg: 699f09c208d6
    │             │          │              │              │
    │             │          │              │              └── message ID (use with /feedback)
    │             │          │              └─────────────── generation time
    │             │          └────────────────────────────── total tokens used
    │             └───────────────────────────────────────── confidence tier (HIGH / MED / LOW)
    └─────────────────────────────────────────────────────── grounded = at least one source cited
```

### Confidence tiers explained
| Tier | Fused Score | Meaning |
|---|---|---|
| HIGH | ≥ 0.025 | Strong match — retrieved chunks are highly relevant |
| MED | ≥ 0.015 | Moderate match — answer is likely correct but verify |
| LOW | < 0.015 | Weak match — model may be reasoning without strong evidence |

---

## 5. Observability Dashboard

### Start the Streamlit dashboard
```bash
streamlit run app/dashboard/streamlit_app.py
```
Opens at `http://localhost:8501` by default.

### Start on a custom port
```bash
streamlit run app/dashboard/streamlit_app.py --server.port 8502
```

### Dashboard pages
| Page | What you see |
|---|---|
| Overview | Total sessions, queries, avg tokens, grounded %, avg rating |
| Messages | Filterable message log with full answer + citations + thinking |
| Retrieval | Source distribution, fused scores, BM25 vs vector rank agreement |
| Performance | Latency (p50/p95/p99), throughput (tok/s), cost estimator (USD + INR) |
| Feedback | Rating distribution, low-rated answers, all feedback with comments |

---

## 6. Git Workflow

### Check current status
```bash
git status
```

### See what's staged (what will be committed)
```bash
git diff --cached --stat
```

### See full diff of staged changes
```bash
git diff --cached
```

### Add specific files to staging
```bash
git add app/cli.py
git add app/retrieval/reranker.py
git add app/generation/response_formatter.py
```

### Add an entire folder
```bash
git add app/dashboard/
git add app/monitoring/
```

### Commit staged changes
```bash
git commit -m "your commit message"
```

### View commit history
```bash
git log --oneline
git log --oneline --reverse         # oldest first
git log --format="%ad %s" --date=format:"%Y-%m-%d"    # with dates
```

### Push to remote (GitHub)
```bash
git push origin main
```

### Check what's tracked by git
```bash
git ls-files app/retrieval/
```

### Move tracked files (preserves git history)
```bash
git mv app.py misc/app.py
git mv parse_regions.py misc/parse_regions.py
```

### See changes between working tree and last commit
```bash
git diff
```

### Undo staged changes (unstage without losing work)
```bash
git restore --staged app/cli.py
```

### See which files git is ignoring
```bash
git status --ignored
```

---

## 7. HuggingFace Hub

### Login (required for upload/create)
```bash
huggingface-cli login
# Paste your write token from https://huggingface.co/settings/tokens
```

### Check login status
```bash
huggingface-cli whoami
```

### Download a model or dataset to cache
```bash
huggingface-cli download BAAI/bge-base-en-v1.5
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2
huggingface-cli download KPrashanth/med-rag-bot-indexes --repo-type dataset
```

### Pre-download the re-ranker model (avoids stalling on first chat run)
```bash
python -c "
from sentence_transformers import CrossEncoder
print('Downloading re-ranker...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Cached — future runs load instantly')
"
```

### Pre-download the embedding model
```bash
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-base-en-v1.5')
print('Cached.')
"
```

### Create a new dataset repo on HF Hub
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-username/repo-name", repo_type="dataset", private=False)
```

### Upload a single file to HF Hub
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="data/indexes/bm25/corpus_bm25_index.pkl",
    path_in_repo="bm25/corpus_bm25_index.pkl",
    repo_id="your-username/med-rag-bot-indexes",
    repo_type="dataset"
)
```

### Upload all three index files at once
```python
from huggingface_hub import HfApi
api = HfApi()

files = [
    ("data/indexes/bm25/corpus_bm25_index.pkl",   "bm25/corpus_bm25_index.pkl"),
    ("data/indexes/vector/faiss_index.bin",         "vector/faiss_index.bin"),
    ("data/indexes/vector/vector_payload.pkl",      "vector/vector_payload.pkl"),
]
for local_path, repo_path in files:
    print(f"Uploading {local_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id="your-username/med-rag-bot-indexes",
        repo_type="dataset"
    )
    print(f"Done: {repo_path}")
```

### Download indexes from HF Hub (for fresh clone / HF Spaces)
```python
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path

files = [
    ("bm25/corpus_bm25_index.pkl",  "data/indexes/bm25/corpus_bm25_index.pkl"),
    ("vector/faiss_index.bin",       "data/indexes/vector/faiss_index.bin"),
    ("vector/vector_payload.pkl",    "data/indexes/vector/vector_payload.pkl"),
]
for repo_path, local_path in files:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(
        "KPrashanth/med-rag-bot-indexes",
        repo_path,
        repo_type="dataset"
    )
    shutil.copy(src, local_path)
    print(f"Ready: {local_path}")
```

### Check HF cache location
```bash
ls ~/.cache/huggingface/hub/
```

---

## 8. MongoDB

MongoDB is optional. The system logs to JSONL files locally and works fully without it.

### Install MongoDB (Ubuntu/WSL2)
```bash
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
```

### Check if MongoDB is running
```bash
sudo systemctl status mongod
```

### Connect with mongosh
```bash
mongosh
```

### Upload JSONL logs to MongoDB
```bash
python -m app.cli upload-logs
```

### Dry run (preview without writing)
```bash
python -m app.cli upload-logs --dry-run
```

### Set MongoDB URI in .env
```bash
# In your .env file:
export MONGODB_URI=mongodb://localhost:27017
export MONGODB_DB=med360
```

---

## 9. Troubleshooting

### Wrong conda environment activated
```bash
# Check which env is active
conda info --envs

# Switch to the correct one
conda activate medrag_all
```

### `ModuleNotFoundError: No module named 'dotenv'`
```bash
pip install python-dotenv
```

### `ModuleNotFoundError: No module named 'bitsandbytes'`
```bash
pip install bitsandbytes
```

### HuggingFace 401 Unauthorized
```bash
# You need to login first
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens (write permission)
```

### Model download stuck at 0%
The download is a one-time operation. Try:
```bash
# Method 1: retry with huggingface-cli (handles resume)
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2

# Method 2: skip reranker for now, use --no-reranker
python -m app.cli chat --no-reranker

# Method 3: download outside WSL2 if network is slow there
```

### `IndexError` or `FileNotFoundError` on indexes
```bash
# Indexes not downloaded yet. Run:
python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path

files = [
    ("bm25/corpus_bm25_index.pkl",  "data/indexes/bm25/corpus_bm25_index.pkl"),
    ("vector/faiss_index.bin",       "data/indexes/vector/faiss_index.bin"),
    ("vector/vector_payload.pkl",    "data/indexes/vector/vector_payload.pkl"),
]
for repo_path, local_path in files:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download("KPrashanth/med-rag-bot-indexes", repo_path, repo_type="dataset")
    shutil.copy(src, local_path)
    print(f"Ready: {local_path}")
EOF
```

### Streamlit dashboard shows "No data yet"
```bash
# Run a chat session first to generate logs
python -m app.cli chat
# Then refresh the dashboard
```

### Check GPU is available to PyTorch
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

*This file is part of the Med RAG Bot project. Started March 16, 2026.*
