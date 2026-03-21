"""
app.py — HuggingFace Spaces entry point for Med RAG Bot.

HF Spaces looks for this file at the repo root to launch the Gradio UI.

On first startup it downloads the pre-built BM25 + FAISS indexes from
HF Hub (KPrashanth/med-rag-bot-indexes) so the Space works immediately
after cloning — no corpus pipeline needed.

Run locally:
    python app.py
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download indexes from HF Hub if not already present
# ---------------------------------------------------------------------------

_INDEX_FILES = [
    ("bm25/corpus_bm25_index.pkl",  "data/indexes/bm25/corpus_bm25_index.pkl"),
    ("vector/faiss_index.bin",       "data/indexes/vector/faiss_index.bin"),
    ("vector/vector_payload.pkl",    "data/indexes/vector/vector_payload.pkl"),
]
_HF_DATASET_REPO = "KPrashanth/med-rag-bot-indexes"


def _download_indexes() -> None:
    """Download pre-built indexes from HF Hub on first run."""
    from huggingface_hub import hf_hub_download

    for repo_path, local_path in _INDEX_FILES:
        dest = Path(local_path)
        if dest.exists():
            logger.info("Index already present: %s", local_path)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s …", repo_path)
        cached = hf_hub_download(
            _HF_DATASET_REPO,
            repo_path,
            repo_type="dataset",
        )
        shutil.copy(cached, dest)
        logger.info("Saved → %s", local_path)


_download_indexes()

# ---------------------------------------------------------------------------
# Launch Gradio app
# ---------------------------------------------------------------------------
# Import after index download so ChatEngine finds the files on load().

from ui.gradio_app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
    )
else:
    # HF Spaces calls demo.launch() via the Gradio SDK — just expose demo.
    pass
