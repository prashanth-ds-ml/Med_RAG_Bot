from __future__ import annotations

"""
vector_index.py — FAISS vector index for semantic retrieval.

Uses BAAI/bge-base-en-v1.5 (HuggingFace) to embed atomic chunks.
Vectors are L2-normalized so IndexFlatIP gives cosine similarity.

Saved artifacts:
  faiss_index.bin    — FAISS binary index (via faiss.write_index)
  vector_payload.pkl — {chunk_ids: list[str], chunk_records: list[dict]}

The chunk_records stored here contain only the fields needed at retrieval
time (chunk_id, chunk_text, doc_id, source_file, metadata) to keep memory
footprint low.

Device selection:
  - Uses CUDA if available (RTX 3060 local / GPU instance on AWS)
  - Falls back to CPU transparently
  - device_map="auto" pattern keeps the same code path for all environments

Why this module mirrors bm25_index.py structure:
  - build / save / load / search follow the same API shape
  - Makes hybrid_retriever.py straightforward to write
  - Easier to test each retrieval mode independently
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only the fields needed at query time.
    Reduces memory footprint of the saved payload (~500 bytes vs ~2KB per chunk).
    """
    return {
        "chunk_id":    record.get("chunk_id", ""),
        "doc_id":      record.get("doc_id", ""),
        "source_file": record.get("source_file", ""),
        "chunk_text":  record.get("chunk_text", ""),
        "metadata":    record.get("metadata", {}),
        "heading_path": record.get("heading_path", []),
        "doc_type":    record.get("doc_type", ""),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_vector_index(
    atomic_chunks_path: Path,
    index_path: Path,
    payload_path: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Embed all atomic chunks and build a FAISS IndexFlatIP index.

    Steps:
      1. Load chunks from JSONL
      2. Encode chunk texts in batches with sentence-transformers
      3. L2-normalize embeddings (cosine similarity via IndexFlatIP)
      4. Build and save FAISS index
      5. Save chunk_ids + compact chunk_records as pickle payload

    Returns a build summary dict.
    """
    if device is None:
        device = _select_device()

    logger.info("Loading atomic chunks from %s", atomic_chunks_path)
    records = _load_jsonl(atomic_chunks_path)
    if not records:
        raise ValueError(f"No chunk records found in {atomic_chunks_path}")

    texts = [r.get("chunk_text", "") for r in records]
    chunk_ids = [r.get("chunk_id", "") for r in records]
    compact_records = [_compact_record(r) for r in records]

    logger.info(
        "Encoding %d chunks with %s on %s (batch_size=%d)",
        len(texts), model_name, device, batch_size,
    )

    model = SentenceTransformer(model_name, device=device)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize for cosine via IndexFlatIP
        convert_to_numpy=True,
    )
    del model  # free VRAM before LLM loads at query time

    dim = embeddings.shape[1]
    logger.info("Embedding shape: %s  dim=%d", embeddings.shape, dim)

    # Build FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save FAISS index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved to %s  (%d vectors)", index_path, index.ntotal)

    # Save payload (chunk_ids + compact records)
    payload = {
        "chunk_ids": chunk_ids,
        "chunk_records": compact_records,
        "model_name": model_name,
        "dim": dim,
        "total": len(chunk_ids),
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with payload_path.open("wb") as fh:
        pickle.dump(payload, fh)
    logger.info("Vector payload saved to %s", payload_path)

    return {
        "total_chunks": len(records),
        "embedding_dim": dim,
        "model_name": model_name,
        "device": device,
        "index_path": str(index_path),
        "payload_path": str(payload_path),
    }


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_vector_index(
    index_path: Path,
    payload_path: Path,
) -> dict[str, Any]:
    """
    Load FAISS index + payload from disk.

    Returns a dict with:
      index, chunk_ids, chunk_records, model_name, dim
    """
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not payload_path.exists():
        raise FileNotFoundError(f"Vector payload not found: {payload_path}")

    index = faiss.read_index(str(index_path))

    with payload_path.open("rb") as fh:
        payload = pickle.load(fh)

    payload["index"] = index
    return payload


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_vector_index(
    index_path: Path,
    payload_path: Path,
    query: str,
    *,
    top_k: int = 10,
    model_name: str | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Encode a query and search the FAISS index. Returns top-k results.

    Each result dict mirrors the BM25 result shape for easy RRF fusion:
      rank, score, chunk_id, chunk_text, record
    """
    if device is None:
        device = _select_device()

    payload = load_vector_index(index_path, payload_path)
    _model_name = model_name or payload.get("model_name", DEFAULT_MODEL)

    model = SentenceTransformer(_model_name, device=device)
    query_vec: np.ndarray = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    del model

    scores, indices = payload["index"].search(
        query_vec.astype(np.float32), top_k
    )
    # scores shape: (1, top_k), indices shape: (1, top_k)
    scores = scores[0]
    indices = indices[0]

    chunk_ids = payload["chunk_ids"]
    chunk_records = payload["chunk_records"]

    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        record = chunk_records[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk_ids[idx],
                "chunk_text": record.get("chunk_text", ""),
                "record": record,
            }
        )

    return results


def search_vector_payload(
    payload: dict[str, Any],
    query_vec: np.ndarray,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Search an already-loaded vector payload with a pre-encoded query vector.
    Used by hybrid_retriever to avoid reloading the index on each call.
    """
    scores, indices = payload["index"].search(
        query_vec.astype(np.float32), top_k
    )
    scores = scores[0]
    indices = indices[0]

    chunk_ids = payload["chunk_ids"]
    chunk_records = payload["chunk_records"]

    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx == -1:
            continue
        record = chunk_records[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk_ids[idx],
                "chunk_text": record.get("chunk_text", ""),
                "record": record,
            }
        )

    return results
