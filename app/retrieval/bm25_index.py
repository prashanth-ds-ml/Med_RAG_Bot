from __future__ import annotations

"""
BM25 retrieval index for atomic chunks.

Why this file exists:
- Provides the first transparent retrieval layer for the markdown corpus
- Gives a strong lexical baseline before vector search and reranking
- Makes it easy to inspect why a chunk matched a query

What this module does:
- Loads atomic chunk records from JSONL
- Tokenizes chunk text and queries
- Builds a BM25Okapi index
- Saves and loads a lightweight serialized BM25 artifact
- Returns top-k chunk matches for a query

Design choice:
- Keep tokenization simple and deterministic
- Persist enough metadata so retrieval remains inspectable
"""

import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent directory for an output file if needed.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text into lowercase lexical tokens.

    Why this matters:
    - BM25 works on tokenized text
    - Simple tokenization keeps the pipeline transparent and reproducible
    """
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """
    Load JSONL records from disk.

    Returns an empty list for empty files.

    Why this matters:
    - Atomic chunks are stored in JSONL
    - Retrieval indexing needs the records as dictionaries
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    if file_path.stat().st_size == 0:
        return []

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def build_bm25_payload(chunk_records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a BM25 payload from atomic chunk records.

    Payload contains:
    - tokenized corpus
    - original chunk records
    - chunk ids for easier inspection

    Why this matters:
    - Keeps retrieval state serializable
    - Makes later querying easy and explicit
    """
    corpus_tokens = [
        tokenize_text(record.get("chunk_text", ""))
        for record in chunk_records
    ]

    bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    return {
        "bm25": bm25,
        "chunk_records": chunk_records,
        "corpus_tokens": corpus_tokens,
        "chunk_ids": [record.get("chunk_id") for record in chunk_records],
    }


def save_bm25_payload(payload: dict[str, Any], path: str | Path) -> None:
    """
    Save a BM25 payload to disk using pickle.

    Why pickle:
    - BM25Okapi is not JSON-serializable
    - This keeps local indexing simple and fast
    """
    ensure_parent_dir(path)

    with Path(path).open("wb") as file_obj:
        pickle.dump(payload, file_obj)


def load_bm25_payload(path: str | Path) -> dict[str, Any]:
    """
    Load a previously saved BM25 payload from disk.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"BM25 index file not found: {file_path}")

    with file_path.open("rb") as file_obj:
        return pickle.load(file_obj)


def build_bm25_index_from_atomic_chunks(
    atomic_chunks_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """
    Build a BM25 index from atomic chunk JSONL and save it to disk.

    Returns
    -------
    dict[str, Any]
        Compact build summary for CLI and monitoring
    """
    chunk_records = load_jsonl_records(atomic_chunks_path)
    payload = build_bm25_payload(chunk_records)
    save_bm25_payload(payload, output_path)

    return {
        "document_count": len(chunk_records),
        "index_path": str(Path(output_path)),
        "source_path": str(Path(atomic_chunks_path)),
    }


def search_bm25_payload(
    payload: dict[str, Any],
    query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Search an in-memory BM25 payload and return top-k chunk matches.

    Returns a list of dicts with:
    - rank
    - score
    - chunk_id
    - chunk_text
    - full record

    Why this matters:
    - Gives a transparent retrieval API for local testing and CLI inspection
    """
    bm25 = payload.get("bm25")
    chunk_records = payload.get("chunk_records", [])

    if not chunk_records:
        return []

    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        record = chunk_records[index]
        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "chunk_id": record.get("chunk_id"),
                "chunk_text": record.get("chunk_text", ""),
                "record": record,
            }
        )

    return results


def search_bm25_index(
    index_path: str | Path,
    query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Load a BM25 index from disk and search it.
    """
    payload = load_bm25_payload(index_path)
    return search_bm25_payload(payload, query, top_k=top_k)