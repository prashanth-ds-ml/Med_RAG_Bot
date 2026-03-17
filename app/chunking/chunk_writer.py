from __future__ import annotations

"""
Writers for chunking outputs.

Why this file exists:
- Chunking results should be persisted as transparent artifacts
- JSONL is easy to inspect, append, diff, and load later
- A separate stats file makes it easy to track corpus growth over time

What this module does:
- Writes atomic chunks to JSONL
- Writes parent chunks to JSONL
- Writes a compact chunking summary to JSON

Design choice:
- Keep serialization explicit and simple
- Do not hide output writing inside the chunker itself
"""

import json
from pathlib import Path
from typing import Any

from app.chunking.chunk_markdown import ChunkingResult


def ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent directory for a file path if needed.

    Why this matters:
    - Writers should be able to bootstrap their output folders safely
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    """
    Write JSONL records to disk.

    Why JSONL:
    - Easy to inspect line by line
    - Easy to stream later during indexing or ingestion
    """
    ensure_parent_dir(path)

    with Path(path).open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json_file(path: str | Path, payload: dict[str, Any]) -> None:
    """
    Write a JSON payload to disk with indentation.

    Why this matters:
    - Stats and summaries should stay human-readable
    """
    ensure_parent_dir(path)

    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def build_chunk_stats_summary(result: ChunkingResult) -> dict[str, Any]:
    """
    Build a compact chunking summary for one document.

    Why this matters:
    - Gives quick visibility into chunk counts and distribution
    - Useful for monitoring chunk growth as the corpus evolves
    """
    atomic_word_counts = [
        chunk.stats.word_count
        for chunk in result.atomic_chunks
        if chunk.stats is not None
    ]
    parent_word_counts = [
        chunk.stats.word_count
        for chunk in result.parent_chunks
        if chunk.stats is not None
    ]

    return {
        "doc_id": result.doc_id,
        "source_file": result.source_file,
        "relative_path": result.relative_path,
        "atomic_chunk_count": len(result.atomic_chunks),
        "parent_chunk_count": len(result.parent_chunks),
        "avg_atomic_words": round(sum(atomic_word_counts) / len(atomic_word_counts), 2)
        if atomic_word_counts
        else 0.0,
        "max_atomic_words": max(atomic_word_counts) if atomic_word_counts else 0,
        "avg_parent_words": round(sum(parent_word_counts) / len(parent_word_counts), 2)
        if parent_word_counts
        else 0.0,
        "max_parent_words": max(parent_word_counts) if parent_word_counts else 0,
        "list_heavy_atomic_chunks": sum(
            1 for chunk in result.atomic_chunks if chunk.is_list_heavy
        ),
        "table_ref_atomic_chunks": sum(
            1 for chunk in result.atomic_chunks if chunk.has_table_ref
        ),
        "image_ref_atomic_chunks": sum(
            1 for chunk in result.atomic_chunks if chunk.has_image_ref
        ),
    }


def write_chunking_outputs(
    result: ChunkingResult,
    *,
    atomic_chunks_path: str | Path,
    parent_chunks_path: str | Path,
    chunk_stats_path: str | Path,
) -> dict[str, Any]:
    """
    Write a complete chunking result to disk.

    Outputs:
    - atomic chunks JSONL
    - parent chunks JSONL
    - chunk stats JSON

    Returns
    -------
    dict[str, Any]
        A summary of what was written.

    Why this matters:
    - Keeps the writing step explicit and easy to test
    """
    atomic_records = [chunk.to_dict() for chunk in result.atomic_chunks]
    parent_records = [chunk.to_dict() for chunk in result.parent_chunks]
    stats_payload = build_chunk_stats_summary(result)

    write_jsonl_records(atomic_chunks_path, atomic_records)
    write_jsonl_records(parent_chunks_path, parent_records)
    write_json_file(chunk_stats_path, stats_payload)

    return {
        "doc_id": result.doc_id,
        "atomic_chunks_written": len(atomic_records),
        "parent_chunks_written": len(parent_records),
        "atomic_chunks_path": str(Path(atomic_chunks_path)),
        "parent_chunks_path": str(Path(parent_chunks_path)),
        "chunk_stats_path": str(Path(chunk_stats_path)),
    }