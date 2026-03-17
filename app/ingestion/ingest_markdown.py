from __future__ import annotations

"""
Bulk markdown ingest pipeline.

Why this file exists:
- Turns the entire markdown source folder into retrieval-ready chunk artifacts
- Keeps ingestion logic separate from CLI and chunking internals
- Produces transparent, inspectable outputs for later indexing and monitoring
"""

from typing import Any
from pathlib import Path

from app.chunking.chunk_markdown import ChunkingConfig, chunk_markdown_file
from app.chunking.chunk_writer import (
    build_chunk_stats_summary,
    write_json_file,
    write_jsonl_records,
)
from app.ingestion.scan_source import scan_markdown_source


def build_corpus_chunk_stats(results: list[Any]) -> dict[str, Any]:
    """
    Build a corpus-level chunking summary across all processed documents.

    Why this matters:
    - Lets us track total corpus growth over time
    - Gives quick visibility into chunk distribution
    """
    document_summaries = [build_chunk_stats_summary(result) for result in results]

    total_atomic = sum(item["atomic_chunk_count"] for item in document_summaries)
    total_parent = sum(item["parent_chunk_count"] for item in document_summaries)

    atomic_word_avgs = [item["avg_atomic_words"] for item in document_summaries if item["atomic_chunk_count"] > 0]
    parent_word_avgs = [item["avg_parent_words"] for item in document_summaries if item["parent_chunk_count"] > 0]

    return {
        "corpus_summary": {
            "document_count": len(results),
            "total_atomic_chunks": total_atomic,
            "total_parent_chunks": total_parent,
            "avg_atomic_words_across_docs": round(sum(atomic_word_avgs) / len(atomic_word_avgs), 2)
            if atomic_word_avgs
            else 0.0,
            "avg_parent_words_across_docs": round(sum(parent_word_avgs) / len(parent_word_avgs), 2)
            if parent_word_avgs
            else 0.0,
        },
        "documents": document_summaries,
    }


def ingest_markdown_corpus(
    *,
    source_dir: str | Path,
    atomic_chunks_path: str | Path,
    parent_chunks_path: str | Path,
    chunk_stats_path: str | Path,
    config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    """
    Chunk all markdown files in the source directory and write combined outputs.

    Outputs:
    - atomic_chunks.jsonl
    - parent_chunks.jsonl
    - chunk_stats.json

    Returns
    -------
    dict[str, Any]
        A compact ingest summary for CLI and monitoring use.
    """
    config = config or ChunkingConfig()
    source_root = Path(source_dir).resolve()

    markdown_files = scan_markdown_source(source_root)

    results = []
    atomic_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []

    for file_path in markdown_files:
        result = chunk_markdown_file(
            file_path,
            source_root=source_root,
            config=config,
        )
        results.append(result)

        atomic_records.extend(chunk.to_dict() for chunk in result.atomic_chunks)
        parent_records.extend(chunk.to_dict() for chunk in result.parent_chunks)

    corpus_stats = build_corpus_chunk_stats(results)

    write_jsonl_records(atomic_chunks_path, atomic_records)
    write_jsonl_records(parent_chunks_path, parent_records)
    write_json_file(chunk_stats_path, corpus_stats)

    return {
        "document_count": len(results),
        "total_atomic_chunks": len(atomic_records),
        "total_parent_chunks": len(parent_records),
        "atomic_chunks_path": str(Path(atomic_chunks_path)),
        "parent_chunks_path": str(Path(parent_chunks_path)),
        "chunk_stats_path": str(Path(chunk_stats_path)),
    }