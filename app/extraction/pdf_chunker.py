from __future__ import annotations

"""
pdf_chunker.py — Chunk extracted PDF text into atomic and parent chunks.

PDF text lacks markdown heading structure, so we use paragraph-aware
sliding-window chunking:

  1. Split each page's text into paragraphs (blank-line boundaries).
  2. Accumulate paragraphs into atomic chunks, respecting word limits.
     When a chunk reaches target_words, close it and start a new one.
     Respect page boundaries: track the first page_num of each chunk.
  3. Group atomic chunks into parent chunks by accumulating up to
     parent_target_words.

Each chunk inherits citation metadata from the source document:
  pdf_url, source_name, doc_type, page_num → stored in chunk.metadata
  so citations are available at every stage: retrieval → generation → response.

Output: corpus_atomic_chunks.jsonl + corpus_parent_chunks.jsonl
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.chunking.models import (
    ChunkStats,
    HeadingContext,
    ParentChunkRecord,
    compute_chunk_stats,
    make_chunk_record,
    make_parent_chunk_record,
)

logger = logging.getLogger(__name__)


@dataclass
class PdfChunkingConfig:
    target_chunk_words: int = 220
    max_chunk_words: int = 300
    min_chunk_words: int = 60       # chunks below this are merged into next
    parent_target_words: int = 700
    parent_max_words: int = 900
    min_paragraph_chars: int = 20   # paragraphs below this are skipped


def _split_paragraphs(text: str, min_chars: int = 20) -> list[str]:
    """
    Split page text into paragraphs by blank-line boundaries.
    Strips very short fragments (headers, page numbers, footers).
    """
    raw_blocks = text.split("\n\n")
    paras: list[str] = []
    for block in raw_blocks:
        cleaned = block.strip()
        if len(cleaned) >= min_chars:
            paras.append(cleaned)
    return paras


def _chunk_id(doc_id: str, index: int, prefix: str = "c") -> str:
    """
    Deterministic chunk ID from doc_id and position.
    Format: {prefix}_{doc_id_hash8}_{index:04d}
    """
    doc_hash = hashlib.md5(doc_id.encode()).hexdigest()[:8]
    return f"{prefix}_{doc_hash}_{index:04d}"


def _parent_chunk_id(doc_id: str, index: int) -> str:
    return _chunk_id(doc_id, index, prefix="p")


def _chunk_pdf_doc(
    doc: dict[str, Any],
    config: PdfChunkingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Chunk a single extracted doc into atomic and parent chunks.

    Returns:
        (atomic_chunks, parent_chunks) as lists of serializable dicts.
    """
    doc_id = doc["doc_id"]
    source_file = doc["file_name"]
    pdf_url = doc.get("pdf_url", "")
    source_name = doc.get("source_name", "")
    doc_type = doc.get("doc_type", "")

    # Collect (paragraph_text, page_num) pairs across all pages
    para_page_pairs: list[tuple[str, int]] = []
    for page in doc.get("pages", []):
        page_num = page["page_num"]
        paragraphs = _split_paragraphs(
            page["text"], min_chars=config.min_paragraph_chars
        )
        for para in paragraphs:
            para_page_pairs.append((para, page_num))

    if not para_page_pairs:
        return [], []

    # --- Build atomic chunks ---
    atomic_chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_words = 0
    current_start_page = para_page_pairs[0][1]
    chunk_index = 0

    def _flush_atomic(parts: list[str], start_page: int, idx: int) -> dict[str, Any]:
        text = "\n\n".join(parts).strip()
        cid = _chunk_id(doc_id, idx)
        record = make_chunk_record(
            chunk_id=cid,
            doc_id=doc_id,
            source_file=source_file,
            relative_path=source_file,
            chunk_text=text,
            chunk_index=idx,
            chunk_type="text",
            section_kind="standalone_block",
            heading_context=HeadingContext(),
            section_title=None,
            metadata={
                "pdf_url": pdf_url,
                "source_name": source_name,
                "doc_type": doc_type,
                "page_num": start_page,
            },
        )
        return record.to_dict()

    for para, page_num in para_page_pairs:
        para_words = len(para.split())

        # If adding this paragraph would exceed max, flush first
        if current_words + para_words > config.max_chunk_words and current_parts:
            if current_words >= config.min_chunk_words:
                atomic_chunks.append(
                    _flush_atomic(current_parts, current_start_page, chunk_index)
                )
                chunk_index += 1
                current_parts = []
                current_words = 0
                current_start_page = page_num

        current_parts.append(para)
        current_words += para_words

        # If we've hit target, flush
        if current_words >= config.target_chunk_words:
            atomic_chunks.append(
                _flush_atomic(current_parts, current_start_page, chunk_index)
            )
            chunk_index += 1
            current_parts = []
            current_words = 0
            current_start_page = page_num

    # Flush any remainder
    if current_parts:
        if current_words >= config.min_chunk_words:
            atomic_chunks.append(
                _flush_atomic(current_parts, current_start_page, chunk_index)
            )
        elif atomic_chunks:
            # Merge tiny tail into the previous chunk
            last = atomic_chunks[-1]
            merged_text = last["chunk_text"] + "\n\n" + "\n\n".join(current_parts)
            last["chunk_text"] = merged_text.strip()
            stats = compute_chunk_stats(last["chunk_text"])
            last["stats"] = {
                "char_count": stats.char_count,
                "word_count": stats.word_count,
                "line_count": stats.line_count,
            }

    # --- Build parent chunks ---
    parent_chunks: list[dict[str, Any]] = []
    parent_index = 0
    current_atomic_batch: list[dict[str, Any]] = []
    current_parent_words = 0

    def _flush_parent(
        batch: list[dict[str, Any]], pidx: int
    ) -> dict[str, Any]:
        combined_text = "\n\n".join(ch["chunk_text"] for ch in batch)
        child_ids = [ch["chunk_id"] for ch in batch]
        start_page = batch[0]["metadata"]["page_num"]
        pid = _parent_chunk_id(doc_id, pidx)

        record = make_parent_chunk_record(
            parent_chunk_id=pid,
            doc_id=doc_id,
            source_file=source_file,
            relative_path=source_file,
            chunk_text=combined_text,
            child_chunk_ids=child_ids,
            heading_context=HeadingContext(),
            section_title=None,
            chunk_type="mixed",
            section_kind="standalone_block",
            metadata={
                "pdf_url": pdf_url,
                "source_name": source_name,
                "doc_type": doc_type,
                "page_num": start_page,
            },
        )
        # Back-fill parent_chunk_id into atomic chunks
        for ch in batch:
            ch["parent_chunk_id"] = pid
        return record.to_dict()

    for atomic in atomic_chunks:
        word_count = atomic.get("stats", {}).get("word_count", 0)

        if (
            current_parent_words + word_count > config.parent_max_words
            and current_atomic_batch
        ):
            parent_chunks.append(
                _flush_parent(current_atomic_batch, parent_index)
            )
            parent_index += 1
            current_atomic_batch = []
            current_parent_words = 0

        current_atomic_batch.append(atomic)
        current_parent_words += word_count

        if current_parent_words >= config.parent_target_words:
            parent_chunks.append(
                _flush_parent(current_atomic_batch, parent_index)
            )
            parent_index += 1
            current_atomic_batch = []
            current_parent_words = 0

    if current_atomic_batch:
        parent_chunks.append(_flush_parent(current_atomic_batch, parent_index))

    return atomic_chunks, parent_chunks


def chunk_extracted_corpus(
    extraction_manifest_path: Path,
    extracted_dir: Path,
    atomic_chunks_path: Path,
    parent_chunks_path: Path,
    chunk_stats_path: Path,
    config: PdfChunkingConfig | None = None,
    *,
    skip_empty: bool = True,
) -> dict[str, Any]:
    """
    Chunk all successfully extracted docs into atomic + parent JSONL files.

    Args:
        extraction_manifest_path: extraction_manifest.jsonl from pdf_extractor
        extracted_dir: directory containing per-doc JSON files
        atomic_chunks_path: output path for atomic chunks JSONL
        parent_chunks_path: output path for parent chunks JSONL
        chunk_stats_path: output path for summary JSON
        config: chunking configuration (uses defaults if None)
        skip_empty: skip docs with extraction_status != "ok"

    Returns:
        Summary dict with counts and paths.
    """
    if config is None:
        config = PdfChunkingConfig()

    # Load extraction manifest — only process "ok" docs
    manifest_entries: list[dict[str, Any]] = []
    with extraction_manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if skip_empty and rec.get("extraction_status") != "ok":
                continue
            manifest_entries.append(rec)

    total_docs = len(manifest_entries)
    total_atomic = 0
    total_parent = 0
    failed_docs = 0

    atomic_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    parent_chunks_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        atomic_chunks_path.open("w", encoding="utf-8") as af,
        parent_chunks_path.open("w", encoding="utf-8") as pf,
    ):
        for i, entry in enumerate(manifest_entries, start=1):
            doc_id = entry["doc_id"]
            doc_json_path = extracted_dir / f"{doc_id}.json"

            if not doc_json_path.exists():
                logger.warning(
                    "[%d/%d] Extracted JSON not found: %s", i, total_docs, doc_json_path
                )
                failed_docs += 1
                continue

            with doc_json_path.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)

            try:
                atomics, parents = _chunk_pdf_doc(doc, config)
            except Exception as exc:
                logger.error(
                    "[%d/%d] Chunking failed for %s: %s", i, total_docs, doc_id, exc
                )
                failed_docs += 1
                continue

            for chunk in atomics:
                af.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            for chunk in parents:
                pf.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total_atomic += len(atomics)
            total_parent += len(parents)

            if i % 100 == 0:
                logger.info(
                    "[%d/%d] atomic=%d parent=%d",
                    i,
                    total_docs,
                    total_atomic,
                    total_parent,
                )

    chunk_stats = {
        "total_docs_processed": total_docs,
        "failed_docs": failed_docs,
        "total_atomic_chunks": total_atomic,
        "total_parent_chunks": total_parent,
        "avg_atomic_per_doc": round(total_atomic / max(total_docs, 1), 2),
        "avg_parent_per_doc": round(total_parent / max(total_docs, 1), 2),
        "config": {
            "target_chunk_words": config.target_chunk_words,
            "max_chunk_words": config.max_chunk_words,
            "min_chunk_words": config.min_chunk_words,
            "parent_target_words": config.parent_target_words,
            "parent_max_words": config.parent_max_words,
        },
        "atomic_chunks_path": str(atomic_chunks_path),
        "parent_chunks_path": str(parent_chunks_path),
    }

    chunk_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with chunk_stats_path.open("w", encoding="utf-8") as fh:
        json.dump(chunk_stats, fh, indent=2, ensure_ascii=False)

    return chunk_stats
