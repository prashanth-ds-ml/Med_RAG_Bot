from __future__ import annotations

import json
from pathlib import Path

from app.chunking.chunk_markdown import chunk_markdown_document
from app.chunking.chunk_writer import (
    build_chunk_stats_summary,
    write_chunking_outputs,
    write_json_file,
    write_jsonl_records,
)


def test_write_jsonl_records_writes_one_record_per_line(tmp_path: Path) -> None:
    """
    What this test checks:
    - JSONL writing creates one JSON object per line.

    Why this matters:
    - Atomic and parent chunk outputs will rely on JSONL format.
    """
    output_path = tmp_path / "chunks.jsonl"
    records = [
        {"chunk_id": "a1", "text": "first"},
        {"chunk_id": "a2", "text": "second"},
    ]

    write_jsonl_records(output_path, records)

    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["chunk_id"] == "a1"
    assert parsed[1]["chunk_id"] == "a2"


def test_write_json_file_writes_pretty_json(tmp_path: Path) -> None:
    """
    What this test checks:
    - JSON file writing produces a readable valid JSON artifact.

    Why this matters:
    - Chunk stats should stay easy to inspect manually.
    """
    output_path = tmp_path / "stats.json"
    payload = {"doc_id": "afib", "atomic_chunk_count": 3}

    write_json_file(output_path, payload)

    assert output_path.exists()

    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["doc_id"] == "afib"
    assert parsed["atomic_chunk_count"] == 3


def test_build_chunk_stats_summary_returns_expected_keys() -> None:
    """
    What this test checks:
    - Chunk stats summary includes the main monitoring fields.

    Why this matters:
    - We want a compact summary per chunked document.
    """
    text = """# Atrial Fibrillation

## Symptoms

Palpitations and fatigue are common symptoms.

## Management

- Rate control
- Rhythm control
- Anticoagulation when indicated
"""

    result = chunk_markdown_document(
        text=text,
        source_file="atrial_fibrillation.md",
        relative_path="atrial_fibrillation.md",
    )

    summary = build_chunk_stats_summary(result)

    assert summary["doc_id"] == "atrial_fibrillation"
    assert "atomic_chunk_count" in summary
    assert "parent_chunk_count" in summary
    assert "avg_atomic_words" in summary
    assert "max_atomic_words" in summary
    assert "list_heavy_atomic_chunks" in summary


def test_write_chunking_outputs_writes_all_expected_files(tmp_path: Path) -> None:
    """
    What this test checks:
    - Full chunking output writer creates atomic JSONL, parent JSONL, and stats JSON.

    Why this matters:
    - This is the persistence step that later indexing and monitoring will depend on.
    """
    text = """# Diabetes

## Overview

Diabetes mellitus is a chronic metabolic disorder.

## Management

- Dietary change
- Exercise
- Medication when needed
"""

    result = chunk_markdown_document(
        text=text,
        source_file="diabetes.md",
        relative_path="diabetes.md",
    )

    atomic_path = tmp_path / "processed" / "atomic_chunks.jsonl"
    parent_path = tmp_path / "processed" / "parent_chunks.jsonl"
    stats_path = tmp_path / "processed" / "chunk_stats.json"

    write_result = write_chunking_outputs(
        result,
        atomic_chunks_path=atomic_path,
        parent_chunks_path=parent_path,
        chunk_stats_path=stats_path,
    )

    assert atomic_path.exists()
    assert parent_path.exists()
    assert stats_path.exists()

    atomic_lines = atomic_path.read_text(encoding="utf-8").strip().splitlines()
    parent_lines = parent_path.read_text(encoding="utf-8").strip().splitlines()
    stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))

    assert len(atomic_lines) == len(result.atomic_chunks)
    assert len(parent_lines) == len(result.parent_chunks)
    assert stats_payload["doc_id"] == result.doc_id

    assert write_result["doc_id"] == result.doc_id
    assert write_result["atomic_chunks_written"] == len(result.atomic_chunks)
    assert write_result["parent_chunks_written"] == len(result.parent_chunks)


def test_write_chunking_outputs_serializes_heading_path_in_atomic_chunks(tmp_path: Path) -> None:
    """
    What this test checks:
    - Serialized atomic chunk records include heading_path.

    Why this matters:
    - Heading-aware metadata must survive the write step.
    """
    text = """# Stroke

## Emergency Care

Rapid assessment and stabilization are important.
"""

    result = chunk_markdown_document(
        text=text,
        source_file="stroke.md",
        relative_path="stroke.md",
    )

    atomic_path = tmp_path / "atomic_chunks.jsonl"
    parent_path = tmp_path / "parent_chunks.jsonl"
    stats_path = tmp_path / "chunk_stats.json"

    write_chunking_outputs(
        result,
        atomic_chunks_path=atomic_path,
        parent_chunks_path=parent_path,
        chunk_stats_path=stats_path,
    )

    first_atomic = json.loads(
        atomic_path.read_text(encoding="utf-8").strip().splitlines()[0]
    )

    assert "heading_path" in first_atomic
    assert first_atomic["heading_path"] == ["Stroke", "Emergency Care"]