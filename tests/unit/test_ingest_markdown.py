from __future__ import annotations

import json
from pathlib import Path

from app.chunking.chunk_markdown import ChunkingConfig
from app.ingestion.ingest_markdown import ingest_markdown_corpus


def test_ingest_markdown_corpus_writes_expected_outputs(tmp_path: Path) -> None:
    """
    What this test checks:
    - Bulk ingest processes markdown files
    - Combined atomic, parent, and stats outputs are written

    Why this matters:
    - This is the first full corpus-processing step after source tracking.
    """
    source_dir = tmp_path / "data" / "test_source"
    source_dir.mkdir(parents=True)

    (source_dir / "doc_a.md").write_text(
        "# A\n\n## Symptoms\n\nPalpitations and dizziness are common.\n",
        encoding="utf-8",
    )
    (source_dir / "doc_b.md").write_text(
        "# B\n\n## Management\n\n- Lifestyle change\n- Medication when needed\n",
        encoding="utf-8",
    )

    atomic_path = tmp_path / "data" / "processed" / "atomic_chunks.jsonl"
    parent_path = tmp_path / "data" / "processed" / "parent_chunks.jsonl"
    stats_path = tmp_path / "data" / "processed" / "chunk_stats.json"

    result = ingest_markdown_corpus(
        source_dir=source_dir,
        atomic_chunks_path=atomic_path,
        parent_chunks_path=parent_path,
        chunk_stats_path=stats_path,
        config=ChunkingConfig(
            min_chunk_words=5,
            target_chunk_words=40,
            max_chunk_words=60,
            overlap_words=5,
            parent_target_words=100,
            parent_max_words=140,
        ),
    )

    assert result["document_count"] == 2
    assert atomic_path.exists()
    assert parent_path.exists()
    assert stats_path.exists()

    atomic_lines = atomic_path.read_text(encoding="utf-8").strip().splitlines()
    parent_lines = parent_path.read_text(encoding="utf-8").strip().splitlines()
    stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))

    assert len(atomic_lines) == result["total_atomic_chunks"]
    assert len(parent_lines) == result["total_parent_chunks"]
    assert stats_payload["corpus_summary"]["document_count"] == 2
    assert len(stats_payload["documents"]) == 2


def test_ingest_markdown_corpus_handles_empty_source_directory(tmp_path: Path) -> None:
    """
    What this test checks:
    - Empty source folders still produce valid empty output artifacts.

    Why this matters:
    - Initial setup and cleanup phases should not break ingest.
    """
    source_dir = tmp_path / "data" / "test_source"
    source_dir.mkdir(parents=True)

    atomic_path = tmp_path / "data" / "processed" / "atomic_chunks.jsonl"
    parent_path = tmp_path / "data" / "processed" / "parent_chunks.jsonl"
    stats_path = tmp_path / "data" / "processed" / "chunk_stats.json"

    result = ingest_markdown_corpus(
        source_dir=source_dir,
        atomic_chunks_path=atomic_path,
        parent_chunks_path=parent_path,
        chunk_stats_path=stats_path,
    )

    assert result["document_count"] == 0
    assert result["total_atomic_chunks"] == 0
    assert result["total_parent_chunks"] == 0

    assert atomic_path.exists()
    assert parent_path.exists()
    assert stats_path.exists()

    assert atomic_path.read_text(encoding="utf-8") == ""
    assert parent_path.read_text(encoding="utf-8") == ""

    stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats_payload["corpus_summary"]["document_count"] == 0