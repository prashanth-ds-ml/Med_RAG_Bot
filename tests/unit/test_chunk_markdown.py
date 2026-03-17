from pathlib import Path

from app.chunking.chunk_markdown import (
    ChunkingConfig,
    chunk_markdown_document,
    chunk_markdown_file,
)


def test_chunk_markdown_document_creates_chunks_with_heading_paths() -> None:
    """
    What this test checks:
    - A structured markdown document produces atomic chunks
    - Chunks preserve heading hierarchy and section titles

    Why this matters:
    - Heading-aware chunk metadata is central to retrieval quality.
    """
    text = """# Atrial Fibrillation

## Symptoms

Palpitations and fatigue are common symptoms in many patients.

## Management

Rate control and rhythm control are both important management strategies.
"""

    result = chunk_markdown_document(
        text=text,
        source_file="atrial_fibrillation.md",
        relative_path="atrial_fibrillation.md",
    )

    assert result.doc_id == "atrial_fibrillation"
    assert len(result.atomic_chunks) >= 2

    first_chunk = result.atomic_chunks[0]
    assert first_chunk.heading_path() == ["Atrial Fibrillation", "Symptoms"]
    assert first_chunk.section_title == "Symptoms"

    second_chunk = result.atomic_chunks[1]
    assert second_chunk.heading_path() == ["Atrial Fibrillation", "Management"]
    assert second_chunk.section_title == "Management"


def test_chunk_markdown_document_marks_list_heavy_chunks() -> None:
    """
    What this test checks:
    - List-heavy markdown blocks are tagged appropriately.

    Why this matters:
    - Lists often carry compact clinical guidance and should be traceable.
    """
    text = """# Hypertension

## Lifestyle

- Reduce salt intake
- Exercise regularly
- Maintain healthy weight
- Limit alcohol
- Track blood pressure
"""

    result = chunk_markdown_document(
        text=text,
        source_file="hypertension.md",
        relative_path="hypertension.md",
    )

    assert len(result.atomic_chunks) >= 1
    assert any(chunk.is_list_heavy for chunk in result.atomic_chunks)
    assert any(chunk.chunk_type == "list" for chunk in result.atomic_chunks)


def test_chunk_markdown_document_splits_large_text_into_multiple_chunks() -> None:
    """
    What this test checks:
    - Large flat sections are split into more than one atomic chunk.

    Why this matters:
    - Retrieval chunks must not grow too large.
    """
    large_paragraph = " ".join(["atrial fibrillation management"] * 160)

    text = f"""# Atrial Fibrillation

## Management

{large_paragraph}
"""

    result = chunk_markdown_document(
        text=text,
        source_file="afib.md",
        relative_path="afib.md",
        config=ChunkingConfig(
            min_chunk_words=20,
            target_chunk_words=60,
            max_chunk_words=80,
            overlap_words=10,
            parent_target_words=140,
            parent_max_words=180,
        ),
    )

    assert len(result.atomic_chunks) >= 2
    assert all(chunk.stats is not None for chunk in result.atomic_chunks)
    assert all(chunk.section_title == "Management" for chunk in result.atomic_chunks)


def test_chunk_markdown_document_builds_parent_chunks() -> None:
    """
    What this test checks:
    - Parent chunks are built from atomic chunks.

    Why this matters:
    - Parent chunks provide broader answer-time context later in the pipeline.
    """
    text = """# Diabetes

## Management

Dietary modification is recommended as an important first step.

Physical activity improves metabolic health.

Medication may be required depending on severity.
"""

    result = chunk_markdown_document(
        text=text,
        source_file="diabetes.md",
        relative_path="diabetes.md",
        config=ChunkingConfig(
            min_chunk_words=5,
            target_chunk_words=20,
            max_chunk_words=30,
            overlap_words=5,
            parent_target_words=50,
            parent_max_words=70,
        ),
    )

    assert len(result.atomic_chunks) >= 1
    assert len(result.parent_chunks) >= 1
    assert all(parent.child_chunk_ids for parent in result.parent_chunks)


def test_chunk_markdown_document_detects_table_and_image_flags() -> None:
    """
    What this test checks:
    - Table placeholders and image references are reflected in chunk metadata.

    Why this matters:
    - Later retrieval or routing may handle these differently.
    """
    text = """# Report Notes

## Summary

[TABLE: Lab Ranges]

![figure](ecg.png)

Important narrative explanation follows here.
"""

    result = chunk_markdown_document(
        text=text,
        source_file="report_notes.md",
        relative_path="report_notes.md",
    )

    assert len(result.atomic_chunks) >= 1
    assert any(chunk.has_table_ref for chunk in result.atomic_chunks)
    assert any(chunk.has_image_ref for chunk in result.atomic_chunks)


def test_chunk_markdown_file_reads_from_disk_and_derives_relative_path(tmp_path: Path) -> None:
    """
    What this test checks:
    - The file-based entrypoint reads markdown correctly
    - Relative path and doc_id are derived predictably

    Why this matters:
    - Future ingestion commands will operate on files, not raw strings.
    """
    source_root = tmp_path / "data" / "test_source"
    nested_dir = source_root / "cardiology"
    nested_dir.mkdir(parents=True)

    file_path = nested_dir / "atrial_fibrillation.md"
    file_path.write_text(
        "# Atrial Fibrillation\n\n## Overview\n\nIrregular rhythm description.",
        encoding="utf-8",
    )

    result = chunk_markdown_file(file_path, source_root=source_root)

    assert result.relative_path == "cardiology/atrial_fibrillation.md"
    assert result.doc_id == "cardiology__atrial_fibrillation"
    assert result.source_file == "atrial_fibrillation.md"
    assert len(result.atomic_chunks) >= 1