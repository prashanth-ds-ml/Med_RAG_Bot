from app.chunking.models import (
    HeadingContext,
    compute_chunk_stats,
    make_chunk_record,
    make_parent_chunk_record,
)


def test_heading_context_to_path_and_max_depth() -> None:
    """
    What this test checks:
    - HeadingContext returns the correct heading path
    - HeadingContext reports the deepest populated level correctly

    Why this matters:
    - Later chunk inspection and retrieval metadata depend on reliable heading paths.
    """
    context = HeadingContext(
        h1="Atrial Fibrillation",
        h2="Management",
        h3="Rate Control",
    )

    assert context.to_path() == [
        "Atrial Fibrillation",
        "Management",
        "Rate Control",
    ]
    assert context.max_depth() == 3


def test_heading_context_handles_empty_values() -> None:
    """
    What this test checks:
    - Empty heading context produces an empty path and zero depth.

    Why this matters:
    - Some chunks may come from flat or weakly structured markdown.
    """
    context = HeadingContext()

    assert context.to_path() == []
    assert context.max_depth() == 0


def test_compute_chunk_stats_counts_basic_lengths() -> None:
    """
    What this test checks:
    - Basic text statistics are computed consistently.

    Why this matters:
    - Chunk stats are used for inspection, debugging, and later filtering.
    """
    text = "Line one\nLine two\n\nLine four"
    stats = compute_chunk_stats(text)

    assert stats.char_count == len(text)
    assert stats.word_count == 6
    assert stats.line_count == 4


def test_make_chunk_record_builds_atomic_chunk_with_stats() -> None:
    """
    What this test checks:
    - Atomic chunk factory returns a well-formed chunk record
    - Stats and heading metadata are attached correctly

    Why this matters:
    - This is the main schema used for retrieval chunks.
    """
    context = HeadingContext(
        h1="Atrial Fibrillation",
        h2="Symptoms",
    )

    chunk = make_chunk_record(
        chunk_id="afib_0001",
        doc_id="atrial_fibrillation",
        source_file="atrial_fibrillation.md",
        relative_path="atrial_fibrillation.md",
        chunk_text="Common symptoms include palpitations and fatigue.",
        chunk_index=0,
        chunk_type="text",
        section_kind="heading_section",
        heading_context=context,
        section_title="Symptoms",
        is_list_heavy=False,
    )

    assert chunk.chunk_id == "afib_0001"
    assert chunk.doc_id == "atrial_fibrillation"
    assert chunk.source_level == "atomic"
    assert chunk.section_title == "Symptoms"
    assert chunk.heading_path() == ["Atrial Fibrillation", "Symptoms"]
    assert chunk.stats is not None
    assert chunk.stats.word_count > 0


def test_chunk_record_to_dict_contains_heading_path() -> None:
    """
    What this test checks:
    - Chunk serialization includes a computed heading_path field.

    Why this matters:
    - JSONL outputs should be easy to inspect without reconstructing paths later.
    """
    context = HeadingContext(
        h1="Hypertension",
        h2="Treatment",
    )

    chunk = make_chunk_record(
        chunk_id="htn_0001",
        doc_id="hypertension",
        source_file="hypertension.md",
        relative_path="hypertension.md",
        chunk_text="Lifestyle modification is recommended.",
        chunk_index=1,
        heading_context=context,
    )

    payload = chunk.to_dict()

    assert payload["chunk_id"] == "htn_0001"
    assert payload["heading_path"] == ["Hypertension", "Treatment"]
    assert payload["stats"]["word_count"] > 0


def test_make_parent_chunk_record_builds_parent_chunk_with_children() -> None:
    """
    What this test checks:
    - Parent chunk factory returns a well-formed parent chunk
    - Child chunk IDs and stats are preserved correctly

    Why this matters:
    - Parent chunks will later provide richer context to the answer model.
    """
    context = HeadingContext(
        h1="Diabetes Mellitus",
        h2="Management",
    )

    parent_chunk = make_parent_chunk_record(
        parent_chunk_id="dm_parent_0001",
        doc_id="diabetes_mellitus",
        source_file="diabetes_mellitus.md",
        relative_path="diabetes_mellitus.md",
        chunk_text="Management includes diet, exercise, and medication when needed.",
        child_chunk_ids=["dm_0001", "dm_0002"],
        heading_context=context,
        section_title="Management",
        chunk_type="mixed",
        section_kind="heading_section",
    )

    assert parent_chunk.parent_chunk_id == "dm_parent_0001"
    assert parent_chunk.source_level == "parent"
    assert parent_chunk.child_chunk_ids == ["dm_0001", "dm_0002"]
    assert parent_chunk.heading_path() == ["Diabetes Mellitus", "Management"]
    assert parent_chunk.stats is not None
    assert parent_chunk.stats.word_count > 0


def test_parent_chunk_to_dict_contains_heading_path_and_children() -> None:
    """
    What this test checks:
    - Parent chunk serialization includes heading_path and child_chunk_ids.

    Why this matters:
    - Serialized parent chunks should remain easy to inspect and trace.
    """
    context = HeadingContext(
        h1="Stroke",
        h2="Emergency Care",
    )

    parent_chunk = make_parent_chunk_record(
        parent_chunk_id="stroke_parent_0001",
        doc_id="stroke",
        source_file="stroke.md",
        relative_path="stroke.md",
        chunk_text="Emergency care focuses on rapid assessment and stabilization.",
        child_chunk_ids=["stroke_0001", "stroke_0002"],
        heading_context=context,
    )

    payload = parent_chunk.to_dict()

    assert payload["parent_chunk_id"] == "stroke_parent_0001"
    assert payload["heading_path"] == ["Stroke", "Emergency Care"]
    assert payload["child_chunk_ids"] == ["stroke_0001", "stroke_0002"]
    assert payload["stats"]["word_count"] > 0