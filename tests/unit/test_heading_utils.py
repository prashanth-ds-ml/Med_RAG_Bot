from app.chunking.models import HeadingContext
from app.chunking.heading_utils import (
    get_current_section_title,
    heading_context_from_lines,
    heading_path_to_string,
    is_heading_line,
    normalize_heading_text,
    parse_heading_line,
    split_lines_into_heading_sections,
    update_heading_context,
)


def test_is_heading_line_detects_valid_markdown_headings() -> None:
    """
    What this test checks:
    - Valid markdown headings are recognized correctly.

    Why this matters:
    - Heading detection is the first gate for hierarchy-aware chunking.
    """
    assert is_heading_line("# Title") is True
    assert is_heading_line("## Section") is True
    assert is_heading_line("###### Deep Heading") is True
    assert is_heading_line("Plain text") is False
    assert is_heading_line("  - bullet item") is False


def test_parse_heading_line_returns_level_and_text() -> None:
    """
    What this test checks:
    - Heading parsing extracts both level and clean heading text.

    Why this matters:
    - The chunker depends on accurate heading level + text metadata.
    """
    parsed = parse_heading_line("### Rate Control")

    assert parsed is not None
    assert parsed.level == 3
    assert parsed.text == "Rate Control"
    assert parsed.raw_line == "### Rate Control"


def test_parse_heading_line_returns_none_for_non_heading() -> None:
    """
    What this test checks:
    - Non-heading lines are rejected cleanly.

    Why this matters:
    - Prevents accidental false heading detection.
    """
    assert parse_heading_line("This is normal text.") is None
    assert parse_heading_line("- bullet item") is None


def test_normalize_heading_text_collapses_extra_spaces() -> None:
    """
    What this test checks:
    - Heading normalization trims and collapses whitespace.

    Why this matters:
    - Source markdown can have inconsistent spacing.
    """
    assert normalize_heading_text("  Rate   Control   ") == "Rate Control"


def test_update_heading_context_sets_current_level_and_clears_deeper_levels() -> None:
    """
    What this test checks:
    - Updating a heading context replaces the target level
    - All deeper heading levels are cleared

    Why this matters:
    - This is the core rule for maintaining valid hierarchy state.
    """
    context = HeadingContext(
        h1="Atrial Fibrillation",
        h2="Management",
        h3="Rate Control",
    )

    updated = update_heading_context(
        context=context,
        heading_level=2,
        heading_text="Symptoms",
    )

    assert updated.h1 == "Atrial Fibrillation"
    assert updated.h2 == "Symptoms"
    assert updated.h3 is None
    assert updated.h4 is None
    assert updated.to_path() == ["Atrial Fibrillation", "Symptoms"]


def test_update_heading_context_rejects_invalid_level() -> None:
    """
    What this test checks:
    - Invalid heading levels raise a clear error.

    Why this matters:
    - Guards against broken parser inputs.
    """
    context = HeadingContext()

    try:
        update_heading_context(context=context, heading_level=0, heading_text="Bad")
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_heading_context_from_lines_builds_final_context() -> None:
    """
    What this test checks:
    - A sequence of heading lines builds the expected final heading context.

    Why this matters:
    - Useful for smaller parsing utilities and validation logic.
    """
    lines = [
        "# Atrial Fibrillation",
        "Some intro text",
        "## Management",
        "### Rate Control",
    ]

    context = heading_context_from_lines(lines)

    assert context.to_path() == [
        "Atrial Fibrillation",
        "Management",
        "Rate Control",
    ]


def test_get_current_section_title_returns_deepest_heading() -> None:
    """
    What this test checks:
    - The current section title is the deepest populated heading.

    Why this matters:
    - Chunk labels should use the most specific heading available.
    """
    context = HeadingContext(
        h1="Diabetes",
        h2="Management",
        h3="Lifestyle",
    )

    assert get_current_section_title(context) == "Lifestyle"


def test_heading_path_to_string_builds_readable_path() -> None:
    """
    What this test checks:
    - Heading path is rendered into a readable string.

    Why this matters:
    - Helpful in CLI inspection and debugging output.
    """
    context = HeadingContext(
        h1="Stroke",
        h2="Emergency Care",
    )

    assert heading_path_to_string(context) == "Stroke > Emergency Care"


def test_split_lines_into_heading_sections_handles_preface_and_headings() -> None:
    """
    What this test checks:
    - Content before the first heading is preserved as its own section
    - Each heading starts a new section with the correct context

    Why this matters:
    - Some markdown files may begin with loose text before the first heading.
    """
    lines = [
        "Intro text before heading",
        "",
        "# Title",
        "Paragraph under title",
        "## Section A",
        "Paragraph A",
        "## Section B",
        "Paragraph B",
    ]

    sections = split_lines_into_heading_sections(lines)

    assert len(sections) == 4

    preface_context, preface_lines = sections[0]
    assert preface_context.to_path() == []
    assert "Intro text before heading" in preface_lines[0]

    title_context, title_lines = sections[1]
    assert title_context.to_path() == ["Title"]
    assert title_lines[0] == "# Title"

    section_a_context, section_a_lines = sections[2]
    assert section_a_context.to_path() == ["Title", "Section A"]
    assert section_a_lines[0] == "## Section A"

    section_b_context, section_b_lines = sections[3]
    assert section_b_context.to_path() == ["Title", "Section B"]
    assert section_b_lines[0] == "## Section B"