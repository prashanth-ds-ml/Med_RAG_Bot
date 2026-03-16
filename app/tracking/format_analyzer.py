from __future__ import annotations

"""
Markdown formatting analyzer for corpus evolution tracking.

Why this file exists:
- We want to measure how source formatting changes over time.
- Better markdown structure often leads to better chunking and retrieval.
- These stats help us understand which cleanup edits actually help the RAG system.

What this module does:
- Counts headings, bullets, numbered items, code fences, table-like lines, and image refs
- Estimates section structure quality
- Detects large flat blocks and broken heading order
- Produces a compact stats dictionary for each markdown document

Important note:
- These are heuristic signals, not perfect markdown parsing.
- That is okay for tracking trends and comparing revisions over time.
"""

from dataclasses import dataclass, asdict
import re
from typing import Any


HEADING_RE = re.compile(r"^(#{1,6})\s+.+$")
BULLET_RE = re.compile(r"^\s*[-*+]\s+.+$")
NUMBERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+.+$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
TABLE_PIPE_RE = re.compile(r"\|")
TABLE_STUB_RE = re.compile(r"\[TABLE:.*?\]", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^\s*```")
BLANK_LINE_RE = re.compile(r"^\s*$")


@dataclass
class FormattingStats:
    """
    Structured formatting stats for a markdown document.

    Why dataclass:
    - Makes the output explicit and easy to test
    - Easy to serialize later with asdict()
    """

    char_count: int
    word_count: int
    line_count: int

    heading_count_h1: int
    heading_count_h2: int
    heading_count_h3: int
    heading_count_h4_plus: int
    heading_depth_max: int

    bullet_count: int
    numbered_list_count: int
    code_block_count: int
    table_stub_count: int
    table_like_line_count: int
    image_ref_count: int

    blank_line_count: int
    blank_line_ratio: float

    section_count: int
    avg_section_length: float
    max_section_length: int
    num_short_sections: int
    num_long_sections: int

    has_clear_hierarchy: bool
    has_broken_heading_order: bool
    has_large_flat_blocks: bool
    has_dense_list_regions: bool


def _count_words(text: str) -> int:
    """
    Count words using simple whitespace splitting.

    Why simple counting is enough:
    - We only need stable trend signals, not linguistically perfect tokenization.
    """
    return len(text.split())


def _extract_section_lengths(lines: list[str]) -> list[int]:
    """
    Split the document into heading-based sections and return section lengths in words.

    Why this matters:
    - Section size is a good proxy for whether the markdown is cleanly structured.
    - Very long sections often indicate flat, hard-to-chunk content.
    """
    sections: list[list[str]] = []
    current_section: list[str] = []

    for line in lines:
        if HEADING_RE.match(line):
            if current_section:
                sections.append(current_section)
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append(current_section)

    # If the document is empty, return no sections.
    if not sections:
        return []

    section_lengths = []
    for section in sections:
        section_text = "\n".join(section).strip()
        section_lengths.append(_count_words(section_text))

    return section_lengths


def _detect_broken_heading_order(lines: list[str]) -> bool:
    """
    Detect suspicious heading jumps like H1 -> H3 without an H2 in between.

    Why this matters:
    - Heading jumps often indicate messy markdown hierarchy.
    - This can hurt chunk grouping and section understanding.
    """
    seen_levels: set[int] = set()
    previous_level: int | None = None

    for line in lines:
        match = HEADING_RE.match(line)
        if not match:
            continue

        level = len(match.group(1))
        seen_levels.add(level)

        if previous_level is not None and level > previous_level + 1:
            return True

        # Also flag cases like first heading being H3 or H4.
        if previous_level is None and level > 2:
            return True

        previous_level = level

    return False


def _detect_dense_list_regions(lines: list[str]) -> bool:
    """
    Detect whether the document contains many consecutive list lines.

    Why this matters:
    - Dense list-heavy regions may need special chunk handling.
    - Lists often contain compact but high-value medical guidance.
    """
    consecutive_list_lines = 0
    max_consecutive_list_lines = 0

    for line in lines:
        if BULLET_RE.match(line) or NUMBERED_LIST_RE.match(line):
            consecutive_list_lines += 1
            max_consecutive_list_lines = max(max_consecutive_list_lines, consecutive_list_lines)
        else:
            consecutive_list_lines = 0

    return max_consecutive_list_lines >= 5


def analyze_markdown_format(text: str) -> dict[str, Any]:
    """
    Analyze markdown text and return formatting statistics.

    Parameters
    ----------
    text : str
        Raw markdown text.

    Returns
    -------
    dict[str, Any]
        Dictionary of formatting statistics.

    Why dictionary output:
    - Easy to write into JSON / JSONL tracking files
    - Easy to compare across source snapshots
    """
    lines = text.splitlines()

    heading_count_h1 = 0
    heading_count_h2 = 0
    heading_count_h3 = 0
    heading_count_h4_plus = 0
    heading_depth_max = 0

    bullet_count = 0
    numbered_list_count = 0
    code_fence_lines = 0
    table_stub_count = 0
    table_like_line_count = 0
    image_ref_count = 0
    blank_line_count = 0

    for line in lines:
        heading_match = HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_depth_max = max(heading_depth_max, level)

            if level == 1:
                heading_count_h1 += 1
            elif level == 2:
                heading_count_h2 += 1
            elif level == 3:
                heading_count_h3 += 1
            else:
                heading_count_h4_plus += 1

        if BULLET_RE.match(line):
            bullet_count += 1

        if NUMBERED_LIST_RE.match(line):
            numbered_list_count += 1

        if CODE_FENCE_RE.match(line):
            code_fence_lines += 1

        if TABLE_STUB_RE.search(line):
            table_stub_count += 1

        if TABLE_PIPE_RE.search(line):
            table_like_line_count += 1

        if IMAGE_RE.search(line):
            image_ref_count += 1

        if BLANK_LINE_RE.match(line):
            blank_line_count += 1

    # Two fence lines generally represent one fenced code block.
    code_block_count = code_fence_lines // 2

    section_lengths = _extract_section_lengths(lines)
    section_count = len(section_lengths)

    avg_section_length = (
        round(sum(section_lengths) / section_count, 2) if section_count > 0 else 0.0
    )
    max_section_length = max(section_lengths) if section_lengths else 0

    # Heuristic thresholds that are simple and stable for tracking.
    num_short_sections = sum(1 for length in section_lengths if length < 80)
    num_long_sections = sum(1 for length in section_lengths if length > 300)

    has_clear_hierarchy = (
        (heading_count_h1 + heading_count_h2 + heading_count_h3) > 0
        and not _detect_broken_heading_order(lines)
        and heading_depth_max <= 4
    )

    has_broken_heading_order = _detect_broken_heading_order(lines)
    has_large_flat_blocks = section_count <= 1 and _count_words(text) > 300
    has_dense_list_regions = _detect_dense_list_regions(lines)

    stats = FormattingStats(
        char_count=len(text),
        word_count=_count_words(text),
        line_count=len(lines),
        heading_count_h1=heading_count_h1,
        heading_count_h2=heading_count_h2,
        heading_count_h3=heading_count_h3,
        heading_count_h4_plus=heading_count_h4_plus,
        heading_depth_max=heading_depth_max,
        bullet_count=bullet_count,
        numbered_list_count=numbered_list_count,
        code_block_count=code_block_count,
        table_stub_count=table_stub_count,
        table_like_line_count=table_like_line_count,
        image_ref_count=image_ref_count,
        blank_line_count=blank_line_count,
        blank_line_ratio=round(blank_line_count / len(lines), 4) if lines else 0.0,
        section_count=section_count,
        avg_section_length=avg_section_length,
        max_section_length=max_section_length,
        num_short_sections=num_short_sections,
        num_long_sections=num_long_sections,
        has_clear_hierarchy=has_clear_hierarchy,
        has_broken_heading_order=has_broken_heading_order,
        has_large_flat_blocks=has_large_flat_blocks,
        has_dense_list_regions=has_dense_list_regions,
    )

    return asdict(stats)