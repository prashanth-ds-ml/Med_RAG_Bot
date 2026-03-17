from __future__ import annotations

"""
Utilities for working with markdown headings.

Why this file exists:
- The chunker needs a reliable way to understand heading hierarchy
- We want chunk records to carry clean heading context metadata
- Later retrieval and answer generation benefit from section-aware chunks

What this module does:
- Detect markdown heading lines
- Extract heading level and text
- Maintain heading context as the parser walks through a document
- Expose helper functions for heading-based section labeling
"""

import re
from dataclasses import dataclass

from app.chunking.models import HeadingContext


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


@dataclass(slots=True)
class HeadingMatch:
    """
    Parsed representation of a markdown heading line.

    Example:
    '### Rate Control' -> level=3, text='Rate Control'
    """

    level: int
    text: str
    raw_line: str


def is_heading_line(line: str) -> bool:
    """
    Return True when a line is a markdown heading.

    Why this matters:
    - The chunker needs a quick filter before attempting deeper parsing
    """
    return HEADING_RE.match(line.strip()) is not None


def parse_heading_line(line: str) -> HeadingMatch | None:
    """
    Parse a markdown heading line into level + text.

    Returns None when the line is not a valid markdown heading.

    Why this matters:
    - Keeps heading parsing logic centralized and testable
    """
    stripped = line.strip()
    match = HEADING_RE.match(stripped)

    if not match:
        return None

    level = len(match.group(1))
    text = match.group(2).strip()

    return HeadingMatch(level=level, text=text, raw_line=stripped)


def normalize_heading_text(text: str) -> str:
    """
    Normalize heading text for cleaner metadata.

    Current behavior:
    - trims leading/trailing whitespace
    - collapses repeated internal whitespace

    Why this matters:
    - Source markdown may have inconsistent spacing
    - Clean heading metadata improves inspection and debugging
    """
    return " ".join(text.strip().split())


def update_heading_context(
    context: HeadingContext,
    heading_level: int,
    heading_text: str,
) -> HeadingContext:
    """
    Return a new HeadingContext updated with the latest heading.

    Behavior:
    - sets the given heading level to the new text
    - clears all deeper heading levels

    Example:
    existing: h1='A', h2='B', h3='C'
    update with level=2, text='D'
    result:   h1='A', h2='D', h3=None, h4=None, ...

    Why this matters:
    - This is the core operation behind hierarchical markdown parsing
    """
    if heading_level < 1 or heading_level > 6:
        raise ValueError("heading_level must be between 1 and 6")

    normalized_text = normalize_heading_text(heading_text)

    values = {
        "h1": context.h1,
        "h2": context.h2,
        "h3": context.h3,
        "h4": context.h4,
        "h5": context.h5,
        "h6": context.h6,
    }

    current_key = f"h{heading_level}"
    values[current_key] = normalized_text

    for level in range(heading_level + 1, 7):
        values[f"h{level}"] = None

    return HeadingContext(**values)


def heading_context_from_lines(lines: list[str]) -> HeadingContext:
    """
    Build heading context from a sequence of lines.

    This helper walks the lines in order and applies all headings found.

    Why this matters:
    - Useful for tests and for smaller parsing utilities
    """
    context = HeadingContext()

    for line in lines:
        parsed = parse_heading_line(line)
        if parsed is not None:
            context = update_heading_context(
                context=context,
                heading_level=parsed.level,
                heading_text=parsed.text,
            )

    return context


def get_current_section_title(context: HeadingContext) -> str | None:
    """
    Return the deepest available heading as the current section title.

    Why this matters:
    - Helpful when labeling chunks with a human-readable section name
    """
    path = context.to_path()
    return path[-1] if path else None


def heading_path_to_string(context: HeadingContext, separator: str = " > ") -> str:
    """
    Convert heading path to a readable string.

    Example:
    ['Atrial Fibrillation', 'Management', 'Rate Control']
    -> 'Atrial Fibrillation > Management > Rate Control'
    """
    return separator.join(context.to_path())


def split_lines_into_heading_sections(lines: list[str]) -> list[tuple[HeadingContext, list[str]]]:
    """
    Split a markdown document into heading-driven sections.

    Returns:
    - a list of tuples: (heading_context, section_lines)

    Behavior:
    - Each new heading starts a new section
    - Non-heading text before the first heading becomes its own section
      with an empty HeadingContext

    Why this matters:
    - This provides the chunker with a structure-aware first pass
    """
    sections: list[tuple[HeadingContext, list[str]]] = []

    current_context = HeadingContext()
    current_lines: list[str] = []

    for line in lines:
        parsed = parse_heading_line(line)

        if parsed is not None:
            if current_lines:
                sections.append((current_context, current_lines))

            current_context = update_heading_context(
                context=current_context,
                heading_level=parsed.level,
                heading_text=parsed.text,
            )
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_context, current_lines))

    return sections