from __future__ import annotations

"""
Data models for markdown chunking.

Why this file exists:
- Gives the chunking pipeline a clear, consistent schema
- Makes chunk records easier to validate, test, log, and serialize
- Keeps later retrieval / reranking / answer generation modules aligned

Design goals:
- Human-readable fields
- Easy JSON/JSONL serialization
- Explicit metadata for tracing chunk origin and hierarchy
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ChunkType = Literal[
    "text",
    "list",
    "table_stub",
    "image_ref",
    "mixed",
]

SectionKind = Literal[
    "heading_section",
    "merged_small_section",
    "split_large_section",
    "standalone_block",
]

ParentSourceType = Literal[
    "atomic",
    "parent",
]


@dataclass(slots=True)
class HeadingContext:
    """
    Stores the heading hierarchy for a chunk.

    Example:
    - h1: 'Atrial Fibrillation'
    - h2: 'Management'
    - h3: 'Rate Control'

    Why this matters:
    - Later retrieval and answer generation benefit from section-aware metadata
    - Helps inspect where a chunk came from in the original markdown
    """

    h1: str | None = None
    h2: str | None = None
    h3: str | None = None
    h4: str | None = None
    h5: str | None = None
    h6: str | None = None

    def to_path(self) -> list[str]:
        """
        Return the non-empty heading hierarchy as a list.

        Example:
        ['Atrial Fibrillation', 'Management', 'Rate Control']
        """
        return [
            heading
            for heading in [self.h1, self.h2, self.h3, self.h4, self.h5, self.h6]
            if heading
        ]

    def max_depth(self) -> int:
        """
        Return the deepest populated heading level.

        Returns 0 when no headings are set.
        """
        levels = [self.h1, self.h2, self.h3, self.h4, self.h5, self.h6]
        depth = 0
        for index, value in enumerate(levels, start=1):
            if value:
                depth = index
        return depth

    def to_dict(self) -> dict[str, str | None]:
        """
        Return a serializable dictionary view.
        """
        return asdict(self)


@dataclass(slots=True)
class ChunkStats:
    """
    Basic length and structural statistics for a chunk.

    Why this matters:
    - Useful for quality checks
    - Helps later filtering, debugging, and monitoring
    """

    char_count: int
    word_count: int
    line_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class ChunkRecord:
    """
    Atomic chunk record.

    This is the main unit used for retrieval.

    Fields are designed so every chunk can be traced back to:
    - source file
    - heading hierarchy
    - section behavior
    - parent chunk
    """

    chunk_id: str
    doc_id: str
    source_file: str
    relative_path: str

    chunk_type: ChunkType
    section_kind: SectionKind
    source_level: ParentSourceType = "atomic"

    heading_context: HeadingContext = field(default_factory=HeadingContext)
    section_title: str | None = None

    chunk_text: str = ""
    chunk_index: int = 0
    parent_chunk_id: str | None = None

    has_table_ref: bool = False
    has_image_ref: bool = False
    has_code_block: bool = False
    is_list_heavy: bool = False

    stats: ChunkStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def heading_path(self) -> list[str]:
        """
        Return the heading path as a list.
        """
        return self.heading_context.to_path()

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable dictionary.

        Why this matters:
        - This is the format we will later write to JSONL
        - Keeps nested dataclasses flattened into plain dictionaries
        """
        payload = asdict(self)
        payload["heading_path"] = self.heading_path()
        return payload


@dataclass(slots=True)
class ParentChunkRecord:
    """
    Parent chunk record built from one or more atomic chunks.

    Why this matters:
    - Atomic chunks help retrieval precision
    - Parent chunks help reasoning models answer with more context
    """

    parent_chunk_id: str
    doc_id: str
    source_file: str
    relative_path: str

    heading_context: HeadingContext = field(default_factory=HeadingContext)
    section_title: str | None = None

    chunk_text: str = ""
    child_chunk_ids: list[str] = field(default_factory=list)

    source_level: ParentSourceType = "parent"
    chunk_type: ChunkType = "mixed"
    section_kind: SectionKind = "heading_section"

    has_table_ref: bool = False
    has_image_ref: bool = False
    has_code_block: bool = False
    is_list_heavy: bool = False

    stats: ChunkStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def heading_path(self) -> list[str]:
        return self.heading_context.to_path()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["heading_path"] = self.heading_path()
        return payload


def compute_chunk_stats(text: str) -> ChunkStats:
    """
    Compute basic stats for a text block.

    Why this helper exists:
    - Keeps stats generation consistent across atomic and parent chunks
    """
    lines = text.splitlines()
    words = text.split()

    return ChunkStats(
        char_count=len(text),
        word_count=len(words),
        line_count=len(lines),
    )


def make_chunk_record(
    *,
    chunk_id: str,
    doc_id: str,
    source_file: str,
    relative_path: str,
    chunk_text: str,
    chunk_index: int,
    chunk_type: ChunkType = "text",
    section_kind: SectionKind = "heading_section",
    heading_context: HeadingContext | None = None,
    section_title: str | None = None,
    parent_chunk_id: str | None = None,
    has_table_ref: bool = False,
    has_image_ref: bool = False,
    has_code_block: bool = False,
    is_list_heavy: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ChunkRecord:
    """
    Factory helper for creating atomic chunk records.

    Why this helper exists:
    - Reduces repeated boilerplate in the chunking module
    - Ensures stats are always computed consistently
    """
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=source_file,
        relative_path=relative_path,
        chunk_type=chunk_type,
        section_kind=section_kind,
        source_level="atomic",
        heading_context=heading_context or HeadingContext(),
        section_title=section_title,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        parent_chunk_id=parent_chunk_id,
        has_table_ref=has_table_ref,
        has_image_ref=has_image_ref,
        has_code_block=has_code_block,
        is_list_heavy=is_list_heavy,
        stats=compute_chunk_stats(chunk_text),
        metadata=metadata or {},
    )


def make_parent_chunk_record(
    *,
    parent_chunk_id: str,
    doc_id: str,
    source_file: str,
    relative_path: str,
    chunk_text: str,
    child_chunk_ids: list[str],
    heading_context: HeadingContext | None = None,
    section_title: str | None = None,
    chunk_type: ChunkType = "mixed",
    section_kind: SectionKind = "heading_section",
    has_table_ref: bool = False,
    has_image_ref: bool = False,
    has_code_block: bool = False,
    is_list_heavy: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ParentChunkRecord:
    """
    Factory helper for creating parent chunk records.

    Why this helper exists:
    - Keeps parent chunk creation consistent
    - Ensures stats are always attached
    """
    return ParentChunkRecord(
        parent_chunk_id=parent_chunk_id,
        doc_id=doc_id,
        source_file=source_file,
        relative_path=relative_path,
        heading_context=heading_context or HeadingContext(),
        section_title=section_title,
        chunk_text=chunk_text,
        child_chunk_ids=child_chunk_ids,
        source_level="parent",
        chunk_type=chunk_type,
        section_kind=section_kind,
        has_table_ref=has_table_ref,
        has_image_ref=has_image_ref,
        has_code_block=has_code_block,
        is_list_heavy=is_list_heavy,
        stats=compute_chunk_stats(chunk_text),
        metadata=metadata or {},
    )