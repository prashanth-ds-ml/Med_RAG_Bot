from __future__ import annotations

"""
Markdown chunking pipeline.

Why this file exists:
- Turns cleaned markdown documents into retrieval-ready atomic chunks
- Builds larger parent chunks for richer answer-time context
- Preserves heading hierarchy and useful structural metadata

Chunking strategy used here:
1. Split document into heading-driven sections
2. Split each section into content blocks
3. Convert blocks into atomic chunks
4. Merge nearby atomic chunks into parent chunks

Design goals:
- Deterministic and inspectable
- Structure-aware rather than naive fixed-window splitting
- Easy to test and evolve as the corpus grows
"""

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Iterable

from app.chunking.heading_utils import (
    get_current_section_title,
    is_heading_line,
    split_lines_into_heading_sections,
)
from app.chunking.models import (
    ChunkRecord,
    ChunkType,
    HeadingContext,
    ParentChunkRecord,
    make_chunk_record,
    make_parent_chunk_record,
)


LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+.+$")
TABLE_STUB_RE = re.compile(r"\[TABLE:.*?\]", re.IGNORECASE)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
CODE_FENCE_RE = re.compile(r"^\s*```")
BLANK_LINE_RE = re.compile(r"^\s*$")


@dataclass(slots=True)
class ChunkingConfig:
    """
    Chunking thresholds and heuristics.

    Why these settings exist:
    - Small chunks improve retrieval precision
    - Larger parent chunks provide better reasoning context
    - Thresholds are kept word-based for simplicity and transparency
    """

    min_chunk_words: int = 80
    target_chunk_words: int = 220
    max_chunk_words: int = 300
    overlap_words: int = 40

    parent_target_words: int = 700
    parent_max_words: int = 900


@dataclass(slots=True)
class ChunkingResult:
    """
    Final output of a markdown chunking run.
    """

    doc_id: str
    source_file: str
    relative_path: str
    atomic_chunks: list[ChunkRecord] = field(default_factory=list)
    parent_chunks: list[ParentChunkRecord] = field(default_factory=list)

    def summary(self) -> dict[str, int | str]:
        """
        Return a compact summary for logging or CLI display.
        """
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "relative_path": self.relative_path,
            "atomic_chunk_count": len(self.atomic_chunks),
            "parent_chunk_count": len(self.parent_chunks),
        }


def _count_words(text: str) -> int:
    return len(text.split())


def _normalize_doc_id(relative_path: str) -> str:
    """
    Build a readable stable doc_id from a relative markdown path.
    """
    doc_id = relative_path.replace("\\", "/").lower()
    if doc_id.endswith(".md"):
        doc_id = doc_id[:-3]
    doc_id = re.sub(r"[^a-z0-9/_-]+", "_", doc_id)
    doc_id = doc_id.replace("/", "__")
    return doc_id.strip("_")


def _strip_empty_edges(lines: list[str]) -> list[str]:
    """
    Trim leading and trailing blank lines from a list of lines.
    """
    start = 0
    end = len(lines)

    while start < end and BLANK_LINE_RE.match(lines[start]):
        start += 1
    while end > start and BLANK_LINE_RE.match(lines[end - 1]):
        end -= 1

    return lines[start:end]


def _group_section_into_blocks(lines: list[str]) -> list[list[str]]:
    """
    Group section lines into content blocks.

    Rules:
    - blank lines split blocks
    - fenced code blocks stay intact
    - heading lines are not expected here and are skipped if seen

    Why this matters:
    - This is the structure-aware middle step before chunk splitting
    """
    blocks: list[list[str]] = []
    current_block: list[str] = []
    in_code_fence = False

    for line in lines:
        if is_heading_line(line):
            continue

        if CODE_FENCE_RE.match(line):
            current_block.append(line)
            in_code_fence = not in_code_fence
            continue

        if BLANK_LINE_RE.match(line) and not in_code_fence:
            trimmed = _strip_empty_edges(current_block)
            if trimmed:
                blocks.append(trimmed)
            current_block = []
            continue

        current_block.append(line)

    trimmed = _strip_empty_edges(current_block)
    if trimmed:
        blocks.append(trimmed)

    return blocks


def _is_list_heavy(lines: list[str]) -> bool:
    """
    Return True when a block is mostly made of list items.
    """
    non_blank = [line for line in lines if not BLANK_LINE_RE.match(line)]
    if not non_blank:
        return False

    list_lines = sum(1 for line in non_blank if LIST_ITEM_RE.match(line))
    return list_lines / len(non_blank) >= 0.5


def _detect_chunk_type(lines: list[str]) -> ChunkType:
    """
    Infer a coarse block type from content lines.
    """
    text = "\n".join(lines)

    has_table_stub = TABLE_STUB_RE.search(text) is not None
    has_image_ref = IMAGE_RE.search(text) is not None
    is_list = _is_list_heavy(lines)

    non_blank = [line for line in lines if not BLANK_LINE_RE.match(line)]
    table_like_line_count = sum(1 for line in non_blank if "|" in line)

    if has_table_stub:
        return "table_stub"

    if has_image_ref and len(non_blank) <= 2:
        return "image_ref"

    if is_list:
        return "list"

    if has_image_ref or table_like_line_count >= 2:
        return "mixed"

    return "text"


def _split_words_with_overlap(
    text: str,
    *,
    target_words: int,
    max_words: int,
    overlap_words: int,
) -> list[str]:
    """
    Split text into overlapping word windows.

    Why this exists:
    - Large paragraphs or flat blocks still need safe chunk boundaries
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= max_words:
        return [" ".join(words)]

    results: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + target_words, len(words))
        if end - start < 1:
            break

        if end < len(words):
            end = min(start + max_words, len(words))

        results.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = max(end - overlap_words, start + 1)

    return results


def _split_list_block(
    lines: list[str],
    *,
    target_words: int,
    max_words: int,
) -> list[str]:
    """
    Split a list-heavy block by list items where possible.

    Why this matters:
    - Preserves list item boundaries better than plain word windows
    """
    items: list[list[str]] = []
    current_item: list[str] = []

    for line in lines:
        if LIST_ITEM_RE.match(line):
            if current_item:
                items.append(current_item)
            current_item = [line]
        else:
            if current_item:
                current_item.append(line)
            else:
                current_item = [line]

    if current_item:
        items.append(current_item)

    if not items:
        return []

    chunks: list[str] = []
    current_lines: list[str] = []
    current_words = 0

    for item in items:
        item_text = "\n".join(item).strip()
        item_words = _count_words(item_text)

        # If a single item is too large, flush current and split the item itself.
        if item_words > max_words:
            if current_lines:
                chunks.append("\n".join(current_lines).strip())
                current_lines = []
                current_words = 0

            chunks.extend(
                _split_words_with_overlap(
                    item_text,
                    target_words=target_words,
                    max_words=max_words,
                    overlap_words=0,
                )
            )
            continue

        if current_lines and current_words + item_words > target_words:
            chunks.append("\n".join(current_lines).strip())
            current_lines = []
            current_words = 0

        current_lines.extend(item)
        current_words += item_words

    if current_lines:
        chunks.append("\n".join(current_lines).strip())

    return [chunk for chunk in chunks if chunk.strip()]


def _split_block_into_atomic_texts(
    lines: list[str],
    *,
    config: ChunkingConfig,
) -> tuple[list[str], ChunkType, bool, bool, bool, bool]:
    """
    Split a block into atomic chunk texts and detect block flags.

    Returns:
    - chunk_texts
    - chunk_type
    - has_table_ref
    - has_image_ref
    - has_code_block
    - is_list_heavy
    """
    text = "\n".join(lines).strip()
    if not text:
        return [], "text", False, False, False, False

    chunk_type = _detect_chunk_type(lines)
    has_table_ref = TABLE_STUB_RE.search(text) is not None
    has_image_ref = IMAGE_RE.search(text) is not None
    has_code_block = "```" in text
    is_list_heavy = _is_list_heavy(lines)

    if chunk_type == "list":
        pieces = _split_list_block(
            lines,
            target_words=config.target_chunk_words,
            max_words=config.max_chunk_words,
        )
        return (
            pieces,
            chunk_type,
            has_table_ref,
            has_image_ref,
            has_code_block,
            is_list_heavy,
        )

    pieces = _split_words_with_overlap(
        text,
        target_words=config.target_chunk_words,
        max_words=config.max_chunk_words,
        overlap_words=config.overlap_words,
    )
    return (
        pieces,
        chunk_type,
        has_table_ref,
        has_image_ref,
        has_code_block,
        is_list_heavy,
    )


def _merge_small_atomic_chunks(
    chunks: list[ChunkRecord],
    *,
    min_chunk_words: int,
    target_chunk_words: int,
) -> list[ChunkRecord]:
    """
    Merge very small adjacent atomic chunks when they share the same heading path.

    Why this matters:
    - Tiny chunks often hurt retrieval usefulness
    - Small clean merges improve context without bloating chunk size
    """
    if not chunks:
        return []

    merged: list[ChunkRecord] = []
    current = chunks[0]

    for next_chunk in chunks[1:]:
        same_heading = current.heading_path() == next_chunk.heading_path()
        both_small = (
            current.stats is not None
            and next_chunk.stats is not None
            and current.stats.word_count < min_chunk_words
            and current.stats.word_count + next_chunk.stats.word_count <= target_chunk_words
        )

        if same_heading and both_small:
            merged_text = f"{current.chunk_text}\n\n{next_chunk.chunk_text}".strip()
            current = make_chunk_record(
                chunk_id=current.chunk_id,
                doc_id=current.doc_id,
                source_file=current.source_file,
                relative_path=current.relative_path,
                chunk_text=merged_text,
                chunk_index=current.chunk_index,
                chunk_type="mixed" if current.chunk_type != next_chunk.chunk_type else current.chunk_type,
                section_kind="merged_small_section",
                heading_context=current.heading_context,
                section_title=current.section_title,
                parent_chunk_id=current.parent_chunk_id,
                has_table_ref=current.has_table_ref or next_chunk.has_table_ref,
                has_image_ref=current.has_image_ref or next_chunk.has_image_ref,
                has_code_block=current.has_code_block or next_chunk.has_code_block,
                is_list_heavy=current.is_list_heavy or next_chunk.is_list_heavy,
                metadata={
                    **current.metadata,
                    "merged_from": current.metadata.get("merged_from", [current.chunk_id]) + [next_chunk.chunk_id],
                },
            )
        else:
            merged.append(current)
            current = next_chunk

    merged.append(current)
    return merged


def _build_parent_chunks(
    atomic_chunks: list[ChunkRecord],
    *,
    config: ChunkingConfig,
) -> list[ParentChunkRecord]:
    """
    Build parent chunks from adjacent atomic chunks.

    Strategy:
    - combine neighboring atomic chunks with the same heading path
    - stop when parent target/max size is reached
    """
    if not atomic_chunks:
        return []

    parent_chunks: list[ParentChunkRecord] = []
    parent_index = 0

    current_group: list[ChunkRecord] = []
    current_word_count = 0

    def flush_group(group: list[ChunkRecord]) -> None:
        nonlocal parent_index

        if not group:
            return

        parent_index += 1
        parent_text = "\n\n".join(chunk.chunk_text for chunk in group).strip()
        first = group[0]

        parent_chunks.append(
            make_parent_chunk_record(
                parent_chunk_id=f"{first.doc_id}_parent_{parent_index:04d}",
                doc_id=first.doc_id,
                source_file=first.source_file,
                relative_path=first.relative_path,
                chunk_text=parent_text,
                child_chunk_ids=[chunk.chunk_id for chunk in group],
                heading_context=first.heading_context,
                section_title=first.section_title,
                chunk_type="mixed" if len({chunk.chunk_type for chunk in group}) > 1 else group[0].chunk_type,
                section_kind=first.section_kind,
                has_table_ref=any(chunk.has_table_ref for chunk in group),
                has_image_ref=any(chunk.has_image_ref for chunk in group),
                has_code_block=any(chunk.has_code_block for chunk in group),
                is_list_heavy=any(chunk.is_list_heavy for chunk in group),
                metadata={"child_count": len(group)},
            )
        )

    for chunk in atomic_chunks:
        chunk_words = chunk.stats.word_count if chunk.stats is not None else _count_words(chunk.chunk_text)

        if not current_group:
            current_group = [chunk]
            current_word_count = chunk_words
            continue

        same_heading = current_group[-1].heading_path() == chunk.heading_path()
        would_exceed = current_word_count + chunk_words > config.parent_max_words

        if same_heading and not would_exceed and current_word_count < config.parent_target_words:
            current_group.append(chunk)
            current_word_count += chunk_words
        else:
            flush_group(current_group)
            current_group = [chunk]
            current_word_count = chunk_words

    flush_group(current_group)
    return parent_chunks


def chunk_markdown_document(
    *,
    text: str,
    source_file: str,
    relative_path: str,
    doc_id: str | None = None,
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """
    Chunk a markdown document into atomic and parent chunks.

    Parameters
    ----------
    text:
        Raw markdown text
    source_file:
        Source file name, e.g. 'atrial_fibrillation.md'
    relative_path:
        Relative path inside the corpus
    doc_id:
        Optional explicit doc_id
    config:
        Optional chunking config

    Returns
    -------
    ChunkingResult
        Atomic and parent chunks for the document
    """
    config = config or ChunkingConfig()
    doc_id = doc_id or _normalize_doc_id(relative_path)

    lines = text.splitlines()
    sections = split_lines_into_heading_sections(lines)

    atomic_chunks: list[ChunkRecord] = []
    chunk_index = 0

    for heading_context, section_lines in sections:
        section_title = get_current_section_title(heading_context)

        content_lines = section_lines[:]
        if content_lines and is_heading_line(content_lines[0]):
            content_lines = content_lines[1:]

        blocks = _group_section_into_blocks(content_lines)

        for block_lines in blocks:
            (
                atomic_texts,
                chunk_type,
                has_table_ref,
                has_image_ref,
                has_code_block,
                is_list_heavy,
            ) = _split_block_into_atomic_texts(block_lines, config=config)

            for piece_index, chunk_text in enumerate(atomic_texts):
                chunk_index += 1

                section_kind = "heading_section"
                if len(atomic_texts) > 1:
                    section_kind = "split_large_section"

                atomic_chunks.append(
                    make_chunk_record(
                        chunk_id=f"{doc_id}_{chunk_index:04d}",
                        doc_id=doc_id,
                        source_file=source_file,
                        relative_path=relative_path,
                        chunk_text=chunk_text,
                        chunk_index=chunk_index - 1,
                        chunk_type=chunk_type,
                        section_kind=section_kind,
                        heading_context=heading_context,
                        section_title=section_title,
                        has_table_ref=has_table_ref,
                        has_image_ref=has_image_ref,
                        has_code_block=has_code_block,
                        is_list_heavy=is_list_heavy,
                        metadata={"piece_index_within_block": piece_index},
                    )
                )

    atomic_chunks = _merge_small_atomic_chunks(
        atomic_chunks,
        min_chunk_words=config.min_chunk_words,
        target_chunk_words=config.target_chunk_words,
    )

    # Reassign clean sequential chunk indexes after merging.
    normalized_atomic_chunks: list[ChunkRecord] = []
    for new_index, chunk in enumerate(atomic_chunks):
        normalized_atomic_chunks.append(
            make_chunk_record(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_file=chunk.source_file,
                relative_path=chunk.relative_path,
                chunk_text=chunk.chunk_text,
                chunk_index=new_index,
                chunk_type=chunk.chunk_type,
                section_kind=chunk.section_kind,
                heading_context=chunk.heading_context,
                section_title=chunk.section_title,
                parent_chunk_id=chunk.parent_chunk_id,
                has_table_ref=chunk.has_table_ref,
                has_image_ref=chunk.has_image_ref,
                has_code_block=chunk.has_code_block,
                is_list_heavy=chunk.is_list_heavy,
                metadata=chunk.metadata,
            )
        )

    parent_chunks = _build_parent_chunks(normalized_atomic_chunks, config=config)

    return ChunkingResult(
        doc_id=doc_id,
        source_file=source_file,
        relative_path=relative_path,
        atomic_chunks=normalized_atomic_chunks,
        parent_chunks=parent_chunks,
    )


def chunk_markdown_file(
    file_path: str | Path,
    *,
    source_root: str | Path | None = None,
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """
    Chunk a markdown file from disk.

    Why this helper exists:
    - Keeps file IO separate from higher-level ingestion flow
    - Useful for manual testing and future CLI commands
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    if source_root is not None:
        relative_path = str(path.resolve().relative_to(Path(source_root).resolve())).replace("\\", "/")
    else:
        relative_path = path.name

    return chunk_markdown_document(
        text=text,
        source_file=path.name,
        relative_path=relative_path,
        config=config,
    )