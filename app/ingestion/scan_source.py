from __future__ import annotations

"""
Source scanning utilities for markdown ingestion.

Why this file exists:
- The ingest pipeline needs a clean, reusable way to discover source markdown files
- We want deterministic file ordering for reproducible outputs
- This keeps source discovery separate from chunking and CLI code
"""

from pathlib import Path


def scan_markdown_source(source_dir: str | Path) -> list[Path]:
    """
    Return all markdown files under the source directory in sorted order.

    Parameters
    ----------
    source_dir:
        Root markdown source directory.

    Returns
    -------
    list[Path]
        Sorted list of markdown file paths.

    Why sorting matters:
    - Makes corpus ingest deterministic across runs and machines
    """
    root = Path(source_dir)

    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory but got: {root}")

    return sorted(root.rglob("*.md"))


def derive_relative_path(file_path: str | Path, source_dir: str | Path) -> str:
    """
    Return a normalized relative path for a markdown file within the source root.

    Why this matters:
    - Relative paths become part of doc identity and output metadata
    """
    path = Path(file_path).resolve()
    root = Path(source_dir).resolve()

    return str(path.relative_to(root)).replace("\\", "/")