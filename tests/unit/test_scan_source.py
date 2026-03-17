from pathlib import Path

import pytest

from app.ingestion.scan_source import derive_relative_path, scan_markdown_source


def test_scan_markdown_source_returns_sorted_markdown_files(tmp_path: Path) -> None:
    """
    What this test checks:
    - Markdown files are discovered recursively
    - Results are returned in deterministic sorted order

    Why this matters:
    - Reproducible ingest order makes debugging and testing easier.
    """
    source_dir = tmp_path / "test_source"
    nested_dir = source_dir / "cardiology"
    nested_dir.mkdir(parents=True)

    (source_dir / "b.md").write_text("# B", encoding="utf-8")
    (source_dir / "a.md").write_text("# A", encoding="utf-8")
    (nested_dir / "c.md").write_text("# C", encoding="utf-8")
    (nested_dir / "ignore.txt").write_text("not markdown", encoding="utf-8")

    files = scan_markdown_source(source_dir)

    assert [path.name for path in files] == ["a.md", "b.md", "c.md"]


def test_scan_markdown_source_raises_for_missing_directory(tmp_path: Path) -> None:
    """
    What this test checks:
    - Missing source directories fail loudly.

    Why this matters:
    - Ingest should not silently continue with bad paths.
    """
    missing_dir = tmp_path / "missing_source"

    with pytest.raises(FileNotFoundError):
        scan_markdown_source(missing_dir)


def test_scan_markdown_source_raises_for_non_directory(tmp_path: Path) -> None:
    """
    What this test checks:
    - Passing a file instead of a directory raises a clear error.

    Why this matters:
    - Guards against user mistakes and confusing ingest failures.
    """
    file_path = tmp_path / "file.md"
    file_path.write_text("# Not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        scan_markdown_source(file_path)


def test_derive_relative_path_returns_normalized_relative_path(tmp_path: Path) -> None:
    """
    What this test checks:
    - Relative paths are derived correctly from the source root.

    Why this matters:
    - Relative paths are used in metadata and doc IDs.
    """
    source_dir = tmp_path / "test_source"
    nested_dir = source_dir / "neurology"
    nested_dir.mkdir(parents=True)

    file_path = nested_dir / "stroke.md"
    file_path.write_text("# Stroke", encoding="utf-8")

    relative_path = derive_relative_path(file_path, source_dir)

    assert relative_path == "neurology/stroke.md"