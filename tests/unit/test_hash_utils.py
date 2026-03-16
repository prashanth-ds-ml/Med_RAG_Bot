from pathlib import Path

import pytest

from app.tracking.hash_utils import hash_file, hash_text, short_hash


def test_hash_text_is_deterministic() -> None:
    """
    What this test checks:
    - The same input text always produces the same hash.

    Why this matters:
    - Our source tracking depends on stable, repeatable hashes.
    """
    text = "# Title\n\nThis is a sample markdown file."
    hash_1 = hash_text(text)
    hash_2 = hash_text(text)

    assert hash_1 == hash_2


def test_hash_text_changes_when_content_changes() -> None:
    """
    What this test checks:
    - Different text content should produce different hashes.

    Why this matters:
    - We need reliable change detection for modified source files.
    """
    text_a = "# Title\n\nParagraph one."
    text_b = "# Title\n\nParagraph two."

    assert hash_text(text_a) != hash_text(text_b)


def test_hash_file_matches_hash_text(tmp_path: Path) -> None:
    """
    What this test checks:
    - Hashing file content from disk should match hashing the same content as text.

    Why this matters:
    - Confirms file-based tracking is consistent with content-based logic.
    """
    content = "## Section\n\nSome markdown content for hashing."
    sample_file = tmp_path / "sample.md"
    sample_file.write_text(content, encoding="utf-8")

    assert hash_file(sample_file) == hash_text(content)


def test_hash_file_differs_for_different_files(tmp_path: Path) -> None:
    """
    What this test checks:
    - Two different files should not produce the same hash.

    Why this matters:
    - Prevents false 'unchanged' detection in corpus tracking.
    """
    file_a = tmp_path / "a.md"
    file_b = tmp_path / "b.md"

    file_a.write_text("File A content", encoding="utf-8")
    file_b.write_text("File B content", encoding="utf-8")

    assert hash_file(file_a) != hash_file(file_b)


def test_hash_file_raises_for_missing_file(tmp_path: Path) -> None:
    """
    What this test checks:
    - A missing file should raise a clear error.

    Why this matters:
    - Helps us fail loudly and transparently during source scans.
    """
    missing_file = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError):
        hash_file(missing_file)


def test_hash_file_raises_for_directory(tmp_path: Path) -> None:
    """
    What this test checks:
    - Passing a directory path instead of a file should raise an error.

    Why this matters:
    - Guards against incorrect scan inputs and makes debugging easier.
    """
    with pytest.raises(IsADirectoryError):
        hash_file(tmp_path)


def test_short_hash_returns_requested_length() -> None:
    """
    What this test checks:
    - short_hash should trim the full hash to the requested size.

    Why this matters:
    - We will use short hashes in CLI tables and tracking logs.
    """
    full_hash = "abcdef1234567890"
    assert short_hash(full_hash, length=8) == "abcdef12"


def test_short_hash_raises_for_invalid_length() -> None:
    """
    What this test checks:
    - Non-positive short hash lengths should be rejected.

    Why this matters:
    - Prevents silent bad output in logs and reports.
    """
    with pytest.raises(ValueError):
        short_hash("abcdef", length=0)