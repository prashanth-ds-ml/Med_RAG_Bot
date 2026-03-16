from __future__ import annotations

"""
Hash utilities for source-file tracking.

Why this file exists:
- We need a stable way to detect whether a markdown file changed.
- Content hashes help us identify new / modified / unchanged files.
- Stable hashes are the backbone of snapshotting and corpus history tracking.

What this module does:
- Hash raw text
- Hash files from disk
- Return short hashes for readable logs / IDs

Important design choice:
- We hash file CONTENT, not file modified time.
  This is much more reliable for reproducible tracking.
"""

from hashlib import new as hashlib_new
from pathlib import Path


DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_CHUNK_SIZE = 64 * 1024  # 64 KB


def hash_text(text: str, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """
    Return a hex digest for a text string.

    Why this is useful:
    - Lets us test hashing logic without touching the filesystem.
    - Helps compare expected content vs saved file content.

    Parameters
    ----------
    text : str
        The input text to hash.
    algorithm : str
        Hash algorithm name supported by hashlib, e.g. 'sha256', 'md5'.

    Returns
    -------
    str
        Hex digest string.
    """
    hasher = hashlib_new(algorithm)
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def hash_file(
    file_path: str | Path,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    Return a hex digest for a file on disk.

    Why chunked reading:
    - Keeps memory usage low for large files.
    - Makes this safe even if later we hash larger text artifacts.

    Parameters
    ----------
    file_path : str | Path
        Path to the file to hash.
    algorithm : str
        Hash algorithm name supported by hashlib.
    chunk_size : int
        Number of bytes to read at a time.

    Returns
    -------
    str
        Hex digest string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    IsADirectoryError
        If the provided path points to a directory instead of a file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.is_dir():
        raise IsADirectoryError(f"Expected a file but got a directory: {path}")

    hasher = hashlib_new(algorithm)

    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def short_hash(hash_value: str, length: int = 12) -> str:
    """
    Return a shortened version of a full hash string.

    Why this exists:
    - Full SHA256 hashes are great for tracking but noisy in CLI output.
    - Short hashes are easier to read in logs, tables, and diff summaries.

    Parameters
    ----------
    hash_value : str
        Full hash string.
    length : int
        Number of characters to keep from the start.

    Returns
    -------
    str
        Shortened hash string.

    Raises
    ------
    ValueError
        If length is not positive.
    """
    if length <= 0:
        raise ValueError("length must be greater than 0")

    return hash_value[:length]