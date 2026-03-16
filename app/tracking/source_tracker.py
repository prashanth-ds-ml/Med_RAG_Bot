from __future__ import annotations

"""
Source tracking for the markdown corpus.

Why this file exists:
- We want to track how the source corpus evolves over time.
- We need to know which files were added, modified, unchanged, or deleted.
- We also want to connect source formatting changes to later RAG quality changes.

What this module does:
- Scans a source directory for markdown files
- Builds a snapshot manifest with hashes and formatting stats
- Compares the current snapshot against the previous snapshot
- Writes tracking artifacts to JSON / JSONL files

Design choice:
- File identity is based on relative path within the source directory.
- Content changes are detected using content hashes, not modified timestamps.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.tracking.format_analyzer import analyze_markdown_format
from app.tracking.hash_utils import hash_file, short_hash


def utc_now_iso() -> str:
    """
    Return the current UTC time in ISO-8601 format.

    Why this matters:
    - Tracking files should have stable, timezone-aware timestamps.
    """
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: str | Path) -> None:
    """
    Create the parent directory for a file path if it does not already exist.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json_file(path: str | Path) -> dict[str, Any] | None:
    """
    Read a JSON file if it exists and is non-empty.

    Returns None when the file does not exist or is empty.

    Why this matters:
    - First tracking run may not have any previous manifest yet.
    """
    file_path = Path(path)

    if not file_path.exists() or file_path.stat().st_size == 0:
        return None

    with file_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json_file(path: str | Path, payload: dict[str, Any]) -> None:
    """
    Write a JSON payload to disk with pretty indentation.
    """
    ensure_parent_dir(path)

    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """
    Append one or more JSON records to a JSONL file.

    Why JSONL:
    - Easy to inspect manually
    - Easy to stream or append over time
    """
    if not records:
        return

    ensure_parent_dir(path)

    with Path(path).open("a", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_doc_id(relative_path: str) -> str:
    """
    Build a stable document ID from the relative path.

    Why this matters:
    - A readable doc_id is easier to follow in logs than raw filenames.
    - Using relative path helps avoid collisions across nested folders.
    """
    doc_id = relative_path.replace("\\", "/").lower()
    doc_id = re.sub(r"\.md$", "", doc_id)
    doc_id = re.sub(r"[^a-z0-9/_-]+", "_", doc_id)
    doc_id = doc_id.replace("/", "__")
    return doc_id.strip("_")


def scan_markdown_files(source_dir: str | Path) -> list[Path]:
    """
    Return all markdown files under the source directory in sorted order.

    Why sorting matters:
    - Makes outputs deterministic across runs and machines.
    """
    root = Path(source_dir)

    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory but got: {root}")

    return sorted(root.rglob("*.md"))


def build_file_record(file_path: str | Path, source_dir: str | Path) -> dict[str, Any]:
    """
    Build a tracking record for a single markdown file.

    Record includes:
    - identity
    - file size
    - content hash
    - formatting statistics
    """
    path = Path(file_path)
    root = Path(source_dir)

    relative_path = str(path.relative_to(root)).replace("\\", "/")
    text = path.read_text(encoding="utf-8")
    content_hash = hash_file(path)
    formatting_stats = analyze_markdown_format(text)

    return {
        "doc_id": make_doc_id(relative_path),
        "file_name": path.name,
        "relative_path": relative_path,
        "absolute_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "content_hash": content_hash,
        "content_hash_short": short_hash(content_hash),
        "char_count": len(text),
        "word_count": formatting_stats["word_count"],
        "formatting_stats": formatting_stats,
        "last_seen_at": utc_now_iso(),
    }


def build_snapshot_manifest(
    source_dir: str | Path,
    snapshot_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """
    Build a complete snapshot manifest for the current source directory.

    Why this matters:
    - This is the main artifact that describes the current state of the corpus.
    """
    root = Path(source_dir)
    files = scan_markdown_files(root)

    created_at = created_at or utc_now_iso()
    snapshot_id = snapshot_id or f"snapshot_{created_at.replace(':', '-')}"

    records = [build_file_record(file_path, root) for file_path in files]
    files_by_path = {record["relative_path"]: record for record in records}

    return {
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "source_dir": str(root.resolve()),
        "file_count": len(records),
        "files": files_by_path,
    }


def _build_formatting_delta(
    old_stats: dict[str, Any],
    new_stats: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Build a compact delta of formatting fields that changed.

    Why this matters:
    - Helps explain *what* changed, not just that a file changed.
    """
    delta: dict[str, dict[str, Any]] = {}

    all_keys = sorted(set(old_stats.keys()) | set(new_stats.keys()))
    for key in all_keys:
        old_value = old_stats.get(key)
        new_value = new_stats.get(key)

        if old_value != new_value:
            delta[key] = {"old": old_value, "new": new_value}

    return delta


def compare_manifests(
    previous_manifest: dict[str, Any] | None,
    current_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Compare the current manifest against the previous manifest.

    Returns a list of change-event records.

    Change types:
    - new_file
    - modified
    - unchanged
    - deleted
    """
    previous_files = (previous_manifest or {}).get("files", {})
    current_files = current_manifest.get("files", {})

    previous_paths = set(previous_files.keys())
    current_paths = set(current_files.keys())

    all_paths = sorted(previous_paths | current_paths)
    events: list[dict[str, Any]] = []

    snapshot_id = current_manifest["snapshot_id"]
    created_at = current_manifest["created_at"]

    for relative_path in all_paths:
        old_record = previous_files.get(relative_path)
        new_record = current_files.get(relative_path)

        if old_record is None and new_record is not None:
            events.append(
                {
                    "event_id": f"{snapshot_id}__{new_record['doc_id']}",
                    "snapshot_id": snapshot_id,
                    "event_time": created_at,
                    "doc_id": new_record["doc_id"],
                    "file_name": new_record["file_name"],
                    "relative_path": relative_path,
                    "change_type": "new_file",
                    "old_hash": None,
                    "new_hash": new_record["content_hash"],
                    "formatting_delta": {},
                }
            )
            continue

        if old_record is not None and new_record is None:
            events.append(
                {
                    "event_id": f"{snapshot_id}__{old_record['doc_id']}",
                    "snapshot_id": snapshot_id,
                    "event_time": created_at,
                    "doc_id": old_record["doc_id"],
                    "file_name": old_record["file_name"],
                    "relative_path": relative_path,
                    "change_type": "deleted",
                    "old_hash": old_record["content_hash"],
                    "new_hash": None,
                    "formatting_delta": {},
                }
            )
            continue

        assert old_record is not None and new_record is not None

        if old_record["content_hash"] == new_record["content_hash"]:
            events.append(
                {
                    "event_id": f"{snapshot_id}__{new_record['doc_id']}",
                    "snapshot_id": snapshot_id,
                    "event_time": created_at,
                    "doc_id": new_record["doc_id"],
                    "file_name": new_record["file_name"],
                    "relative_path": relative_path,
                    "change_type": "unchanged",
                    "old_hash": old_record["content_hash"],
                    "new_hash": new_record["content_hash"],
                    "formatting_delta": {},
                }
            )
        else:
            events.append(
                {
                    "event_id": f"{snapshot_id}__{new_record['doc_id']}",
                    "snapshot_id": snapshot_id,
                    "event_time": created_at,
                    "doc_id": new_record["doc_id"],
                    "file_name": new_record["file_name"],
                    "relative_path": relative_path,
                    "change_type": "modified",
                    "old_hash": old_record["content_hash"],
                    "new_hash": new_record["content_hash"],
                    "formatting_delta": _build_formatting_delta(
                        old_record.get("formatting_stats", {}),
                        new_record.get("formatting_stats", {}),
                    ),
                }
            )

    return events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, int]:
    """
    Build a small count summary of change events.

    Why this matters:
    - Useful for CLI output and tracking history.
    """
    summary = {
        "new_file": 0,
        "modified": 0,
        "unchanged": 0,
        "deleted": 0,
    }

    for event in events:
        summary[event["change_type"]] += 1

    return summary


def track_source_directory(
    source_dir: str | Path,
    manifest_current_path: str | Path,
    source_history_path: str | Path,
    formatting_stats_path: str | Path,
    change_events_path: str | Path,
    snapshot_dir: str | Path,
) -> dict[str, Any]:
    """
    Run the complete source tracking flow.

    Steps:
    1. Load previous manifest
    2. Build current manifest
    3. Compare old vs new
    4. Write current manifest
    5. Write snapshot copy
    6. Append summary/history rows
    7. Append per-file formatting rows
    8. Append change events

    Returns
    -------
    dict[str, Any]
        A compact summary of the tracking run.
    """
    previous_manifest = read_json_file(manifest_current_path)
    current_manifest = build_snapshot_manifest(source_dir=source_dir)
    events = compare_manifests(previous_manifest, current_manifest)
    event_summary = summarize_events(events)

    # Persist the current manifest as the canonical "latest" state.
    write_json_file(manifest_current_path, current_manifest)

    # Persist a point-in-time snapshot for later forensic inspection.
    snapshot_id = current_manifest["snapshot_id"]
    snapshot_path = Path(snapshot_dir) / f"{snapshot_id}.json"
    write_json_file(snapshot_path, current_manifest)

    # Append a high-level history row for this tracking run.
    history_record = {
        "snapshot_id": snapshot_id,
        "created_at": current_manifest["created_at"],
        "source_dir": current_manifest["source_dir"],
        "file_count": current_manifest["file_count"],
        "change_summary": event_summary,
    }
    append_jsonl(source_history_path, [history_record])

    # Append per-file formatting rows for downstream analysis.
    formatting_records = []
    for record in current_manifest["files"].values():
        formatting_records.append(
            {
                "snapshot_id": snapshot_id,
                "created_at": current_manifest["created_at"],
                "doc_id": record["doc_id"],
                "file_name": record["file_name"],
                "relative_path": record["relative_path"],
                **record["formatting_stats"],
            }
        )
    append_jsonl(formatting_stats_path, formatting_records)

    # Append per-file change events.
    append_jsonl(change_events_path, events)

    return {
        "snapshot_id": snapshot_id,
        "created_at": current_manifest["created_at"],
        "file_count": current_manifest["file_count"],
        "event_summary": event_summary,
        "snapshot_path": str(snapshot_path),
    }