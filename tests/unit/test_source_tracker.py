from __future__ import annotations

import json
from pathlib import Path

from app.tracking.source_tracker import (
    build_snapshot_manifest,
    compare_manifests,
    summarize_events,
    track_source_directory,
)


def test_build_snapshot_manifest_collects_markdown_files(tmp_path: Path) -> None:
    """
    What this test checks:
    - Markdown files are discovered and included in the manifest.

    Why this matters:
    - If source scanning is wrong, the whole tracking system becomes unreliable.
    """
    source_dir = tmp_path / "test_source"
    source_dir.mkdir()

    (source_dir / "doc_a.md").write_text("# A\n\nSome content", encoding="utf-8")
    (source_dir / "doc_b.md").write_text("# B\n\nMore content", encoding="utf-8")

    manifest = build_snapshot_manifest(source_dir)

    assert manifest["file_count"] == 2
    assert "doc_a.md" in manifest["files"]
    assert "doc_b.md" in manifest["files"]


def test_compare_manifests_marks_new_files_on_first_run(tmp_path: Path) -> None:
    """
    What this test checks:
    - On the first run, files should be marked as new_file.

    Why this matters:
    - Initial corpus ingestion should clearly distinguish newly seen files.
    """
    source_dir = tmp_path / "test_source"
    source_dir.mkdir()

    (source_dir / "doc_a.md").write_text("# A\n\nSome content", encoding="utf-8")
    (source_dir / "doc_b.md").write_text("# B\n\nMore content", encoding="utf-8")

    current_manifest = build_snapshot_manifest(source_dir)
    events = compare_manifests(previous_manifest=None, current_manifest=current_manifest)
    summary = summarize_events(events)

    assert summary["new_file"] == 2
    assert summary["modified"] == 0
    assert summary["unchanged"] == 0
    assert summary["deleted"] == 0


def test_compare_manifests_detects_modified_and_deleted_files(tmp_path: Path) -> None:
    """
    What this test checks:
    - Manifest comparison detects modified and deleted files correctly.

    Why this matters:
    - We want trustworthy change tracking as the corpus evolves.
    """
    source_dir = tmp_path / "test_source"
    source_dir.mkdir()

    doc_a = source_dir / "doc_a.md"
    doc_b = source_dir / "doc_b.md"

    doc_a.write_text("# A\n\nOriginal content", encoding="utf-8")
    doc_b.write_text("# B\n\nOriginal content", encoding="utf-8")

    previous_manifest = build_snapshot_manifest(source_dir)

    doc_a.write_text("# A\n\nUpdated content with more structure\n\n## New Section", encoding="utf-8")
    doc_b.unlink()
    (source_dir / "doc_c.md").write_text("# C\n\nBrand new file", encoding="utf-8")

    current_manifest = build_snapshot_manifest(source_dir)
    events = compare_manifests(previous_manifest=previous_manifest, current_manifest=current_manifest)
    summary = summarize_events(events)

    assert summary["modified"] == 1
    assert summary["deleted"] == 1
    assert summary["new_file"] == 1
    assert summary["unchanged"] == 0

    modified_events = [event for event in events if event["change_type"] == "modified"]
    assert len(modified_events) == 1
    assert modified_events[0]["relative_path"] == "doc_a.md"
    assert modified_events[0]["formatting_delta"] != {}


def test_track_source_directory_writes_expected_artifacts(tmp_path: Path) -> None:
    """
    What this test checks:
    - A full tracking run writes manifest, snapshot, history, formatting logs,
      and change-event logs.

    Why this matters:
    - The tracking system should leave behind transparent, inspectable artifacts.
    """
    source_dir = tmp_path / "test_source"
    tracking_dir = tmp_path / "tracking"
    snapshot_dir = tracking_dir / "source_snapshots"

    source_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)

    (source_dir / "doc_a.md").write_text("# A\n\nSome content", encoding="utf-8")

    manifest_current_path = tracking_dir / "source_manifest_current.json"
    source_history_path = tracking_dir / "source_history.jsonl"
    formatting_stats_path = tracking_dir / "formatting_stats.jsonl"
    change_events_path = tracking_dir / "change_events.jsonl"

    result = track_source_directory(
        source_dir=source_dir,
        manifest_current_path=manifest_current_path,
        source_history_path=source_history_path,
        formatting_stats_path=formatting_stats_path,
        change_events_path=change_events_path,
        snapshot_dir=snapshot_dir,
    )

    assert result["file_count"] == 1
    assert manifest_current_path.exists()
    assert source_history_path.exists()
    assert formatting_stats_path.exists()
    assert change_events_path.exists()

    snapshot_path = Path(result["snapshot_path"])
    assert snapshot_path.exists()

    manifest_payload = json.loads(manifest_current_path.read_text(encoding="utf-8"))
    assert manifest_payload["file_count"] == 1

    history_lines = source_history_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 1

    formatting_lines = formatting_stats_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(formatting_lines) == 1

    event_lines = change_events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(event_lines) == 1