from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from app.cli import app


runner = CliRunner()


def test_show_settings_command_runs_successfully(tmp_path: Path) -> None:
    """
    What this test checks:
    - The show-settings command exits successfully and prints key path info.

    Why this matters:
    - This is the first line of visibility into repo configuration.
    """
    result = runner.invoke(
        app,
        ["show-settings", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Resolved Paths" in result.output
    assert "project_root" in result.output
    assert "data_dir" in result.output
    assert "Settings loaded successfully." in result.output


def test_init_dirs_command_creates_expected_directories(tmp_path: Path) -> None:
    """
    What this test checks:
    - The init-dirs command creates the expected repo folders.

    Why this matters:
    - New environments should be bootstrappable from the CLI.
    """
    result = runner.invoke(
        app,
        ["init-dirs", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Initialization Summary" in result.output
    assert (tmp_path / "data").exists()
    assert (tmp_path / "data" / "test_source").exists()
    assert (tmp_path / "data" / "tracking").exists()
    assert (tmp_path / "data" / "processed").exists()
    assert (tmp_path / "data" / "logs").exists()


def test_track_source_command_creates_tracking_artifacts(tmp_path: Path) -> None:
    """
    What this test checks:
    - The track-source command runs end-to-end and writes tracking files.

    Why this matters:
    - This is the first actual pipeline command the user will rely on locally.
    """
    source_dir = tmp_path / "data" / "test_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    sample_file = source_dir / "sample_doc.md"
    sample_file.write_text(
        "# Sample Title\n\n## Section\n\n- point one\n- point two\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["track-source", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Tracking Result" in result.output
    assert "Detected Changes" in result.output
    assert "Source tracking completed successfully." in result.output

    assert (tmp_path / "data" / "tracking" / "source_manifest_current.json").exists()
    assert (tmp_path / "data" / "tracking" / "source_history.jsonl").exists()
    assert (tmp_path / "data" / "tracking" / "formatting_stats.jsonl").exists()
    assert (tmp_path / "data" / "tracking" / "change_events.jsonl").exists()
    assert (tmp_path / "data" / "tracking" / "source_snapshots").exists()


def test_track_source_command_handles_empty_source_directory(tmp_path: Path) -> None:
    """
    What this test checks:
    - The track-source command still runs when the source folder exists but is empty.

    Why this matters:
    - Empty corpora are common during initial setup or cleanup phases.
    """
    source_dir = tmp_path / "data" / "test_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        app,
        ["track-source", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Files scanned: 0" in result.output
    assert (tmp_path / "data" / "tracking" / "source_manifest_current.json").exists()

def test_chunk_preview_command_runs_successfully_for_relative_path(tmp_path: Path) -> None:
    """
    What this test checks:
    - chunk-preview works with a path relative to data/test_source.

    Why this matters:
    - This is the most convenient local workflow for corpus inspection.
    """
    source_dir = tmp_path / "data" / "test_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    sample_file = source_dir / "sample_doc.md"
    sample_file.write_text(
        "# Sample Title\n\n## Section\n\nThis is a test paragraph for chunk preview.\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "chunk-preview",
            "sample_doc.md",
            "--project-root",
            str(tmp_path),
            "--max-chunks",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "Chunk Preview" in result.output
    assert "Document" in result.output
    assert "Chunking Summary" in result.output
    assert "Atomic Chunk 1" in result.output
    assert "Chunk preview completed successfully." in result.output


def test_chunk_preview_command_fails_for_missing_file(tmp_path: Path) -> None:
    """
    What this test checks:
    - chunk-preview fails clearly when the requested markdown file does not exist.

    Why this matters:
    - Good error messages make local debugging much easier.
    """
    result = runner.invoke(
        app,
        [
            "chunk-preview",
            "missing_file.md",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "Markdown file not found" in result.output