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

def test_ingest_command_creates_processed_outputs(tmp_path: Path) -> None:
    """
    What this test checks:
    - The ingest command processes markdown files and writes combined outputs.

    Why this matters:
    - This is the first bulk corpus-processing command exposed in the CLI.
    """
    source_dir = tmp_path / "data" / "test_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    (source_dir / "doc_a.md").write_text(
        "# A\n\n## Symptoms\n\nPalpitations and fatigue are common.\n",
        encoding="utf-8",
    )
    (source_dir / "doc_b.md").write_text(
        "# B\n\n## Management\n\n- Diet\n- Exercise\n- Medication\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["ingest", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Ingest Result" in result.output
    assert "Written Artifacts" in result.output
    assert "Markdown ingest completed successfully." in result.output

    assert (tmp_path / "data" / "processed" / "atomic_chunks.jsonl").exists()
    assert (tmp_path / "data" / "processed" / "parent_chunks.jsonl").exists()
    assert (tmp_path / "data" / "processed" / "chunk_stats.json").exists()


def test_ingest_command_handles_empty_source_directory(tmp_path: Path) -> None:
    """
    What this test checks:
    - The ingest command still succeeds when the source directory is empty.

    Why this matters:
    - Empty corpus states should not crash the CLI.
    """
    source_dir = tmp_path / "data" / "test_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        app,
        ["ingest", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Documents processed: 0" in result.output
    assert (tmp_path / "data" / "processed" / "atomic_chunks.jsonl").exists()
    assert (tmp_path / "data" / "processed" / "parent_chunks.jsonl").exists()
    assert (tmp_path / "data" / "processed" / "chunk_stats.json").exists()


def test_search_bm25_command_returns_ranked_results(tmp_path: Path) -> None:
    """
    What this test checks:
    - search-bm25 reads the saved BM25 index and returns ranked chunk matches.

    Why this matters:
    - This is the first end-to-end retrieval command in the CLI.
    """
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    atomic_chunks_path = processed_dir / "atomic_chunks.jsonl"
    atomic_chunks_path.write_text(
        '{"chunk_id":"afib_1","chunk_text":"atrial fibrillation symptoms palpitations fatigue","heading_path":["Atrial Fibrillation","Symptoms"]}\n'
        '{"chunk_id":"stroke_1","chunk_text":"stroke emergency care assessment stabilization","heading_path":["Stroke","Emergency Care"]}\n',
        encoding="utf-8",
    )

    build_result = runner.invoke(
        app,
        ["build-bm25", "--project-root", str(tmp_path)],
    )
    assert build_result.exit_code == 0

    search_result = runner.invoke(
        app,
        ["search-bm25", "atrial fibrillation symptoms", "--project-root", str(tmp_path), "--top-k", "2"],
    )

    assert search_result.exit_code == 0
    assert "BM25 Search" in search_result.output
    assert "Search Summary" in search_result.output
    assert "BM25 Result 1" in search_result.output
    assert "afib_1" in search_result.output
    assert "BM25 search completed successfully." in search_result.output


def test_search_bm25_command_fails_when_index_missing(tmp_path: Path) -> None:
    """
    What this test checks:
    - search-bm25 fails clearly when the BM25 index does not exist yet.

    Why this matters:
    - Good failure messages make the local workflow easier to use.
    """
    result = runner.invoke(
        app,
        ["search-bm25", "atrial fibrillation", "--project-root", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "BM25 index not found" in result.output


def test_search_bm25_command_handles_empty_results(tmp_path: Path) -> None:
    """
    What this test checks:
    - search-bm25 handles an empty or irrelevant query result set cleanly.

    Why this matters:
    - Retrieval debugging should stay predictable even when no matches are found.
    """
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    atomic_chunks_path = processed_dir / "atomic_chunks.jsonl"
    atomic_chunks_path.write_text(
        '{"chunk_id":"c1","chunk_text":"atrial fibrillation symptoms","heading_path":["Atrial Fibrillation","Symptoms"]}\n',
        encoding="utf-8",
    )

    build_result = runner.invoke(
        app,
        ["build-bm25", "--project-root", str(tmp_path)],
    )
    assert build_result.exit_code == 0

    search_result = runner.invoke(
        app,
        ["search-bm25", "", "--project-root", str(tmp_path)],
    )

    assert search_result.exit_code == 0
    assert "No BM25 results found." in search_result.output