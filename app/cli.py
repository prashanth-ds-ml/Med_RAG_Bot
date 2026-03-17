from __future__ import annotations

"""
Main Typer CLI for the Med360 RAG local workflow.
"""

from pathlib import Path

import typer

from app.chunking.chunk_markdown import ChunkingConfig, chunk_markdown_file
from app.console import (
    print_change_summary,
    print_info,
    print_kv_summary,
    print_panel,
    print_path_summary,
    print_rule,
    print_success,
    print_warning,
)
from app.settings import AppSettings, settings
from app.tracking.source_tracker import track_source_directory


app = typer.Typer(
    help="Local CLI for the Med360 RAG pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


def get_settings(project_root: str | None = None) -> AppSettings:
    """
    Return an AppSettings instance.
    """
    if project_root:
        return AppSettings(project_root=project_root)
    return settings


def _truncate_text(text: str, max_chars: int = 220) -> str:
    """
    Build a readable preview snippet for CLI display.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


@app.command("show-settings")
def show_settings(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    )
) -> None:
    """
    Show the key repository paths.
    """
    active_settings = get_settings(project_root)

    print_rule("Med360 RAG Settings")
    print_path_summary(active_settings.to_dict(), title="Resolved Paths")
    print_success("Settings loaded successfully.")


@app.command("init-dirs")
def init_dirs(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    )
) -> None:
    """
    Create the required repository directories if they do not already exist.
    """
    active_settings = get_settings(project_root)

    print_rule("Initializing Repository Directories")
    active_settings.ensure_directories()

    summary_rows = {
        "Project root": active_settings.project_root,
        "Directories ensured": len(active_settings.required_directories()),
        "Tracking dir": active_settings.tracking_dir,
        "Processed dir": active_settings.processed_dir,
        "Logs dir": active_settings.logs_dir,
    }

    print_kv_summary(summary_rows, title="Initialization Summary")
    print_success("Required directories are ready.")


@app.command("track-source")
def track_source(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    )
) -> None:
    """
    Track markdown source changes in the test_source folder.
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    print_rule("Source Tracking Run")
    print_info(f"Scanning source directory: {active_settings.test_source_dir}")

    result = track_source_directory(
        source_dir=active_settings.test_source_dir,
        manifest_current_path=active_settings.source_manifest_current_path,
        source_history_path=active_settings.source_history_path,
        formatting_stats_path=active_settings.formatting_stats_path,
        change_events_path=active_settings.change_events_path,
        snapshot_dir=active_settings.source_snapshots_dir,
    )

    print_panel(
        (
            f"Snapshot ID: {result['snapshot_id']}\n"
            f"Files scanned: {result['file_count']}\n"
            f"Snapshot saved to: {result['snapshot_path']}"
        ),
        title="Tracking Result",
        border_style="success",
    )

    print_change_summary(result["event_summary"], title="Detected Changes")
    print_success("Source tracking completed successfully.")


@app.command("chunk-preview")
def chunk_preview(
    file_path: str = typer.Argument(
        ...,
        help="Markdown file path, either absolute or relative to data/test_source.",
    ),
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    ),
    max_chunks: int = typer.Option(
        5,
        "--max-chunks",
        min=1,
        help="Maximum number of atomic chunks to preview.",
    ),
    target_words: int = typer.Option(
        220,
        "--target-words",
        min=20,
        help="Target word count for atomic chunks.",
    ),
    max_words: int = typer.Option(
        300,
        "--max-words",
        min=30,
        help="Maximum word count for atomic chunks.",
    ),
    min_words: int = typer.Option(
        80,
        "--min-words",
        min=1,
        help="Minimum word count before small-chunk merging.",
    ),
    parent_target_words: int = typer.Option(
        700,
        "--parent-target-words",
        min=50,
        help="Target word count for parent chunks.",
    ),
    parent_max_words: int = typer.Option(
        900,
        "--parent-max-words",
        min=80,
        help="Maximum word count for parent chunks.",
    ),
) -> None:
    """
    Preview how a markdown file will be chunked.

    Why this command matters:
    - Lets us inspect chunk boundaries before bulk ingestion
    - Helps compare formatting changes against chunking behavior
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    candidate_path = Path(file_path)
    if not candidate_path.is_absolute():
        candidate_path = active_settings.test_source_dir / candidate_path

    candidate_path = candidate_path.resolve()

    if not candidate_path.exists():
        raise typer.BadParameter(f"Markdown file not found: {candidate_path}")

    config = ChunkingConfig(
        min_chunk_words=min_words,
        target_chunk_words=target_words,
        max_chunk_words=max_words,
        overlap_words=40,
        parent_target_words=parent_target_words,
        parent_max_words=parent_max_words,
    )

    result = chunk_markdown_file(
        candidate_path,
        source_root=active_settings.test_source_dir,
        config=config,
    )

    print_rule("Chunk Preview")
    print_panel(
        (
            f"Source file: {result.source_file}\n"
            f"Relative path: {result.relative_path}\n"
            f"Doc ID: {result.doc_id}"
        ),
        title="Document",
        border_style="info",
    )

    print_kv_summary(
        {
            "Atomic chunks": len(result.atomic_chunks),
            "Parent chunks": len(result.parent_chunks),
            "Previewing first N atomic chunks": min(max_chunks, len(result.atomic_chunks)),
            "Target atomic words": target_words,
            "Max atomic words": max_words,
        },
        title="Chunking Summary",
    )

    if not result.atomic_chunks:
        print_warning("No atomic chunks were produced for this file.")
        return

    for index, chunk in enumerate(result.atomic_chunks[:max_chunks], start=1):
        heading_path = " > ".join(chunk.heading_path()) if chunk.heading_path() else "(no heading path)"
        word_count = chunk.stats.word_count if chunk.stats is not None else 0

        flags = []
        if chunk.is_list_heavy:
            flags.append("list-heavy")
        if chunk.has_table_ref:
            flags.append("table-ref")
        if chunk.has_image_ref:
            flags.append("image-ref")
        if chunk.has_code_block:
            flags.append("code-block")

        flags_text = ", ".join(flags) if flags else "none"

        print_panel(
            (
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Heading path: {heading_path}\n"
                f"Section title: {chunk.section_title or '(none)'}\n"
                f"Chunk type: {chunk.chunk_type}\n"
                f"Section kind: {chunk.section_kind}\n"
                f"Word count: {word_count}\n"
                f"Flags: {flags_text}\n\n"
                f"Preview:\n{_truncate_text(chunk.chunk_text)}"
            ),
            title=f"Atomic Chunk {index}",
            border_style="success",
        )

    print_success("Chunk preview completed successfully.")


@app.callback()
def main() -> None:
    """
    Root CLI callback.
    """
    return


if __name__ == "__main__":
    app()