from __future__ import annotations

"""
Main Typer CLI for the Med360 RAG local workflow.

Why this file exists:
- Gives us one consistent command-line entrypoint for local development
- Lets us test pipeline stages before building any web UI
- Makes the repo easier to reproduce and operate across machines

What this module does:
- Exposes a Typer app
- Provides basic commands for:
  - showing settings
  - creating required directories
  - tracking source-folder changes

Design choice:
- Start small and keep commands reliable
- Add ingest / chunk / ask / eval commands only after the foundations are stable
"""

import typer

from app.console import (
    print_change_summary,
    print_info,
    print_kv_summary,
    print_panel,
    print_path_summary,
    print_rule,
    print_success,
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

    Why this helper exists:
    - Lets tests override the project root cleanly
    - Keeps command functions small and readable
    """
    if project_root:
        return AppSettings(project_root=project_root)
    return settings


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

    Why this command matters:
    - Helps confirm the CLI is pointing at the expected repo folders
    - Useful during debugging and test setup
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

    Why this command matters:
    - Gives a deterministic bootstrap step for new environments
    - Avoids repeated manual mkdir calls
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

    What this command does:
    - Scans markdown files
    - Computes hashes and formatting stats
    - Compares against the previous manifest
    - Writes tracking artifacts

    Why this command matters:
    - Gives visibility into corpus growth and formatting evolution
    - Helps connect source edits to later chunking / retrieval quality
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


@app.callback()
def main() -> None:
    """
    Root CLI callback.

    Why this exists:
    - Keeps a place for future global CLI setup
    - Makes the Typer app easier to extend later
    """
    return


if __name__ == "__main__":
    app()