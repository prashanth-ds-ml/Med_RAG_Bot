from __future__ import annotations

"""
Manual smoke runner for source tracking.

What this script does:
- Runs the source tracker against your local markdown source folder
- Prints a compact summary so you can quickly inspect what changed

Why this exists:
- Unit tests verify logic.
- This manual runner verifies the real local workflow and output artifacts.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add the project root to sys.path so imports like `from app...` work
# even when this file is run directly with `python tests/manual/...`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tracking.source_tracker import track_source_directory


console = Console()


def main() -> None:
    source_dir = PROJECT_ROOT / "data" / "test_source"
    tracking_dir = PROJECT_ROOT / "data" / "tracking"
    snapshot_dir = tracking_dir / "source_snapshots"

    result = track_source_directory(
        source_dir=source_dir,
        manifest_current_path=tracking_dir / "source_manifest_current.json",
        source_history_path=tracking_dir / "source_history.jsonl",
        formatting_stats_path=tracking_dir / "formatting_stats.jsonl",
        change_events_path=tracking_dir / "change_events.jsonl",
        snapshot_dir=snapshot_dir,
    )

    summary = result["event_summary"]

    table = Table(title="Source Tracking Summary")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Snapshot ID", result["snapshot_id"])
    table.add_row("Files scanned", str(result["file_count"]))
    table.add_row("New files", str(summary["new_file"]))
    table.add_row("Modified", str(summary["modified"]))
    table.add_row("Unchanged", str(summary["unchanged"]))
    table.add_row("Deleted", str(summary["deleted"]))
    table.add_row("Snapshot path", result["snapshot_path"])

    console.print(Panel.fit("[bold blue]Source tracking completed[/bold blue]"))
    console.print(table)


if __name__ == "__main__":
    main()