from __future__ import annotations

from io import StringIO

from rich.console import Console

from app.console import (
    build_kv_table,
    build_message_panel,
    print_change_summary,
    print_error,
    print_info,
    print_kv_summary,
    print_panel,
    print_path_summary,
    print_rule,
    print_success,
    print_warning,
)


def make_test_console() -> tuple[Console, StringIO]:
    """
    Create a Rich console that writes into an in-memory string buffer.

    Why this matters:
    - Lets us test printed output without depending on a real terminal
    """
    buffer = StringIO()
    test_console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=120,
    )
    return test_console, buffer


def test_build_message_panel_keeps_message_and_title() -> None:
    """
    What this test checks:
    - Panel builder preserves message and title.

    Why this matters:
    - Panels are used for important status output in manual runs and CLI commands.
    """
    panel = build_message_panel("Tracking completed", title="Done", border_style="success")

    assert panel.title == "Done"
    assert "Tracking completed" in str(panel.renderable)


def test_build_kv_table_contains_expected_rows() -> None:
    """
    What this test checks:
    - Key-value table builder creates rows from input data.

    Why this matters:
    - Summary tables are one of the most common console render patterns in the repo.
    """
    table = build_kv_table({"Files": 3, "Modified": 1}, title="Tracking")

    assert table.title == "Tracking"
    assert len(table.columns) == 2
    assert len(table.rows) == 2


def test_print_helpers_emit_text() -> None:
    """
    What this test checks:
    - Basic print helpers write readable output.

    Why this matters:
    - Confirms our console helpers are actually sending content to the terminal layer.
    """
    test_console, buffer = make_test_console()

    print_rule("Stage Start", target_console=test_console)
    print_info("Scanning source files", target_console=test_console)
    print_success("Completed successfully", target_console=test_console)
    print_warning("One file was skipped", target_console=test_console)
    print_error("Example error message", target_console=test_console)

    output = buffer.getvalue()

    assert "Stage Start" in output
    assert "Scanning source files" in output
    assert "Completed successfully" in output
    assert "One file was skipped" in output
    assert "Example error message" in output


def test_print_panel_outputs_title_and_message() -> None:
    """
    What this test checks:
    - Panel printing includes both title and message content.

    Why this matters:
    - Many manual scripts use panels for important status blocks.
    """
    test_console, buffer = make_test_console()

    print_panel(
        "Source tracking completed",
        title="Tracker",
        border_style="success",
        target_console=test_console,
    )

    output = buffer.getvalue()

    assert "Tracker" in output
    assert "Source tracking completed" in output


def test_print_kv_summary_outputs_rows() -> None:
    """
    What this test checks:
    - Summary table printing includes the expected keys and values.

    Why this matters:
    - Used across stats, dashboard, and tracking views.
    """
    test_console, buffer = make_test_console()

    print_kv_summary(
        {"Files scanned": 5, "Modified": 2},
        title="Run Summary",
        target_console=test_console,
    )

    output = buffer.getvalue()

    assert "Run Summary" in output
    assert "Files scanned" in output
    assert "5" in output
    assert "Modified" in output
    assert "2" in output


def test_print_path_summary_outputs_names_and_paths() -> None:
    """
    What this test checks:
    - Path summary printing includes path labels and values.

    Why this matters:
    - Path visibility is important during local pipeline development.
    """
    test_console, buffer = make_test_console()

    print_path_summary(
        {
            "Source": "data/test_source",
            "Tracking": "data/tracking",
        },
        title="Key Paths",
        target_console=test_console,
    )

    output = buffer.getvalue()

    assert "Key Paths" in output
    assert "Source" in output
    assert "data/test_source" in output
    assert "Tracking" in output
    assert "data/tracking" in output


def test_print_change_summary_outputs_standard_change_fields() -> None:
    """
    What this test checks:
    - Change summary prints the standard source tracking counters.

    Why this matters:
    - This output will be reused in the source tracking CLI command.
    """
    test_console, buffer = make_test_console()

    print_change_summary(
        {
            "new_file": 2,
            "modified": 1,
            "unchanged": 4,
            "deleted": 0,
        },
        target_console=test_console,
    )

    output = buffer.getvalue()

    assert "Change Summary" in output
    assert "New files" in output
    assert "2" in output
    assert "Modified" in output
    assert "1" in output
    assert "Unchanged" in output
    assert "4" in output
    assert "Deleted" in output
    assert "0" in output