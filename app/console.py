from __future__ import annotations

"""
Rich console helpers for the Med360 RAG repository.

Why this file exists:
- Gives all scripts one consistent terminal style
- Makes logs easier to read during local development
- Avoids repeating Rich setup and print patterns in every module

What this module does:
- Creates a shared Rich console with a small custom theme
- Provides helper functions for common output patterns:
  banners, status messages, key-value tables, and change summaries

Design choice:
- Keep rendering helpers small and reusable
- Separate "build" helpers from "print" helpers so tests can verify output cleanly
- Resolve semantic style names into real Rich style strings so the helpers work
  even with plain test consoles that do not load the custom theme
"""

from pathlib import Path
from typing import Iterable, Mapping

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme


STYLE_MAP = {
    "info": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "muted": "dim",
    "title": "bold blue",
    "path": "magenta",
    "label": "bold cyan",
    "value": "bold green",
}


APP_THEME = Theme(STYLE_MAP)


def resolve_style(style_name: str) -> str:
    """
    Resolve a semantic style name into a concrete Rich style string.

    Why this matters:
    - Our production console uses a custom theme
    - Test consoles may not use that theme
    - Resolving here makes the helpers work in both cases
    """
    return STYLE_MAP.get(style_name, style_name)


def build_console() -> Console:
    """
    Create the shared Rich console instance.

    Why this exists:
    - Keeps styling centralized
    - Makes future tweaks to terminal appearance easy
    """
    return Console(theme=APP_THEME)


console = build_console()


def build_message_panel(
    message: str,
    *,
    title: str | None = None,
    border_style: str = "info",
) -> Panel:
    """
    Build a styled panel for a message.

    Why this matters:
    - Panels make important output blocks stand out
    - Useful for stage starts, completion messages, and warnings
    """
    return Panel.fit(
        message,
        title=title,
        border_style=resolve_style(border_style),
    )


def build_kv_table(
    rows: Mapping[str, object] | Iterable[tuple[str, object]],
    *,
    title: str = "Summary",
    key_header: str = "Metric",
    value_header: str = "Value",
) -> Table:
    """
    Build a two-column key-value table.

    Parameters
    ----------
    rows:
        Either a mapping or an iterable of (key, value) pairs.
    title:
        Table title.
    key_header:
        Header for the left column.
    value_header:
        Header for the right column.
    """
    table = Table(title=title)
    table.add_column(key_header, style=resolve_style("label"))
    table.add_column(value_header, style=resolve_style("value"))

    if isinstance(rows, Mapping):
        items = rows.items()
    else:
        items = rows

    for key, value in items:
        table.add_row(str(key), str(value))

    return table


def print_rule(title: str, *, style: str = "title", target_console: Console | None = None) -> None:
    """
    Print a section rule line.
    """
    active_console = target_console or console
    active_console.rule(title, style=resolve_style(style))


def print_info(message: str, *, target_console: Console | None = None) -> None:
    """
    Print an informational message.
    """
    active_console = target_console or console
    active_console.print(message, style=resolve_style("info"))


def print_success(message: str, *, target_console: Console | None = None) -> None:
    """
    Print a success message.
    """
    active_console = target_console or console
    active_console.print(message, style=resolve_style("success"))


def print_warning(message: str, *, target_console: Console | None = None) -> None:
    """
    Print a warning message.
    """
    active_console = target_console or console
    active_console.print(message, style=resolve_style("warning"))


def print_error(message: str, *, target_console: Console | None = None) -> None:
    """
    Print an error message.
    """
    active_console = target_console or console
    active_console.print(message, style=resolve_style("error"))


def print_panel(
    message: str,
    *,
    title: str | None = None,
    border_style: str = "info",
    target_console: Console | None = None,
) -> None:
    """
    Print a message panel.
    """
    active_console = target_console or console
    active_console.print(
        build_message_panel(
            message=message,
            title=title,
            border_style=border_style,
        )
    )


def print_kv_summary(
    rows: Mapping[str, object] | Iterable[tuple[str, object]],
    *,
    title: str = "Summary",
    target_console: Console | None = None,
) -> None:
    """
    Print a key-value summary table.
    """
    active_console = target_console or console
    active_console.print(build_kv_table(rows, title=title))


def print_path_summary(
    rows: Mapping[str, str | Path] | Iterable[tuple[str, str | Path]],
    *,
    title: str = "Paths",
    target_console: Console | None = None,
) -> None:
    """
    Print a path-focused summary table.
    """
    active_console = target_console or console

    normalized_rows: list[tuple[str, object]] = []
    if isinstance(rows, Mapping):
        items = rows.items()
    else:
        items = rows

    for key, value in items:
        normalized_rows.append((key, Path(value) if not isinstance(value, Path) else value))

    table = Table(title=title)
    table.add_column("Name", style=resolve_style("label"))
    table.add_column("Path", style=resolve_style("path"))

    for key, value in normalized_rows:
        table.add_row(str(key), str(value))

    active_console.print(table)


def print_change_summary(
    summary: Mapping[str, int],
    *,
    title: str = "Change Summary",
    target_console: Console | None = None,
) -> None:
    """
    Print a standard summary for source tracking event counts.
    """
    active_console = target_console or console

    ordered_rows = [
        ("New files", summary.get("new_file", 0)),
        ("Modified", summary.get("modified", 0)),
        ("Unchanged", summary.get("unchanged", 0)),
        ("Deleted", summary.get("deleted", 0)),
    ]

    active_console.print(build_kv_table(ordered_rows, title=title))