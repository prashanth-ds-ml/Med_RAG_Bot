from __future__ import annotations

"""
Main Typer CLI for the Med360 RAG local workflow.
"""

from pathlib import Path

import typer
from app.generation.answer import build_baseline_answer
from app.retrieval.bm25_index import (
    build_bm25_index_from_atomic_chunks,
    search_bm25_index,
)
from app.retrieval.bm25_index import build_bm25_index_from_atomic_chunks
from app.ingestion.ingest_markdown import ingest_markdown_corpus
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

@app.command("ingest")
def ingest(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
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
    Ingest all markdown files in data/test_source and write chunk artifacts.

    Why this command matters:
    - This is the first bulk-processing stage for the corpus
    - Produces combined chunk artifacts for later retrieval and indexing
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    print_rule("Markdown Ingest Run")
    print_info(f"Source directory: {active_settings.test_source_dir}")

    config = ChunkingConfig(
        min_chunk_words=min_words,
        target_chunk_words=target_words,
        max_chunk_words=max_words,
        overlap_words=40,
        parent_target_words=parent_target_words,
        parent_max_words=parent_max_words,
    )

    result = ingest_markdown_corpus(
        source_dir=active_settings.test_source_dir,
        atomic_chunks_path=active_settings.atomic_chunks_path,
        parent_chunks_path=active_settings.parent_chunks_path,
        chunk_stats_path=active_settings.chunk_stats_path,
        config=config,
    )

    print_panel(
        (
            f"Documents processed: {result['document_count']}\n"
            f"Atomic chunks written: {result['total_atomic_chunks']}\n"
            f"Parent chunks written: {result['total_parent_chunks']}"
        ),
        title="Ingest Result",
        border_style="success",
    )

    print_path_summary(
        {
            "Atomic chunks": result["atomic_chunks_path"],
            "Parent chunks": result["parent_chunks_path"],
            "Chunk stats": result["chunk_stats_path"],
        },
        title="Written Artifacts",
    )

    print_success("Markdown ingest completed successfully.")

@app.command("build-bm25")
def build_bm25(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    ),
) -> None:
    """
    Build a BM25 index from atomic chunk artifacts.

    Why this command matters:
    - Provides the first real retrieval layer for the local pipeline
    - Uses transparent lexical matching that is easy to inspect and debug
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    print_rule("BM25 Index Build")
    print_info(f"Atomic chunks source: {active_settings.atomic_chunks_path}")

    result = build_bm25_index_from_atomic_chunks(
        atomic_chunks_path=active_settings.atomic_chunks_path,
        output_path=active_settings.bm25_index_path,
    )

    print_panel(
        (
            f"Chunk records indexed: {result['document_count']}\n"
            f"BM25 index saved to: {result['index_path']}"
        ),
        title="BM25 Build Result",
        border_style="success",
    )

    print_success("BM25 index built successfully.")

@app.command("search-bm25")
def search_bm25(
    query: str = typer.Argument(..., help="Query to search against the BM25 index."),
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        min=1,
        help="Number of top results to return.",
    ),
) -> None:
    """
    Search the local BM25 index and print top matching chunks.
    """
    active_settings = get_settings(project_root)

    if not active_settings.bm25_index_path.exists():
        raise typer.BadParameter(
            f"BM25 index not found: {active_settings.bm25_index_path}. Run 'build-bm25' first."
        )

    print_rule("BM25 Search")
    print_info(f"Query: {query}")

    results = search_bm25_index(
        active_settings.bm25_index_path,
        query,
        top_k=top_k,
    )

    if not results:
        print_warning("No BM25 results found.")
        return

    print_kv_summary(
        {
            "Top results returned": len(results),
            "Index path": active_settings.bm25_index_path,
        },
        title="Search Summary",
    )

    for result in results:
        record = result["record"]
        heading_path = " > ".join(record.get("heading_path", [])) or "(no heading path)"
        preview = _truncate_text(result["chunk_text"], max_chars=220)

        print_panel(
            (
                f"Rank: {result['rank']}\n"
                f"Score: {result['score']:.4f}\n"
                f"Chunk ID: {result['chunk_id']}\n"
                f"Heading path: {heading_path}\n\n"
                f"Preview:\n{preview}"
            ),
            title=f"BM25 Result {result['rank']}",
            border_style="success",
        )

    print_success("BM25 search completed successfully.")

@app.command("ask")
def ask(
    query: str = typer.Argument(..., help="Question to ask against the local corpus."),
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        min=1,
        help="Number of BM25 results to use for the answer.",
    ),
) -> None:
    """
    Ask a question against the local corpus using BM25 retrieval.

    Why this command matters:
    - Closes the loop from source files to a first grounded QA workflow
    - Helps debug retrieval quality before adding vector search or an LLM
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    if not active_settings.bm25_index_path.exists():
        raise typer.BadParameter(
            f"BM25 index not found: {active_settings.bm25_index_path}. Run 'build-bm25' first."
        )

    print_rule("Local QA Run")
    print_info(f"Question: {query}")

    results = search_bm25_index(
        active_settings.bm25_index_path,
        query,
        top_k=top_k,
    )

    answer = build_baseline_answer(query, results)

    print_panel(
        answer["answer_text"],
        title="Answer",
        border_style="success" if answer["grounded"] else "warning",
    )

    if results:
        print_kv_summary(
            {
                "Retrieved chunks used": len(results),
                "Grounded": answer["grounded"],
            },
            title="Answer Summary",
        )

        for index, item in enumerate(results, start=1):
            record = item["record"]
            heading_path = " > ".join(record.get("heading_path", [])) or "(no heading path)"
            preview = _truncate_text(item["chunk_text"], max_chars=220)

            print_panel(
                (
                    f"Rank: {item['rank']}\n"
                    f"Score: {item['score']:.4f}\n"
                    f"Chunk ID: {item['chunk_id']}\n"
                    f"Heading path: {heading_path}\n\n"
                    f"Preview:\n{preview}"
                ),
                title=f"Evidence Chunk {index}",
                border_style="info",
            )

    print_success("Local QA completed successfully.")

@app.callback()
def main() -> None:
    """
    Root CLI callback.
    """
    return


if __name__ == "__main__":
    app()