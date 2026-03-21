from __future__ import annotations

"""
Main Typer CLI for the Med360 RAG local workflow.
"""

import uuid
from datetime import datetime
from pathlib import Path

import typer
from rich.live import Live
from rich.panel import Panel as RichPanel
from app.engine import ChatEngine
from app.generation.llm_client import QwenClient
from app.generation.prompt_builder import build_messages
from app.generation.response_formatter import (
    format_response,
    render_citations_text,
    render_deduplicated_citations,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_index import (
    build_bm25_index_from_atomic_chunks,
    search_bm25_index,
)
from app.retrieval.bm25_index import build_bm25_index_from_atomic_chunks
from app.ingestion.ingest_markdown import ingest_markdown_corpus
from app.chunking.chunk_markdown import ChunkingConfig, chunk_markdown_file
from app.extraction.corpus_filter import build_corpus_manifest
from app.extraction.pdf_extractor import extract_corpus
from app.extraction.pdf_chunker import PdfChunkingConfig, chunk_extracted_corpus
from app.retrieval.vector_index import build_vector_index, search_vector_index
from app.retrieval.hybrid_retriever import search_hybrid
from app.console import (
    console,
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
from app.monitoring.db_client import ping_db
from app.monitoring.logger import log_session_start, log_session_end, log_turn, log_feedback


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
    use_corpus: bool = typer.Option(
        False,
        "--use-corpus/--use-markdown",
        help=(
            "Build from the main PDF corpus chunks (corpus_atomic_chunks.jsonl). "
            "Default builds from the markdown test source (atomic_chunks.jsonl)."
        ),
    ),
) -> None:
    """
    Build a BM25 index from atomic chunk artifacts.

    Use --use-corpus to build from the full PDF corpus (for production retrieval).
    Default builds from the markdown test source (for development/testing).
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    if use_corpus:
        chunks_path = active_settings.corpus_atomic_chunks_path
        index_path = active_settings.corpus_bm25_index_path
        label = "PDF Corpus"
    else:
        chunks_path = active_settings.atomic_chunks_path
        index_path = active_settings.bm25_index_path
        label = "Markdown Test Source"

    print_rule(f"BM25 Index Build — {label}")
    print_info(f"Chunks source: {chunks_path}")
    print_info(f"Index output:  {index_path}")

    if not chunks_path.exists():
        raise typer.BadParameter(
            f"Chunks file not found: {chunks_path}. "
            + ("Run 'chunk-corpus' first." if use_corpus else "Run 'ingest' first.")
        )

    result = build_bm25_index_from_atomic_chunks(
        atomic_chunks_path=chunks_path,
        output_path=index_path,
    )

    print_panel(
        (
            f"Chunks indexed: {result['document_count']}\n"
            f"BM25 index saved to: {result['index_path']}"
        ),
        title="BM25 Build Result",
        border_style="success",
    )

    print_success("BM25 index built successfully.")


@app.command("build-vector")
def build_vector(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    ),
    model_name: str = typer.Option(
        "BAAI/bge-base-en-v1.5",
        "--model",
        help="HuggingFace sentence-transformer model for embeddings.",
    ),
    batch_size: int = typer.Option(
        128,
        "--batch-size",
        min=1,
        help="Embedding batch size. Reduce if you run out of VRAM.",
    ),
) -> None:
    """
    Embed all corpus atomic chunks and build a FAISS vector index.

    Uses BAAI/bge-base-en-v1.5 by default (good medical domain performance,
    ~0.4 GB VRAM). Vectors are L2-normalized so IndexFlatIP gives cosine
    similarity scores in [0, 1].

    Saves:
      data/indexes/vector/faiss_index.bin
      data/indexes/vector/vector_payload.pkl

    Run after 'chunk-corpus'. Typically takes 10-20 min for ~42k chunks on GPU.
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    print_rule("FAISS Vector Index Build")

    if not active_settings.corpus_atomic_chunks_path.exists():
        raise typer.BadParameter(
            f"Corpus atomic chunks not found: {active_settings.corpus_atomic_chunks_path}. "
            "Run 'chunk-corpus' first."
        )

    print_info(f"Chunks:        {active_settings.corpus_atomic_chunks_path}")
    print_info(f"Model:         {model_name}")
    print_info(f"Batch size:    {batch_size}")
    print_info(f"FAISS index:   {active_settings.faiss_index_path}")
    print_info(f"Payload:       {active_settings.vector_payload_path}")

    result = build_vector_index(
        atomic_chunks_path=active_settings.corpus_atomic_chunks_path,
        index_path=active_settings.faiss_index_path,
        payload_path=active_settings.vector_payload_path,
        model_name=model_name,
        batch_size=batch_size,
    )

    print_kv_summary(
        {
            "Chunks indexed":  result["total_chunks"],
            "Embedding dim":   result["embedding_dim"],
            "Model":           result["model_name"],
            "Device":          result["device"],
        },
        title="Vector Index Summary",
    )
    print_path_summary(
        {
            "FAISS index":  result["index_path"],
            "Payload":      result["payload_path"],
        },
        title="Output",
    )
    print_success("Vector index built successfully.")

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

@app.command("filter-corpus")
def filter_corpus(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    ),
) -> None:
    """
    Filter the raw PDF corpus to a clean working set and write corpus_manifest.jsonl.

    Applies three filters:
      - Layout class: text_heavy only
      - Language: English (en) only
      - Doc type: excludes form_template, administrative_document, unknown
    Joins profiles + language tags + download manifest to attach pdf_url for citations.
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    print_rule("Corpus Filter")
    print_info(f"Profiles:       {active_settings.profiles_path}")
    print_info(f"Language tags:  {active_settings.language_tags_path}")
    print_info(f"Downloads:      {active_settings.downloaded_files_path}")

    for path in (
        active_settings.profiles_path,
        active_settings.language_tags_path,
        active_settings.downloaded_files_path,
    ):
        if not path.exists():
            raise typer.BadParameter(f"Required file not found: {path}")

    result = build_corpus_manifest(
        profiles_path=active_settings.profiles_path,
        language_tags_path=active_settings.language_tags_path,
        downloaded_files_path=active_settings.downloaded_files_path,
        output_path=active_settings.corpus_manifest_path,
    )

    print_kv_summary(
        {
            "Total profiles":       result["total_profiles"],
            "Excluded (layout)":    result["excluded_layout"],
            "Excluded (language)":  result["excluded_language"],
            "Excluded (doc type)":  result["excluded_doc_type"],
            "Excluded (no URL)":    result["excluded_no_url"],
            "Kept":                 result["kept"],
        },
        title="Filter Summary",
    )
    print_kv_summary(result["source_breakdown"], title="Source Breakdown")
    print_kv_summary(result["doc_type_breakdown"], title="Doc Type Breakdown")
    print_path_summary({"Manifest": result["output_path"]}, title="Output")
    print_success("Corpus manifest written successfully.")


@app.command("extract-corpus")
def extract_corpus_cmd(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip PDFs whose extracted JSON already exists.",
    ),
) -> None:
    """
    Extract text from filtered PDFs using PyMuPDF.

    Reads corpus_manifest.jsonl (run filter-corpus first).
    Writes one JSON file per document to data/corpus_pipeline/extracted/
    and an extraction_manifest.jsonl tracking status and paths.
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    print_rule("PDF Text Extraction")

    if not active_settings.corpus_manifest_path.exists():
        raise typer.BadParameter(
            f"corpus_manifest.jsonl not found: {active_settings.corpus_manifest_path}. "
            "Run 'filter-corpus' first."
        )

    print_info(f"Manifest:    {active_settings.corpus_manifest_path}")
    print_info(f"Output dir:  {active_settings.extracted_corpus_dir}")
    print_info(f"Skip existing: {skip_existing}")

    result = extract_corpus(
        corpus_manifest_path=active_settings.corpus_manifest_path,
        output_dir=active_settings.extracted_corpus_dir,
        project_root=active_settings.project_root,
        skip_existing=skip_existing,
    )

    print_kv_summary(
        {
            "Total in manifest": result["total_in_manifest"],
            "Extracted (ok)":    result["ok"],
            "Empty":             result["empty"],
            "Failed":            result["failed"],
            "Skipped":           result["skipped"],
        },
        title="Extraction Summary",
    )
    print_path_summary(
        {
            "Output dir":           result["output_dir"],
            "Extraction manifest":  result["extraction_manifest"],
        },
        title="Output",
    )
    print_success("PDF extraction completed.")


@app.command("chunk-corpus")
def chunk_corpus(
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Optional override for the project root directory.",
    ),
    target_words: int = typer.Option(
        220, "--target-words", min=20,
        help="Target word count for atomic chunks.",
    ),
    max_words: int = typer.Option(
        300, "--max-words", min=30,
        help="Maximum word count for atomic chunks.",
    ),
    min_words: int = typer.Option(
        60, "--min-words", min=1,
        help="Minimum word count before merging a small tail chunk.",
    ),
    parent_target_words: int = typer.Option(
        700, "--parent-target-words", min=50,
        help="Target word count for parent chunks.",
    ),
    parent_max_words: int = typer.Option(
        900, "--parent-max-words", min=80,
        help="Maximum word count for parent chunks.",
    ),
) -> None:
    """
    Chunk extracted PDF corpus into atomic and parent chunks.

    Reads extraction_manifest.jsonl (run extract-corpus first).
    Writes corpus_atomic_chunks.jsonl and corpus_parent_chunks.jsonl
    to data/processed/, plus corpus_chunk_stats.json.

    Each chunk carries pdf_url, source_name, doc_type, and page_num
    in its metadata for citation at retrieval and generation time.
    """
    active_settings = get_settings(project_root)
    active_settings.ensure_directories()

    print_rule("PDF Corpus Chunking")

    if not active_settings.extraction_manifest_path.exists():
        raise typer.BadParameter(
            f"extraction_manifest.jsonl not found: {active_settings.extraction_manifest_path}. "
            "Run 'extract-corpus' first."
        )

    print_info(f"Extraction manifest: {active_settings.extraction_manifest_path}")
    print_info(f"Atomic chunks out:   {active_settings.corpus_atomic_chunks_path}")
    print_info(f"Parent chunks out:   {active_settings.corpus_parent_chunks_path}")

    config = PdfChunkingConfig(
        target_chunk_words=target_words,
        max_chunk_words=max_words,
        min_chunk_words=min_words,
        parent_target_words=parent_target_words,
        parent_max_words=parent_max_words,
    )

    result = chunk_extracted_corpus(
        extraction_manifest_path=active_settings.extraction_manifest_path,
        extracted_dir=active_settings.extracted_corpus_dir,
        atomic_chunks_path=active_settings.corpus_atomic_chunks_path,
        parent_chunks_path=active_settings.corpus_parent_chunks_path,
        chunk_stats_path=active_settings.corpus_chunk_stats_path,
        config=config,
    )

    print_kv_summary(
        {
            "Docs processed":      result["total_docs_processed"],
            "Failed docs":         result["failed_docs"],
            "Atomic chunks":       result["total_atomic_chunks"],
            "Parent chunks":       result["total_parent_chunks"],
            "Avg atomic / doc":    result["avg_atomic_per_doc"],
            "Avg parent / doc":    result["avg_parent_per_doc"],
        },
        title="Chunking Summary",
    )
    print_path_summary(
        {
            "Atomic chunks":  result["atomic_chunks_path"],
            "Parent chunks":  result["parent_chunks_path"],
        },
        title="Output",
    )
    print_success("Corpus chunking completed.")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query."),
    project_root: str | None = typer.Option(
        None, "--project-root",
        help="Optional override for the project root directory.",
    ),
    top_k: int = typer.Option(
        5, "--top-k", min=1,
        help="Number of results to return after RRF fusion.",
    ),
    fetch_k: int = typer.Option(
        20, "--fetch-k", min=1,
        help="Candidates fetched from each index before fusion.",
    ),
) -> None:
    """
    Hybrid search: BM25 + FAISS fused with Reciprocal Rank Fusion (RRF).

    Shows per-result BM25 rank, vector rank, and fused score so retrieval
    quality is fully inspectable before wiring up generation.

    Run 'build-bm25 --use-corpus' and 'build-vector' first.
    """
    active_settings = get_settings(project_root)

    for path, name in [
        (active_settings.corpus_bm25_index_path, "BM25 index"),
        (active_settings.faiss_index_path, "FAISS index"),
        (active_settings.vector_payload_path, "Vector payload"),
    ]:
        if not path.exists():
            raise typer.BadParameter(
                f"{name} not found: {path}. "
                "Run 'build-bm25 --use-corpus' and 'build-vector' first."
            )

    print_rule("Hybrid Search")
    print_info(f"Query: {query}")
    print_info(f"top_k={top_k}  fetch_k={fetch_k}")

    results = search_hybrid(
        bm25_index_path=active_settings.corpus_bm25_index_path,
        faiss_index_path=active_settings.faiss_index_path,
        vector_payload_path=active_settings.vector_payload_path,
        query=query,
        top_k=top_k,
        fetch_k=fetch_k,
    )

    if not results:
        print_warning("No results found.")
        return

    print_kv_summary(
        {"Results returned": len(results), "Fetch k per index": fetch_k},
        title="Search Summary",
    )

    for result in results:
        record = result["record"]
        metadata = record.get("metadata", {})
        source_name = metadata.get("source_name", "")
        doc_type = metadata.get("doc_type", "")
        page_num = metadata.get("page_num", "?")
        pdf_url = metadata.get("pdf_url", "")

        bm25_rank = result.get("bm25_rank")
        vec_rank = result.get("vector_rank")
        bm25_tag = f"BM25 rank {bm25_rank}" if bm25_rank else "BM25 —"
        vec_tag = f"Vector rank {vec_rank}" if vec_rank else "Vector —"

        print_panel(
            (
                f"Fused score: {result['fused_score']:.5f}  "
                f"({bm25_tag}, {vec_tag})\n"
                f"Chunk ID:    {result['chunk_id']}\n"
                f"Source:      {source_name.upper()} | {doc_type} | Page {page_num}\n"
                f"URL:         {pdf_url}\n\n"
                f"Preview:\n{_truncate_text(result['chunk_text'], max_chars=300)}"
            ),
            title=f"Result {result['rank']}",
            border_style="success",
        )

    print_success("Hybrid search completed.")


@app.command("ask")
def ask(
    query: str = typer.Argument(..., help="Question to ask the medical corpus."),
    project_root: str | None = typer.Option(
        None, "--project-root",
        help="Optional override for the project root directory.",
    ),
    top_k: int = typer.Option(
        5, "--top-k", min=1,
        help="Number of chunks to retrieve and pass to the model.",
    ),
    fetch_k: int = typer.Option(
        20, "--fetch-k", min=1,
        help="Candidates fetched from each index before RRF fusion.",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-4B", "--model",
        help="HuggingFace model ID for generation.",
    ),
    thinking: bool = typer.Option(
        False, "--thinking/--no-thinking",
        help="Enable Qwen3 thinking mode (better reasoning, slower).",
    ),
    show_chunks: bool = typer.Option(
        False, "--show-chunks/--no-show-chunks",
        help="Print retrieved chunks alongside the answer.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose/--no-verbose",
        help="Print full run stats table (tokens, time, grounded).",
    ),
) -> None:
    """
    Ask a question — hybrid retrieval (BM25 + FAISS) + Qwen3 generation.

    Retrieves the most relevant chunks from the medical corpus, builds a
    grounded prompt, and generates a structured answer with citations.

    Run 'build-bm25 --use-corpus' and 'build-vector' first.
    """
    active_settings = get_settings(project_root)

    for path, name in [
        (active_settings.corpus_bm25_index_path, "BM25 index"),
        (active_settings.faiss_index_path, "FAISS index"),
        (active_settings.vector_payload_path, "Vector payload"),
    ]:
        if not path.exists():
            raise typer.BadParameter(
                f"{name} not found: {path}. "
                "Run 'build-bm25 --use-corpus' and 'build-vector' first."
            )

    print_rule("Med360 RAG")
    print_info(f"Query:   {query}")
    print_info(f"Model:   {model_name}  (thinking={'on' if thinking else 'off'})")

    # --- Retrieval ---
    print_info("Loading indexes and retrieving context...")
    retriever = HybridRetriever.load(
        bm25_index_path=active_settings.corpus_bm25_index_path,
        faiss_index_path=active_settings.faiss_index_path,
        vector_payload_path=active_settings.vector_payload_path,
        fetch_k=fetch_k,
    )
    results = retriever.search(query, top_k=top_k)

    if not results:
        print_warning("No relevant context found in the corpus.")
        return

    # --- Generation ---
    print_info(f"Generating answer from {len(results)} chunks...")
    messages = build_messages(query, results)

    client = QwenClient(model_name=model_name)
    client.load()
    gen = client.generate(messages, enable_thinking=thinking)

    # --- Format ---
    response = format_response(
        raw_answer=gen.answer_text,
        retrieved_chunks=results,
        query=query,
        generation_time_ms=gen.generation_time_ms,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        model_name=gen.model_name,
        thinking_text=gen.thinking_text,
    )

    # --- Display ---
    border = "success" if response["grounded"] else "warning"
    print_panel(response["answer_text"], title=f"Answer  [{response['query_type']}]", border_style=border)

    print_panel(
        render_deduplicated_citations(response["citations"]),
        title="Sources",
        border_style="info",
    )

    if verbose:
        print_kv_summary(
            {
                "Query type":         response["query_type"],
                "Grounded":           response["grounded"],
                "Chunks used":        len(results),
                "Prompt tokens":      response["prompt_tokens"],
                "Completion tokens":  response["completion_tokens"],
                "Generation time":    f"{response['generation_time_ms']} ms",
            },
            title="Run Stats",
        )
    else:
        gen_s = response["generation_time_ms"] / 1000
        grounded_mark = "✓" if response["grounded"] else "✗"
        print_info(
            f"{response['query_type']}  |  grounded {grounded_mark}  |  "
            f"{response['total_tokens']} tokens  |  {gen_s:.1f}s"
        )

    if show_chunks:
        for r in results:
            rec = r.get("record", {})
            meta = rec.get("metadata", {})
            print_panel(
                (
                    f"Fused: {r.get('fused_score', 0):.5f}  "
                    f"BM25 rank: {r.get('bm25_rank', '—')}  "
                    f"Vec rank: {r.get('vector_rank', '—')}\n"
                    f"Source: {meta.get('source_name','').upper()} | "
                    f"{meta.get('doc_type','')} | Page {meta.get('page_num','?')}\n\n"
                    f"{_truncate_text(rec.get('chunk_text', ''), max_chars=250)}"
                ),
                title=f"Chunk {r.get('rank')}",
                border_style="info",
            )

    if thinking and gen.thinking_text:
        print_panel(gen.thinking_text, title="Thinking (internal reasoning)", border_style="info")

    print_success("Done.")


@app.command("chat")
def chat(
    project_root: str | None = typer.Option(
        None, "--project-root",
        help="Optional override for the project root directory.",
    ),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Chunks per query."),
    fetch_k: int = typer.Option(20, "--fetch-k", min=1, help="Candidates per index before re-ranking."),
    model_name: str = typer.Option("Qwen/Qwen3-4B", "--model", help="Generation model."),
    thinking: bool = typer.Option(False, "--thinking/--no-thinking", help="Enable thinking mode."),
    thinking_budget: int = typer.Option(512, "--thinking-budget", min=0, help="Max tokens for <think> block (0 = unlimited). Only used when --thinking is on."),
    reranker: bool = typer.Option(False, "--reranker/--no-reranker", help="Enable cross-encoder re-ranking (runs on CPU, ~200-400ms/query). Requires model download on first use."),
    reranker_model: str = typer.Option("cross-encoder/ms-marco-MiniLM-L-6-v2", "--reranker-model", help="Cross-encoder model for re-ranking."),
) -> None:
    """
    Interactive multi-turn chat session with the medical corpus.

    Loads indexes and model once, then accepts queries in a loop.
    Type 'exit' or 'quit' to end the session.
    """
    active_settings = get_settings(project_root)

    for path, name in [
        (active_settings.corpus_bm25_index_path, "BM25 index"),
        (active_settings.faiss_index_path, "FAISS index"),
        (active_settings.vector_payload_path, "Vector payload"),
    ]:
        if not path.exists():
            raise typer.BadParameter(f"{name} not found: {path}.")

    # --- Session setup ---
    session_id = uuid.uuid4().hex
    db_online = ping_db()
    db_status = "connected" if db_online else "offline (logging disabled)"

    print_rule("Med RAG Bot — Chat Session")
    print_info(f"Model: {model_name}  |  top_k={top_k}  |  thinking={'on' if thinking else 'off'}  |  reranker={'on' if reranker else 'off'}")
    print_info(f"Session: {session_id[:12]}  |  db: {'connected' if ping_db() else 'MongoDB offline | JSONL logging active'}")
    print_info("Loading indexes and model (first load may take a moment)...")

    engine = ChatEngine(
        app_settings=active_settings,
        top_k=top_k,
        fetch_k=fetch_k,
        model_name=model_name,
        use_reranker=reranker,
        reranker_model=reranker_model,
    )
    engine.load()
    engine.start_session(session_id=session_id, thinking_on=thinking)

    _HELP_TEXT = (
        "  /think              — toggle thinking computation on/off\n"
        "  /show-think         — toggle showing the thinking block (when thinking=ON)\n"
        "  /budget <N>         — set thinking token budget (0 = unlimited, default 512)\n"
        "  /chunks             — toggle showing retrieved chunks\n"
        "  /verbose            — toggle full token/time stats\n"
        "  /feedback <1-5> [comment] — rate the last answer\n"
        "  /export             — save this session as a markdown file\n"
        "  /help               — show this message\n"
        "  exit | quit | /q    — end session"
    )
    print_success("Ready. Type your question or a /command for options.")
    print_panel(_HELP_TEXT, title="Commands", border_style="info")

    # mutable session state
    think_on    = thinking
    show_think  = thinking   # display thinking block (only relevant when think_on=True)
    think_budget: int | None = None if thinking_budget == 0 else thinking_budget
    show_chunks = False
    verbose     = False
    turn        = 0
    last_message_id: str | None = None
    history:  list[tuple[str, str]] = []   # (query, answer) for conversation memory
    turn_log: list[dict] = []              # full turn records for /export

    while True:
        if think_on and show_think:
            prompt_label = "You [thinking]"
        elif think_on and not show_think:
            prompt_label = "You [thinking/hidden]"
        else:
            prompt_label = "You"
        try:
            query = input(f"\n{prompt_label}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        # --- slash commands ---
        if query.lower() in {"exit", "quit", "/q", "q"}:
            break
        if query == "/think":
            think_on = not think_on
            if not think_on:
                show_think = False
            print_info(f"Thinking mode: {'ON' if think_on else 'OFF'}")
            continue
        if query == "/show-think":
            if not think_on:
                print_warning("Thinking mode is OFF — enable it first with /think")
                continue
            show_think = not show_think
            print_info(f"Show thinking block: {'ON' if show_think else 'OFF'}")
            continue
        if query.startswith("/budget"):
            parts = query.split(maxsplit=1)
            if len(parts) < 2:
                budget_display = "unlimited" if think_budget is None else str(think_budget)
                print_info(f"Current thinking budget: {budget_display} tokens. Usage: /budget <N>  (0 = unlimited)")
                continue
            try:
                n = int(parts[1])
                if n < 0:
                    raise ValueError
            except ValueError:
                print_warning("Usage: /budget <N>  where N ≥ 0  (0 = unlimited)")
                continue
            think_budget = None if n == 0 else n
            label = "unlimited" if think_budget is None else f"{think_budget} tokens"
            print_info(f"Thinking budget set to: {label}")
            continue
        if query == "/chunks":
            show_chunks = not show_chunks
            print_info(f"Show chunks: {'ON' if show_chunks else 'OFF'}")
            continue
        if query == "/verbose":
            verbose = not verbose
            print_info(f"Verbose stats: {'ON' if verbose else 'OFF'}")
            continue
        if query == "/help":
            print_panel(_HELP_TEXT, title="Commands", border_style="info")
            continue
        if query.startswith("/feedback"):
            parts = query.split(maxsplit=2)
            if len(parts) < 2:
                print_warning("Usage: /feedback <1-5> [optional comment]")
                continue
            try:
                rating = int(parts[1])
                if not 1 <= rating <= 5:
                    raise ValueError
            except ValueError:
                print_warning("Rating must be 1–5, e.g.  /feedback 4  or  /feedback 2 missing dosage")
                continue
            if last_message_id is None:
                print_warning("No answer to rate yet — ask a question first.")
                continue
            comment = parts[2] if len(parts) > 2 else ""
            saved = log_feedback(
                message_id=last_message_id,
                session_id=session_id,
                rating=rating,
                comment=comment,
            )
            if saved:
                print_success(f"Feedback saved  (rating: {rating}/5).")
            else:
                print_warning("MongoDB offline — feedback not saved.")
            continue

        if query == "/export":
            if not turn_log:
                print_warning("Nothing to export yet — ask a question first.")
                continue
            active_settings.ensure_directories()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = active_settings.exports_dir / f"chat_{session_id[:8]}_{ts}.md"
            lines = [f"# Med360 Chat Session\n\n**Session:** `{session_id[:12]}`  |  **Model:** {model_name}\n"]
            for t in turn_log:
                lines.append(f"---\n\n## Turn {t['turn']}  —  `{t['query_type']}`\n")
                lines.append(f"**You:** {t['query']}\n\n")
                lines.append(f"**Answer:**\n\n{t['answer']}\n\n")
                if t.get("sources"):
                    lines.append(f"**Sources:**\n\n{t['sources']}\n\n")
                if t.get("follow_ups"):
                    lines.append("**Follow-up questions:**\n")
                    for fq in t["follow_ups"]:
                        lines.append(f"- {fq}\n")
                    lines.append("\n")
                lines.append(f"*{t['time_s']:.1f}s  |  {t['tokens']} tokens  |  confidence: {t['confidence']}*\n")
            export_path.write_text("".join(lines), encoding="utf-8")
            print_success(f"Exported to: {export_path}")
            continue

        # --- RAG turn ---
        turn += 1
        think_label = "Thinking then generating..." if think_on else "Generating..."
        with console.status(f"[dim cyan]{think_label}[/dim cyan]", spinner="dots"):
            resp = engine.ask(
                query,
                session_id=session_id,
                turn=turn,
                history=history[-3:] if history else None,
                enable_thinking=think_on,
                thinking_budget=think_budget,
            )

        if not resp.retrieved_chunks:
            print_warning("No relevant context found for this query.")
            continue

        confidence      = resp.confidence
        last_message_id = resp.message_id

        border = "success" if resp.grounded else "warning"
        print_panel(resp.answer_text, title=f"[{turn}] {resp.query_type}", border_style=border)
        print_panel(resp.citations_text, title="Sources", border_style="info")

        if resp.follow_ups:
            follow_text = "\n".join(f"  • {q}" for q in resp.follow_ups)
            print_panel(follow_text, title="You might also ask", border_style="muted")

        # Update conversation memory
        history.append((query, resp.answer_text))
        if len(history) > 3:
            history.pop(0)

        # Save turn for /export
        turn_log.append({
            "turn":       turn,
            "query":      query,
            "query_type": resp.query_type,
            "answer":     resp.answer_text,
            "sources":    resp.citations_text,
            "follow_ups": resp.follow_ups,
            "tokens":     resp.total_tokens,
            "time_s":     resp.generation_time_ms / 1000,
            "confidence": confidence,
        })

        if verbose:
            print_kv_summary(
                {
                    "Query type":        resp.query_type,
                    "Confidence":        resp.confidence,
                    "Grounded":          resp.grounded,
                    "Chunks used":       len(resp.retrieved_chunks),
                    "Prompt tokens":     resp.prompt_tokens,
                    "Completion tokens": resp.completion_tokens,
                    "Generation time":   f"{resp.generation_time_ms} ms",
                    "Message ID":        resp.message_id[:12],
                },
                title="Stats",
            )
        else:
            gen_s         = resp.generation_time_ms / 1000
            grounded_mark = "✓" if resp.grounded else "✗"
            conf_color    = {"HIGH": "green", "MED": "yellow", "LOW": "red"}.get(confidence, "white")
            print_info(
                f"grounded {grounded_mark}  |  [{conf_color}]{confidence}[/{conf_color}]  |  "
                f"{resp.total_tokens} tokens  |  {gen_s:.1f}s  |  msg: {resp.message_id[:12]}"
            )

        if show_chunks:
            for r in resp.retrieved_chunks:
                rec  = r.get("record", {})
                meta = rec.get("metadata", {})
                rerank_line = f"  Rerank: {r['rerank_score']:.4f}" if "rerank_score" in r else ""
                print_panel(
                    (
                        f"Fused: {r.get('fused_score', 0):.5f}{rerank_line}  "
                        f"BM25 rank: {r.get('bm25_rank', '—')}  "
                        f"Vec rank: {r.get('vector_rank', '—')}\n"
                        f"Source: {meta.get('source_name','').upper()} | "
                        f"{meta.get('doc_type','')} | Page {meta.get('page_num','?')}\n\n"
                        f"{_truncate_text(rec.get('chunk_text', ''), max_chars=250)}"
                    ),
                    title=f"Chunk {r.get('rank')}",
                    border_style="info",
                )

        if think_on and show_think and resp.thinking_text:
            print_panel(resp.thinking_text, title="Thinking", border_style="info")

    engine.unload()
    engine.end_session(session_id=session_id, turn_count=turn)
    print_success(f"Session ended after {turn} turn{'s' if turn != 1 else ''}.")

@app.command("upload-logs")
def upload_logs(
    project_root: str | None = typer.Option(
        None, "--project-root",
        help="Optional override for the project root directory.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="Show what would be uploaded without writing to MongoDB.",
    ),
) -> None:
    """
    Upload local JSONL log files to MongoDB.

    Reads data/logs/{chat_sessions,messages,retrieval_logs,feedback}.jsonl
    and bulk-inserts records into the med360 MongoDB database.

    Safe to run multiple times — skips records already present (upsert by ID).
    Run 'upload-logs --dry-run' first to preview counts.
    """
    import orjson
    from app.monitoring.db_client import get_db

    active_settings = get_settings(project_root)

    print_rule("Upload Logs → MongoDB")

    db = get_db()
    if db is None and not dry_run:
        print_warning(
            "MongoDB is not reachable. Start MongoDB and check MONGODB_URI in .env, "
            "or use --dry-run to preview counts."
        )
        raise typer.Exit(1)

    log_files = [
        (active_settings.chat_sessions_log_path,  "chat_sessions",  "session_id"),
        (active_settings.messages_log_path,        "messages",       "message_id"),
        (active_settings.retrieval_logs_log_path,  "retrieval_logs", "message_id"),
        (active_settings.feedback_log_path,        "feedback",       "feedback_id"),
    ]

    total_uploaded = 0

    for log_path, collection_name, id_field in log_files:
        if not log_path.exists():
            print_info(f"{collection_name}: file not found, skipping.")
            continue

        records = []
        with open(log_path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = orjson.loads(line)
                    # Skip session_end update events — they're merged into chat_sessions
                    if doc.get("_event") == "session_end":
                        continue
                    records.append(doc)
                except Exception:
                    continue

        if not records:
            print_info(f"{collection_name}: 0 records found.")
            continue

        if dry_run:
            print_info(f"{collection_name}: {len(records)} records would be uploaded.")
            continue

        # Upsert by the natural ID field to avoid duplicates on re-runs
        inserted = 0
        skipped = 0
        collection = db[collection_name]
        for doc in records:
            key = doc.get(id_field)
            if key is None:
                continue
            result = collection.update_one(
                {id_field: key},
                {"$setOnInsert": doc},
                upsert=True,
            )
            if result.upserted_id is not None:
                inserted += 1
            else:
                skipped += 1

        print_info(f"{collection_name}: {inserted} inserted, {skipped} already existed.")
        total_uploaded += inserted

    if dry_run:
        print_info("Dry run complete — nothing was written to MongoDB.")
    else:
        print_success(f"Upload complete. {total_uploaded} new records written to MongoDB.")


@app.callback()
def main() -> None:
    """
    Root CLI callback.
    """
    return


if __name__ == "__main__":
    app()