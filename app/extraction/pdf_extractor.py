from __future__ import annotations

"""
pdf_extractor.py — Extract text from filtered PDFs using PyMuPDF.

Input:  corpus_manifest.jsonl (produced by corpus_filter.py)
Output: one JSON file per document in extracted_corpus/
        + extraction_manifest.jsonl (status, paths, stats)

Each extracted doc JSON:
{
  "doc_id":       str,        # file stem, used as primary key
  "file_name":    str,
  "source_name":  str,        # icmr | ncdc | who
  "doc_type":     str,
  "pdf_url":      str,        # original URL for citations
  "source_page":  str,
  "layout_class": str,
  "page_count":   int,
  "pages": [
    {"page_num": int, "text": str, "char_count": int, "word_count": int},
    ...
  ],
  "total_chars":  int,
  "total_words":  int,
  "extraction_status": "ok" | "empty" | "failed"
}
"""

import json
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def _make_doc_id(file_name: str) -> str:
    """
    Derive a stable doc_id from the file name.
    Strips the .pdf extension; used as primary key across all artifacts.
    """
    return Path(file_name).stem


def _clean_text(text: str) -> str:
    """
    Normalize extracted page text.
    - Collapse runs of 3+ blank lines to two
    - Strip trailing whitespace per line
    - Strip leading/trailing whitespace from the whole block
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        stripped = line.rstrip()
        if stripped == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def extract_single_pdf(
    file_path: Path,
    *,
    min_page_chars: int = 30,
) -> tuple[list[dict[str, Any]], str]:
    """
    Extract text from a single PDF file using PyMuPDF.

    Returns:
        (pages, status) where status is "ok" | "empty" | "failed"
        pages: list of {page_num, text, char_count, word_count}
    """
    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        logger.error("Failed to open %s: %s", file_path, exc)
        return [], "failed"

    pages: list[dict[str, Any]] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")
            text = _clean_text(raw_text)
            pages.append(
                {
                    "page_num": page_num + 1,
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
            )
    finally:
        doc.close()

    # Determine status
    total_chars = sum(p["char_count"] for p in pages)
    meaningful_pages = sum(
        1 for p in pages if p["char_count"] >= min_page_chars
    )

    if total_chars == 0 or meaningful_pages == 0:
        return pages, "empty"

    return pages, "ok"


def extract_corpus(
    corpus_manifest_path: Path,
    output_dir: Path,
    project_root: Path,
    *,
    skip_existing: bool = True,
    min_page_chars: int = 30,
) -> dict[str, Any]:
    """
    Extract text from all PDFs in corpus_manifest.jsonl.

    Args:
        corpus_manifest_path: path to corpus_manifest.jsonl
        output_dir: directory to write per-doc JSON files
        project_root: project root for resolving relative file_path values
        skip_existing: if True, skip PDFs whose output JSON already exists
        min_page_chars: pages below this char count are treated as empty

    Returns:
        Summary dict with counts and paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out_path = output_dir / "extraction_manifest.jsonl"

    # Load existing manifest entries to support skip_existing correctly
    existing_doc_ids: set[str] = set()
    if skip_existing and manifest_out_path.exists():
        with manifest_out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("extraction_status") == "ok":
                        existing_doc_ids.add(rec["doc_id"])

    # Load corpus manifest
    entries: list[dict[str, Any]] = []
    with corpus_manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    total = len(entries)
    stats = {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}

    manifest_fh = manifest_out_path.open("a", encoding="utf-8")

    try:
        for i, entry in enumerate(entries, start=1):
            doc_id = _make_doc_id(entry["file_name"])

            if skip_existing and doc_id in existing_doc_ids:
                stats["skipped"] += 1
                continue

            # Resolve absolute file path
            rel_path = entry.get("file_path", "")
            abs_path = (project_root / rel_path).resolve()

            if not abs_path.exists():
                logger.warning(
                    "[%d/%d] File not found: %s — skipping", i, total, abs_path
                )
                stats["failed"] += 1
                manifest_fh.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "file_name": entry["file_name"],
                            "extraction_status": "failed",
                            "reason": "file_not_found",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                manifest_fh.flush()
                continue

            pages, status = extract_single_pdf(
                abs_path, min_page_chars=min_page_chars
            )

            total_chars = sum(p["char_count"] for p in pages)
            total_words = sum(p["word_count"] for p in pages)

            doc_record: dict[str, Any] = {
                "doc_id": doc_id,
                "file_name": entry["file_name"],
                "source_name": entry["source_name"],
                "doc_type": entry["doc_type"],
                "pdf_url": entry["pdf_url"],
                "source_page": entry["source_page"],
                "layout_class": entry["layout_class"],
                "page_count": len(pages),
                "pages": pages,
                "total_chars": total_chars,
                "total_words": total_words,
                "extraction_status": status,
            }

            # Write per-doc JSON
            out_path = output_dir / f"{doc_id}.json"
            with out_path.open("w", encoding="utf-8") as out_fh:
                json.dump(doc_record, out_fh, ensure_ascii=False)

            # Write manifest entry (without pages to keep it small)
            manifest_entry = {k: v for k, v in doc_record.items() if k != "pages"}
            manifest_entry["output_path"] = str(out_path)
            manifest_fh.write(
                json.dumps(manifest_entry, ensure_ascii=False) + "\n"
            )
            manifest_fh.flush()

            stats[status] += 1

            if i % 100 == 0:
                logger.info(
                    "[%d/%d] ok=%d empty=%d failed=%d skipped=%d",
                    i,
                    total,
                    stats["ok"],
                    stats["empty"],
                    stats["failed"],
                    stats["skipped"],
                )
    finally:
        manifest_fh.close()

    return {
        "total_in_manifest": total,
        "ok": stats["ok"],
        "empty": stats["empty"],
        "failed": stats["failed"],
        "skipped": stats["skipped"],
        "output_dir": str(output_dir),
        "extraction_manifest": str(manifest_out_path),
    }
