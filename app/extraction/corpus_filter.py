from __future__ import annotations

"""
corpus_filter.py — Build the authoritative corpus manifest for PDF processing.

Joins three data sources:
  1. per_pdf_profiles.jsonl  — layout classification (text_heavy, scan_heavy, etc.)
  2. updated_keep_after_second_pass.jsonl — language + doc_type tags (post language filter)
  3. downloaded_files.jsonl  — original PDF URLs for citations

Filter criteria (all three must pass):
  - layout_class == "text_heavy"
  - dominant_language == "en"
  - doc_type NOT IN EXCLUDED_DOC_TYPES

Output: corpus_manifest.jsonl — one entry per document that passes all filters.
Each entry carries pdf_url so citations are available from chunk → response.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Doc types that pollute retrieval or have no medical RAG value
EXCLUDED_DOC_TYPES: frozenset[str] = frozenset(
    {"form_template", "administrative_document", "unknown"}
)

INCLUDED_LAYOUT_CLASSES: frozenset[str] = frozenset({"text_heavy"})

INCLUDED_LANGUAGES: frozenset[str] = frozenset({"en"})


@dataclass
class CorpusEntry:
    """
    A single document that has passed all corpus filters.
    Carries everything needed for extraction and citation.
    """

    file_name: str
    file_path: str          # relative path from project root
    source_name: str        # icmr | ncdc | who
    pdf_url: str            # original URL — used in citations
    source_page: str        # web page where PDF was discovered
    doc_type: str
    dominant_language: str
    language_confidence: float
    layout_class: str
    page_count: int
    total_chars: int
    total_words: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_url_lookup(downloaded_files_path: Path) -> dict[str, dict[str, str]]:
    """
    Build file_name → {pdf_url, source_page} from downloaded_files.jsonl.
    The local_path field contains the relative path; basename is the file_name.
    """
    lookup: dict[str, dict[str, str]] = {}
    records = _load_jsonl(downloaded_files_path)
    for rec in records:
        local_path = rec.get("local_path", "")
        fname = Path(local_path).name
        if fname:
            lookup[fname] = {
                "pdf_url": rec.get("pdf_url", ""),
                "source_page": rec.get("source_page", ""),
            }
    return lookup


def build_corpus_manifest(
    profiles_path: Path,
    language_tags_path: Path,
    downloaded_files_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Filter the corpus and write corpus_manifest.jsonl.

    Returns a summary dict with counts at each filter stage.
    """
    logger.info("Loading per-PDF profiles from %s", profiles_path)
    profiles = _load_jsonl(profiles_path)

    logger.info("Loading language/doc-type tags from %s", language_tags_path)
    lang_tags = _load_jsonl(language_tags_path)

    logger.info("Building URL lookup from %s", downloaded_files_path)
    url_lookup = _build_url_lookup(downloaded_files_path)

    # Index language tags by file_name for O(1) join
    lang_index: dict[str, dict[str, Any]] = {
        rec["file_name"]: rec for rec in lang_tags if "file_name" in rec
    }

    total = len(profiles)
    excluded_layout = 0
    excluded_language = 0
    excluded_doc_type = 0
    excluded_no_url = 0
    kept: list[CorpusEntry] = []

    for profile in profiles:
        fname = profile.get("file_name", "")

        # --- Layout filter ---
        layout = profile.get("layout_class", "")
        if layout not in INCLUDED_LAYOUT_CLASSES:
            excluded_layout += 1
            continue

        # --- Language + doc_type filter ---
        tag = lang_index.get(fname)
        if tag is None:
            # No language tag means it wasn't processed — skip
            excluded_language += 1
            continue

        lang = tag.get("dominant_language", "")
        if lang not in INCLUDED_LANGUAGES:
            excluded_language += 1
            continue

        doc_type = tag.get("doc_type", "unknown")
        if doc_type in EXCLUDED_DOC_TYPES:
            excluded_doc_type += 1
            continue

        # --- URL join ---
        url_info = url_lookup.get(fname)
        if not url_info or not url_info.get("pdf_url"):
            excluded_no_url += 1
            logger.warning("No PDF URL found for %s — skipping", fname)
            continue

        entry = CorpusEntry(
            file_name=fname,
            file_path=profile.get("file_path", ""),
            source_name=tag.get("source_name", profile.get("source_name", "")),
            pdf_url=url_info["pdf_url"],
            source_page=url_info["source_page"],
            doc_type=doc_type,
            dominant_language=lang,
            language_confidence=tag.get("language_confidence", 0.0),
            layout_class=layout,
            page_count=profile.get("page_count", 0),
            total_chars=profile.get("total_chars", 0),
            total_words=profile.get("total_words", 0),
        )
        kept.append(entry)

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for entry in kept:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    summary = {
        "total_profiles": total,
        "excluded_layout": excluded_layout,
        "excluded_language": excluded_language,
        "excluded_doc_type": excluded_doc_type,
        "excluded_no_url": excluded_no_url,
        "kept": len(kept),
        "output_path": str(output_path),
        "doc_type_breakdown": _count_field(kept, "doc_type"),
        "source_breakdown": _count_field(kept, "source_name"),
    }

    logger.info(
        "Corpus filter complete: %d / %d documents kept → %s",
        len(kept),
        total,
        output_path,
    )
    return summary


def _count_field(entries: list[CorpusEntry], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        val = getattr(entry, field, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
