from __future__ import annotations

"""
response_formatter.py — Parse and structure LLM output for CLI display and storage.

Responsibilities:
  1. Extract the query_type from the model's first line ("Query type: GUIDELINE")
  2. Separate the answer body from the Sources block
  3. Parse the Sources block into structured citation dicts
  4. Build the final response dict that goes to CLI display + MongoDB logging

Why a separate formatter:
  - The LLM output is text; everything downstream (CLI, MongoDB, FastAPI) needs
    structured data with citations as first-class fields
  - Keeps generation code clean — llm_client just generates text
  - Easy to test independently from the LLM
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Query type detection
# ---------------------------------------------------------------------------

KNOWN_QUERY_TYPES = {
    "FACTUAL", "GUIDELINE", "PROCEDURE",
    "COMPARISON", "STATISTICS", "MULTI_ASPECT", "INSUFFICIENT",
}

_QUERY_TYPE_RE = re.compile(
    r"^Query\s+type\s*[:：]\s*(\w+)", re.IGNORECASE | re.MULTILINE
)


def extract_query_type(text: str) -> str:
    """Extract the query type declared by the model on its first line."""
    match = _QUERY_TYPE_RE.search(text)
    if match:
        candidate = match.group(1).upper()
        if candidate in KNOWN_QUERY_TYPES:
            return candidate
    return "FACTUAL"  # safe default


# ---------------------------------------------------------------------------
# Sources block parser
# ---------------------------------------------------------------------------

_SOURCES_SPLIT_RE = re.compile(
    r"^(?:Sources?(?:\s+reviewed)?|Citations?)\s*[:：]",
    re.IGNORECASE | re.MULTILINE,
)

# Normalize [N1] / [N4] → [1] / [4] (model sometimes uses N-prefixed indices)
_N_INDEX_RE = re.compile(r"\[N(\d+)\]")

_SOURCE_ENTRY_RE = re.compile(
    r"\[(\d+)\]\s+(.+?)(?=\n\[|\Z)",
    re.DOTALL,
)

_URL_RE = re.compile(r"https?://\S+")


def _parse_sources_block(sources_text: str) -> list[dict[str, Any]]:
    """
    Parse the Sources block into a list of citation dicts.

    Expected format per entry:
        [N] Document description — SOURCE | doc type, Page X
            https://...
    """
    citations: list[dict[str, Any]] = []
    for match in _SOURCE_ENTRY_RE.finditer(sources_text):
        num = int(match.group(1))
        body = match.group(2).strip()

        # Extract URL if present
        url_match = _URL_RE.search(body)
        url = url_match.group(0) if url_match else ""
        description = _URL_RE.sub("", body).strip().rstrip("\n").strip()

        citations.append(
            {
                "index": num,
                "description": description,
                "url": url,
            }
        )

    return citations


def split_answer_and_sources(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Split model output into (answer_body, citations_list).

    Splits at the FIRST 'Sources:' or 'Sources reviewed:' line, handling both:
      - Inline:  Sources: [1][2][3]
      - Block:   Sources:\n[1] description\n    url
    """
    match = _SOURCES_SPLIT_RE.search(text)
    if match:
        answer_body = text[: match.start()].strip()
        sources_text = text[match.end():]
        citations = _parse_sources_block(sources_text)
    else:
        answer_body = text.strip()
        citations = []

    return answer_body, citations


# ---------------------------------------------------------------------------
# Query type line and template artifact cleanup
# ---------------------------------------------------------------------------

def _strip_query_type_line(text: str) -> str:
    """Remove the 'Query type: X' line from the answer body."""
    return _QUERY_TYPE_RE.sub("", text).strip()


# Matches literal template placeholder lines the model sometimes echoes,
# e.g. "[Answer in 1-3 clear paragraphs with inline citations]"
_TEMPLATE_ARTIFACT_RE = re.compile(
    r"^\[(?:Answer|Write|Provide|List|Describe)[^\]]{0,80}\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Matches bold header artifacts the model sometimes emits, e.g. "**Answer:**"
# These are template echo artifacts that should not appear in the final answer.
_BOLD_HEADER_ARTIFACT_RE = re.compile(
    r"^\*\*(?:Answer|Response)\s*:\*\*\s*\n?",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_template_artifacts(text: str) -> str:
    """Remove literal template instruction brackets and bold headers echoed by the model."""
    text = _TEMPLATE_ARTIFACT_RE.sub("", text)
    text = _BOLD_HEADER_ARTIFACT_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Follow-up question extraction
# ---------------------------------------------------------------------------

_FOLLOWUP_BLOCK_RE = re.compile(
    r"\*\*Follow-?up questions?:?\*\*\s*\n((?:\s*[-•*]\s*.+\n?)+)",
    re.IGNORECASE,
)

_FOLLOWUP_ITEM_RE = re.compile(r"^\s*[-•*]\s*(.+)$", re.MULTILINE)


def extract_followups(text: str) -> tuple[str, list[str]]:
    """
    Extract follow-up questions from the answer body.

    Returns (cleaned_text, [question1, question2, question3])
    where cleaned_text has the follow-up block removed.
    """
    match = _FOLLOWUP_BLOCK_RE.search(text)
    if not match:
        return text.strip(), []

    items = _FOLLOWUP_ITEM_RE.findall(match.group(1))
    cleaned = text[: match.start()].strip()
    return cleaned, [q.strip() for q in items if q.strip()]


# ---------------------------------------------------------------------------
# Main formatter
# ---------------------------------------------------------------------------

def format_response(
    raw_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    *,
    query: str = "",
    generation_time_ms: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model_name: str = "",
    thinking_text: str = "",
) -> dict[str, Any]:
    """
    Convert raw LLM output + retrieval results into a structured response dict.

    This is the object that gets:
      - Displayed in the CLI
      - Stored in MongoDB messages collection
      - Returned by the FastAPI endpoint (Phase 8)

    Returns:
        {
          query_type, answer_text, citations,
          retrieved_chunks (compact),
          prompt_tokens, completion_tokens, generation_time_ms,
          model_name, thinking_text, grounded
        }
    """
    query_type = extract_query_type(raw_answer)
    # Normalize [N1] → [1] before any parsing
    raw_answer = _N_INDEX_RE.sub(r"[\1]", raw_answer)
    # Strip follow-ups from the full output BEFORE splitting at Sources —
    # the model appends follow-ups after the Sources block, so they end up
    # in sources_text (not answer_body) and would otherwise be mis-parsed
    # as a citation entry (e.g. "[5] **Follow-up questions:**…").
    raw_answer, follow_ups = extract_followups(raw_answer)
    answer_body, citations = split_answer_and_sources(raw_answer)
    answer_body = _strip_query_type_line(answer_body)
    answer_body = _strip_template_artifacts(answer_body)

    # If no citations were parsed from LLM output, build them from retrieved chunks
    # (fallback — ensures citations are always present even if model skipped Sources)
    if not citations:
        citations = _build_fallback_citations(retrieved_chunks)

    # Compact chunk list for storage (no full text — text is in the answer)
    compact_chunks = [
        {
            "chunk_id": r.get("chunk_id", r.get("record", {}).get("chunk_id", "")),
            "fused_score": r.get("fused_score", r.get("score", 0.0)),
            "bm25_rank": r.get("bm25_rank"),
            "vector_rank": r.get("vector_rank"),
            "source_name": r.get("record", {}).get("metadata", {}).get("source_name", ""),
            "doc_type": r.get("record", {}).get("metadata", {}).get("doc_type", ""),
            "page_num": r.get("record", {}).get("metadata", {}).get("page_num"),
            "pdf_url": r.get("record", {}).get("metadata", {}).get("pdf_url", ""),
        }
        for r in retrieved_chunks
    ]

    grounded = bool(citations) and query_type != "INSUFFICIENT"

    return {
        "query": query,
        "query_type": query_type,
        "answer_text": answer_body,
        "follow_ups": follow_ups,
        "citations": citations,
        "retrieved_chunks": compact_chunks,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "generation_time_ms": generation_time_ms,
        "model_name": model_name,
        "thinking_text": thinking_text,
        "grounded": grounded,
    }


def _build_fallback_citations(
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build citations directly from retrieved chunks when the model
    failed to include a Sources block.
    """
    citations: list[dict[str, Any]] = []
    for i, r in enumerate(retrieved_chunks, start=1):
        record = r.get("record", {})
        metadata = record.get("metadata", {})
        source_name = metadata.get("source_name", "").upper()
        doc_type = metadata.get("doc_type", "").replace("_", " ")
        page_num = metadata.get("page_num", "?")
        pdf_url = metadata.get("pdf_url", "")

        description = f"{source_name} | {doc_type} | Page {page_num}"
        citations.append({"index": i, "description": description, "url": pdf_url})

    return citations


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------

def render_citations_text(citations: list[dict[str, Any]]) -> str:
    """Render citation list as plain text for CLI display (one line per entry)."""
    if not citations:
        return "(no citations)"
    lines: list[str] = []
    for c in citations:
        lines.append(f"[{c['index']}] {c['description']}")
        if c.get("url"):
            lines.append(f"    {c['url']}")
    return "\n".join(lines)


_PAGE_STRIP_RE = re.compile(r"\s*[|,]\s*[Pp]age\s+\d+.*$")
_PAGE_EXTRACT_RE = re.compile(r"[Pp]age\s+(\d+)")


def render_deduplicated_citations(citations: list[dict[str, Any]]) -> str:
    """
    Render citations grouped by PDF (same URL = same document).

    When multiple chunks reference the same PDF, collapse them into one entry
    showing all pages referenced.  Entries without a URL are shown individually.

    Example output:
        [1-5] ICMR | guideline  (pp. 10, 22, 25, 53, 59)
              https://www.icmr.gov.in/...
    """
    if not citations:
        return "(no citations)"

    from collections import defaultdict

    url_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_url: list[dict[str, Any]] = []

    for c in citations:
        url = (c.get("url") or "").strip()
        if url:
            url_groups[url].append(c)
        else:
            no_url.append(c)

    lines: list[str] = []

    for url, group in url_groups.items():
        idxs = sorted(c["index"] for c in group)
        if len(idxs) == 1:
            idx_label = f"[{idxs[0]}]"
        elif idxs == list(range(idxs[0], idxs[-1] + 1)):
            idx_label = f"[{idxs[0]}-{idxs[-1]}]"
        else:
            idx_label = f"[{', '.join(str(i) for i in idxs)}]"

        # Extract page numbers from all descriptions in the group
        pages: list[int] = []
        for c in group:
            m = _PAGE_EXTRACT_RE.search(c.get("description", ""))
            if m:
                pages.append(int(m.group(1)))
        pages_str = f"  (pp. {', '.join(str(p) for p in sorted(set(pages)))})" if pages else ""

        # Clean base description: strip "| Page N" suffix
        raw_desc = group[0].get("description", "")
        clean_desc = _PAGE_STRIP_RE.sub("", raw_desc).strip()

        lines.append(f"{idx_label} {clean_desc}{pages_str}")
        lines.append(f"      {url}")

    for c in no_url:
        lines.append(f"[{c['index']}] {c['description']}")

    return "\n".join(lines)
