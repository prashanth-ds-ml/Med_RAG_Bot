from __future__ import annotations

"""
Baseline grounded answer generation.

Why this file exists:
- Gives us a first local QA loop before integrating a real reasoning model
- Keeps answer formatting separate from retrieval and CLI code
- Makes output easy to inspect and test
"""

from typing import Any


def build_baseline_answer(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a simple grounded answer from BM25 retrieval results.

    Current behavior:
    - If no results: return an insufficiency response
    - Otherwise: summarize which chunks look relevant and show a grounded answer stub

    Why this matters:
    - This is a transparent first answer stage
    - Later we can replace only this function with an LLM-backed generator
    """
    if not results:
        return {
            "answer_text": (
                "I could not find enough relevant context in the current corpus to answer this confidently."
            ),
            "citations": [],
            "used_chunk_ids": [],
            "grounded": False,
        }

    used_chunk_ids = [item["chunk_id"] for item in results]
    citation_lines = []

    for item in results:
        record = item["record"]
        heading_path = " > ".join(record.get("heading_path", [])) or "(no heading path)"
        citation_lines.append(f"- {item['chunk_id']} — {heading_path}")

    answer_text = (
        f"Question: {query}\n\n"
        "Based on the retrieved corpus context, the most relevant information appears in the following sections:\n\n"
        + "\n".join(citation_lines)
        + "\n\n"
        "Baseline answer:\n"
        "This answer is currently retrieval-grounded and inspection-first. "
        "Use the cited chunk previews below to verify whether the retrieval looks correct before we attach a full reasoning model."
    )

    return {
        "answer_text": answer_text,
        "citations": citation_lines,
        "used_chunk_ids": used_chunk_ids,
        "grounded": True,
    }