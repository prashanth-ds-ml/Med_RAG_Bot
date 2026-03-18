from __future__ import annotations

"""
Prompt/context builder for grounded answering.

Why this file exists:
- Keeps retrieval context formatting separate from CLI code
- Makes it easier to later swap in a real LLM or reasoning model
- Ensures the answer stage stays inspectable and deterministic
"""

from typing import Any


def build_context_block(results: list[dict[str, Any]]) -> str:
    """
    Build a readable context block from retrieval results.

    Each result is expected to include:
    - chunk_id
    - record
    - chunk_text
    """
    if not results:
        return ""

    parts: list[str] = []

    for item in results:
        record = item["record"]
        chunk_id = record.get("chunk_id", "unknown_chunk")
        heading_path = " > ".join(record.get("heading_path", [])) or "(no heading path)"
        chunk_text = record.get("chunk_text", "").strip()

        parts.append(
            f"[{chunk_id}] {heading_path}\n{chunk_text}"
        )

    return "\n\n".join(parts)


def build_grounded_prompt(query: str, results: list[dict[str, Any]]) -> str:
    """
    Build a grounded prompt for a future reasoning model.

    Why this matters:
    - Lets us keep prompt design explicit from the beginning
    - Makes later LLM integration smoother
    """
    context_block = build_context_block(results)

    return (
        "You are a careful medical-domain retrieval assistant.\n"
        "Answer only from the provided context.\n"
        "If the context is insufficient, say so clearly.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_block}\n"
    )