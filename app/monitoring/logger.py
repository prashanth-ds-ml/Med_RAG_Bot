from __future__ import annotations

"""
logger.py — Observability logging for Med360 chat sessions.

Storage strategy:
  All events are written to local JSONL files in data/logs/.
  The schema exactly mirrors the MongoDB collection structure so that
  the 'upload-logs' CLI command can bulk-insert the files later.

  JSONL files:
    data/logs/chat_sessions.jsonl   — one line per session
    data/logs/messages.jsonl        — one line per Q&A turn
    data/logs/retrieval_logs.jsonl  — one line per turn (chunks + scores)
    data/logs/feedback.jsonl        — one line per user rating

All functions are silent on failure — a logging error never interrupts
the chat session.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import orjson

from app.settings import settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def _append(path, doc: dict) -> None:
    """Append one JSON record as a line to a JSONL file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = orjson.dumps(doc) + b"\n"
        with open(path, "ab") as f:
            f.write(line)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session logging
# ---------------------------------------------------------------------------

def log_session_start(
    *,
    session_id: str,
    model_name: str,
    top_k: int,
    fetch_k: int,
    thinking_on: bool,
) -> None:
    """Append a new session record when a chat session begins."""
    _append(
        settings.chat_sessions_log_path,
        {
            "session_id":  session_id,
            "model_name":  model_name,
            "top_k":       top_k,
            "fetch_k":     fetch_k,
            "thinking_on": thinking_on,
            "turn_count":  0,
            "started_at":  _now_iso(),
            "ended_at":    None,
        },
    )


def log_session_end(*, session_id: str, turn_count: int) -> None:
    """
    Append an end-of-session update record.

    Note: JSONL is append-only, so we write a separate 'session_end' event
    rather than updating the original record. The upload-logs command merges
    these before inserting to MongoDB.
    """
    _append(
        settings.chat_sessions_log_path,
        {
            "_event":      "session_end",
            "session_id":  session_id,
            "turn_count":  turn_count,
            "ended_at":    _now_iso(),
        },
    )


# ---------------------------------------------------------------------------
# Turn logging
# ---------------------------------------------------------------------------

def log_turn(
    *,
    session_id: str,
    turn: int,
    query: str,
    response: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    top_k: int,
    fetch_k: int,
) -> str:
    """
    Append one Q&A turn to messages.jsonl and retrieval_logs.jsonl.

    Returns the message_id so the CLI can show it and accept /feedback for it.
    """
    message_id = _new_id()

    _append(
        settings.messages_log_path,
        {
            "message_id":         message_id,
            "session_id":         session_id,
            "turn":               turn,
            "query":              query,
            "query_type":         response.get("query_type", ""),
            "answer_text":        response.get("answer_text", ""),
            "grounded":           response.get("grounded", False),
            "citations":          response.get("citations", []),
            "thinking_text":      response.get("thinking_text", ""),
            "prompt_tokens":      response.get("prompt_tokens", 0),
            "completion_tokens":  response.get("completion_tokens", 0),
            "total_tokens":       response.get("total_tokens", 0),
            "generation_time_ms": response.get("generation_time_ms", 0),
            "model_name":         response.get("model_name", ""),
            "created_at":         _now_iso(),
        },
    )

    chunks_log = []
    for r in retrieved_chunks:
        rec  = r.get("record", {})
        meta = rec.get("metadata", {})
        chunks_log.append(
            {
                "rank":         r.get("rank"),
                "chunk_id":     r.get("chunk_id", rec.get("chunk_id", "")),
                "fused_score":  r.get("fused_score", 0.0),
                "bm25_rank":    r.get("bm25_rank"),
                "bm25_score":   r.get("bm25_score"),
                "vector_rank":  r.get("vector_rank"),
                "vector_score": r.get("vector_score"),
                "source_name":  meta.get("source_name", ""),
                "doc_type":     meta.get("doc_type", ""),
                "page_num":     meta.get("page_num"),
                "pdf_url":      meta.get("pdf_url", ""),
            }
        )

    _append(
        settings.retrieval_logs_log_path,
        {
            "message_id": message_id,
            "session_id": session_id,
            "turn":       turn,
            "query":      query,
            "top_k":      top_k,
            "fetch_k":    fetch_k,
            "chunks":     chunks_log,
            "created_at": _now_iso(),
        },
    )

    return message_id


# ---------------------------------------------------------------------------
# Feedback logging
# ---------------------------------------------------------------------------

def log_feedback(
    *,
    message_id: str,
    session_id: str,
    rating: int,
    comment: str = "",
    tier: str = "user",
) -> bool:
    """
    Append a user feedback entry to feedback.jsonl.

    rating: 1–5
    tier:   "user" | "professional" | "model_judge"

    Returns True always (local write; no connectivity to fail on).
    """
    _append(
        settings.feedback_log_path,
        {
            "feedback_id": _new_id(),
            "message_id":  message_id,
            "session_id":  session_id,
            "rating":      max(1, min(5, rating)),
            "comment":     comment.strip(),
            "tier":        tier,
            "created_at":  _now_iso(),
        },
    )
    return True
