from __future__ import annotations

"""
engine.py — ChatEngine: the stateless RAG core for Med RAG Bot.

This class owns model loading and the ask() method.
It has no knowledge of CLI, Gradio, or any UI layer.

Both the CLI and the Gradio UI import ChatEngine directly:

    engine = ChatEngine(settings=settings, use_reranker=True)
    engine.load()
    response = engine.ask("how to treat snake bite", enable_thinking=True)

Session state (turn count, history, session_id) is managed by the caller:
  - CLI:   local variables in the chat loop
  - Gradio: gr.State() per browser session
"""

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.settings import AppSettings, settings as _default_settings
from app.generation.llm_client import QwenClient
from app.generation.prompt_builder import build_messages
from app.generation.response_formatter import (
    format_response,
    render_deduplicated_citations,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import DEFAULT_RERANKER_MODEL
from app.monitoring.logger import (
    log_session_start,
    log_session_end,
    log_turn,
    log_feedback as _log_feedback,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """Everything the UI needs to render one RAG turn."""

    # Core answer
    query:            str
    query_type:       str
    answer_text:      str
    thinking_text:    str | None

    # Citations and follow-ups
    citations:        list[dict[str, Any]]
    citations_text:   str          # pre-rendered deduped citation block
    follow_ups:       list[str]

    # Retrieval metadata (for /chunks display or Gradio sources tab)
    retrieved_chunks: list[dict[str, Any]]

    # Confidence
    confidence:       str          # "HIGH" | "MED" | "LOW"
    grounded:         bool

    # Token / latency stats
    prompt_tokens:    int
    completion_tokens: int
    total_tokens:     int
    generation_time_ms: int

    # Logging
    message_id:       str
    session_id:       str
    turn:             int


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

class ChatEngine:
    """
    Stateless RAG engine.

    Load once, call ask() for each query.
    Thread safety: NOT thread-safe — use one engine per process/thread.
    For multi-user (FastAPI), load one engine per worker or use a lock.
    """

    def __init__(
        self,
        *,
        app_settings: AppSettings | None = None,
        top_k: int = 5,
        fetch_k: int = 20,
        model_name: str | None = None,
        use_reranker: bool = False,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self._settings    = app_settings or _default_settings
        self.top_k        = top_k
        self.fetch_k      = fetch_k
        self.model_name   = model_name  # None → QwenClient uses its own default
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model

        self._retriever: HybridRetriever | None = None
        self._client:    QwenClient | None = None
        self._loaded     = False

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all models and indexes into memory. Call once before ask()."""
        if self._loaded:
            return

        logger.info("Loading retriever...")
        self._retriever = HybridRetriever.load(
            bm25_index_path=self._settings.corpus_bm25_index_path,
            faiss_index_path=self._settings.faiss_index_path,
            vector_payload_path=self._settings.vector_payload_path,
            fetch_k=self.fetch_k,
            reranker_model=self.reranker_model if self.use_reranker else None,
            reranker_device="cpu",
        )

        logger.info("Loading LLM...")
        kwargs: dict[str, Any] = {}
        if self.model_name:
            kwargs["model_name"] = self.model_name
        self._client = QwenClient(**kwargs)
        self._client.load()

        self._loaded = True
        logger.info("ChatEngine ready.")

    def unload(self) -> None:
        """Release models from memory."""
        if self._client:
            self._client.unload()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Core ask()
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        *,
        session_id: str,
        turn: int = 1,
        history: list[tuple[str, str]] | None = None,
        enable_thinking: bool = False,
        thinking_budget: int | None = 512,
    ) -> ChatResponse:
        """
        Run one RAG turn and return a ChatResponse.

        Args:
            query:           user question
            session_id:      caller-managed session ID (for logging)
            turn:            1-based turn number within the session
            history:         list of (query, answer) pairs for conversation memory
                             (pass last 3 turns max)
            enable_thinking: enable Qwen3 thinking mode
            thinking_budget: max tokens for the <think> block (None = unlimited)

        Returns:
            ChatResponse with answer, citations, confidence, stats, etc.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before ask().")

        # 1. Retrieve + re-rank (reranker is embedded in HybridRetriever)
        results = self._retriever.search(query, top_k=self.top_k, fetch_k=self.fetch_k)

        # 3. Confidence from top fused score
        top_score  = results[0].get("fused_score", 0.0) if results else 0.0
        confidence = (
            "HIGH" if top_score >= 0.025
            else ("MED" if top_score >= 0.015 else "LOW")
        )

        # 4. Build prompt
        messages = build_messages(
            query,
            results,
            history=history[-3:] if history else None,
        )

        # 5. Generate
        gen = self._client.generate(
            messages,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )

        # 6. Format response
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

        # 7. Log turn
        message_id = log_turn(
            session_id=session_id,
            turn=turn,
            query=query,
            response=response,
            retrieved_chunks=results,
            top_k=self.top_k,
            fetch_k=self.fetch_k,
        )

        return ChatResponse(
            query=query,
            query_type=response["query_type"],
            answer_text=response["answer_text"],
            thinking_text=gen.thinking_text,
            citations=response["citations"],
            citations_text=render_deduplicated_citations(response["citations"]),
            follow_ups=response.get("follow_ups", []),
            retrieved_chunks=results,
            confidence=confidence,
            grounded=response["grounded"],
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
            total_tokens=response["total_tokens"],
            generation_time_ms=response["generation_time_ms"],
            message_id=message_id,
            session_id=session_id,
            turn=turn,
        )

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def submit_feedback(
        self,
        *,
        message_id: str,
        session_id: str,
        rating: int,
        comment: str = "",
    ) -> bool:
        """Log user feedback for a message. Returns True if saved."""
        return _log_feedback(
            message_id=message_id,
            session_id=session_id,
            rating=rating,
            comment=comment,
            tier="gradio",
        )

    # ------------------------------------------------------------------
    # Session lifecycle helpers
    # ------------------------------------------------------------------

    def start_session(
        self,
        *,
        session_id: str | None = None,
        thinking_on: bool = False,
    ) -> str:
        """
        Create a new session ID and log its start.
        Returns the session_id (useful if you pass None to auto-generate).
        """
        sid = session_id or uuid.uuid4().hex
        log_session_start(
            session_id=sid,
            model_name=self._client.model_name if self._client else "unknown",
            top_k=self.top_k,
            fetch_k=self.fetch_k,
            thinking_on=thinking_on,
        )
        return sid

    def end_session(self, *, session_id: str, turn_count: int) -> None:
        """Log session end."""
        log_session_end(session_id=session_id, turn_count=turn_count)
