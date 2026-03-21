from __future__ import annotations

"""
reranker.py — Cross-encoder re-ranking for Med RAG Bot.

Why a re-ranker:
  Bi-encoders (our FAISS embedding model) encode query and document
  separately and compare via cosine similarity — fast, but approximate.
  A cross-encoder sees (query, document) together in one forward pass,
  giving a much more accurate relevance score.

Pipeline position:
  BM25 + FAISS → RRF fusion (top-20 candidates) → Cross-encoder re-rank → top-5 to LLM

  We run the cross-encoder only on the 20 RRF candidates, not the full
  corpus — so latency is low (~100-500ms depending on device).

Device choice:
  Defaults to CPU to avoid competing with Qwen3-4B for VRAM.
  On CPU, scoring 20 short medical chunks takes ~200-400ms — acceptable.
  Pass device="cuda" if you have VRAM headroom (RTX 3060 with 8GB+).

Model choice:
  cross-encoder/ms-marco-MiniLM-L-6-v2 (~22MB, very fast, good English quality) ← default
  BAAI/bge-reranker-base  (~1.1GB, English, stronger quality)
  BAAI/bge-reranker-v2-m3 (~570MB, multilingual, best quality)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for second-pass re-ranking.

    Usage:
        reranker = CrossEncoderReranker()
        reranker.load()
        results = reranker.rerank(query, rrf_candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the cross-encoder model into memory."""
        if self.is_loaded:
            return
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading cross-encoder re-ranker: %s on %s", self.model_name, self.device
        )
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )
        logger.info("Re-ranker loaded.")

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Re-rank retrieval results using cross-encoder scores.

        Args:
            query:   the user query string
            results: RRF-fused candidates (any length, typically fetch_k=20)
            top_k:   number of results to return after re-ranking

        Returns:
            Top-k results sorted by cross-encoder score descending.
            Each result gets a new "rerank_score" field and updated "rank".
            The original "fused_score" is preserved for debugging/logging.
        """
        if not self.is_loaded:
            raise RuntimeError("Call load() before rerank().")
        if not results:
            return results

        # Build (query, document_text) pairs for the cross-encoder
        pairs = [
            (query, r.get("chunk_text", r.get("record", {}).get("chunk_text", "")))
            for r in results
        ]

        # Score all pairs in one batch — fast even on CPU for 20 pairs
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach rerank_score to each result (preserve fused_score for logging)
        for result, score in zip(results, scores):
            result["rerank_score"] = round(float(score), 6)

        # Sort by rerank_score descending, assign new ranks
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        for i, result in enumerate(reranked[:top_k], start=1):
            result["rank"] = i

        logger.debug(
            "Re-ranked %d candidates → top %d  (top score: %.4f)",
            len(results),
            top_k,
            reranked[0]["rerank_score"] if reranked else 0,
        )

        return reranked[:top_k]

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Re-ranker unloaded.")
