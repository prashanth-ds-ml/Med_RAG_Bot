from __future__ import annotations

"""
hybrid_retriever.py — Reciprocal Rank Fusion of BM25 + FAISS results.

RRF formula:
    fused_score(doc) = Σ  1 / (k + rank_i)
where k=60 (empirically robust default) and rank_i is the 1-based rank
in each individual list.

Why RRF:
  - No tunable parameters (k=60 is standard across IR literature)
  - Handles mismatched score scales (BM25 raw vs cosine [0,1]) naturally
  - Robust: a strong signal in either modality boosts the final rank
  - Simple to inspect and debug — every score component is preserved

Retrieval flow:
  1. Get top-N BM25 results     (N = fetch_k, default 20)
  2. Get top-N FAISS results    (N = fetch_k)
  3. RRF-fuse → re-rank
  4. Return top-k results with full citation metadata

Result dict shape (mirrors BM25/FAISS shape for consistency):
  rank, fused_score, bm25_rank, bm25_score, vector_rank, vector_score,
  chunk_id, chunk_text, record {chunk_id, doc_id, source_file, metadata}
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieval.bm25_index import (
    load_bm25_payload,
    search_bm25_payload,
    tokenize_text,
)
from app.retrieval.vector_index import (
    _select_device,
    load_vector_index,
    search_vector_payload,
)
from app.retrieval.reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant — do not change without good reason


# ---------------------------------------------------------------------------
# RRF core
# ---------------------------------------------------------------------------

def _rrf_fuse(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    *,
    k: int = RRF_K,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Fuse BM25 and vector results using Reciprocal Rank Fusion.

    Chunks appearing in only one list still get the contribution from
    that list; chunks in both get both contributions added.
    """
    # Build per-chunk contribution maps
    bm25_map: dict[str, dict[str, Any]] = {}
    for r in bm25_results:
        cid = r["chunk_id"]
        bm25_map[cid] = {"rank": r["rank"], "score": r["score"], "record": r["record"]}

    vector_map: dict[str, dict[str, Any]] = {}
    for r in vector_results:
        cid = r["chunk_id"]
        vector_map[cid] = {"rank": r["rank"], "score": r["score"], "record": r["record"]}

    # Union of all seen chunk ids
    all_chunk_ids = set(bm25_map.keys()) | set(vector_map.keys())

    fused: list[dict[str, Any]] = []
    for cid in all_chunk_ids:
        bm25_entry = bm25_map.get(cid)
        vec_entry = vector_map.get(cid)

        bm25_rank = bm25_entry["rank"] if bm25_entry else None
        vector_rank = vec_entry["rank"] if vec_entry else None
        bm25_score = bm25_entry["score"] if bm25_entry else 0.0
        vector_score = vec_entry["score"] if vec_entry else 0.0

        rrf_score = (
            (1.0 / (k + bm25_rank) if bm25_rank else 0.0)
            + (1.0 / (k + vector_rank) if vector_rank else 0.0)
        )

        # Prefer the record from vector results (compact, has metadata);
        # fall back to BM25 record if only in BM25 list.
        record = (vec_entry or bm25_entry)["record"]  # type: ignore[index]

        fused.append(
            {
                "chunk_id": cid,
                "fused_score": round(rrf_score, 6),
                "bm25_rank": bm25_rank,
                "bm25_score": round(bm25_score, 4),
                "vector_rank": vector_rank,
                "vector_score": round(float(vector_score), 4),
                "chunk_text": record.get("chunk_text", ""),
                "record": record,
            }
        )

    # Sort by fused score descending, assign final ranks
    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    for i, item in enumerate(fused[:top_k], start=1):
        item["rank"] = i

    return fused[:top_k]


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Stateful retriever that holds both indexes in memory.

    Load once, query many times — avoids re-loading indexes on every call,
    which matters for the interactive 'chat' command and future FastAPI wrapping.

    Usage:
        retriever = HybridRetriever.from_settings(settings)
        results = retriever.search("What is the treatment for dengue?", top_k=5)
    """

    def __init__(
        self,
        bm25_payload: dict[str, Any],
        vector_payload: dict[str, Any],
        model: SentenceTransformer,
        *,
        fetch_k: int = 20,
        rrf_k: int = RRF_K,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._bm25 = bm25_payload
        self._vector = vector_payload
        self._model = model
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k
        self._reranker = reranker

    @classmethod
    def load(
        cls,
        bm25_index_path: Path,
        faiss_index_path: Path,
        vector_payload_path: Path,
        *,
        model_name: str = "BAAI/bge-base-en-v1.5",
        fetch_k: int = 20,
        rrf_k: int = RRF_K,
        device: str | None = None,
        reranker_model: str | None = DEFAULT_RERANKER_MODEL,
        reranker_device: str = "cpu",
    ) -> "HybridRetriever":
        """
        Load both indexes, the embedding model, and optionally the re-ranker.

        Args:
            reranker_model: cross-encoder model name, or None to disable re-ranking.
            reranker_device: device for the re-ranker ("cpu" recommended to
                             preserve VRAM for the LLM).
        """
        if device is None:
            device = _select_device()

        logger.info("Loading BM25 index from %s", bm25_index_path)
        bm25_payload = load_bm25_payload(bm25_index_path)

        logger.info("Loading FAISS index from %s", faiss_index_path)
        vector_payload = load_vector_index(faiss_index_path, vector_payload_path)

        _model_name = vector_payload.get("model_name", model_name)
        logger.info("Loading embedding model %s on %s", _model_name, device)
        model = SentenceTransformer(_model_name, device=device)

        reranker: CrossEncoderReranker | None = None
        if reranker_model:
            reranker = CrossEncoderReranker(
                model_name=reranker_model,
                device=reranker_device,
            )
            reranker.load()

        return cls(
            bm25_payload,
            vector_payload,
            model,
            fetch_k=fetch_k,
            rrf_k=rrf_k,
            reranker=reranker,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        fetch_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run hybrid BM25 + FAISS retrieval with RRF fusion and optional re-ranking.

        Pipeline:
          1. BM25 top-fetch_k  +  FAISS top-fetch_k
          2. RRF fusion → top-fetch_k candidates
          3. Cross-encoder re-rank (if loaded) → top-k final results

        Args:
            query:   natural language query string
            top_k:   number of results to return
            fetch_k: candidates per index before fusion/re-ranking
                     (defaults to self.fetch_k, typically 20)

        Returns:
            List of result dicts sorted by rerank_score (if re-ranker active)
            or fused_score descending.
        """
        _fetch = fetch_k or self.fetch_k

        # BM25 retrieval
        bm25_results = search_bm25_payload(
            self._bm25, query, top_k=_fetch
        )

        # Vector retrieval — encode query, search FAISS
        query_vec: np.ndarray = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vector_results = search_vector_payload(
            self._vector, query_vec, top_k=_fetch
        )

        # RRF fusion — get top-fetch_k candidates for re-ranker,
        # or top-k directly if no re-ranker
        rrf_top = _fetch if self._reranker else top_k
        candidates = _rrf_fuse(
            bm25_results,
            vector_results,
            k=self.rrf_k,
            top_k=rrf_top,
        )

        # Cross-encoder re-rank (optional second pass)
        if self._reranker:
            return self._reranker.rerank(query, candidates, top_k=top_k)

        return candidates


# ---------------------------------------------------------------------------
# Stateless convenience function (for single-shot CLI calls)
# ---------------------------------------------------------------------------

def search_hybrid(
    bm25_index_path: Path,
    faiss_index_path: Path,
    vector_payload_path: Path,
    query: str,
    *,
    top_k: int = 5,
    fetch_k: int = 20,
    model_name: str = "BAAI/bge-base-en-v1.5",
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    One-shot hybrid search — loads indexes, searches, returns results.
    Convenience wrapper around HybridRetriever for CLI use.
    """
    retriever = HybridRetriever.load(
        bm25_index_path=bm25_index_path,
        faiss_index_path=faiss_index_path,
        vector_payload_path=vector_payload_path,
        model_name=model_name,
        fetch_k=fetch_k,
        device=device,
    )
    return retriever.search(query, top_k=top_k, fetch_k=fetch_k)
