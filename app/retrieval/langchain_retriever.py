from __future__ import annotations

"""
langchain_retriever.py — LangChain BaseRetriever adapter for HybridRetriever.

Why this adapter exists:
- LangChain's LCEL pipeline expects retrievers to return list[Document].
- HybridRetriever returns list[dict] with rich citation metadata.
- This wrapper bridges the two without touching the retrieval logic at all.

Design decisions:
- _last_raw_results stores the original dicts so engine.py can still do
  confidence scoring and JSONL logging without reaching into Document metadata
  to reconstruct what was already computed.
- arbitrary_types_allowed is required because HybridRetriever is not a
  Pydantic model.
- top_k and fetch_k are exposed as fields so callers can tune them at
  construction time rather than monkey-patching after the fact.
"""

from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.retrieval.hybrid_retriever import HybridRetriever


class LangChainHybridRetriever(BaseRetriever):
    """LangChain-compatible wrapper around HybridRetriever (BM25+FAISS+RRF).

    Keeps all retrieval logic intact — this class only converts result dicts
    to Document objects. The raw_results property gives access to the original
    dicts for confidence scoring and JSONL logging in engine.py.
    """

    hybrid_retriever: Any = Field(description="Loaded HybridRetriever instance")
    top_k: int = Field(default=5)
    fetch_k: int = Field(default=20)

    # Private storage for raw results — underscore prefix keeps Pydantic from
    # treating it as a model field while still making it instance-accessible.
    _last_raw_results: list[dict] = []

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = self.hybrid_retriever.search(
            query, top_k=self.top_k, fetch_k=self.fetch_k
        )
        self._last_raw_results = results

        docs: list[Document] = []
        for r in results:
            metadata = r.get("record", {}).get("metadata", {})
            docs.append(
                Document(
                    page_content=r.get("chunk_text", ""),
                    metadata={
                        "chunk_id": r.get("chunk_id", ""),
                        "fused_score": r.get("fused_score", 0.0),
                        "rerank_score": r.get("rerank_score"),
                        "rank": r.get("rank", 0),
                        "bm25_rank": r.get("bm25_rank"),
                        "vector_rank": r.get("vector_rank"),
                        "source_name": metadata.get("source_name", ""),
                        "doc_type": metadata.get("doc_type", ""),
                        "page_num": metadata.get("page_num"),
                        "pdf_url": metadata.get("pdf_url", ""),
                    },
                )
            )
        return docs

    @property
    def last_raw_results(self) -> list[dict]:
        """Raw result dicts from the last search — for confidence scoring and logging."""
        return self._last_raw_results
