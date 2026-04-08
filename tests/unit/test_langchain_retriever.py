from __future__ import annotations

"""
test_langchain_retriever.py — Unit tests for LangChainHybridRetriever.

No real indexes are loaded; HybridRetriever.search() is mocked so these tests
run in any environment and finish in milliseconds.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.retrieval.langchain_retriever import LangChainHybridRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    rank: int = 1,
    chunk_id: str = "c1",
    chunk_text: str = "Some medical text.",
    fused_score: float = 0.032,
    rerank_score: float | None = None,
    bm25_rank: int | None = 1,
    vector_rank: int | None = 2,
    source_name: str = "WHO Guideline",
    doc_type: str = "guideline",
    page_num: int | None = 3,
    pdf_url: str = "https://example.com/doc.pdf",
) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "fused_score": fused_score,
        "rerank_score": rerank_score,
        "bm25_rank": bm25_rank,
        "vector_rank": vector_rank,
        "record": {
            "chunk_id": chunk_id,
            "doc_id": "doc_001",
            "source_file": "doc.pdf",
            "chunk_text": chunk_text,
            "metadata": {
                "source_name": source_name,
                "doc_type": doc_type,
                "page_num": page_num,
                "pdf_url": pdf_url,
            },
            "heading_path": ["Introduction"],
            "doc_type": doc_type,
        },
    }


def _make_retriever(raw_results: list[dict], top_k: int = 5, fetch_k: int = 20) -> LangChainHybridRetriever:
    mock_hr = MagicMock()
    mock_hr.search.return_value = raw_results
    return LangChainHybridRetriever(
        hybrid_retriever=mock_hr, top_k=top_k, fetch_k=fetch_k
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDocumentConversion:
    def test_page_content_is_chunk_text(self) -> None:
        r = _make_result(chunk_text="Dengue treatment protocol.")
        retriever = _make_retriever([r])
        docs = retriever.invoke("dengue treatment")
        assert docs[0].page_content == "Dengue treatment protocol."

    def test_metadata_chunk_id(self) -> None:
        r = _make_result(chunk_id="abc-123")
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        assert docs[0].metadata["chunk_id"] == "abc-123"

    def test_metadata_fused_score(self) -> None:
        r = _make_result(fused_score=0.0456)
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        assert docs[0].metadata["fused_score"] == pytest.approx(0.0456)

    def test_metadata_rerank_score_none_when_absent(self) -> None:
        r = _make_result(rerank_score=None)
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        assert docs[0].metadata["rerank_score"] is None

    def test_metadata_rerank_score_present(self) -> None:
        r = _make_result(rerank_score=0.91)
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        assert docs[0].metadata["rerank_score"] == pytest.approx(0.91)

    def test_metadata_rank(self) -> None:
        r = _make_result(rank=2)
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        assert docs[0].metadata["rank"] == 2

    def test_metadata_bm25_and_vector_ranks(self) -> None:
        r = _make_result(bm25_rank=3, vector_rank=5)
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        assert docs[0].metadata["bm25_rank"] == 3
        assert docs[0].metadata["vector_rank"] == 5

    def test_metadata_source_fields(self) -> None:
        r = _make_result(
            source_name="CDC Report",
            doc_type="report",
            page_num=10,
            pdf_url="https://cdc.gov/report.pdf",
        )
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        m = docs[0].metadata
        assert m["source_name"] == "CDC Report"
        assert m["doc_type"] == "report"
        assert m["page_num"] == 10
        assert m["pdf_url"] == "https://cdc.gov/report.pdf"

    def test_returns_document_objects(self) -> None:
        r = _make_result()
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        assert all(isinstance(d, Document) for d in docs)


class TestMultipleResults:
    def test_correct_number_of_docs_returned(self) -> None:
        results = [_make_result(rank=i, chunk_id=f"c{i}") for i in range(1, 6)]
        retriever = _make_retriever(results)
        docs = retriever.invoke("query")
        assert len(docs) == 5

    def test_order_preserved(self) -> None:
        results = [_make_result(rank=i, chunk_id=f"c{i}") for i in range(1, 4)]
        retriever = _make_retriever(results)
        docs = retriever.invoke("query")
        chunk_ids = [d.metadata["chunk_id"] for d in docs]
        assert chunk_ids == ["c1", "c2", "c3"]


class TestLastRawResults:
    def test_last_raw_results_stored_after_invoke(self) -> None:
        raw = [_make_result(chunk_id="x1"), _make_result(chunk_id="x2", rank=2)]
        retriever = _make_retriever(raw)
        retriever.invoke("malaria symptoms")
        assert retriever.last_raw_results == raw

    def test_last_raw_results_updated_on_each_invoke(self) -> None:
        retriever = _make_retriever([_make_result(chunk_id="first")])
        retriever.invoke("first query")

        new_raw = [_make_result(chunk_id="second")]
        retriever.hybrid_retriever.search.return_value = new_raw
        retriever.invoke("second query")

        assert retriever.last_raw_results[0]["chunk_id"] == "second"

    def test_last_raw_results_initially_empty(self) -> None:
        mock_hr = MagicMock()
        retriever = LangChainHybridRetriever(hybrid_retriever=mock_hr)
        assert retriever.last_raw_results == []


class TestSearchCallForwarding:
    def test_top_k_forwarded_to_search(self) -> None:
        mock_hr = MagicMock()
        mock_hr.search.return_value = []
        retriever = LangChainHybridRetriever(
            hybrid_retriever=mock_hr, top_k=3, fetch_k=15
        )
        retriever.invoke("query")
        mock_hr.search.assert_called_once_with("query", top_k=3, fetch_k=15)

    def test_empty_results_returns_empty_list(self) -> None:
        retriever = _make_retriever([])
        docs = retriever.invoke("query")
        assert docs == []


class TestMissingFields:
    def test_missing_chunk_text_defaults_to_empty_string(self) -> None:
        r = _make_result()
        del r["chunk_text"]
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        assert docs[0].page_content == ""

    def test_missing_metadata_fields_default_gracefully(self) -> None:
        r = _make_result()
        r["record"]["metadata"] = {}
        retriever = _make_retriever([r])
        docs = retriever.invoke("query")
        m = docs[0].metadata
        assert m["source_name"] == ""
        assert m["doc_type"] == ""
        assert m["page_num"] is None
        assert m["pdf_url"] == ""
