from __future__ import annotations

import json
from pathlib import Path

from app.retrieval.bm25_index import (
    build_bm25_index_from_atomic_chunks,
    build_bm25_payload,
    load_bm25_payload,
    load_jsonl_records,
    save_bm25_payload,
    search_bm25_index,
    search_bm25_payload,
    tokenize_text,
)


def test_tokenize_text_returns_lowercase_tokens() -> None:
    """
    What this test checks:
    - Tokenization is deterministic and lowercase.

    Why this matters:
    - BM25 behavior depends on consistent tokenization.
    """
    tokens = tokenize_text("Atrial Fibrillation, Rate-Control 2026!")

    assert tokens == ["atrial", "fibrillation", "rate", "control", "2026"]


def test_load_jsonl_records_reads_records_correctly(tmp_path: Path) -> None:
    """
    What this test checks:
    - JSONL chunk records are loaded correctly from disk.

    Why this matters:
    - BM25 indexing starts from atomic chunk JSONL artifacts.
    """
    path = tmp_path / "atomic_chunks.jsonl"
    records = [
        {"chunk_id": "c1", "chunk_text": "atrial fibrillation symptoms"},
        {"chunk_id": "c2", "chunk_text": "stroke emergency management"},
    ]

    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record) + "\n")

    loaded = load_jsonl_records(path)

    assert len(loaded) == 2
    assert loaded[0]["chunk_id"] == "c1"
    assert loaded[1]["chunk_id"] == "c2"


def test_build_and_save_bm25_payload_roundtrip(tmp_path: Path) -> None:
    """
    What this test checks:
    - BM25 payload can be built, saved, and loaded back successfully.

    Why this matters:
    - Local retrieval depends on index persistence across runs.
    """
    records = [
        {"chunk_id": "c1", "chunk_text": "atrial fibrillation symptoms"},
        {"chunk_id": "c2", "chunk_text": "stroke emergency management"},
    ]

    payload = build_bm25_payload(records)
    index_path = tmp_path / "bm25.pkl"

    save_bm25_payload(payload, index_path)
    loaded_payload = load_bm25_payload(index_path)

    assert index_path.exists()
    assert len(loaded_payload["chunk_records"]) == 2
    assert loaded_payload["chunk_ids"] == ["c1", "c2"]


def test_search_bm25_payload_returns_relevant_top_result() -> None:
    """
    What this test checks:
    - BM25 retrieval ranks the most relevant lexical match first.

    Why this matters:
    - This is the first retrieval baseline in the pipeline.
    """
    records = [
        {"chunk_id": "afib_1", "chunk_text": "atrial fibrillation symptoms palpitations fatigue"},
        {"chunk_id": "stroke_1", "chunk_text": "stroke emergency management rapid assessment"},
        {"chunk_id": "htn_1", "chunk_text": "hypertension lifestyle management"},
    ]

    payload = build_bm25_payload(records)
    results = search_bm25_payload(payload, "atrial fibrillation symptoms", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "afib_1"
    assert results[0]["score"] >= results[1]["score"]


def test_build_bm25_index_from_atomic_chunks_and_search(tmp_path: Path) -> None:
    """
    What this test checks:
    - Full BM25 build from atomic chunk JSONL works end to end.

    Why this matters:
    - This is the real local workflow for retrieval indexing.
    """
    atomic_chunks_path = tmp_path / "atomic_chunks.jsonl"
    index_path = tmp_path / "bm25" / "bm25_index.pkl"

    records = [
        {
            "chunk_id": "c1",
            "chunk_text": "atrial fibrillation management rate control anticoagulation",
        },
        {
            "chunk_id": "c2",
            "chunk_text": "stroke emergency care assessment and stabilization",
        },
    ]

    with atomic_chunks_path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record) + "\n")

    build_result = build_bm25_index_from_atomic_chunks(
        atomic_chunks_path=atomic_chunks_path,
        output_path=index_path,
    )

    assert build_result["document_count"] == 2
    assert index_path.exists()

    results = search_bm25_index(index_path, "atrial fibrillation rate control", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"


def test_search_bm25_payload_returns_empty_for_empty_query() -> None:
    """
    What this test checks:
    - Empty queries do not produce meaningless results.

    Why this matters:
    - Query validation should be predictable during local testing.
    """
    payload = build_bm25_payload(
        [{"chunk_id": "c1", "chunk_text": "atrial fibrillation symptoms"}]
    )

    results = search_bm25_payload(payload, "", top_k=3)

    assert results == []