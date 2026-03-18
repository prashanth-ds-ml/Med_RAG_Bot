from app.generation.answer import build_baseline_answer


def test_build_baseline_answer_handles_empty_results() -> None:
    """
    What this test checks:
    - Empty retrieval results produce a safe insufficiency response.

    Why this matters:
    - QA should fail clearly when retrieval is weak.
    """
    result = build_baseline_answer("What is atrial fibrillation?", [])

    assert result["grounded"] is False
    assert result["used_chunk_ids"] == []
    assert "could not find enough relevant context" in result["answer_text"].lower()


def test_build_baseline_answer_returns_used_chunk_ids_and_citations() -> None:
    """
    What this test checks:
    - Answer builder preserves citations and used chunk IDs.

    Why this matters:
    - We want answers to remain inspectable and traceable.
    """
    retrieval_results = [
        {
            "chunk_id": "afib_1",
            "record": {
                "chunk_id": "afib_1",
                "heading_path": ["Atrial Fibrillation", "Symptoms"],
                "chunk_text": "Palpitations and fatigue are common symptoms.",
            },
        },
        {
            "chunk_id": "afib_2",
            "record": {
                "chunk_id": "afib_2",
                "heading_path": ["Atrial Fibrillation", "Management"],
                "chunk_text": "Rate control and rhythm control are treatment strategies.",
            },
        },
    ]

    result = build_baseline_answer("What is atrial fibrillation?", retrieval_results)

    assert result["grounded"] is True
    assert result["used_chunk_ids"] == ["afib_1", "afib_2"]
    assert len(result["citations"]) == 2
    assert "afib_1" in result["answer_text"]
    assert "afib_2" in result["answer_text"]