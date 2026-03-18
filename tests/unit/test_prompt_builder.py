from app.generation.prompt_builder import build_context_block, build_grounded_prompt


def test_build_context_block_formats_results_cleanly() -> None:
    """
    What this test checks:
    - Retrieved results are formatted into a readable context block.

    Why this matters:
    - Later answer generation depends on a stable context layout.
    """
    results = [
        {
            "chunk_id": "afib_1",
            "record": {
                "chunk_id": "afib_1",
                "heading_path": ["Atrial Fibrillation", "Symptoms"],
                "chunk_text": "Palpitations and fatigue are common symptoms.",
            },
        }
    ]

    context = build_context_block(results)

    assert "[afib_1]" in context
    assert "Atrial Fibrillation > Symptoms" in context
    assert "Palpitations and fatigue" in context


def test_build_grounded_prompt_contains_question_and_context() -> None:
    """
    What this test checks:
    - Prompt builder includes both the user question and retrieval context.

    Why this matters:
    - Prompt layout should stay explicit and inspectable.
    """
    results = [
        {
            "chunk_id": "afib_1",
            "record": {
                "chunk_id": "afib_1",
                "heading_path": ["Atrial Fibrillation", "Symptoms"],
                "chunk_text": "Palpitations and fatigue are common symptoms.",
            },
        }
    ]

    prompt = build_grounded_prompt("What are common symptoms?", results)

    assert "Question:" in prompt
    assert "What are common symptoms?" in prompt
    assert "Context:" in prompt
    assert "[afib_1]" in prompt