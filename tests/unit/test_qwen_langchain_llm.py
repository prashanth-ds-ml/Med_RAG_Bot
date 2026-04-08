from __future__ import annotations

"""
test_qwen_langchain_llm.py — Unit tests for QwenLangChainLLM and helpers.

Real model weights are never loaded; QwenClient.generate() is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.generation.llm_client import (
    GenerationResult,
    QwenLangChainLLM,
    _lc_messages_to_dicts,
)


# ---------------------------------------------------------------------------
# _lc_messages_to_dicts
# ---------------------------------------------------------------------------


def test_lc_messages_to_dicts_basic():
    messages = [
        SystemMessage(content="You are a doctor."),
        HumanMessage(content="What is TB?"),
    ]
    result = _lc_messages_to_dicts(messages)
    assert result == [
        {"role": "system", "content": "You are a doctor."},
        {"role": "user", "content": "What is TB?"},
    ]


def test_lc_messages_to_dicts_ai_message():
    from langchain_core.messages import AIMessage

    messages = [AIMessage(content="TB is tuberculosis.")]
    result = _lc_messages_to_dicts(messages)
    assert result == [{"role": "assistant", "content": "TB is tuberculosis."}]


def test_lc_messages_to_dicts_unknown_type_falls_back_to_user():
    """Unknown message types should default to 'user' role."""
    msg = MagicMock()
    msg.type = "tool"
    msg.content = "some tool output"
    result = _lc_messages_to_dicts([msg])
    assert result[0]["role"] == "user"


def test_lc_messages_to_dicts_empty():
    assert _lc_messages_to_dicts([]) == []


# ---------------------------------------------------------------------------
# QwenLangChainLLM — RuntimeError before load()
# ---------------------------------------------------------------------------


def test_generate_raises_if_not_loaded():
    llm = QwenLangChainLLM()
    with pytest.raises(RuntimeError, match="load()"):
        llm._generate([HumanMessage(content="hello")])


# ---------------------------------------------------------------------------
# QwenLangChainLLM._generate — happy path with mocked QwenClient
# ---------------------------------------------------------------------------


def _make_fake_result(answer: str = "TB is tuberculosis.", thinking: str = "") -> GenerationResult:
    return GenerationResult(
        answer_text=answer,
        thinking_text=thinking,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        generation_time_ms=100,
    )


def test_generate_returns_ai_message():
    llm = QwenLangChainLLM()

    mock_client = MagicMock()
    mock_client.generate.return_value = _make_fake_result("TB is tuberculosis.")
    llm._client = mock_client

    chat_result = llm._generate([HumanMessage(content="What is TB?")])

    assert len(chat_result.generations) == 1
    ai_msg = chat_result.generations[0].message
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "TB is tuberculosis."


def test_generate_token_usage_in_additional_kwargs():
    llm = QwenLangChainLLM()

    mock_client = MagicMock()
    mock_client.generate.return_value = _make_fake_result()
    llm._client = mock_client

    chat_result = llm._generate([HumanMessage(content="What is TB?")])
    usage = chat_result.generations[0].message.additional_kwargs["usage"]

    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30


def test_generate_thinking_text_in_additional_kwargs():
    llm = QwenLangChainLLM(enable_thinking=True)

    mock_client = MagicMock()
    mock_client.generate.return_value = _make_fake_result(
        answer="TB is a lung disease.", thinking="Let me think about this..."
    )
    llm._client = mock_client

    chat_result = llm._generate([HumanMessage(content="Explain TB")])
    ai_msg = chat_result.generations[0].message

    assert ai_msg.additional_kwargs["thinking"] == "Let me think about this..."


def test_generate_no_thinking_key_when_empty():
    """When thinking_text is empty, 'thinking' key must not appear."""
    llm = QwenLangChainLLM()

    mock_client = MagicMock()
    mock_client.generate.return_value = _make_fake_result(thinking="")
    llm._client = mock_client

    chat_result = llm._generate([HumanMessage(content="hello")])
    assert "thinking" not in chat_result.generations[0].message.additional_kwargs


def test_generate_passes_messages_as_dicts():
    """QwenClient.generate must receive plain dicts, not LangChain objects."""
    llm = QwenLangChainLLM()

    mock_client = MagicMock()
    mock_client.generate.return_value = _make_fake_result()
    llm._client = mock_client

    messages = [SystemMessage(content="Be helpful."), HumanMessage(content="Hi")]
    llm._generate(messages)

    call_args = mock_client.generate.call_args
    passed_messages = call_args[0][0]
    assert all(isinstance(m, dict) for m in passed_messages)
    assert passed_messages[0] == {"role": "system", "content": "Be helpful."}
    assert passed_messages[1] == {"role": "user", "content": "Hi"}


# ---------------------------------------------------------------------------
# _llm_type and _identifying_params
# ---------------------------------------------------------------------------


def test_llm_type():
    llm = QwenLangChainLLM()
    assert llm._llm_type == "qwen-local"


def test_identifying_params():
    llm = QwenLangChainLLM(enable_thinking=True, load_in_4bit=False)
    params = llm._identifying_params
    assert params["enable_thinking"] is True
    assert params["load_in_4bit"] is False
    assert "model_name" in params


# ---------------------------------------------------------------------------
# load() / unload() delegation
# ---------------------------------------------------------------------------


def test_unload_clears_client():
    llm = QwenLangChainLLM()
    mock_client = MagicMock()
    llm._client = mock_client

    llm.unload()

    mock_client.unload.assert_called_once()
    assert llm._client is None


def test_unload_no_op_when_not_loaded():
    llm = QwenLangChainLLM()
    # Should not raise even though _client is None
    llm.unload()


@patch("app.generation.llm_client.QwenClient")
def test_load_creates_and_loads_client(mock_qwen_cls):
    llm = QwenLangChainLLM(model_name="test-model", temperature=0.5, max_tokens=512)
    llm.load()

    mock_qwen_cls.assert_called_once_with(
        model_name="test-model",
        load_in_4bit=True,
        temperature=0.5,
        max_new_tokens=512,
        thinking_budget=512,
    )
    mock_qwen_cls.return_value.load.assert_called_once()
    assert llm._client is mock_qwen_cls.return_value
