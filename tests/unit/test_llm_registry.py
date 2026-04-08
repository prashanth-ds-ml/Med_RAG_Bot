from __future__ import annotations
"""
Tests for app/generation/llm_registry.py.

Strategy: monkeypatch env vars and mock the LangChain provider imports so
no real API keys are needed. We test provider selection logic, model name
resolution, and the RuntimeError raised when no keys are present.
"""

from importlib import reload

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all LLM API key env vars so tests start from a clean slate."""
    for var in ("GROQ_API_KEY", "GROQ_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# get_provider_name
# ---------------------------------------------------------------------------

class TestGetProviderName:
    def test_groq_api_key_wins(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        from app.generation.llm_registry import get_provider_name
        assert get_provider_name() == "groq"

    def test_groq_token_wins(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_TOKEN", "gsk_test")
        from app.generation.llm_registry import get_provider_name
        assert get_provider_name() == "groq"

    def test_openai_when_no_groq(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from app.generation.llm_registry import get_provider_name
        assert get_provider_name() == "openai"

    def test_anthropic_when_only_anthropic(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from app.generation.llm_registry import get_provider_name
        assert get_provider_name() == "anthropic"

    def test_none_when_no_keys(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        from app.generation.llm_registry import get_provider_name
        assert get_provider_name() is None

    def test_groq_takes_priority_over_openai(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from app.generation.llm_registry import get_provider_name
        assert get_provider_name() == "groq"


# ---------------------------------------------------------------------------
# get_api_llm — RuntimeError when no keys
# ---------------------------------------------------------------------------

class TestGetApiLlmNoKeys:
    def test_raises_runtime_error(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        from app.generation.llm_registry import get_api_llm
        with pytest.raises(RuntimeError, match="No LLM API key found"):
            get_api_llm()

    def test_error_mentions_groq(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        from app.generation.llm_registry import get_api_llm
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_api_llm()


# ---------------------------------------------------------------------------
# get_api_llm — provider auto-detection + model defaults
# ---------------------------------------------------------------------------

class TestGetApiLlmAutoDetect:
    def test_groq_selected_with_groq_api_key(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        mock_chat_groq = MagicMock()
        mock_instance = MagicMock()
        mock_chat_groq.return_value = mock_instance

        with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=mock_chat_groq)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            result = reg.get_api_llm()

        mock_chat_groq.assert_called_once()
        call_kwargs = mock_chat_groq.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.1-8b-instant"
        assert call_kwargs["api_key"] == "gsk_test"

    def test_groq_token_used_as_api_key(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_TOKEN", "gsk_token_test")

        mock_chat_groq = MagicMock()
        with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=mock_chat_groq)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm()

        call_kwargs = mock_chat_groq.call_args.kwargs
        assert call_kwargs["api_key"] == "gsk_token_test"

    def test_openai_selected_when_only_openai_key(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_chat_openai = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm()

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_anthropic_selected_when_only_anthropic_key(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_chat_anthropic = MagicMock()
        with patch.dict("sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_chat_anthropic)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm()

        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# get_api_llm — explicit provider override
# ---------------------------------------------------------------------------

class TestGetApiLlmExplicitProvider:
    def test_force_groq_provider(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        mock_chat_groq = MagicMock()
        with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=mock_chat_groq)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm(provider="groq")

        mock_chat_groq.assert_called_once()

    def test_force_openai_even_with_groq_key(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        # Groq key is set but we explicitly ask for openai
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_chat_openai = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm(provider="openai")

        mock_chat_openai.assert_called_once()

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        from importlib import reload
        import app.generation.llm_registry as reg
        reload(reg)

        with pytest.raises(ValueError, match="Unknown provider"):
            reg.get_api_llm(provider="cohere")


# ---------------------------------------------------------------------------
# get_api_llm — model name override
# ---------------------------------------------------------------------------

class TestGetApiLlmModelOverride:
    def test_custom_model_name_passed_to_groq(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        mock_chat_groq = MagicMock()
        with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=mock_chat_groq)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm("llama-3.3-70b-versatile")

        call_kwargs = mock_chat_groq.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    def test_custom_model_name_passed_to_openai(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_chat_openai = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm("gpt-4o")

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# get_api_llm — temperature and max_tokens forwarded
# ---------------------------------------------------------------------------

class TestGetApiLlmParams:
    def test_temperature_and_max_tokens_forwarded(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        mock_chat_groq = MagicMock()
        with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=mock_chat_groq)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm(temperature=0.7, max_tokens=512)

        call_kwargs = mock_chat_groq.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 512

    def test_extra_kwargs_forwarded(self, monkeypatch):
        _clear_all_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        mock_chat_groq = MagicMock()
        with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=mock_chat_groq)}):
            from importlib import reload
            import app.generation.llm_registry as reg
            reload(reg)

            reg.get_api_llm(streaming=True)

        call_kwargs = mock_chat_groq.call_args.kwargs
        assert call_kwargs["streaming"] is True
