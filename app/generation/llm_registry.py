from __future__ import annotations
"""
llm_registry.py — Pluggable LLM factory for Med RAG Bot.

Returns a LangChain BaseChatModel based on available API keys.
Priority: Groq → OpenAI → Anthropic.

Why a registry:
  Swapping LLM providers is a one-env-var change.
  No code changes needed to move from Groq to OpenAI to Anthropic.
  All returned models share the BaseChatModel interface (invoke, stream, batch).

Usage:
    llm = get_api_llm()                     # auto-detect from env
    llm = get_api_llm("llama-3.1-8b-instant")  # override model name
    llm = get_api_llm(provider="openai")    # force a specific provider
"""

import logging
import os
from typing import Any
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


def get_api_llm(
    model_name: str | None = None,
    *,
    provider: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Return a configured LangChain BaseChatModel.

    Provider selection (checked in order):
      1. `provider` argument (explicit override)
      2. GROQ_API_KEY or GROQ_TOKEN  → ChatGroq
      3. OPENAI_API_KEY              → ChatOpenAI
      4. ANTHROPIC_API_KEY           → ChatAnthropic

    Args:
        model_name: Override the default model for the chosen provider.
        provider:   Force a specific provider ("groq", "openai", "anthropic").
        temperature: Sampling temperature (default 0.3 for consistent answers).
        max_tokens: Max completion tokens.

    Returns:
        A LangChain BaseChatModel ready to call with .invoke() or .stream().
    """
    if provider is None:
        provider = get_provider_name()
        if provider is None:
            raise RuntimeError(
                "No LLM API key found. Set one of:\n"
                "  GROQ_API_KEY   (free tier — console.groq.com)\n"
                "  OPENAI_API_KEY\n"
                "  ANTHROPIC_API_KEY"
            )

    if provider not in _DEFAULTS:
        raise ValueError(f"Unknown provider: {provider!r}. Choose from: groq, openai, anthropic")

    model = model_name or _DEFAULTS[provider]
    logger.info("LLM provider: %s | model: %s", provider, model)

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_TOKEN")
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    raise ValueError(f"Unknown provider: {provider!r}. Choose from: groq, openai, anthropic")


def get_provider_name() -> str | None:
    """Return the name of the provider that would be auto-selected, or None."""
    if os.getenv("GROQ_API_KEY") or os.getenv("GROQ_TOKEN"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None
