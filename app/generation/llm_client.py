from __future__ import annotations

"""
llm_client.py — Qwen3-3B inference client via HuggingFace transformers.

Why HuggingFace transformers (not Ollama):
  - Same code runs locally and on AWS EC2 / SageMaker with device_map="auto"
  - Model is swappable by changing model_name — no Ollama layer in the way
  - Full control over quantization, generation params, and token counting

Quantization:
  - 4-bit NF4 with double quantization via bitsandbytes
  - Qwen3-3B 4-bit: ~2 GB VRAM — fits alongside the embedding model (~0.4 GB)
    on the RTX 3060 with headroom to spare

Thinking mode (Qwen3 feature):
  - When enable_thinking=True the model generates <think>...</think> blocks
    before the answer. Useful for complex clinical reasoning queries.
  - When False (default) thinking tokens are suppressed for lower latency.
  - The thinking block is stripped from the final response but returned
    separately in case you want to log or display it.

Token tracking:
  - prompt_tokens, completion_tokens, total_tokens are returned on every call
  - Used by observability layer (MongoDB / Streamlit) for cost estimation
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Any, Generator

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field as PydanticField

# Load .env from project root (two levels up from this file: app/generation/ → root)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-4B"


@dataclass
class GenerationResult:
    """
    Structured output from a single LLM generation call.
    Carries everything the observability layer needs to log.
    """
    answer_text: str                    # clean answer, thinking stripped
    thinking_text: str = ""             # raw <think> block if present
    query_type: str = "unknown"         # detected by the model in its response
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: int = 0
    model_name: str = DEFAULT_MODEL
    thinking_enabled: bool = False
    raw_output: str = ""                # full model output before post-processing


def _strip_thinking(text: str) -> tuple[str, str]:
    """
    Separate <think>...</think> block from the answer.
    Returns (answer_text, thinking_text).
    """
    import re
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = think_pattern.search(text)
    if match:
        thinking = match.group(1).strip()
        answer = think_pattern.sub("", text).strip()
        return answer, thinking
    return text.strip(), ""


class QwenClient:
    """
    Wrapper around Qwen3-3B for grounded medical QA.

    Lazy-loads the model on first call — keeps import time fast and
    lets the CLI start without waiting for model loading.

    Usage:
        client = QwenClient()
        client.load()
        result = client.generate(messages)
        client.unload()   # frees VRAM if needed before other heavy ops
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        repetition_penalty: float = 1.1,
        thinking_budget: int | None = 512,
    ) -> None:
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        # thinking_budget: max tokens the model may spend in <think> blocks.
        # None = no limit (full thinking).  0 = same effect as enable_thinking=False.
        # For RAG, 512 is a sensible default — facts come from chunks, not reasoning.
        self.thinking_budget = thinking_budget

        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load model + tokenizer into memory."""
        if self.is_loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_token = os.environ.get("HF_TOKEN") or None

        logger.info("Loading tokenizer: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, token=hf_token
        )

        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info(
                "Loading model %s with 4-bit NF4 quantization, device_map=%s",
                self.model_name, self.device_map,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
                token=hf_token,
            )
        else:
            logger.info(
                "Loading model %s in full precision, device_map=%s",
                self.model_name, self.device_map,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=hf_token,
            )

        self._model.eval()
        logger.info("Model loaded.")

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> Generator[tuple[str, Any], None, None]:
        """
        Stream generation token by token.

        Yields (phase, payload) tuples:
          ("thinking", str)         — tokens from the <think> block
          ("answer",   str)         — tokens from the actual answer
          ("done",     GenerationResult) — final yield with complete result

        Handles the <think>...</think> boundary automatically:
          - While thinking is active: yields thinking tokens (caller shows indicator)
          - After </think>: yields answer tokens live
          - If </think> never appears (thinking=False or stripped): all tokens are answer

        The caller should collect answer tokens to build the display string.
        GenerationResult in the "done" tuple is fully post-processed (same as generate()).
        """
        if not self.is_loaded:
            raise RuntimeError("Call load() before generate_stream().")

        import torch
        from transformers import TextIteratorStreamer

        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )

        t0 = time.time()
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        # Stream tokens; detect <think>...</think> boundary
        in_thinking = enable_thinking
        buffer = ""
        think_accumulated = ""
        answer_accumulated = ""
        THINK_END = "</think>"

        for token_text in streamer:
            buffer += token_text

            if in_thinking:
                if THINK_END in buffer:
                    idx = buffer.index(THINK_END) + len(THINK_END)
                    think_part = buffer[:idx].replace("<think>", "").replace(THINK_END, "").strip()
                    remainder = buffer[idx:].lstrip("\n")
                    if think_part:
                        think_accumulated += think_part
                        yield ("thinking", think_part)
                    in_thinking = False
                    buffer = remainder
                    if remainder:
                        answer_accumulated += remainder
                        yield ("answer", remainder)
                        buffer = ""
                else:
                    clean = token_text.replace("<think>", "")
                    think_accumulated += clean
                    if clean:
                        yield ("thinking", clean)
                    buffer = ""
            else:
                answer_accumulated += buffer
                if buffer:
                    yield ("answer", buffer)
                buffer = ""

        thread.join()
        generation_time_ms = int((time.time() - t0) * 1000)

        # Rebuild proper raw_output for post-processing (same path as generate())
        if enable_thinking and think_accumulated:
            raw_output = f"<think>{think_accumulated}</think>\n\n{answer_accumulated}"
        else:
            raw_output = answer_accumulated

        answer_text, thinking_text = _strip_thinking(raw_output)
        completion_tokens = len(self._tokenizer.encode(answer_accumulated, add_special_tokens=False))

        yield ("done", GenerationResult(
            answer_text=answer_text,
            thinking_text=thinking_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            generation_time_ms=generation_time_ms,
            model_name=self.model_name,
            thinking_enabled=enable_thinking,
            raw_output=raw_output,
        ))

    def unload(self) -> None:
        """Release model from memory and free VRAM."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Model unloaded.")

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool = False,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = ...,  # type: ignore[assignment]
    ) -> GenerationResult:
        """
        Generate a response from a list of chat messages.

        Args:
            messages: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
            enable_thinking: if True, allow Qwen3 to emit <think> blocks
            max_new_tokens: overrides instance default
            temperature: overrides instance default

        Returns:
            GenerationResult with answer, token counts, and timing.
        """
        if not self.is_loaded:
            raise RuntimeError("Call client.load() before generate().")

        import torch

        _max_tokens = max_new_tokens or self.max_new_tokens
        _temp = temperature or self.temperature
        # Sentinel ... means "use the instance default"
        _budget: int | None = self.thinking_budget if thinking_budget is ... else thinking_budget  # type: ignore[comparison-overlap]

        # Apply chat template — pass thinking_budget only when thinking is on
        # and a budget is set; fall back gracefully if the tokenizer doesn't
        # support these kwargs (older transformers versions).
        template_kwargs: dict = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if enable_thinking and _budget is not None:
            template_kwargs["thinking_budget"] = _budget

        try:
            text = self._tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Older tokenizer — strip unsupported kwargs and retry
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=_max_tokens,
                temperature=_temp,
                do_sample=_temp > 0,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generation_time_ms = int((time.time() - t0) * 1000)

        # Decode only the new tokens
        generated_ids = output_ids[0][prompt_tokens:]
        raw_output = self._tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        completion_tokens = len(generated_ids)

        # Strip thinking block if present
        answer_text, thinking_text = _strip_thinking(raw_output)

        return GenerationResult(
            answer_text=answer_text,
            thinking_text=thinking_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            generation_time_ms=generation_time_ms,
            model_name=self.model_name,
            thinking_enabled=enable_thinking,
            raw_output=raw_output,
        )


# ---------------------------------------------------------------------------
# Module-level singleton — shared across CLI commands in a session
# ---------------------------------------------------------------------------

_client: QwenClient | None = None


def get_client(model_name: str = DEFAULT_MODEL) -> QwenClient:
    """
    Return the module-level QwenClient, loading it if needed.
    This avoids reloading the model on every CLI call within a session.
    """
    global _client
    if _client is None or _client.model_name != model_name:
        _client = QwenClient(model_name=model_name)
        _client.load()
    return _client


# ---------------------------------------------------------------------------
# LangChain adapter — lets Qwen3 participate in LCEL chains
# ---------------------------------------------------------------------------


def _lc_messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """
    Convert LangChain message objects to the dict format QwenClient expects.

    Why: QwenClient takes {"role": ..., "content": ...} dicts, matching
    the OpenAI chat format. LangChain uses typed message objects instead.
    """
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }
    return [
        {"role": role_map.get(msg.type, "user"), "content": msg.content}
        for msg in messages
    ]


class QwenLangChainLLM(BaseChatModel):
    """
    LangChain BaseChatModel wrapper around QwenClient (local Qwen3-4B).

    Why this wrapper:
      Lets the local Qwen3 model participate in LCEL chains alongside
      ChatGroq, ChatOpenAI etc — same invoke/stream interface.
      The underlying QwenClient handles 4-bit NF4 quantization, thinking
      mode, and token tracking. This wrapper just adapts the interface.

    Usage:
        llm = QwenLangChainLLM()
        llm.load()
        result = llm.invoke([HumanMessage(content="What is TB?")])

    Thinking mode:
        llm = QwenLangChainLLM(enable_thinking=True, thinking_budget=512)
    """

    model_name: str = PydanticField(default=DEFAULT_MODEL)
    enable_thinking: bool = PydanticField(default=False)
    thinking_budget: int = PydanticField(default=512)
    load_in_4bit: bool = PydanticField(default=True)
    temperature: float = PydanticField(default=0.3)
    max_tokens: int = PydanticField(default=1024)

    # Private — excluded from Pydantic schema; managed by load()/unload()
    _client: QwenClient | None = None

    class Config:
        arbitrary_types_allowed = True

    def load(self) -> None:
        """Load the Qwen3 model into memory. Call once before invoke()."""
        self._client = QwenClient(
            model_name=self.model_name,
            load_in_4bit=self.load_in_4bit,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            thinking_budget=self.thinking_budget,
        )
        self._client.load()

    def unload(self) -> None:
        """Release model from memory."""
        if self._client:
            self._client.unload()
            self._client = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> ChatResult:
        if self._client is None:
            raise RuntimeError("Call load() before using QwenLangChainLLM.")

        dict_messages = _lc_messages_to_dicts(messages)
        enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)

        result: GenerationResult = self._client.generate(
            dict_messages,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            max_new_tokens=self.max_tokens,
        )

        # Store thinking text and token usage for optional display / observability
        additional_kwargs: dict = {}
        if result.thinking_text:
            additional_kwargs["thinking"] = result.thinking_text
        additional_kwargs["usage"] = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }

        ai_message = AIMessage(
            content=result.answer_text,
            additional_kwargs=additional_kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @property
    def _llm_type(self) -> str:
        return "qwen-local"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "enable_thinking": self.enable_thinking,
            "load_in_4bit": self.load_in_4bit,
        }
