from __future__ import annotations

"""
prompt_builder.py — System prompt and context formatting for grounded medical QA.

Design goals:
  1. Answer ONLY from retrieved context — never hallucinate
  2. Cite every factual claim with [N]
  3. Detect query type and apply the matching response template
  4. Flag clearly when context is insufficient

Query types and their response templates:
  FACTUAL       "What is dengue?"
                → Clear answer paragraph(s) + Sources

  GUIDELINE     "What is the treatment for TB?"
                → Indication | Recommendation | Dosage/Regimen | Cautions | Sources

  PROCEDURE     "How to collect a blood sample for malaria diagnosis?"
                → Numbered step-by-step | Notes | Sources

  COMPARISON    "Difference between Type 1 and Type 2 diabetes?"
                → Parallel structure or table | Sources

  STATISTICS    "TB case burden in India 2023?"
                → Key figures | Context | Data period | Sources

  MULTI_ASPECT  "Explain malaria prevention and treatment"
                → Separate headed sections | Sources per section

  INSUFFICIENT  (no relevant context found)
                → Honest statement + what was found

The system prompt instructs Qwen to identify the query type first, then
use the matching template — no external classifier needed.
"""

from typing import Any


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical information assistant for Indian public health. \
You answer questions using ONLY the provided source documents from ICMR, NCDC, WHO, and MOHFW.

STRICT RULES:
1. Use ONLY the provided context. Never use outside knowledge.
2. Cite every factual claim with [N] where N is the source number.
3. If context is insufficient, say so explicitly — do not guess or fill gaps.
4. Be precise. Do not over-summarise or pad answers.

STEP 1 — Identify the query type (choose exactly one):
  FACTUAL      – definition, explanation, or factual question
  GUIDELINE    – treatment, diagnosis, or clinical recommendation query
  PROCEDURE    – how-to, step-by-step, or protocol query
  COMPARISON   – comparing two or more conditions, drugs, or approaches
  STATISTICS   – data, numbers, case counts, prevalence, or trends
  MULTI_ASPECT – query spans multiple distinct subtopics
  INSUFFICIENT – context does not contain enough information to answer

STEP 2 — Use the matching response template:

FACTUAL:
  Write 1-3 clear paragraphs. Cite every claim with [N].
  Sources: [numbered list]

GUIDELINE:
  **Indication:** [when this applies] [citation]
  **Recommendation:** [what to do] [citation]
  **Regimen / Dosage:** [specifics if available] [citation]
  **Cautions / Contraindications:** [if mentioned] [citation]
  Sources: [numbered list]

PROCEDURE:
  **Steps:**
  1. [step] [citation]
  2. [step] [citation]
  ...
  **Notes:** [any important caveats] [citation]
  Sources: [numbered list]

COMPARISON:
  | Aspect | [Item A] | [Item B] |
  |--------|----------|----------|
  | [row]  | [detail] | [detail] |
  ...
  [Brief summary paragraph] [citations]
  Sources: [numbered list]

STATISTICS:
  **Key figures:** [data points with inline citations]
  **Context:** [what the data means]
  **Data period:** [year/period of the data]
  Sources: [numbered list]

MULTI_ASPECT:
  ## [Subtopic 1]
  [answer with citations]

  ## [Subtopic 2]
  [answer with citations]
  ...
  Sources: [numbered list]

INSUFFICIENT:
  The available documents do not contain sufficient information to answer this query.
  [Describe what was found, if anything relevant was retrieved.]
  Sources reviewed: [numbered list of what was checked]

SOURCES FORMAT (always at the end):
Sources:
[1] [Document title or description] — [Source name] | [Doc type], Page [N]
    [PDF URL]
[2] ...

FOLLOW-UP QUESTIONS (always after Sources):
Add exactly 3 brief follow-up questions the user might want to ask next.
**Follow-up questions:**
- [question]
- [question]
- [question]

Start your response by stating the query type on the first line:
Query type: [TYPE]

Then give the response in the matching template."""


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------

def build_context_block(results: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks as a numbered source list for the prompt.

    Format per source:
        [N] Source: ICMR | guideline | Page 12
            URL: https://...
            Content:
            <chunk text>

    The [N] numbers match the citation markers the model will use in its response.
    """
    if not results:
        return "No relevant context found."

    parts: list[str] = []
    for i, item in enumerate(results, start=1):
        record = item.get("record", {})
        metadata = record.get("metadata", {})

        source_name = metadata.get("source_name", "unknown").upper()
        doc_type = metadata.get("doc_type", "").replace("_", " ")
        page_num = metadata.get("page_num", "?")
        pdf_url = metadata.get("pdf_url", "")
        chunk_text = record.get("chunk_text", item.get("chunk_text", "")).strip()

        source_line = f"Source: {source_name} | {doc_type} | Page {page_num}"
        url_line = f"URL: {pdf_url}" if pdf_url else "URL: not available"

        parts.append(
            f"[{i}] {source_line}\n"
            f"    {url_line}\n"
            f"    Content:\n"
            f"    {chunk_text}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Full prompt builder
# ---------------------------------------------------------------------------

def build_messages(
    query: str,
    results: list[dict[str, Any]],
    *,
    history: list[tuple[str, str]] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """
    Build the full chat messages list for the LLM.

    Args:
        history: Previous (user_query, assistant_answer) pairs for conversation
                 memory. Only the clean Q&A is passed — context blocks are not
                 repeated for prior turns to keep token count manageable.
                 Pass the last 3 turns maximum.

    Returns:
        [system, (user, assistant)*, user]  — standard multi-turn chat format
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Inject prior turns as plain Q&A (no context blocks — saves tokens)
    if history:
        for prev_query, prev_answer in history:
            messages.append({"role": "user",      "content": prev_query})
            messages.append({"role": "assistant", "content": prev_answer})

    context_block = build_context_block(results)
    user_message = (
        f"CONTEXT:\n"
        f"{context_block}\n\n"
        f"QUESTION:\n{query}"
    )
    messages.append({"role": "user", "content": user_message})

    return messages


def build_grounded_prompt(query: str, results: list[dict[str, Any]]) -> str:
    """
    Legacy helper — returns the user message as a single string.
    Kept for backwards compatibility with existing tests.
    """
    messages = build_messages(query, results)
    return messages[-1]["content"]
