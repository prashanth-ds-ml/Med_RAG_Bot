from __future__ import annotations

"""
gradio_app.py — Gradio chat UI for Med RAG Bot.

Run locally:
    python ui/gradio_app.py

For HuggingFace Spaces, this file must be named app.py at the repo root,
or referenced in README.md as the entry point.

Architecture:
    Gradio UI  →  ChatEngine  →  HybridRetriever + Qwen3-4B
    No FastAPI needed — Gradio and ChatEngine run in the same Python process.

Session state:
    Each browser tab gets its own gr.State() for history and session_id.
    The ChatEngine is loaded once at startup (shared across all tabs — not
    thread-safe; for multi-user production, wrap with a lock or use FastAPI).
"""

import os
import uuid
import logging
from pathlib import Path

import gradio as gr

from app.engine import ChatEngine, ChatResponse
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine — loaded once at startup
# ---------------------------------------------------------------------------

USE_RERANKER  = os.getenv("USE_RERANKER", "false").lower() == "true"
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "512"))

engine = ChatEngine(
    app_settings=settings,
    top_k=5,
    fetch_k=20,
    use_reranker=USE_RERANKER,
)

def _load_engine() -> str:
    """Load engine at startup. Returns status string for the UI."""
    try:
        engine.load()
        reranker_status = "re-ranker ON" if USE_RERANKER else "re-ranker OFF"
        return f"Model loaded · {reranker_status}"
    except Exception as e:
        logger.exception("Failed to load engine")
        return f"Load failed: {e}"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _new_session() -> tuple[str, list]:
    """Return a fresh (session_id, history) pair."""
    sid = engine.start_session(thinking_on=ENABLE_THINKING)
    return sid, []


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------

def chat_fn(
    message: str,
    chat_history: list[list[str | None]],
    session_id: str,
    history: list[tuple[str, str]],
    think_on: bool,
    thinking_budget: int,
) -> tuple[
    list[list[str | None]],  # updated chat_history for Chatbot
    str,                      # sources markdown
    str,                      # thinking text
    str,                      # follow-ups markdown
    str,                      # stats line
    str,                      # session_id (pass-through)
    list[tuple[str, str]],    # updated history
]:
    if not message.strip():
        return chat_history, "", "", "", "", session_id, history

    if not engine.is_loaded:
        chat_history.append([message, "Engine not loaded yet — please wait."])
        return chat_history, "", "", "", "", session_id, history

    turn = len(history) + 1
    budget: int | None = None if thinking_budget == 0 else thinking_budget

    try:
        resp: ChatResponse = engine.ask(
            message,
            session_id=session_id,
            turn=turn,
            history=history[-3:] if history else None,
            enable_thinking=think_on,
            thinking_budget=budget,
        )
    except Exception as e:
        logger.exception("Error during ask()")
        chat_history.append([message, f"Error: {e}"])
        return chat_history, "", "", "", "", session_id, history

    # Update conversation memory
    updated_history = history + [(message, resp.answer_text)]

    # Build Chatbot message pair
    chat_history.append([message, resp.answer_text])

    # Sources block
    sources_md = resp.citations_text if resp.citations_text.strip() else "_No sources cited._"

    # Thinking block
    thinking_md = (
        f"```\n{resp.thinking_text}\n```"
        if resp.thinking_text else "_Thinking not enabled or not available._"
    )

    # Follow-ups
    if resp.follow_ups:
        followups_md = "\n".join(f"- {q}" for q in resp.follow_ups)
    else:
        followups_md = "_No follow-up suggestions._"

    # Stats
    gen_s  = resp.generation_time_ms / 1000
    conf_emoji = {"HIGH": "🟢", "MED": "🟡", "LOW": "🔴"}.get(resp.confidence, "⚪")
    grounded = "✓ Grounded" if resp.grounded else "✗ Ungrounded"
    stats_md = (
        f"{grounded} · {conf_emoji} {resp.confidence} · "
        f"{resp.total_tokens:,} tokens · {gen_s:.1f}s · "
        f"`msg: {resp.message_id[:12]}`"
    )

    return (
        chat_history,
        sources_md,
        thinking_md,
        followups_md,
        stats_md,
        session_id,
        updated_history,
    )


def feedback_fn(
    rating: int,
    comment: str,
    session_id: str,
    chat_history: list[list[str | None]],
) -> str:
    """Submit feedback for the last message."""
    if not chat_history:
        return "No message to rate yet."
    # We don't store message_id in Gradio state for simplicity —
    # feedback is logged at the session level with the comment.
    success = engine.submit_feedback(
        message_id="last",     # simplified — full impl stores last_message_id in gr.State
        session_id=session_id,
        rating=rating,
        comment=comment,
    )
    return f"{'Saved' if success else 'Not saved (logging offline)'} — rating {rating}/5"


def clear_fn(
    session_id: str,
    turn_count: int,
) -> tuple[list, str, str, str, str, str, list]:
    """Clear conversation and start a new session."""
    engine.end_session(session_id=session_id, turn_count=turn_count)
    new_sid, new_history = _new_session()
    return [], "", "", "", "", new_sid, new_history


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Med RAG Bot",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
        .stats-box { font-size: 0.85em; color: #666; margin-top: 4px; }
        footer { display: none !important; }
    """,
) as demo:

    # ---- Header ----
    gr.Markdown(
        """
        # Med RAG Bot
        **Grounded medical QA over ICMR · NCDC · WHO · MOHFW documents**

        Ask any Indian public health question — answers are grounded with source citations
        and page numbers from real government guidelines.
        """
    )

    # ---- Session state ----
    session_id_state = gr.State(value="")
    history_state    = gr.State(value=[])

    with gr.Row():
        # ---- Left: Chat column ----
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
                show_copy_button=True,
                bubble_full_width=False,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a medical or public health question…",
                    show_label=False,
                    scale=9,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            stats_box = gr.Markdown(
                value="",
                elem_classes=["stats-box"],
            )

            with gr.Row():
                clear_btn = gr.Button("New Session", variant="secondary")

        # ---- Right: Info column ----
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Sources"):
                    sources_box = gr.Markdown(value="_Sources will appear here after your first question._")

                with gr.TabItem("Follow-ups"):
                    followups_box = gr.Markdown(value="_Follow-up suggestions will appear here._")

                with gr.TabItem("Thinking"):
                    thinking_box = gr.Markdown(value="_Internal reasoning will appear here (when thinking is enabled)._")

                with gr.TabItem("Feedback"):
                    gr.Markdown("Rate the last answer:")
                    rating_slider = gr.Slider(1, 5, value=3, step=1, label="Rating (1-5)")
                    comment_box   = gr.Textbox(label="Comment (optional)", placeholder="What was missing or wrong?")
                    submit_feedback_btn = gr.Button("Submit Feedback")
                    feedback_status = gr.Markdown("")

                with gr.TabItem("Settings"):
                    think_toggle   = gr.Checkbox(label="Enable thinking mode", value=ENABLE_THINKING)
                    budget_slider  = gr.Slider(
                        0, 2048, value=THINKING_BUDGET, step=128,
                        label="Thinking token budget (0 = unlimited)",
                        info="Lower = faster. 512 is good for most RAG queries.",
                    )
                    gr.Markdown(
                        "_Thinking mode makes the model reason before answering. "
                        "Produces better answers on complex queries but takes longer._"
                    )

    # ---- Status bar ----
    status_bar = gr.Markdown(value="⏳ Loading model…")

    # ---------------------------------------------------------------------------
    # Event wiring
    # ---------------------------------------------------------------------------

    # Initialize session on page load
    demo.load(
        fn=lambda: _new_session(),
        inputs=[],
        outputs=[session_id_state, history_state],
    )

    # Load engine status
    demo.load(
        fn=_load_engine,
        inputs=[],
        outputs=[status_bar],
    )

    # Send message — triggered by button or Enter
    send_inputs = [
        msg_box, chatbot, session_id_state, history_state,
        think_toggle, budget_slider,
    ]
    send_outputs = [
        chatbot, sources_box, thinking_box, followups_box,
        stats_box, session_id_state, history_state,
    ]

    send_btn.click(
        fn=chat_fn,
        inputs=send_inputs,
        outputs=send_outputs,
    ).then(fn=lambda: "", outputs=[msg_box])

    msg_box.submit(
        fn=chat_fn,
        inputs=send_inputs,
        outputs=send_outputs,
    ).then(fn=lambda: "", outputs=[msg_box])

    # Clear / new session
    clear_btn.click(
        fn=clear_fn,
        inputs=[session_id_state, history_state],
        outputs=[
            chatbot, sources_box, thinking_box,
            followups_box, stats_box, session_id_state, history_state,
        ],
    )

    # Feedback
    submit_feedback_btn.click(
        fn=feedback_fn,
        inputs=[rating_slider, comment_box, session_id_state, chatbot],
        outputs=[feedback_status],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # set True to get a public gradio.live URL
        show_error=True,
    )
