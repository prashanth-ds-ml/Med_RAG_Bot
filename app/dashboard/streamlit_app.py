"""
Med360 RAG — Observability Dashboard
=====================================
Run with:
    streamlit run app/dashboard/streamlit_app.py

Reads from data/logs/*.jsonl (written by the chat command).
No MongoDB required — works entirely from local files.
Refreshes every 30 seconds automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Med360 RAG — Observability",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Resolve log file paths without importing the full app stack
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGS_DIR     = _PROJECT_ROOT / "data" / "logs"

CHAT_SESSIONS_PATH  = _LOGS_DIR / "chat_sessions.jsonl"
MESSAGES_PATH       = _LOGS_DIR / "messages.jsonl"
RETRIEVAL_LOGS_PATH = _LOGS_DIR / "retrieval_logs.jsonl"
FEEDBACK_PATH       = _LOGS_DIR / "feedback.jsonl"


# ---------------------------------------------------------------------------
# Data loaders  (cached — auto-refresh every 30 s)
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(orjson.loads(line))
                except Exception:
                    pass
    return records


@st.cache_data(ttl=30)
def load_sessions() -> pd.DataFrame:
    rows = _read_jsonl(CHAT_SESSIONS_PATH)
    # Merge session_end events into their start record
    starts = {r["session_id"]: r for r in rows if r.get("_event") != "session_end"}
    for r in rows:
        if r.get("_event") == "session_end":
            sid = r["session_id"]
            if sid in starts:
                starts[sid]["ended_at"]   = r.get("ended_at")
                starts[sid]["turn_count"] = r.get("turn_count", 0)
    df = pd.DataFrame(list(starts.values()))
    if df.empty:
        return df
    for col in ("started_at", "ended_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


@st.cache_data(ttl=30)
def load_messages() -> pd.DataFrame:
    rows = _read_jsonl(MESSAGES_PATH)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "generation_time_s" not in df.columns and "generation_time_ms" in df.columns:
        df["generation_time_s"] = df["generation_time_ms"] / 1000
    return df


@st.cache_data(ttl=30)
def load_retrieval() -> pd.DataFrame:
    """Explode retrieval_logs into one row per chunk for easy analysis."""
    rows = _read_jsonl(RETRIEVAL_LOGS_PATH)
    if not rows:
        return pd.DataFrame()
    chunk_rows: list[dict] = []
    for r in rows:
        for chunk in r.get("chunks", []):
            chunk_rows.append(
                {
                    "message_id": r["message_id"],
                    "session_id": r["session_id"],
                    "query":      r["query"],
                    "created_at": r["created_at"],
                    **chunk,
                }
            )
    df = pd.DataFrame(chunk_rows)
    if not df.empty and "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return df


@st.cache_data(ttl=30)
def load_feedback() -> pd.DataFrame:
    rows = _read_jsonl(FEEDBACK_PATH)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _no_data(label: str = "No data yet") -> None:
    st.info(f"**{label}** — start a chat session to populate this view.", icon="💡")


def _metric_row(metrics: dict[str, Any]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.header("Overview")

    sessions = load_sessions()
    messages = load_messages()
    feedback = load_feedback()

    if messages.empty:
        _no_data("No chat data yet")
        return

    # --- KPI row ---
    total_sessions = len(sessions)
    total_queries  = len(messages)
    avg_tokens     = int(messages["total_tokens"].mean()) if "total_tokens" in messages.columns else 0
    avg_gen_s      = round(messages["generation_time_s"].mean(), 1) if "generation_time_s" in messages.columns else 0
    grounded_pct   = (
        f"{100 * messages['grounded'].mean():.0f}%"
        if "grounded" in messages.columns else "—"
    )
    avg_rating = (
        f"{feedback['rating'].mean():.1f} / 5"
        if not feedback.empty and "rating" in feedback.columns else "—"
    )

    _metric_row({
        "Sessions":       total_sessions,
        "Total Queries":  total_queries,
        "Avg Tokens":     avg_tokens,
        "Avg Gen Time":   f"{avg_gen_s}s",
        "Grounded":       grounded_pct,
        "Avg Rating":     avg_rating,
    })

    st.divider()

    col1, col2 = st.columns(2)

    # --- Query type distribution ---
    with col1:
        st.subheader("Query Type Distribution")
        if "query_type" in messages.columns:
            counts = messages["query_type"].value_counts().reset_index()
            counts.columns = ["Query Type", "Count"]
            st.bar_chart(counts.set_index("Query Type"))
        else:
            _no_data()

    # --- Grounded vs ungrounded ---
    with col2:
        st.subheader("Grounded Answers")
        if "grounded" in messages.columns:
            grounded_counts = messages["grounded"].map(
                {True: "Grounded", False: "Ungrounded"}
            ).value_counts().reset_index()
            grounded_counts.columns = ["Status", "Count"]
            st.bar_chart(grounded_counts.set_index("Status"))
        else:
            _no_data()

    st.divider()

    # --- Queries over time ---
    st.subheader("Queries Over Time")
    if "created_at" in messages.columns and not messages["created_at"].isna().all():
        time_df = (
            messages.set_index("created_at")
            .resample("1h")
            .size()
            .reset_index(name="Queries")
            .rename(columns={"created_at": "Hour"})
        )
        st.line_chart(time_df.set_index("Hour"))
    else:
        _no_data()


# ---------------------------------------------------------------------------
# Page: Messages
# ---------------------------------------------------------------------------

def page_messages() -> None:
    st.header("Message Log")

    messages = load_messages()
    if messages.empty:
        _no_data()
        return

    # --- Filters ---
    col1, col2, col3 = st.columns(3)
    with col1:
        qt_options = ["All"] + sorted(messages["query_type"].dropna().unique().tolist()) if "query_type" in messages.columns else ["All"]
        qt_filter = st.selectbox("Query Type", qt_options)
    with col2:
        grounded_options = ["All", "Grounded", "Ungrounded"]
        grounded_filter = st.selectbox("Grounded", grounded_options)
    with col3:
        search = st.text_input("Search queries", placeholder="keyword…")

    df = messages.copy()
    if qt_filter != "All" and "query_type" in df.columns:
        df = df[df["query_type"] == qt_filter]
    if grounded_filter == "Grounded" and "grounded" in df.columns:
        df = df[df["grounded"] == True]
    elif grounded_filter == "Ungrounded" and "grounded" in df.columns:
        df = df[df["grounded"] == False]
    if search and "query" in df.columns:
        df = df[df["query"].str.contains(search, case=False, na=False)]

    st.caption(f"Showing {len(df)} of {len(messages)} messages")

    # --- Table ---
    display_cols = [c for c in ["created_at", "query_type", "grounded", "total_tokens",
                                 "generation_time_s", "query", "message_id"] if c in df.columns]
    if not df.empty:
        st.dataframe(
            df[display_cols].sort_values("created_at", ascending=False) if "created_at" in df.columns else df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "created_at":       st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                "query_type":       st.column_config.TextColumn("Type", width="small"),
                "grounded":         st.column_config.CheckboxColumn("Grounded", width="small"),
                "total_tokens":     st.column_config.NumberColumn("Tokens", width="small"),
                "generation_time_s": st.column_config.NumberColumn("Time (s)", format="%.1f", width="small"),
                "query":            st.column_config.TextColumn("Query", width="large"),
                "message_id":       st.column_config.TextColumn("Msg ID", width="medium"),
            },
        )

    # --- Expandable detail ---
    st.divider()
    st.subheader("Message Detail")
    if not df.empty and "message_id" in df.columns:
        msg_ids = df["message_id"].tolist()
        selected_id = st.selectbox("Select message ID", msg_ids, format_func=lambda x: x[:16])
        row = df[df["message_id"] == selected_id].iloc[0]
        st.markdown(f"**Query:** {row.get('query', '')}")
        st.markdown(f"**Type:** `{row.get('query_type', '')}` | **Grounded:** {row.get('grounded', '')} | "
                    f"**Tokens:** {row.get('total_tokens', '')} | **Time:** {row.get('generation_time_s', 0):.1f}s")
        st.markdown("**Answer:**")
        st.markdown(row.get("answer_text", ""))
        if row.get("thinking_text"):
            with st.expander("Thinking (internal reasoning)"):
                st.text(row["thinking_text"])
        if row.get("citations"):
            with st.expander("Citations"):
                for c in row["citations"]:
                    st.markdown(f"**[{c.get('index')}]** {c.get('description', '')}  \n{c.get('url', '')}")


# ---------------------------------------------------------------------------
# Page: Retrieval Inspector
# ---------------------------------------------------------------------------

def page_retrieval() -> None:
    st.header("Retrieval Inspector")

    retrieval = load_retrieval()
    if retrieval.empty:
        _no_data()
        return

    col1, col2 = st.columns(2)

    # --- Source distribution ---
    with col1:
        st.subheader("Source Distribution")
        if "source_name" in retrieval.columns:
            src = retrieval["source_name"].str.upper().value_counts().reset_index()
            src.columns = ["Source", "Chunk Hits"]
            st.bar_chart(src.set_index("Source"))

    # --- Doc type distribution ---
    with col2:
        st.subheader("Doc Type Distribution")
        if "doc_type" in retrieval.columns:
            dt = retrieval["doc_type"].str.replace("_", " ").value_counts().reset_index()
            dt.columns = ["Doc Type", "Count"]
            st.bar_chart(dt.set_index("Doc Type"))

    st.divider()

    col3, col4 = st.columns(2)

    # --- Avg fused score by source ---
    with col3:
        st.subheader("Avg Fused Score by Source")
        if "source_name" in retrieval.columns and "fused_score" in retrieval.columns:
            score_df = (
                retrieval.groupby(retrieval["source_name"].str.upper())["fused_score"]
                .mean()
                .reset_index()
            )
            score_df.columns = ["Source", "Avg Fused Score"]
            st.bar_chart(score_df.set_index("Source"))

    # --- BM25 vs Vector rank correlation ---
    with col4:
        st.subheader("BM25 vs Vector Rank")
        if "bm25_rank" in retrieval.columns and "vector_rank" in retrieval.columns:
            rank_df = retrieval[["bm25_rank", "vector_rank"]].dropna()
            if not rank_df.empty:
                # Count how often BM25/vector rank agree (both ≤ 5)
                both_top = ((rank_df["bm25_rank"] <= 5) & (rank_df["vector_rank"] <= 5)).sum()
                only_bm25 = ((rank_df["bm25_rank"] <= 5) & (rank_df["vector_rank"] > 5)).sum()
                only_vec  = ((rank_df["bm25_rank"] > 5) & (rank_df["vector_rank"] <= 5)).sum()
                agreement_df = pd.DataFrame({
                    "Category": ["Both top-5", "BM25 only top-5", "Vector only top-5"],
                    "Count":    [both_top, only_bm25, only_vec],
                })
                st.bar_chart(agreement_df.set_index("Category"))

    st.divider()

    # --- Most cited pages ---
    st.subheader("Most Cited Pages")
    if all(c in retrieval.columns for c in ["source_name", "doc_type", "page_num", "pdf_url"]):
        cited = (
            retrieval.groupby(["source_name", "doc_type", "page_num", "pdf_url"])
            .size()
            .reset_index(name="Times Retrieved")
            .sort_values("Times Retrieved", ascending=False)
            .head(20)
        )
        st.dataframe(cited, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Feedback
# ---------------------------------------------------------------------------

def page_feedback() -> None:
    st.header("Feedback Tracker")

    feedback = load_feedback()
    messages = load_messages()

    if feedback.empty:
        _no_data("No feedback yet — use /feedback in the chat to rate answers")
        return

    # --- KPIs ---
    avg_rating = round(feedback["rating"].mean(), 2) if "rating" in feedback.columns else 0
    total      = len(feedback)
    low_count  = len(feedback[feedback["rating"] <= 2]) if "rating" in feedback.columns else 0

    _metric_row({
        "Total Ratings":  total,
        "Avg Rating":     f"{avg_rating} / 5",
        "Low Ratings (≤2)": low_count,
    })

    st.divider()
    col1, col2 = st.columns(2)

    # --- Rating distribution ---
    with col1:
        st.subheader("Rating Distribution")
        if "rating" in feedback.columns:
            dist = feedback["rating"].value_counts().sort_index().reset_index()
            dist.columns = ["Rating", "Count"]
            dist["Rating"] = dist["Rating"].astype(str) + " ★"
            st.bar_chart(dist.set_index("Rating"))

    # --- Ratings over time ---
    with col2:
        st.subheader("Ratings Over Time")
        if "created_at" in feedback.columns and not feedback["created_at"].isna().all():
            time_fb = feedback.set_index("created_at")["rating"].resample("1h").mean().reset_index()
            time_fb.columns = ["Hour", "Avg Rating"]
            st.line_chart(time_fb.set_index("Hour"))

    st.divider()

    # --- Low-rated answers (1–2 stars) — needs improvement ---
    st.subheader("Low-Rated Answers (1–2 stars)")
    if "rating" in feedback.columns:
        low = feedback[feedback["rating"] <= 2].copy()
        if low.empty:
            st.success("No low-rated answers yet.")
        else:
            for _, row in low.iterrows():
                mid = row.get("message_id", "")
                comment = row.get("comment", "")
                rating = row.get("rating", "?")
                # Try to find the original query
                query = ""
                if not messages.empty and "message_id" in messages.columns:
                    match = messages[messages["message_id"] == mid]
                    if not match.empty:
                        query = match.iloc[0].get("query", "")
                with st.expander(f"★{rating} — {query[:80] or mid[:16]}"):
                    st.markdown(f"**Message ID:** `{mid}`")
                    if query:
                        st.markdown(f"**Query:** {query}")
                    if comment:
                        st.markdown(f"**Comment:** {comment}")
                    if not messages.empty and "message_id" in messages.columns:
                        match = messages[messages["message_id"] == mid]
                        if not match.empty:
                            st.markdown("**Answer:**")
                            st.markdown(match.iloc[0].get("answer_text", ""))

    st.divider()

    # --- All feedback table ---
    st.subheader("All Feedback")
    display_cols = [c for c in ["created_at", "rating", "comment", "tier", "message_id"] if c in feedback.columns]
    st.dataframe(
        feedback[display_cols].sort_values("created_at", ascending=False) if "created_at" in feedback.columns else feedback[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
            "rating":     st.column_config.NumberColumn("Rating", format="%d ★"),
            "comment":    st.column_config.TextColumn("Comment", width="large"),
            "tier":       st.column_config.TextColumn("Tier", width="small"),
            "message_id": st.column_config.TextColumn("Msg ID", width="medium"),
        },
    )


# ---------------------------------------------------------------------------
# Page: Performance
# ---------------------------------------------------------------------------

# Cloud API pricing (per 1M tokens, approximate mid-2025 rates)
# Add new models here as needed — the cost table picks them up automatically.
_CLOUD_PRICING: dict[str, dict[str, float]] = {
    "GPT-4o":             {"input": 2.50,  "output": 10.00},
    "GPT-4o mini":        {"input": 0.15,  "output": 0.60},
    "Claude Opus 4.6":    {"input": 15.00, "output": 75.00},
    "Claude Sonnet 4.6":  {"input": 3.00,  "output": 15.00},
    "Claude Haiku 4.5":   {"input": 0.80,  "output": 4.00},
    "Gemini 2.0 Flash":   {"input": 0.10,  "output": 0.40},
    "Gemini 1.5 Flash":   {"input": 0.075, "output": 0.30},
}

# When you switch to a real API, set the model key here and actual spend
# will be tracked automatically in the "Current API" row.
_CURRENT_API_MODEL: str | None = None  # e.g. "Claude Sonnet 4.6"


def page_performance() -> None:
    st.header("Performance & Cost")

    messages = load_messages()
    if messages.empty:
        _no_data()
        return

    df = messages.copy()

    # Derived columns
    if "generation_time_ms" in df.columns and "completion_tokens" in df.columns:
        df["gen_time_s"]     = df["generation_time_ms"] / 1000
        df["tokens_per_sec"] = df.apply(
            lambda r: r["completion_tokens"] / r["gen_time_s"]
            if r["gen_time_s"] > 0 else 0,
            axis=1,
        )

    # ---- Latency KPIs ----
    st.subheader("Latency")
    if "gen_time_s" in df.columns:
        lat = df["gen_time_s"]
        _metric_row({
            "Avg Latency":  f"{lat.mean():.1f}s",
            "p50":          f"{lat.quantile(0.50):.1f}s",
            "p95":          f"{lat.quantile(0.95):.1f}s",
            "p99":          f"{lat.quantile(0.99):.1f}s",
            "Min":          f"{lat.min():.1f}s",
            "Max":          f"{lat.max():.1f}s",
        })

        st.caption("Generation time per query")
        time_df = df[["created_at", "gen_time_s"]].dropna().set_index("created_at").sort_index()
        st.line_chart(time_df.rename(columns={"gen_time_s": "Generation time (s)"}))
    else:
        _no_data("No latency data")

    st.divider()

    # ---- Throughput ----
    st.subheader("Throughput  (tokens / sec)")
    if "tokens_per_sec" in df.columns:
        tps = df["tokens_per_sec"]
        _metric_row({
            "Avg Tokens/sec":  f"{tps.mean():.1f}",
            "p50":             f"{tps.quantile(0.50):.1f}",
            "p95":             f"{tps.quantile(0.95):.1f}",
            "Total Tokens":    int(df["total_tokens"].sum()) if "total_tokens" in df.columns else "—",
        })

        st.caption("Completion tokens / generation time per query")
        tps_df = df[["created_at", "tokens_per_sec"]].dropna().set_index("created_at").sort_index()
        st.line_chart(tps_df.rename(columns={"tokens_per_sec": "Tokens/sec"}))
    else:
        _no_data("No throughput data")

    st.divider()

    # ---- Token breakdown ----
    st.subheader("Token Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        if all(c in df.columns for c in ["prompt_tokens", "completion_tokens"]):
            avg_prompt     = int(df["prompt_tokens"].mean())
            avg_completion = int(df["completion_tokens"].mean())
            total_prompt   = int(df["prompt_tokens"].sum())
            total_comp     = int(df["completion_tokens"].sum())
            token_breakdown = pd.DataFrame({
                "Type":  ["Prompt", "Completion"],
                "Avg":   [avg_prompt, avg_completion],
                "Total": [total_prompt, total_comp],
            })
            st.dataframe(token_breakdown, hide_index=True, use_container_width=True)

    with col2:
        if "query_type" in df.columns and "total_tokens" in df.columns:
            st.caption("Avg tokens by query type")
            qt_tokens = df.groupby("query_type")["total_tokens"].mean().reset_index()
            qt_tokens.columns = ["Query Type", "Avg Tokens"]
            st.bar_chart(qt_tokens.set_index("Query Type"))

    st.divider()

    # ---- Cumulative token totals ----
    st.subheader("Cumulative Token Usage (all-time)")
    if all(c in df.columns for c in ["prompt_tokens", "completion_tokens"]):
        total_prompt = int(df["prompt_tokens"].sum())
        total_comp   = int(df["completion_tokens"].sum())
        total_all    = total_prompt + total_comp
        avg_prompt   = int(df["prompt_tokens"].mean())
        avg_comp     = int(df["completion_tokens"].mean())

        _metric_row({
            "Total Prompt Tokens":     f"{total_prompt:,}",
            "Total Completion Tokens": f"{total_comp:,}",
            "Grand Total Tokens":      f"{total_all:,}",
            "Avg Prompt / Query":      f"{avg_prompt:,}",
            "Avg Completion / Query":  f"{avg_comp:,}",
            "Queries Logged":          f"{len(df):,}",
        })
    else:
        _no_data("No token data")

    st.divider()

    # ---- Cloud cost estimator ----
    st.subheader("Cloud Cost Estimator")
    st.caption(
        "What these queries **would have cost** on cloud APIs — both cumulative (all-time) "
        "and per-query average. Your local Qwen3-4B costs ₹0 / $0."
    )

    # INR rate comes from the sidebar widget stored in session_state
    usd_inr: float = st.session_state.get("usd_inr", 84.0)

    if all(c in df.columns for c in ["prompt_tokens", "completion_tokens"]):
        total_prompt_m = df["prompt_tokens"].sum() / 1_000_000
        total_comp_m   = df["completion_tokens"].sum() / 1_000_000
        avg_prompt_m   = df["prompt_tokens"].mean() / 1_000_000
        avg_comp_m     = df["completion_tokens"].mean() / 1_000_000

        rows = []
        for model, prices in _CLOUD_PRICING.items():
            cumul_usd      = total_prompt_m * prices["input"] + total_comp_m   * prices["output"]
            per_query_usd  = avg_prompt_m   * prices["input"] + avg_comp_m     * prices["output"]
            is_current     = (model == _CURRENT_API_MODEL)
            rows.append({
                "Model":                model + (" ◀ active" if is_current else ""),
                "Input $/1M":           f"${prices['input']:.3f}",
                "Output $/1M":          f"${prices['output']:.2f}",
                "Total (USD)":          f"${cumul_usd:.4f}",
                "Total (INR ₹)":        f"₹{cumul_usd * usd_inr:.2f}",
                "Per Query (USD)":      f"${per_query_usd:.6f}",
                "Per Query (INR ₹)":    f"₹{per_query_usd * usd_inr:.4f}",
            })

        rows.append({
            "Model":             "Qwen3-4B (local)" + (" ◀ active" if _CURRENT_API_MODEL is None else ""),
            "Input $/1M":        "$0.000",
            "Output $/1M":       "$0.00",
            "Total (USD)":       "$0.0000",
            "Total (INR ₹)":     "₹0.00",
            "Per Query (USD)":   "$0.000000",
            "Per Query (INR ₹)": "₹0.0000",
        })

        cost_df = pd.DataFrame(rows)
        st.dataframe(cost_df, hide_index=True, use_container_width=True)

        st.caption(
            f"Based on **{int(df['prompt_tokens'].sum()):,}** prompt + "
            f"**{int(df['completion_tokens'].sum()):,}** completion = "
            f"**{int(df['prompt_tokens'].sum()) + int(df['completion_tokens'].sum()):,} total tokens** "
            f"across **{len(df):,}** queries. "
            f"Exchange rate: $1 = ₹{usd_inr:.2f}"
        )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Med360 RAG")
st.sidebar.caption("Observability Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Messages", "Retrieval", "Performance", "Feedback"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(f"Logs: `{_LOGS_DIR}`")

# Log file sizes as quick health check
for label, path in [
    ("chat_sessions", CHAT_SESSIONS_PATH),
    ("messages",      MESSAGES_PATH),
    ("retrieval",     RETRIEVAL_LOGS_PATH),
    ("feedback",      FEEDBACK_PATH),
]:
    if path.exists():
        lines = sum(1 for _ in open(path, "rb"))
        st.sidebar.caption(f"{label}: {lines} records")
    else:
        st.sidebar.caption(f"{label}: no file")

st.sidebar.divider()
st.sidebar.caption("Cost Settings")
usd_inr_input = st.sidebar.number_input(
    "USD → INR rate",
    min_value=1.0,
    max_value=200.0,
    value=st.session_state.get("usd_inr", 84.0),
    step=0.5,
    help="Used to convert cloud API cost estimates to Indian Rupees.",
)
st.session_state["usd_inr"] = usd_inr_input

st.sidebar.divider()
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Route to selected page
# ---------------------------------------------------------------------------

if page == "Overview":
    page_overview()
elif page == "Messages":
    page_messages()
elif page == "Retrieval":
    page_retrieval()
elif page == "Performance":
    page_performance()
elif page == "Feedback":
    page_feedback()
