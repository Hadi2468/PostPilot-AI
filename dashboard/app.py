"""Streamlit dashboard: generate a post, inspect the critique, and approve / edit / revise it.

The dashboard is a thin client over the FastAPI service, so all agent state
(including paused human-review runs) lives server-side.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_meeting.json"
SERIES_COLOR = "#2a78d6"
MUTED_COLOR = "#8a8984"
REQUEST_TIMEOUT_S = 600  # a full reflexion loop can take a few minutes

STATUS_LABELS = {
    "awaiting_review": ("⏸️", "Awaiting your review"),
    "approved": ("✅", "Approved"),
    "edited": ("✏️", "Approved with your edits"),
    "rejected": ("🚫", "Rejected"),
    "completed": ("🤖", "Completed automatically"),
}
RUBRIC = [
    ("Hook", "hook_score"),
    ("Clarity", "clarity_score"),
    ("Engagement", "engagement_score"),
    ("Originality", "originality_score"),
    ("Faithfulness", "faithfulness_score"),
]

st.set_page_config(page_title="PostPilot AI", page_icon="✈️", layout="wide")


# ============================================================
# API client
# ============================================================

def api(method: str, path: str, **kwargs: Any) -> dict[str, Any] | None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.request(
            method, f"{api_url}{path}", headers=headers, timeout=REQUEST_TIMEOUT_S, **kwargs
        )
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {api_url}. Is it running?")
        return None
    if not resp.ok:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return None
    return resp.json()


# ============================================================
# Rendering helpers
# ============================================================

def score_chart(history: list[dict[str, Any]], threshold: float) -> alt.LayerChart:
    df = pd.DataFrame(history)[["iteration", "score"]]
    base = alt.Chart(df).encode(
        x=alt.X("iteration:O", title="Revision round", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("score:Q", title="Overall score", scale=alt.Scale(domain=[0, 10])),
        tooltip=[
            alt.Tooltip("iteration:O", title="Round"),
            alt.Tooltip("score:Q", title="Score", format=".2f"),
        ],
    )
    line = base.mark_line(color=SERIES_COLOR, strokeWidth=2)
    points = base.mark_point(color=SERIES_COLOR, filled=True, size=90, opacity=1)
    rule_df = pd.DataFrame({"y": [threshold], "label": [f"Acceptance threshold ({threshold})"]})
    rule = alt.Chart(rule_df).mark_rule(color=MUTED_COLOR, strokeDash=[4, 4]).encode(y="y:Q")
    label = (
        alt.Chart(rule_df)
        .mark_text(align="left", dx=4, dy=-6, color=MUTED_COLOR, fontSize=11)
        .encode(y="y:Q", text="label:N", x=alt.value(0))
    )
    return (rule + label + line + points).properties(height=260)


def render_run(run: dict[str, Any]) -> None:
    icon, label = STATUS_LABELS[run["status"]]
    st.subheader(f"{icon} {label}")

    evaluation = run.get("evaluation") or {}
    cols = st.columns(len(RUBRIC) + 1)
    cols[0].metric("Overall", f"{run['score']:.2f}" if run["score"] is not None else "—")
    for col, (name, key) in zip(cols[1:], RUBRIC, strict=True):
        col.metric(name, f"{evaluation.get(key, 0):.1f}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Post**")
        if run["status"] == "awaiting_review":
            st.text_area(
                "Edit before approving (optional)",
                value=run["post"] or "",
                height=380,
                key=f"edit_{run['thread_id']}_{len(run['history'])}",
                label_visibility="collapsed",
            )
        elif run["post"]:
            st.code(run["post"], language=None, wrap_lines=True)
        st.caption(f"{len(run['post'] or '')} / 3000 characters")

    with right:
        st.markdown("**Score by revision round**")
        st.altair_chart(score_chart(run["history"], run["score_threshold"]), width="stretch")
        with st.expander("Critique of this draft", expanded=True):
            for title, key in [
                ("Strengths", "strengths"),
                ("Weaknesses", "weaknesses"),
                ("Suggestions", "improvement_suggestions"),
            ]:
                st.markdown(f"**{title}**")
                st.markdown("\n".join(f"- {i}" for i in evaluation.get(key, [])) or "- (none)")


def render_review_controls(run: dict[str, Any]) -> None:
    thread_id = run["thread_id"]
    edited = st.session_state.get(f"edit_{thread_id}_{len(run['history'])}", run["post"])

    st.divider()
    st.markdown("### Your decision")
    c1, c2, c3 = st.columns(3)
    decision: dict[str, Any] | None = None

    if c1.button("✅ Approve", type="primary", width="stretch"):
        # Approving after an in-place edit is an "edit" decision.
        if edited == run["post"]:
            decision = {"action": "approve"}
        else:
            decision = {"action": "edit", "post": edited}
    if c2.button("🚫 Reject", width="stretch"):
        decision = {"action": "reject"}
    with c3.popover("🔁 Request revision", width="stretch"):
        feedback = st.text_area("What should change?", key=f"fb_{thread_id}")
        if st.button("Send to writer", disabled=not feedback.strip()):
            decision = {"action": "revise", "feedback": feedback}

    if decision:
        with st.spinner("Applying your decision…"):
            result = api("POST", f"/runs/{thread_id}/review", json=decision)
        if result:
            st.session_state.run = result
            st.rerun()


# ============================================================
# Page
# ============================================================

with st.sidebar:
    st.title("✈️ PostPilot AI")
    st.caption("Reflexion agent: generate → critique → research → revise → human review")
    api_url = st.text_input("API URL", os.getenv("POSTPILOT_API_URL", "http://localhost:8000"))
    token = st.text_input("API token", os.getenv("POSTPILOT_API_TOKEN", ""), type="password")
    health = api("GET", "/health")
    if health:
        st.success(f"API online · v{health['version']}")

st.header("Meeting summary → LinkedIn post")

source = st.radio("Input", ["Sample meeting", "Upload JSON"], horizontal=True)
meeting_data: dict[str, Any] | None = None
if source == "Sample meeting":
    meeting_data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
else:
    uploaded = st.file_uploader("Meeting summary (.json)", type="json")
    if uploaded:
        try:
            meeting_data = json.load(uploaded)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

if meeting_data:
    with st.expander("Preview input"):
        st.json(meeting_data, expanded=False)

if st.button("Generate post", type="primary", disabled=meeting_data is None):
    with st.spinner("Extracting insights, drafting, critiquing and revising…"):
        result = api("POST", "/runs", json={"meeting_data": meeting_data})
    if result:
        st.session_state.run = result

run = st.session_state.get("run")
if run:
    st.divider()
    render_run(run)
    if run["status"] == "awaiting_review":
        render_review_controls(run)
