"""Conditional routing after each evaluation: accept, research, or revise."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from postpilot.config import LINKEDIN_CHAR_LIMIT, Settings
from postpilot.schemas import Evaluation
from postpilot.state import GraphState

Route = Literal["research", "improve", "review"]


def is_acceptable(evaluation: Evaluation, post: str, settings: Settings) -> bool:
    """Quality gate: good enough overall, faithful to the source, and fits LinkedIn."""
    return (
        evaluation.overall_score >= settings.score_threshold
        and evaluation.faithfulness_score >= settings.min_faithfulness
        and len(post) <= LINKEDIN_CHAR_LIMIT
    )


def make_router(settings: Settings) -> Callable[[GraphState], Route]:
    def route_after_evaluation(state: GraphState) -> Route:
        evaluation = Evaluation.model_validate(state["evaluation"])
        if is_acceptable(evaluation, state["draft_post"], settings):
            return "review"
        if state["iteration"] >= settings.max_iterations:
            return "review"
        # Research once, then reuse the cached results on later revisions.
        return "improve" if "search_results" in state else "research"

    return route_after_evaluation
