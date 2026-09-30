"""The shared state that flows through the LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

FinalStatus = Literal["approved", "edited", "rejected", "completed"]


class GraphState(TypedDict, total=False):
    # Input
    meeting_data: dict[str, Any]

    # Working memory
    insights: dict[str, Any]
    draft_post: str
    evaluation: dict[str, Any]
    search_results: list[dict[str, str]]  # fetched once, then reused across revisions
    iteration: int
    human_feedback: str

    # Best-so-far tracking: a revision can make the post worse
    best_post: str
    best_score: float
    best_evaluation: dict[str, Any]

    # Output
    final_post: str
    status: FinalStatus

    # Append-only audit trail of every evaluated draft
    history: Annotated[list[dict[str, Any]], operator.add]
