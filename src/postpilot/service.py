"""Run lifecycle: start a run, inspect it, and resume it after human review.

This is the single entry point used by the CLI, the FastAPI app, and (via the API)
the Streamlit dashboard, so all three behave identically.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from postpilot.config import Settings, get_settings
from postpilot.observability import run_config
from postpilot.schemas import ReviewDecision

RunStatus = Literal["awaiting_review", "approved", "edited", "rejected", "completed"]


class RunNotFoundError(KeyError):
    pass


class NotAwaitingReviewError(RuntimeError):
    pass


class RunResult(BaseModel):
    thread_id: str
    status: RunStatus
    post: str | None
    score: float | None
    evaluation: dict[str, Any] | None
    iteration: int
    history: list[dict[str, Any]]
    score_threshold: float


class PostPilotService:
    def __init__(self, graph: CompiledStateGraph, settings: Settings | None = None):
        self.graph = graph
        self.settings = settings or get_settings()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> PostPilotService:
        from postpilot.graph import build_default_graph

        settings = settings or get_settings()
        return cls(build_default_graph(settings), settings)

    def start(self, meeting_data: dict[str, Any], thread_id: str | None = None) -> RunResult:
        thread_id = thread_id or uuid.uuid4().hex
        self.graph.invoke({"meeting_data": meeting_data}, run_config(thread_id, self.settings))
        return self.get(thread_id)

    def resume(self, thread_id: str, decision: ReviewDecision) -> RunResult:
        if self.get(thread_id).status != "awaiting_review":
            raise NotAwaitingReviewError(thread_id)
        self.graph.invoke(
            Command(resume=decision.model_dump()), run_config(thread_id, self.settings)
        )
        return self.get(thread_id)

    def get(self, thread_id: str) -> RunResult:
        snapshot = self.graph.get_state(run_config(thread_id, self.settings))
        values = snapshot.values
        if not values:
            raise RunNotFoundError(thread_id)

        status: RunStatus = "awaiting_review" if snapshot.interrupts else values["status"]
        post = values.get("best_post") if status == "awaiting_review" else values.get("final_post")
        return RunResult(
            thread_id=thread_id,
            status=status,
            post=post,
            score=values.get("best_score"),
            evaluation=values.get("best_evaluation"),
            iteration=values.get("iteration", 0),
            history=values.get("history", []),
            score_threshold=self.settings.score_threshold,
        )
