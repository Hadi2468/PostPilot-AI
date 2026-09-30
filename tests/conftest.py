"""Shared fixtures. Everything runs offline: no API keys, no network, no cost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from postpilot.config import Settings
from postpilot.graph import build_graph
from postpilot.schemas import ContentInsights, Evaluation, PostDraft

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_meeting.json"


def make_evaluation(score: float, faithfulness: float | None = None) -> Evaluation:
    return Evaluation(
        hook_score=score,
        clarity_score=score,
        engagement_score=score,
        originality_score=score,
        faithfulness_score=score if faithfulness is None else faithfulness,
        strengths=["clear"],
        weaknesses=["generic hook"],
        improvement_suggestions=["be more specific"],
    )


class FakeLLM:
    """Stands in for ChatOpenAI. Returns scripted structured outputs per schema.

    `scores` is consumed one per evaluation. Drafts are named draft0, draft1, ...
    Every prompt the LLM receives is recorded in `prompts` for assertions.
    """

    def __init__(self, scores: list[float] | None = None, post_prefix: str = "draft"):
        self.scores = list(scores or [])
        self.post_prefix = post_prefix
        self.revisions = 0
        self.prompts: list[tuple[type, str]] = []

    def with_structured_output(self, schema: type) -> RunnableLambda:
        def respond(prompt_value: Any) -> Any:
            self.prompts.append((schema, prompt_value.to_string()))
            if schema is ContentInsights:
                return ContentInsights(
                    title="LLM agents in production",
                    participants=["Maya", "Daniel"],
                    key_sections=["Evaluation", "Observability"],
                    key_summaries=["Evals gate releases.", "Tracing finds regressions."],
                    key_topics=["LLM evaluation", "agent observability", "guardrails"],
                    hooks=["Your agent demo works. Your agent in production doesn't."],
                )
            if schema is PostDraft:
                if "Current post:" in prompt_value.to_string():  # improve prompt
                    self.revisions += 1
                return PostDraft(post=f"{self.post_prefix}{self.revisions}")
            if schema is Evaluation:
                return make_evaluation(self.scores.pop(0))
            raise AssertionError(f"Unexpected schema {schema}")

        return RunnableLambda(respond)

    def prompts_for(self, schema: type) -> list[str]:
        return [p for s, p in self.prompts if s is schema]


class FakeSearch:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[dict[str, str]]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("search provider down")
        return [{"title": "Result", "content": "Some context", "url": "https://example.com"}]


@pytest.fixture
def meeting_data() -> dict[str, Any]:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def settings() -> Settings:
    # _env_file=None: tests must not depend on the developer's local .env
    return Settings(_env_file=None, human_review=False, score_threshold=8.5, max_iterations=3)


@pytest.fixture
def review_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"human_review": True})


@pytest.fixture
def make_graph():
    def _make(llm: FakeLLM, search: FakeSearch | None = None, settings: Settings | None = None):
        return build_graph(llm, llm, search or FakeSearch(), settings)

    return _make
