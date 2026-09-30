"""Pydantic schemas: structured LLM outputs and the human review contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from postpilot.config import SCORE_WEIGHTS


class ContentInsights(BaseModel):
    """What the extractor pulls out of raw meeting data."""

    title: str = Field(description="Short title of the meeting")
    participants: list[str]
    key_sections: list[str] = Field(description="Names of the main discussion sections")
    key_summaries: list[str] = Field(description="One-sentence summary per section")
    key_topics: list[str] = Field(
        description="3-5 short, search-friendly topic phrases (2-6 words each)"
    )
    hooks: list[str] = Field(description="2-3 strong opening hooks for a LinkedIn post")


class PostDraft(BaseModel):
    """A LinkedIn post produced by the writer (first draft or revision)."""

    post: str = Field(description="The full LinkedIn post text, ready to publish")


class Evaluation(BaseModel):
    """The critic's rubric scores. `overall_score` is computed in code, not by the LLM."""

    hook_score: float = Field(description="0-10: does the first line stop the scroll?")
    clarity_score: float = Field(description="0-10: is it easy to read and follow?")
    engagement_score: float = Field(description="0-10: will readers comment or share?")
    originality_score: float = Field(description="0-10: is it fresh rather than generic?")
    faithfulness_score: float = Field(
        description="0-10: are ALL claims supported by the meeting insights? "
        "Penalise invented numbers, names, or outcomes heavily."
    )
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]

    @field_validator(*SCORE_WEIGHTS, mode="before")
    @classmethod
    def _clamp_score(cls, v: Any) -> float:
        return max(0.0, min(10.0, float(v)))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall_score(self) -> float:
        # LLMs are unreliable at arithmetic and tend to anchor an "overall" score
        # independently of the sub-scores, so we compute it deterministically.
        return round(sum(getattr(self, k) * w for k, w in SCORE_WEIGHTS.items()), 2)


class ReviewDecision(BaseModel):
    """What a human reviewer sends back when the graph pauses for approval."""

    action: Literal["approve", "edit", "revise", "reject"]
    post: str | None = Field(None, description="Edited post text (required for 'edit')")
    feedback: str | None = Field(None, description="Revision instructions (required for 'revise')")

    @model_validator(mode="after")
    def _check_payload(self) -> ReviewDecision:
        if self.action == "edit" and not (self.post and self.post.strip()):
            raise ValueError("'edit' requires a non-empty 'post'")
        if self.action == "revise" and not (self.feedback and self.feedback.strip()):
            raise ValueError("'revise' requires non-empty 'feedback'")
        return self
