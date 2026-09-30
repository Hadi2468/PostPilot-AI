import pytest
from pydantic import ValidationError

from postpilot.config import SCORE_WEIGHTS
from postpilot.schemas import Evaluation, ReviewDecision
from tests.conftest import make_evaluation


def test_score_weights_sum_to_one():
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def test_scores_are_clamped_to_0_10():
    e = Evaluation(
        hook_score=12, clarity_score=-3, engagement_score=5, originality_score=5,
        faithfulness_score=5, strengths=[], weaknesses=[], improvement_suggestions=[],
    )
    assert e.hook_score == 10.0
    assert e.clarity_score == 0.0


def test_overall_score_is_weighted_mean():
    e = Evaluation(
        hook_score=10, clarity_score=0, engagement_score=0, originality_score=0,
        faithfulness_score=0, strengths=[], weaknesses=[], improvement_suggestions=[],
    )
    assert e.overall_score == pytest.approx(10 * SCORE_WEIGHTS["hook_score"])


def test_overall_score_survives_serialisation_round_trip():
    e = make_evaluation(7.0)
    dumped = e.model_dump()
    assert dumped["overall_score"] == 7.0
    assert Evaluation.model_validate(dumped).overall_score == 7.0


def test_overall_score_is_not_requested_from_the_llm():
    # The LLM-facing schema must not ask the model to compute the overall score.
    assert "overall_score" not in Evaluation.model_json_schema()["properties"]


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "approve"},
        {"action": "reject"},
        {"action": "edit", "post": "Edited text"},
        {"action": "revise", "feedback": "Shorter hook"},
    ],
)
def test_valid_review_decisions(payload):
    assert ReviewDecision.model_validate(payload).action == payload["action"]


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "edit"},
        {"action": "edit", "post": "   "},
        {"action": "revise"},
        {"action": "publish"},
    ],
)
def test_invalid_review_decisions(payload):
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(payload)
