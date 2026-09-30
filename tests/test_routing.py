import pytest

from postpilot.config import LINKEDIN_CHAR_LIMIT
from postpilot.routing import is_acceptable, make_router
from tests.conftest import make_evaluation


def _state(score, iteration=0, faithfulness=None, post="short post", search_results=None):
    state = {
        "evaluation": make_evaluation(score, faithfulness).model_dump(),
        "draft_post": post,
        "iteration": iteration,
    }
    if search_results is not None:
        state["search_results"] = search_results
    return state


def test_accepts_high_quality_post(settings):
    assert make_router(settings)(_state(9.0)) == "review"


def test_first_revision_triggers_research(settings):
    assert make_router(settings)(_state(6.0)) == "research"


def test_later_revisions_reuse_cached_research(settings):
    results = [{"title": "t", "content": "c", "url": "u"}]
    assert make_router(settings)(_state(6.0, iteration=1, search_results=results)) == "improve"


def test_empty_search_results_still_count_as_cached(settings):
    # A failed search stores [] — we must not retry it on every iteration.
    assert make_router(settings)(_state(6.0, iteration=1, search_results=[])) == "improve"


def test_stops_when_iteration_budget_is_spent(settings):
    assert make_router(settings)(_state(2.0, iteration=settings.max_iterations)) == "review"


def test_unfaithful_post_is_rejected_even_with_high_overall(settings):
    evaluation = make_evaluation(10.0, faithfulness=settings.min_faithfulness - 1)
    assert evaluation.overall_score >= settings.score_threshold
    assert not is_acceptable(evaluation, "post", settings)


def test_post_over_linkedin_limit_is_rejected(settings):
    too_long = "x" * (LINKEDIN_CHAR_LIMIT + 1)
    assert not is_acceptable(make_evaluation(10.0), too_long, settings)


@pytest.mark.parametrize("threshold, expected", [(8.0, True), (8.01, False)])
def test_threshold_boundary(settings, threshold, expected):
    s = settings.model_copy(update={"score_threshold": threshold})
    assert is_acceptable(make_evaluation(8.0), "post", s) is expected
