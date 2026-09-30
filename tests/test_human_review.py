"""Human-in-the-loop: interrupt, then resume with each kind of decision."""

import pytest

from postpilot.schemas import PostDraft, ReviewDecision
from postpilot.service import NotAwaitingReviewError, PostPilotService, RunNotFoundError
from tests.conftest import FakeLLM


@pytest.fixture
def service_for(make_graph, review_settings):
    def _make(llm: FakeLLM) -> PostPilotService:
        return PostPilotService(make_graph(llm, settings=review_settings), review_settings)

    return _make


def test_run_pauses_for_review_with_best_draft(service_for, meeting_data):
    run = service_for(FakeLLM(scores=[9.0])).start(meeting_data)

    assert run.status == "awaiting_review"
    assert run.post == "draft0"
    assert run.score == 9.0
    assert run.evaluation["faithfulness_score"] == 9.0


def test_approve_finalises_best_draft(service_for, meeting_data):
    svc = service_for(FakeLLM(scores=[9.0]))
    run = svc.start(meeting_data)
    run = svc.resume(run.thread_id, ReviewDecision(action="approve"))

    assert run.status == "approved"
    assert run.post == "draft0"


def test_edit_uses_human_text(service_for, meeting_data):
    svc = service_for(FakeLLM(scores=[9.0]))
    run = svc.start(meeting_data)
    run = svc.resume(run.thread_id, ReviewDecision(action="edit", post="My own words"))

    assert run.status == "edited"
    assert run.post == "My own words"


def test_reject_ends_without_a_post(service_for, meeting_data):
    svc = service_for(FakeLLM(scores=[9.0]))
    run = svc.start(meeting_data)
    run = svc.resume(run.thread_id, ReviewDecision(action="reject"))

    assert run.status == "rejected"
    assert run.post is None


def test_revise_feeds_human_feedback_to_writer_then_pauses_again(service_for, meeting_data):
    llm = FakeLLM(scores=[9.0, 8.8])
    svc = service_for(llm)
    run = svc.start(meeting_data)

    run = svc.resume(
        run.thread_id, ReviewDecision(action="revise", feedback="Mention the eval harness")
    )

    improve_prompt = llm.prompts_for(PostDraft)[-1]
    assert "Mention the eval harness" in improve_prompt
    assert "Current post:\ndraft0" in improve_prompt
    # The revised draft becomes the new best even if it scores lower than the
    # pre-feedback draft: the human's feedback redefined what "good" means.
    assert run.status == "awaiting_review"
    assert run.post == "draft1"
    assert run.score == 8.8


def test_resume_after_final_decision_is_rejected(service_for, meeting_data):
    svc = service_for(FakeLLM(scores=[9.0]))
    run = svc.start(meeting_data)
    svc.resume(run.thread_id, ReviewDecision(action="approve"))

    with pytest.raises(NotAwaitingReviewError):
        svc.resume(run.thread_id, ReviewDecision(action="approve"))


def test_unknown_thread_raises(service_for):
    with pytest.raises(RunNotFoundError):
        service_for(FakeLLM()).get("does-not-exist")


def test_autonomous_mode_completes_without_interrupt(make_graph, settings, meeting_data):
    svc = PostPilotService(make_graph(FakeLLM(scores=[9.0]), settings=settings), settings)
    run = svc.start(meeting_data)

    assert run.status == "completed"
    assert run.post == "draft0"
