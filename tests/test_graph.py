"""End-to-end graph behaviour in autonomous mode (no human review)."""

from postpilot.schemas import Evaluation, PostDraft
from tests.conftest import FakeLLM, FakeSearch

CONFIG = {"configurable": {"thread_id": "t1"}}


def test_accepts_first_draft_without_research(make_graph, settings, meeting_data):
    llm, search = FakeLLM(scores=[9.0]), FakeSearch()
    result = make_graph(llm, search, settings).invoke({"meeting_data": meeting_data}, CONFIG)

    assert result["final_post"] == "draft0"
    assert result["status"] == "completed"
    assert result["iteration"] == 0
    assert search.queries == []


def test_loop_stops_at_budget_and_returns_best_not_last(make_graph, settings, meeting_data):
    # Scores never pass the gate; the 2nd draft is the best, the last is the worst.
    llm, search = FakeLLM(scores=[5.0, 7.0, 6.0, 4.0]), FakeSearch()
    result = make_graph(llm, search, settings).invoke({"meeting_data": meeting_data}, CONFIG)

    assert result["iteration"] == settings.max_iterations
    assert [h["score"] for h in result["history"]] == [5.0, 7.0, 6.0, 4.0]
    assert result["final_post"] == "draft1"
    assert result["best_score"] == 7.0


def test_search_runs_once_across_revisions(make_graph, settings, meeting_data):
    llm, search = FakeLLM(scores=[5.0, 6.0, 7.0, 9.0]), FakeSearch()
    make_graph(llm, search, settings).invoke({"meeting_data": meeting_data}, CONFIG)

    assert len(search.queries) == 1
    assert "LLM evaluation" in search.queries[0]


def test_search_failure_degrades_gracefully(make_graph, settings, meeting_data):
    llm, search = FakeLLM(scores=[5.0, 9.0]), FakeSearch(fail=True)
    result = make_graph(llm, search, settings).invoke({"meeting_data": meeting_data}, CONFIG)

    assert result["search_results"] == []
    assert result["final_post"] == "draft1"


def test_critic_sees_source_insights_for_faithfulness(make_graph, settings, meeting_data):
    llm = FakeLLM(scores=[9.0])
    make_graph(llm, settings=settings).invoke({"meeting_data": meeting_data}, CONFIG)

    [critic_prompt] = llm.prompts_for(Evaluation)
    assert "Source meeting insights" in critic_prompt
    assert "LLM agents in production" in critic_prompt


def test_improve_prompt_includes_critique(make_graph, settings, meeting_data):
    llm = FakeLLM(scores=[5.0, 9.0])
    make_graph(llm, settings=settings).invoke({"meeting_data": meeting_data}, CONFIG)

    improve_prompt = llm.prompts_for(PostDraft)[-1]
    assert "- generic hook" in improve_prompt
    assert "- be more specific" in improve_prompt
