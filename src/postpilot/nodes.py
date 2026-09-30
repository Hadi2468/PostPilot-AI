"""Graph nodes. Each node reads the state and returns a partial state update."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END
from langgraph.types import Command, interrupt

from postpilot.config import LINKEDIN_CHAR_LIMIT
from postpilot.prompts import EVALUATE_PROMPT, EXTRACT_PROMPT, GENERATE_PROMPT, IMPROVE_PROMPT
from postpilot.schemas import ContentInsights, Evaluation, PostDraft, ReviewDecision
from postpilot.state import GraphState
from postpilot.tools.search import SearchFn

logger = logging.getLogger(__name__)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) or "- (none)"


class Nodes:
    """Holds the LLM chains and search tool; its methods are the graph's nodes."""

    def __init__(self, writer_llm: BaseChatModel, critic_llm: BaseChatModel, search_fn: SearchFn):
        # The critic (temperature 0) also does extraction: both are analytical tasks.
        self._extract_chain = EXTRACT_PROMPT | critic_llm.with_structured_output(ContentInsights)
        self._generate_chain = GENERATE_PROMPT | writer_llm.with_structured_output(PostDraft)
        self._evaluate_chain = EVALUATE_PROMPT | critic_llm.with_structured_output(Evaluation)
        self._improve_chain = IMPROVE_PROMPT | writer_llm.with_structured_output(PostDraft)
        self._search = search_fn

    # --- 1. Extract structured insights from raw meeting data ---
    def extract(self, state: GraphState) -> GraphState:
        result: ContentInsights = self._extract_chain.invoke(
            {"meeting_data": _json(state["meeting_data"])}
        )
        return {"insights": result.model_dump(), "iteration": 0, "best_score": -1.0}

    # --- 2. Write the first draft ---
    def generate(self, state: GraphState) -> GraphState:
        result: PostDraft = self._generate_chain.invoke(
            {"insights": _json(state["insights"]), "char_limit": LINKEDIN_CHAR_LIMIT}
        )
        return {"draft_post": result.post}

    # --- 3. Critique the current draft and keep the best one ---
    def evaluate(self, state: GraphState) -> GraphState:
        post = state["draft_post"]
        result: Evaluation = self._evaluate_chain.invoke(
            {"insights": _json(state["insights"]), "post": post}
        )
        if len(post) > LINKEDIN_CHAR_LIMIT:
            result.weaknesses.append(
                f"Post is {len(post)} characters; LinkedIn's limit is {LINKEDIN_CHAR_LIMIT}."
            )

        score = result.overall_score
        evaluation = result.model_dump()
        logger.info(
            "iteration=%d overall=%.2f faithfulness=%.1f",
            state["iteration"], score, result.faithfulness_score,
        )

        update: GraphState = {
            "evaluation": evaluation,
            "history": [{"iteration": state["iteration"], "post": post, "score": score}],
        }
        if score > state.get("best_score", -1.0):
            update |= {"best_post": post, "best_score": score, "best_evaluation": evaluation}
        return update

    # --- 4. Web research (runs once; failures degrade gracefully) ---
    def research(self, state: GraphState) -> GraphState:
        insights = state["insights"]
        query = " ".join([insights.get("title", ""), *insights.get("key_topics", [])[:3]]).strip()
        try:
            results = self._search(query)
        except Exception:
            logger.warning(
                "Search failed for %r; continuing without research", query, exc_info=True
            )
            results = []
        return {"search_results": results}

    # --- 5. Revise the draft using human feedback, critique, and research ---
    def improve(self, state: GraphState) -> GraphState:
        evaluation = state["evaluation"]
        result: PostDraft = self._improve_chain.invoke({
            "insights": _json(state["insights"]),
            "post": state["draft_post"],
            "human_feedback": state.get("human_feedback") or "(none)",
            "weaknesses": _bullets(evaluation["weaknesses"]),
            "suggestions": _bullets(evaluation["improvement_suggestions"]),
            "search_results": _json(state.get("search_results", [])),
            "char_limit": LINKEDIN_CHAR_LIMIT,
        })
        return {"draft_post": result.post, "iteration": state["iteration"] + 1}

    # --- 6a. Human-in-the-loop: pause and wait for a reviewer's decision ---
    def human_review(self, state: GraphState) -> Command[Literal["improve", "__end__"]]:
        # interrupt() checkpoints the graph and surfaces this payload to the caller.
        # On resume, the node re-runs from the top and interrupt() returns the decision.
        raw = interrupt({
            "post": state["best_post"],
            "score": state["best_score"],
            "evaluation": state["best_evaluation"],
            "iteration": state["iteration"],
        })
        decision = ReviewDecision.model_validate(raw)
        logger.info("Human review decision: %s", decision.action)

        if decision.action == "approve":
            return Command(
                goto=END, update={"final_post": state["best_post"], "status": "approved"}
            )
        if decision.action == "edit":
            return Command(goto=END, update={"final_post": decision.post, "status": "edited"})
        if decision.action == "reject":
            return Command(goto=END, update={"status": "rejected"})

        # "revise": the reviewer's feedback redefines "good", so restart best-tracking
        # from the reviewed post and let the reflexion loop continue from there.
        return Command(
            goto="improve",
            update={
                "human_feedback": decision.feedback,
                "draft_post": state["best_post"],
                "evaluation": state["best_evaluation"],
                "best_score": -1.0,
            },
        )

    # --- 6b. Autonomous mode: accept the best draft without a human ---
    def finalize(self, state: GraphState) -> GraphState:
        return {"final_post": state["best_post"], "status": "completed"}
