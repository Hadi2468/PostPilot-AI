"""Graph assembly. Dependencies are injected so the graph is testable without API keys."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from postpilot.config import Settings, get_settings
from postpilot.nodes import Nodes
from postpilot.routing import make_router
from postpilot.state import GraphState
from postpilot.tools.search import SearchFn


def build_graph(
    writer_llm: BaseChatModel,
    critic_llm: BaseChatModel,
    search_fn: SearchFn,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """
    START -> extract -> generate -> evaluate --(accept / budget spent)--> review -> END
                                       ^   \\--(revise, 1st time)--> research -> improve
                                       |   \\--(revise, cached)-----------------> improve
                                       \\-----------------------------------------/
    `review` is `human_review` (interrupt) or `finalize` (autonomous), per settings.
    A human "revise" decision routes from `human_review` back to `improve`.
    """
    settings = settings or get_settings()
    nodes = Nodes(writer_llm, critic_llm, search_fn)
    review_node = "human_review" if settings.human_review else "finalize"

    builder = StateGraph(GraphState)
    builder.add_node("extract", nodes.extract)
    builder.add_node("generate", nodes.generate)
    builder.add_node("evaluate", nodes.evaluate)
    builder.add_node("research", nodes.research)
    builder.add_node("improve", nodes.improve)
    if settings.human_review:
        builder.add_node("human_review", nodes.human_review, destinations=("improve", END))
    else:
        builder.add_node("finalize", nodes.finalize)
        builder.add_edge("finalize", END)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        make_router(settings),
        {"research": "research", "improve": "improve", "review": review_node},
    )
    builder.add_edge("research", "improve")
    builder.add_edge("improve", "evaluate")

    # A checkpointer is required for interrupt/resume and lets us inspect any run by
    # thread_id. InMemorySaver suits a single process; use SqliteSaver/PostgresSaver
    # for durability across restarts or multiple workers.
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def build_default_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Wire the production dependencies: OpenAI models and Tavily search."""
    from postpilot.llm import build_llms
    from postpilot.tools.search import make_tavily_search

    settings = settings or get_settings()
    writer, critic = build_llms(settings)
    return build_graph(writer, critic, make_tavily_search(settings.search_max_results), settings)
