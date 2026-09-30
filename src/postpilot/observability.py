"""LangSmith tracing and per-run config.

Tracing is enabled purely through environment variables (LANGSMITH_TRACING=true,
LANGSMITH_API_KEY, LANGSMITH_PROJECT); LangChain/LangGraph pick them up automatically.
Here we only attach a run name, tags, and metadata so traces are easy to filter.
"""

from __future__ import annotations

import logging
import os

from langchain_core.runnables import RunnableConfig

from postpilot import __version__
from postpilot.config import Settings

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    return (
        os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


def log_tracing_status() -> None:
    if tracing_enabled():
        logger.info("LangSmith tracing ON (project=%s)", os.getenv("LANGSMITH_PROJECT", "default"))
    else:
        logger.info("LangSmith tracing OFF")


def run_config(thread_id: str, settings: Settings) -> RunnableConfig:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "postpilot",
        "tags": ["postpilot", settings.openai_model],
        "metadata": {
            "thread_id": thread_id,
            "app_version": __version__,
            "model": settings.openai_model,
            "score_threshold": settings.score_threshold,
            "max_iterations": settings.max_iterations,
            "human_review": settings.human_review,
        },
    }
