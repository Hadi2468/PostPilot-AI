"""LLM factory: one creative writer, one deterministic critic."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from postpilot.config import Settings


def build_llms(settings: Settings) -> tuple[ChatOpenAI, ChatOpenAI]:
    common = {
        "model": settings.openai_model,
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
    }
    writer = ChatOpenAI(temperature=settings.writer_temperature, **common)
    critic = ChatOpenAI(temperature=settings.critic_temperature, **common)
    return writer, critic
