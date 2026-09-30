"""Application settings, loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LINKEDIN_CHAR_LIMIT = 3000

# Weights for the deterministic overall score (must sum to 1.0).
SCORE_WEIGHTS: dict[str, float] = {
    "hook_score": 0.25,
    "clarity_score": 0.20,
    "engagement_score": 0.20,
    "originality_score": 0.15,
    "faithfulness_score": 0.20,
}


class Settings(BaseSettings):
    """App settings. Every field can be overridden with a `POSTPILOT_<FIELD>` env var.

    Provider keys (OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_*) are read directly
    by their SDKs from the environment, so they are not duplicated here.
    """

    model_config = SettingsConfigDict(env_prefix="POSTPILOT_", env_file=".env", extra="ignore")

    # LLM
    openai_model: str = "gpt-4o-mini"
    writer_temperature: float = 0.7
    critic_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # Reflexion loop
    score_threshold: float = Field(8.5, ge=0, le=10)
    min_faithfulness: float = Field(7.0, ge=0, le=10)
    max_iterations: int = Field(3, ge=0)
    search_max_results: int = Field(3, ge=1, le=10)

    # Human-in-the-loop
    human_review: bool = True

    # Serving
    api_token: str | None = None  # when set, the API requires `Authorization: Bearer <token>`
    api_url: str = "http://localhost:8000"  # used by the Streamlit dashboard


@lru_cache
def get_settings() -> Settings:
    # Export .env into os.environ so the OpenAI / Tavily / LangSmith SDKs can see their keys.
    load_dotenv()
    return Settings()
