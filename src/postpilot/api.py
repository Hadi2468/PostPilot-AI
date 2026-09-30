"""FastAPI service exposing the agent with a human-in-the-loop review endpoint.

Run locally:  uvicorn postpilot.api:app --reload
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from postpilot import __version__
from postpilot.config import Settings, get_settings
from postpilot.observability import log_tracing_status
from postpilot.schemas import ReviewDecision
from postpilot.service import (
    NotAwaitingReviewError,
    PostPilotService,
    RunNotFoundError,
    RunResult,
)

logger = logging.getLogger(__name__)


class StartRunRequest(BaseModel):
    meeting_data: dict[str, Any]


# ============================================================
# Dependencies
# ============================================================

def get_service(request: Request) -> PostPilotService:
    return request.app.state.service


def require_token(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))],
) -> None:
    expected = request.app.state.settings.api_token
    if not expected:
        return
    if creds is None or not secrets.compare_digest(creds.credentials, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing bearer token")


ServiceDep = Annotated[PostPilotService, Depends(get_service)]


# ============================================================
# Routes
# ============================================================

# Sync handlers: FastAPI runs them in a threadpool, so a long LLM run
# doesn't block the event loop.
runs = APIRouter(prefix="/runs", tags=["runs"], dependencies=[Depends(require_token)])


@runs.post("", response_model=RunResult)
def start_run(body: StartRunRequest, svc: ServiceDep) -> RunResult:
    """Start a run. Returns `awaiting_review` (human review on) or a final status."""
    return svc.start(body.meeting_data)


@runs.get("/{thread_id}", response_model=RunResult)
def get_run(thread_id: str, svc: ServiceDep) -> RunResult:
    try:
        return svc.get(thread_id)
    except RunNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found") from None


@runs.post("/{thread_id}/review", response_model=RunResult)
def review_run(thread_id: str, decision: ReviewDecision, svc: ServiceDep) -> RunResult:
    """Resume a paused run with approve / edit / revise / reject."""
    try:
        return svc.resume(thread_id, decision)
    except RunNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found") from None
    except NotAwaitingReviewError:
        raise HTTPException(status.HTTP_409_CONFLICT, "Run is not awaiting review") from None


# ============================================================
# App factory
# ============================================================

def create_app(
    service: PostPilotService | None = None, settings: Settings | None = None
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO)
        log_tracing_status()
        # Build lazily so importing this module never needs API keys.
        app.state.service = service or PostPilotService.from_settings(settings)
        yield

    app = FastAPI(title="PostPilot AI", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(runs)
    return app


app = create_app()
