import pytest
from fastapi.testclient import TestClient

from postpilot.api import create_app
from postpilot.service import PostPilotService
from tests.conftest import FakeLLM


@pytest.fixture
def client_for(make_graph, review_settings):
    def _make(scores, api_token=None):
        settings = review_settings.model_copy(update={"api_token": api_token})
        service = PostPilotService(make_graph(FakeLLM(scores=scores), settings=settings), settings)
        return TestClient(create_app(service=service, settings=settings))

    return _make


def test_health(client_for):
    with client_for([]) as client:
        assert client.get("/health").json()["status"] == "ok"


def test_full_review_flow(client_for, meeting_data):
    with client_for([9.0]) as client:
        run = client.post("/runs", json={"meeting_data": meeting_data}).json()
        assert run["status"] == "awaiting_review"

        fetched = client.get(f"/runs/{run['thread_id']}").json()
        assert fetched == run

        done = client.post(f"/runs/{run['thread_id']}/review", json={"action": "approve"})
        assert done.status_code == 200
        assert done.json()["status"] == "approved"

        again = client.post(f"/runs/{run['thread_id']}/review", json={"action": "approve"})
        assert again.status_code == 409


def test_unknown_run_returns_404(client_for):
    with client_for([]) as client:
        assert client.get("/runs/nope").status_code == 404
        assert client.post("/runs/nope/review", json={"action": "approve"}).status_code == 404


def test_invalid_decision_returns_422(client_for, meeting_data):
    with client_for([9.0]) as client:
        run = client.post("/runs", json={"meeting_data": meeting_data}).json()
        resp = client.post(f"/runs/{run['thread_id']}/review", json={"action": "edit"})
        assert resp.status_code == 422


def test_bearer_token_is_enforced_when_configured(client_for, meeting_data):
    with client_for([9.0], api_token="s3cret") as client:
        body = {"meeting_data": meeting_data}
        assert client.post("/runs", json=body).status_code == 401
        assert client.post("/runs", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.post("/runs", json=body, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        # /health stays open for container health checks
        assert client.get("/health").status_code == 200
