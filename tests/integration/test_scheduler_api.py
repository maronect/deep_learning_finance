"""
Integration tests for the /scheduler API endpoints.

Uses FastAPI TestClient. The APScheduler is started and stopped via the
lifespan handler as in production, but with a very long interval so the
pipeline is never actually executed during tests.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture(scope="module")
def client():
    """Return a TestClient wrapping a freshly created app instance.

    Uses the context manager form so the lifespan handler (scheduler startup)
    runs before any test in this module executes.
    """
    from src.scheduler import scheduler as sched_module
    sched_module._scheduler = None  # reset singleton so lifespan builds a fresh one

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /scheduler/status
# ---------------------------------------------------------------------------


class TestSchedulerStatus:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/scheduler/status")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        body = client.get("/scheduler/status").json()
        assert "enabled" in body
        assert "running" in body
        assert "job_count" in body
        assert "jobs" in body

    def test_enabled_is_bool(self, client: TestClient) -> None:
        body = client.get("/scheduler/status").json()
        assert isinstance(body["enabled"], bool)

    def test_running_is_bool(self, client: TestClient) -> None:
        body = client.get("/scheduler/status").json()
        assert isinstance(body["running"], bool)

    def test_job_count_non_negative(self, client: TestClient) -> None:
        body = client.get("/scheduler/status").json()
        assert body["job_count"] >= 0

    def test_jobs_is_list(self, client: TestClient) -> None:
        body = client.get("/scheduler/status").json()
        assert isinstance(body["jobs"], list)

    def test_job_schema_when_present(self, client: TestClient) -> None:
        body = client.get("/scheduler/status").json()
        for job in body["jobs"]:
            assert "id" in job
            assert "name" in job
            assert "trigger" in job
            assert "next_run_time" in job

    def test_scheduler_running_when_enabled(self, client: TestClient) -> None:
        """If enabled=true in config, running should be true after app startup."""
        body = client.get("/scheduler/status").json()
        if body["enabled"]:
            assert body["running"] is True


# ---------------------------------------------------------------------------
# POST /scheduler/trigger
# ---------------------------------------------------------------------------


class TestSchedulerTrigger:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.post("/scheduler/trigger")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        body = client.post("/scheduler/trigger").json()
        assert "message" in body
        assert "run_id" in body

    def test_run_id_is_string(self, client: TestClient) -> None:
        body = client.post("/scheduler/trigger").json()
        assert isinstance(body["run_id"], str)
        assert len(body["run_id"]) > 0

    def test_message_is_string(self, client: TestClient) -> None:
        body = client.post("/scheduler/trigger").json()
        assert isinstance(body["message"], str)


# ---------------------------------------------------------------------------
# GET /scheduler/logs
# ---------------------------------------------------------------------------


class TestSchedulerLogs:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/scheduler/logs")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        body = client.get("/scheduler/logs").json()
        assert "logs" in body
        assert "count" in body

    def test_count_matches_logs_length(self, client: TestClient) -> None:
        body = client.get("/scheduler/logs").json()
        assert body["count"] == len(body["logs"])

    def test_log_entry_schema(self, client: TestClient) -> None:
        body = client.get("/scheduler/logs").json()
        for entry in body["logs"]:
            assert "run_id" in entry
            assert "filename" in entry
            assert "size_bytes" in entry
            assert "modified_at" in entry


# ---------------------------------------------------------------------------
# GET /scheduler/logs/{run_id}
# ---------------------------------------------------------------------------


class TestSchedulerLogDetail:
    def test_404_for_unknown_run(self, client: TestClient) -> None:
        r = client.get("/scheduler/logs/nonexistent_run_id_xyz")
        assert r.status_code == 404

    def test_404_detail_message(self, client: TestClient) -> None:
        r = client.get("/scheduler/logs/nonexistent_run_id_xyz")
        body = r.json()
        assert "detail" in body
