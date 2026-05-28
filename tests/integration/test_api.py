"""
Integration tests for the FastAPI application.

Uses FastAPI's TestClient to verify that all endpoints return expected status
codes, schema-conformant responses, and correct error handling.
Requires httpx (listed in requirements-dev.txt).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app

# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a TestClient wrapping a freshly created app instance."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert "status" in body
        assert "version" in body
        assert "runs_available" in body

    def test_status_value(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_runs_available_is_int(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert isinstance(body["runs_available"], int)
        assert body["runs_available"] >= 0


# ---------------------------------------------------------------------------
# GET /assets
# ---------------------------------------------------------------------------

class TestAssets:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/assets")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        body = client.get("/assets").json()
        assert "source" in body
        assert "assets" in body
        assert "count" in body

    def test_count_matches_assets_length(self, client: TestClient) -> None:
        body = client.get("/assets").json()
        assert body["count"] == len(body["assets"])

    def test_assets_is_list(self, client: TestClient) -> None:
        body = client.get("/assets").json()
        assert isinstance(body["assets"], list)


# ---------------------------------------------------------------------------
# GET /predictions — no completed run → 404
# ---------------------------------------------------------------------------

class TestPredictions:
    def test_no_runs_returns_404(self, client: TestClient) -> None:
        # If no completed run exists for the requested run_id, expect 404
        r = client.get("/predictions?run_id=NONEXISTENT_RUN_XYZ")
        assert r.status_code == 404

    def test_404_has_detail(self, client: TestClient) -> None:
        r = client.get("/predictions?run_id=NONEXISTENT_RUN_XYZ")
        assert "detail" in r.json()


# ---------------------------------------------------------------------------
# GET /pipeline/stages
# ---------------------------------------------------------------------------

class TestPipelineStages:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/pipeline/stages")
        assert r.status_code == 200

    def test_contains_all_stages(self, client: TestClient) -> None:
        body = client.get("/pipeline/stages").json()
        expected = [
            "ingest", "compute_returns", "build_features",
            "train_models", "predict_returns", "optimize_portfolio",
            "evaluate", "export_artifacts",
        ]
        assert body["stages"] == expected


# ---------------------------------------------------------------------------
# GET /pipeline/runs
# ---------------------------------------------------------------------------

class TestPipelineRuns:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/pipeline/runs")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        body = client.get("/pipeline/runs").json()
        assert "runs" in body
        assert "count" in body
        assert isinstance(body["runs"], list)


# ---------------------------------------------------------------------------
# POST /pipeline/run
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_empty_body_returns_200(self, client: TestClient) -> None:
        r = client.post("/pipeline/run", json={})
        assert r.status_code == 200

    def test_response_has_run_id(self, client: TestClient) -> None:
        body = client.post("/pipeline/run", json={}).json()
        assert "run_id" in body
        assert isinstance(body["run_id"], str)
        assert len(body["run_id"]) > 0

    def test_status_is_running(self, client: TestClient) -> None:
        body = client.post("/pipeline/run", json={}).json()
        assert body["status"] == "running"

    def test_invalid_stage_returns_422(self, client: TestClient) -> None:
        r = client.post("/pipeline/run", json={"stages": ["not_a_real_stage"]})
        assert r.status_code == 422

    def test_valid_stage_subset_accepted(self, client: TestClient) -> None:
        r = client.post("/pipeline/run", json={"stages": ["ingest", "compute_returns"]})
        assert r.status_code == 200

    def test_concurrent_run_rejected_with_429(self, client: TestClient) -> None:
        from src.api.routers.pipeline import _run_lock, _run_status
        r = client.post("/pipeline/run", json={})
        run_id = r.json().get("run_id", "")
        # Force the run to appear still active, then verify 429 is returned.
        with _run_lock:
            _run_status[run_id] = "running"
        r2 = client.post("/pipeline/run", json={})
        assert r2.status_code == 429
        assert "detail" in r2.json()
        # Restore so subsequent tests are not blocked.
        with _run_lock:
            _run_status[run_id] = "completed"

    def test_run_id_is_unique(self, client: TestClient) -> None:
        r1 = client.post("/pipeline/run", json={}).json()["run_id"]
        r2 = client.post("/pipeline/run", json={}).json()["run_id"]
        assert r1 != r2


# ---------------------------------------------------------------------------
# GET /portfolio/weights — no completed run → 404
# ---------------------------------------------------------------------------

class TestPortfolioWeights:
    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        r = client.get("/portfolio/weights?run_id=NONEXISTENT_RUN_XYZ")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /portfolio/frontier — no completed run → 404
# ---------------------------------------------------------------------------

class TestPortfolioFrontier:
    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        r = client.get("/portfolio/frontier?run_id=NONEXISTENT_RUN_XYZ")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /metrics/portfolio — no completed run → 404
# ---------------------------------------------------------------------------

class TestMetricsPortfolio:
    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        r = client.get("/metrics/portfolio?run_id=NONEXISTENT_RUN_XYZ")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /metrics/model — no completed run → 404
# ---------------------------------------------------------------------------

class TestMetricsModel:
    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        r = client.get("/metrics/model?run_id=NONEXISTENT_RUN_XYZ")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /history/runs
# ---------------------------------------------------------------------------

class TestHistoryRuns:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/history/runs")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        body = client.get("/history/runs").json()
        assert "runs" in body
        assert "count" in body

    def test_status_filter_completed(self, client: TestClient) -> None:
        r = client.get("/history/runs?status=completed")
        assert r.status_code == 200

    def test_status_filter_invalid_still_200(self, client: TestClient) -> None:
        # Non-existing status returns empty list, not an error
        r = client.get("/history/runs?status=unknown_status_xyz")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_limit_param_respected(self, client: TestClient) -> None:
        r = client.get("/history/runs?limit=1")
        assert r.status_code == 200
        assert len(r.json()["runs"]) <= 1


# ---------------------------------------------------------------------------
# POST /history/sync
# ---------------------------------------------------------------------------

class TestHistorySync:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.post("/history/sync")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        body = client.post("/history/sync").json()
        assert "synced" in body
        assert "message" in body
        assert isinstance(body["synced"], int)
        assert body["synced"] >= 0


# ---------------------------------------------------------------------------
# GET /portfolio/compare
# ---------------------------------------------------------------------------

class TestPortfolioCompare:
    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        r = client.get("/portfolio/compare?run_id=NONEXISTENT_RUN_XYZ")
        assert r.status_code == 404

    def test_missing_run_id_uses_latest(self, client: TestClient) -> None:
        r = client.get("/portfolio/compare")
        # Either 200 (runs exist) or 404 (no runs) — both are valid.
        assert r.status_code in (200, 404)

    def test_response_schema_when_run_exists(self, client: TestClient) -> None:
        # Only validate schema if a completed run is available.
        r = client.get("/portfolio/compare")
        if r.status_code == 200:
            body = r.json()
            assert "run_id" in body
            assert "models" in body
            assert "primary_model" in body
            assert "comparison" in body
            assert isinstance(body["models"], list)
            assert isinstance(body["comparison"], dict)


# ---------------------------------------------------------------------------
# GET /history/runs/compare — unknown IDs → 404
# ---------------------------------------------------------------------------

class TestHistoryCompare:
    def test_unknown_run_ids_returns_404(self, client: TestClient) -> None:
        r = client.get("/history/runs/compare?run_ids=FAKE_ID_1,FAKE_ID_2")
        assert r.status_code == 404

    def test_empty_run_ids_returns_400(self, client: TestClient) -> None:
        r = client.get("/history/runs/compare?run_ids=")
        assert r.status_code == 400

    def test_missing_run_ids_param_returns_422(self, client: TestClient) -> None:
        r = client.get("/history/runs/compare")
        assert r.status_code == 422
