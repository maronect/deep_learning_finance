"""
Router for POST /pipeline/run, GET /pipeline/runs, GET /pipeline/runs/{run_id}.

POST /pipeline/run triggers a full or partial pipeline execution as a
background task and returns the run ID immediately for status polling.
"""
from __future__ import annotations

import threading
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from src.api.schemas.requests import PipelineRunRequest
from src.api.schemas.responses import (
    PipelineRunResponse,
    RunManifestResponse,
    RunSummary,
    RunsListResponse,
)
from src.pipeline.registry import list_runs, load_run_manifest
from src.pipeline.runner import STAGE_ORDER, run_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# In-memory run status tracker. Entries are set to "running" when a background
# task starts and updated to "completed" or "failed" when it finishes.
# On API restart the dict is empty; existing manifests in artifacts/runs/ remain
# the source of truth for historical runs.
_run_status: dict[str, str] = {}
_run_lock = threading.Lock()


def _execute_pipeline(run_id: str, stages: Optional[list[str]]) -> None:
    """Background task: execute the pipeline and update _run_status."""
    try:
        ctx = run_pipeline(stages=stages)
        with _run_lock:
            _run_status[run_id] = ctx.status  # "completed" or "failed"
    except Exception:
        with _run_lock:
            _run_status[run_id] = "failed"


@router.post("/run", response_model=PipelineRunResponse, summary="Trigger pipeline execution")
def run_pipeline_endpoint(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
) -> PipelineRunResponse:
    """Start a full or partial pipeline execution as a background task.

    The pipeline runs asynchronously. Use GET /pipeline/runs/{run_id} to
    poll for completion.

    Args:
        request: Optional list of stage names to run. If omitted, all stages run.

    Returns:
        The run_id generated for this execution.
    """
    import datetime

    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    stages = request.stages

    with _run_lock:
        _run_status[run_id] = "running"

    background_tasks.add_task(_execute_pipeline, run_id, stages)

    stages_desc = ", ".join(stages) if stages else "all stages"
    return PipelineRunResponse(
        run_id=run_id,
        status="running",
        message=f"Pipeline started with stages: {stages_desc}. Poll GET /pipeline/runs/{run_id} for status.",
    )


@router.get("/runs", response_model=RunsListResponse, summary="List all pipeline runs")
def list_pipeline_runs(
    status_filter: Optional[str] = Query(
        default=None,
        description="Filter runs by status. Options: completed, failed, running.",
    ),
) -> RunsListResponse:
    """Return a summary of all recorded pipeline runs.

    Reads from the artifact registry (artifacts/runs/) plus any currently
    running in-memory tasks. Sorted chronologically by run_id.
    """
    df = list_runs()

    rows: list[RunSummary] = []

    # Include runs currently tracked in-memory (may not have manifests yet).
    with _run_lock:
        in_memory = dict(_run_status)

    for run_id, mem_status in in_memory.items():
        if df.empty or run_id not in df["run_id"].values:
            if status_filter is None or mem_status == status_filter:
                rows.append(
                    RunSummary(
                        run_id=run_id,
                        started_at="",
                        status=mem_status,
                        model="",
                        n_assets=0,
                        sharpe=None,
                        annualized_return=None,
                    )
                )

    if not df.empty:
        for _, row in df.iterrows():
            run_id = row["run_id"]
            # Prefer in-memory status for runs still tracked in this process.
            effective_status = in_memory.get(run_id, row.get("status", ""))
            if status_filter is not None and effective_status != status_filter:
                continue
            rows.append(
                RunSummary(
                    run_id=run_id,
                    started_at=str(row.get("started_at", "")),
                    status=effective_status,
                    model=str(row.get("model", "")),
                    n_assets=int(row.get("selected_assets_count", 0)),
                    sharpe=row.get("sharpe"),
                    annualized_return=row.get("annualized_return"),
                )
            )

    return RunsListResponse(runs=rows, count=len(rows))


@router.get(
    "/runs/{run_id}",
    response_model=RunManifestResponse,
    summary="Retrieve a specific run manifest",
)
def get_run(run_id: str) -> RunManifestResponse:
    """Return the full manifest for a specific pipeline run.

    Checks the in-memory status table first (for runs still in progress).
    Falls back to the manifest file written by stage_export_artifacts.

    Raises:
        404: If the run_id is unknown.
        202: (via 200 with status='running') If the run has not completed yet.
    """
    with _run_lock:
        mem_status = _run_status.get(run_id)

    if mem_status == "running":
        return RunManifestResponse(
            run_id=run_id,
            started_at="",
            status="running",
            error=None,
            selected_assets=None,
            artifacts_written=[],
            metrics=None,
        )

    try:
        manifest = load_run_manifest(run_id)
    except FileNotFoundError as exc:
        if mem_status is not None:
            return RunManifestResponse(
                run_id=run_id,
                started_at="",
                status=mem_status,
                error="Pipeline failed before writing manifest.",
                selected_assets=None,
                artifacts_written=[],
                metrics=None,
            )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return RunManifestResponse(
        run_id=manifest.get("run_id", run_id),
        started_at=str(manifest.get("started_at", "")),
        status=manifest.get("status", ""),
        error=manifest.get("error"),
        selected_assets=manifest.get("selected_assets"),
        artifacts_written=manifest.get("artifacts_written", []),
        metrics=manifest.get("metrics"),
    )


@router.get("/stages", summary="List available pipeline stages")
def list_stages() -> dict:
    """Return the ordered list of available pipeline stage names."""
    return {"stages": STAGE_ORDER}
