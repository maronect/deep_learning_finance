"""
Router for /scheduler endpoints.

Provides visibility into the APScheduler state and allows manual pipeline
triggering and log retrieval.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.utils.config_loader import get_config

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class JobInfo(BaseModel):
    id: str
    name: str
    next_run_time: Optional[str]
    trigger: str


class SchedulerStatusResponse(BaseModel):
    enabled: bool
    running: bool
    job_count: int
    jobs: list[JobInfo]


class TriggerResponse(BaseModel):
    message: str
    run_id: str


class LogEntry(BaseModel):
    run_id: str
    filename: str
    size_bytes: int
    modified_at: str


class LogsListResponse(BaseModel):
    logs: list[LogEntry]
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_dir() -> Path:
    cfg = get_config("scheduler")
    return Path(cfg["logging"]["dir"])


def _scheduler_enabled() -> bool:
    cfg = get_config("scheduler")
    return bool(cfg.get("scheduler", {}).get("enabled", True))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status", response_model=SchedulerStatusResponse, summary="Scheduler status")
def scheduler_status() -> SchedulerStatusResponse:
    """Return current scheduler state: running flag, job list, next run times."""
    from src.scheduler.scheduler import get_scheduler

    sched = get_scheduler()
    enabled = _scheduler_enabled()

    if sched is None:
        return SchedulerStatusResponse(enabled=enabled, running=False, job_count=0, jobs=[])

    jobs: list[JobInfo] = []
    for job in sched.get_jobs():
        next_run_time = getattr(job, "next_run_time", None)
        next_run = str(next_run_time) if next_run_time else None
        jobs.append(
            JobInfo(
                id=job.id,
                name=job.name,
                next_run_time=next_run,
                trigger=str(job.trigger),
            )
        )

    return SchedulerStatusResponse(
        enabled=enabled,
        running=sched.running,
        job_count=len(jobs),
        jobs=jobs,
    )


@router.post("/trigger", response_model=TriggerResponse, summary="Manually trigger pipeline")
def trigger_pipeline(background_tasks: BackgroundTasks) -> TriggerResponse:
    """Fire the scheduled pipeline job immediately as a one-shot background task.

    This endpoint does not wait for the pipeline to complete. Use
    GET /pipeline/runs/{run_id} to poll the status.
    """
    from src.scheduler.jobs import run_scheduled_pipeline

    import datetime

    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") + "_manual"

    # Patch the run_id into the job by running it as a background task.
    # run_scheduled_pipeline generates its own run_id internally, so we
    # delegate entirely to it and surface the approximate timestamp.
    background_tasks.add_task(_fire_pipeline)

    return TriggerResponse(
        message="Pipeline triggered. Check GET /scheduler/logs for execution log.",
        run_id=run_id,
    )


def _fire_pipeline() -> None:
    """Thin wrapper so BackgroundTasks can call run_scheduled_pipeline."""
    from src.scheduler.jobs import run_scheduled_pipeline

    try:
        run_scheduled_pipeline()
    except Exception:
        pass  # errors are already written to the log file


@router.get("/logs", response_model=LogsListResponse, summary="List execution logs")
def list_logs() -> LogsListResponse:
    """Return metadata for all log files in artifacts/logs/."""
    log_dir = _log_dir()

    if not log_dir.exists():
        return LogsListResponse(logs=[], count=0)

    entries: list[LogEntry] = []
    for f in sorted(log_dir.glob("*.log"), reverse=True):
        stat = f.stat()
        import datetime

        modified = datetime.datetime.fromtimestamp(
            stat.st_mtime, tz=datetime.timezone.utc
        ).isoformat()
        entries.append(
            LogEntry(
                run_id=f.stem,
                filename=f.name,
                size_bytes=stat.st_size,
                modified_at=modified,
            )
        )

    return LogsListResponse(logs=entries, count=len(entries))


@router.get("/logs/{run_id}", summary="Retrieve log content for a run")
def get_log(run_id: str) -> dict:
    """Return the full text content of the log file for a specific run.

    Args:
        run_id: The run identifier (filename without .log extension).

    Raises:
        404: If no log file exists for the given run_id.
    """
    log_file = _log_dir() / f"{run_id}.log"

    if not log_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No log file found for run_id '{run_id}'.",
        )

    content = log_file.read_text(encoding="utf-8")
    size = os.path.getsize(str(log_file))

    return {"run_id": run_id, "content": content, "size_bytes": size}
