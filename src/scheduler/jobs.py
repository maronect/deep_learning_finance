"""
Scheduled job definitions for periodic pipeline execution.

Each job function is registered with APScheduler and runs in a background
thread. Execution progress and errors are written to per-run log files under
artifacts/logs/{run_id}.log.
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config_loader import get_config


def _get_log_dir() -> Path:
    """Resolve the log output directory from scheduler config."""
    cfg = get_config("scheduler")
    log_dir = Path(cfg["logging"]["dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _build_run_logger(run_id: str) -> tuple[logging.Logger, Path]:
    """Create a logger that writes to artifacts/logs/{run_id}.log.

    Args:
        run_id: Unique identifier for this pipeline run.

    Returns:
        Tuple of (logger, log_file_path).
    """
    log_dir = _get_log_dir()
    log_file = log_dir / f"{run_id}.log"

    logger = logging.getLogger(f"scheduler.run.{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def run_scheduled_pipeline() -> None:
    """Execute the full pipeline as a scheduled job.

    Generates a timestamped run_id, writes a per-run log file, and persists
    the result in the SQLite database via upsert_run(). On failure, the
    exception is logged and re-raised so APScheduler can record the error.

    This function is designed to be registered directly with APScheduler.
    It is also safe to call manually for testing.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_scheduled"
    logger, log_file = _build_run_logger(run_id)

    logger.info("Scheduled pipeline run started. run_id=%s", run_id)
    logger.info("Log file: %s", log_file)

    try:
        # Import here to avoid circular imports at module load time.
        from src.pipeline.runner import run_pipeline

        logger.info("Executing pipeline (all stages)...")
        ctx = run_pipeline()

        logger.info("Pipeline finished with status=%s", ctx.status)

        if ctx.metrics:
            for model, m in ctx.metrics.items():
                sharpe = m.get("Sharpe", "n/a")
                ret = m.get("Annualized_Return", "n/a")
                logger.info(
                    "Model=%s  Sharpe=%.4f  Annualized_Return=%.4f",
                    model,
                    sharpe if isinstance(sharpe, float) else float("nan"),
                    ret if isinstance(ret, float) else float("nan"),
                )

        if ctx.status == "failed":
            logger.error("Pipeline failed. error=%s", ctx.error)
            raise RuntimeError(f"Pipeline run {run_id} failed: {ctx.error}")

        logger.info("Run completed successfully. run_id=%s", run_id)

    except Exception:
        logger.error("Unhandled exception in scheduled run:\n%s", traceback.format_exc())
        raise
    finally:
        # Flush and close the file handler so the log is readable immediately.
        for handler in logging.getLogger(f"scheduler.run.{run_id}").handlers:
            handler.flush()
            handler.close()
