"""
Unit tests for src/scheduler/scheduler.py and src/scheduler/jobs.py.

Tests verify scheduler construction, trigger configuration, and job execution
without running the full pipeline (run_pipeline is mocked).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from apscheduler.schedulers import SchedulerNotRunningError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


def _safe_shutdown(sched: BackgroundScheduler) -> None:
    """Shut down a scheduler only if it is running."""
    try:
        sched.shutdown(wait=False)
    except SchedulerNotRunningError:
        pass


# ---------------------------------------------------------------------------
# build_scheduler tests
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    def test_returns_background_scheduler_when_enabled(self, tmp_path, monkeypatch):
        cfg = {
            "scheduler": {
                "enabled": True,
                "trigger": "interval",
                "interval_hours": 12,
                "max_instances": 1,
                "misfire_grace_seconds": 3600,
            },
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }
        with patch("src.scheduler.scheduler.get_config", return_value=cfg):
            from src.scheduler import scheduler as sched_module
            sched_module._scheduler = None
            result = sched_module.build_scheduler()

        assert isinstance(result, BackgroundScheduler)
        _safe_shutdown(result)

    def test_returns_none_when_disabled(self, tmp_path):
        cfg = {
            "scheduler": {
                "enabled": False,
                "trigger": "interval",
                "interval_hours": 24,
            },
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }
        with patch("src.scheduler.scheduler.get_config", return_value=cfg):
            from src.scheduler import scheduler as sched_module
            sched_module._scheduler = None
            result = sched_module.build_scheduler()

        assert result is None

    def test_interval_trigger_is_used(self, tmp_path):
        cfg = {
            "scheduler": {
                "enabled": True,
                "trigger": "interval",
                "interval_hours": 6,
                "max_instances": 1,
                "misfire_grace_seconds": 3600,
            },
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }
        with patch("src.scheduler.scheduler.get_config", return_value=cfg):
            from src.scheduler import scheduler as sched_module
            sched_module._scheduler = None
            sched = sched_module.build_scheduler()

        assert sched is not None
        job = sched.get_job("pipeline_retrain")
        assert job is not None
        assert isinstance(job.trigger, IntervalTrigger)
        _safe_shutdown(sched)

    def test_cron_trigger_is_used(self, tmp_path):
        cfg = {
            "scheduler": {
                "enabled": True,
                "trigger": "cron",
                "cron": "0 9 * * *",
                "max_instances": 1,
                "misfire_grace_seconds": 3600,
            },
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }
        with patch("src.scheduler.scheduler.get_config", return_value=cfg):
            from src.scheduler import scheduler as sched_module
            sched_module._scheduler = None
            sched = sched_module.build_scheduler()

        assert sched is not None
        job = sched.get_job("pipeline_retrain")
        assert job is not None
        assert isinstance(job.trigger, CronTrigger)
        _safe_shutdown(sched)

    def test_get_scheduler_returns_built_instance(self, tmp_path):
        cfg = {
            "scheduler": {
                "enabled": True,
                "trigger": "interval",
                "interval_hours": 24,
                "max_instances": 1,
                "misfire_grace_seconds": 3600,
            },
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }
        with patch("src.scheduler.scheduler.get_config", return_value=cfg):
            from src.scheduler import scheduler as sched_module
            sched_module._scheduler = None
            built = sched_module.build_scheduler()
            fetched = sched_module.get_scheduler()

        assert built is fetched
        _safe_shutdown(built)


# ---------------------------------------------------------------------------
# run_scheduled_pipeline tests
# ---------------------------------------------------------------------------


class TestRunScheduledPipeline:
    def test_calls_run_pipeline(self, tmp_path):
        """run_scheduled_pipeline should call run_pipeline once."""
        mock_ctx = MagicMock()
        mock_ctx.status = "completed"
        mock_ctx.metrics = {}
        mock_ctx.error = None

        sched_cfg = {
            "scheduler": {"enabled": True},
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }

        with patch("src.scheduler.jobs.get_config", return_value=sched_cfg), \
             patch("src.pipeline.runner.run_pipeline", return_value=mock_ctx) as mock_run:
            from src.scheduler.jobs import run_scheduled_pipeline
            run_scheduled_pipeline()

        mock_run.assert_called_once_with()

    def test_log_file_created(self, tmp_path):
        """A .log file should appear in artifacts/logs/ after execution."""
        mock_ctx = MagicMock()
        mock_ctx.status = "completed"
        mock_ctx.metrics = {}
        mock_ctx.error = None

        sched_cfg = {
            "scheduler": {"enabled": True},
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }

        with patch("src.scheduler.jobs.get_config", return_value=sched_cfg), \
             patch("src.pipeline.runner.run_pipeline", return_value=mock_ctx):
            from src.scheduler.jobs import run_scheduled_pipeline
            run_scheduled_pipeline()

        log_files = list((tmp_path / "logs").glob("*.log"))
        assert len(log_files) == 1

    def test_log_contains_run_id(self, tmp_path):
        """Log file content should mention the run_id."""
        mock_ctx = MagicMock()
        mock_ctx.status = "completed"
        mock_ctx.metrics = {}
        mock_ctx.error = None

        sched_cfg = {
            "scheduler": {"enabled": True},
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }

        with patch("src.scheduler.jobs.get_config", return_value=sched_cfg), \
             patch("src.pipeline.runner.run_pipeline", return_value=mock_ctx):
            from src.scheduler.jobs import run_scheduled_pipeline
            run_scheduled_pipeline()

        log_file = next((tmp_path / "logs").glob("*.log"))
        content = log_file.read_text()
        assert "run_id=" in content or "_scheduled" in content

    def test_raises_on_pipeline_failure(self, tmp_path):
        """If pipeline returns status=failed, a RuntimeError should be raised."""
        mock_ctx = MagicMock()
        mock_ctx.status = "failed"
        mock_ctx.metrics = {}
        mock_ctx.error = "Something went wrong"

        sched_cfg = {
            "scheduler": {"enabled": True},
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }

        with patch("src.scheduler.jobs.get_config", return_value=sched_cfg), \
             patch("src.pipeline.runner.run_pipeline", return_value=mock_ctx):
            from src.scheduler.jobs import run_scheduled_pipeline
            with pytest.raises(RuntimeError):
                run_scheduled_pipeline()

    def test_raises_on_exception(self, tmp_path):
        """Unhandled exceptions from run_pipeline propagate to the caller."""
        sched_cfg = {
            "scheduler": {"enabled": True},
            "logging": {"dir": str(tmp_path / "logs"), "level": "INFO"},
        }

        with patch("src.scheduler.jobs.get_config", return_value=sched_cfg), \
             patch("src.pipeline.runner.run_pipeline", side_effect=ValueError("boom")):
            from src.scheduler.jobs import run_scheduled_pipeline
            with pytest.raises(ValueError, match="boom"):
                run_scheduled_pipeline()
