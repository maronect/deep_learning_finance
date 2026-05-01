"""
APScheduler configuration and singleton management.

Reads config/scheduler.yaml and builds a BackgroundScheduler with the
appropriate trigger (interval or cron). The scheduler is a module-level
singleton so it can be started in the FastAPI lifespan and queried by
the /scheduler API router.
"""
from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

# Module-level singleton — populated by build_scheduler().
_scheduler: Optional[BackgroundScheduler] = None


def build_scheduler() -> Optional[BackgroundScheduler]:
    """Create and store a BackgroundScheduler from config/scheduler.yaml.

    Returns None if scheduler.enabled is false in the config, allowing the
    FastAPI lifespan to skip startup without errors.

    Returns:
        Configured BackgroundScheduler instance, or None if disabled.
    """
    global _scheduler

    cfg = get_config("scheduler")
    sched_cfg = cfg.get("scheduler", {})

    if not sched_cfg.get("enabled", True):
        logger.info("Scheduler disabled via config. Skipping.")
        _scheduler = None
        return None

    _scheduler = BackgroundScheduler(
        job_defaults={
            "max_instances": sched_cfg.get("max_instances", 1),
            "misfire_grace_time": sched_cfg.get("misfire_grace_seconds", 3600),
            "coalesce": True,
        }
    )

    trigger_type = sched_cfg.get("trigger", "interval")

    if trigger_type == "cron":
        cron_expr = sched_cfg.get("cron", "0 9 * * *")
        parts = cron_expr.split()
        if len(parts) == 5:
            minute, hour, day, month, day_of_week = parts
        else:
            logger.warning("Invalid cron expression '%s', using default '0 9 * * *'", cron_expr)
            minute, hour, day, month, day_of_week = "0", "9", "*", "*", "*"
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        )
    else:
        hours = float(sched_cfg.get("interval_hours", 24))
        trigger = IntervalTrigger(hours=hours)

    from src.scheduler.jobs import run_scheduled_pipeline

    _scheduler.add_job(
        run_scheduled_pipeline,
        trigger=trigger,
        id="pipeline_retrain",
        name="Automated pipeline retrain",
        replace_existing=True,
    )

    logger.info(
        "Scheduler built. trigger=%s  enabled=%s",
        trigger_type,
        sched_cfg.get("enabled"),
    )
    return _scheduler


def get_scheduler() -> Optional[BackgroundScheduler]:
    """Return the module-level scheduler instance.

    Returns:
        The BackgroundScheduler if it was built, or None if disabled/not yet built.
    """
    return _scheduler
