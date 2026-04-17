"""
Persistence layer for pipeline run records.

Provides a SQLite-backed store for run metadata, metrics, and artifact paths.
Complements the file-based artifact system (artifacts/) with queryable, indexed
storage that survives across API restarts and enables efficient comparison.
"""
from src.persistence.database import (
    compare_runs,
    get_run,
    init_db,
    list_runs,
    sync_from_manifests,
    upsert_run,
)

__all__ = [
    "init_db",
    "upsert_run",
    "get_run",
    "list_runs",
    "compare_runs",
    "sync_from_manifests",
]
