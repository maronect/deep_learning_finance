"""
FastAPI application factory.

Initializes the app with settings from config/api.yaml, registers CORS middleware,
and includes all routers. This is the entry point for uvicorn.

Environment variable overrides
-------------------------------
CORS_ORIGINS  Comma-separated list of allowed origins. Overrides cors.allow_origins
              in config/api.yaml at startup. Takes precedence over the config file.
              Example: CORS_ORIGINS="https://dlfinance-api.fly.dev,https://example.com"

API_VERSION   Application version string. Overrides docs.version in config/api.yaml.
              Reflected in the OpenAPI schema and in the GET /health response.
              Example: API_VERSION="1.2.0"
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.routers import assets, health, history, metrics, pipeline, portfolio, predictions, scheduler
from src.persistence.database import init_db, sync_from_manifests
from src.scheduler.scheduler import build_scheduler
from src.utils.config_loader import get_config


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    On startup: ensures the SQLite schema exists, syncs any manifest files
    that predate the persistence layer, and starts the APScheduler for
    periodic pipeline retraining.
    On shutdown: gracefully stops the scheduler.
    """
    init_db()
    sync_from_manifests()
    sched = build_scheduler()
    if sched is not None:
        sched.start()
    yield
    if sched is not None:
        sched.shutdown(wait=False)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Loads all settings from config/api.yaml. Registers CORS middleware and
    all API routers. Returns a ready-to-serve FastAPI instance.

    Returns:
        Configured FastAPI application.
    """
    cfg = get_config("api")
    docs_cfg = cfg.get("docs", {})
    cors_cfg = cfg.get("cors", {})

    _cors_env = os.environ.get("CORS_ORIGINS")
    allow_origins = (
        [o.strip() for o in _cors_env.split(",") if o.strip()]
        if _cors_env
        else cors_cfg.get("allow_origins", ["*"])
    )

    version = os.environ.get("API_VERSION") or docs_cfg.get("version", "0.1.0")

    app = FastAPI(
        title=docs_cfg.get("title", "Deep Learning Finance API"),
        description=docs_cfg.get(
            "description",
            "Portfolio optimization and ML return prediction service",
        ),
        version=version,
        docs_url=docs_cfg.get("swagger_url", "/docs"),
        redoc_url=docs_cfg.get("redoc_url", "/redoc"),
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=cors_cfg.get("allow_methods", ["GET", "POST"]),
        allow_headers=cors_cfg.get("allow_headers", ["*"]),
    )

    app.include_router(health.router)
    app.include_router(assets.router)
    app.include_router(predictions.router)
    app.include_router(pipeline.router)
    app.include_router(portfolio.router)
    app.include_router(metrics.router)
    app.include_router(history.router)
    app.include_router(scheduler.router)

    _dashboard = Path(__file__).parents[2] / "dashboard.html"

    @app.get("/", include_in_schema=False, summary="Portfolio dashboard")
    def dashboard() -> FileResponse:
        if not _dashboard.exists():
            raise HTTPException(status_code=404, detail="dashboard.html not found")
        return FileResponse(_dashboard, media_type="text/html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    server_cfg = get_config("api").get("server", {})
    uvicorn.run(
        "src.api.main:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=int(server_cfg.get("port", 8000)),
        reload=bool(server_cfg.get("reload", False)),
    )
