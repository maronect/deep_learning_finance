"""
FastAPI application factory.

Initializes the app with settings from config/api.yaml, registers CORS middleware,
and includes all routers. This is the entry point for uvicorn.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import assets, health, history, metrics, pipeline, portfolio, predictions
from src.persistence.database import init_db, sync_from_manifests
from src.utils.config_loader import get_config


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    On startup: ensures the SQLite schema exists and syncs any manifest files
    that predate the persistence layer (e.g. from Stage 1/2 runs).
    """
    init_db()
    sync_from_manifests()
    yield


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

    app = FastAPI(
        title=docs_cfg.get("title", "Deep Learning Finance API"),
        description=docs_cfg.get(
            "description",
            "Portfolio optimization and ML return prediction service",
        ),
        version=docs_cfg.get("version", "0.1.0"),
        docs_url=docs_cfg.get("swagger_url", "/docs"),
        redoc_url=docs_cfg.get("redoc_url", "/redoc"),
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_cfg.get("allow_origins", ["*"]),
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
