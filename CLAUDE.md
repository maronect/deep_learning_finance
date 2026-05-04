# CLAUDE.md

## Language
- Conversations can be in Portuguese
- All code, comments, docstrings, and git commits must be in English
- You may respond to me in Portuguese
- Do not use any emoji

---

## Project Overview

Quantitative research project evolving into a full ML Engineering application.
Domain: portfolio optimization for Brazilian stocks (B3, 2010–2025) using Modern Portfolio Theory.
Three strategies compared: Classic Markowitz, Markowitz + Ridge Regression, Markowitz + MLP.
Best result: Ridge Regression — Sharpe 0.591 vs 0.543 classic.

Technical architecture reference: see **ARCHITECTURE.md**.

---

## Current Stage: Stage 10 — COMPLETE (all stages done)

### Stage 1 — COMPLETE
- Directory structure, YAML config system (`config/pipeline.yaml`, `models.yaml`, `optimization.yaml`, `api.yaml`)
- `src/utils/config_loader.py`, `src/utils/export.py`, `src/utils/portfolio_utils.py`
- `src/ingestion/` — `downloader.py`, `validators.py`, `__init__.py` (`DataLayerResult` + `run_data_ingestion()`)
- `src/features/returns.py` — `compute_returns`, `ajustar_risk_free`, `converter_periodo`
- `src/features/asset_selection.py` — all 4 strategies + `select_assets()` + `select_assets_from_config()`
- `src/features/lag_features.py` — `build_lag_features`, `make_walk_forward_splits`
- `src/models/base.py` — `BaseReturnModel` (ABC)
- `src/models/ridge.py` — `RidgeReturnModel`
- `src/models/mlp.py` — `MLPReturnModel`
- `src/models/blending.py` — `blend_predictions`, `blend_from_config`
- `src/pipeline/context.py` — `PipelineContext` dataclass
- `src/pipeline/stages.py` — 8 stage functions (ingest → export)
- `src/pipeline/runner.py` — `run_pipeline()`, `STAGE_REGISTRY`, `STAGE_ORDER`
- `tests/smoke_test_data_layer.py` — 22 passing assert-based smoke tests
- CI workflow, `docker-compose.yml`, `pyproject.toml`

### Stage 2 — COMPLETE
- `src/utils/export.py` — extended: `save_returns`, `save_features`, `save_model`, `load_model`
- `src/pipeline/registry.py` — `list_runs()`, `load_run_manifest()`, `compare_runs()`
- `src/pipeline/stages.py:stage_export_artifacts` — saves ALL artifact types:
  - `artifacts/data/{run_id}_returns.csv` — processed returns
  - `artifacts/data/{run_id}_features.csv` + `_targets.csv` — feature/target matrices
  - `artifacts/models/{run_id}_{model}.joblib` — trained model (reloadable via `load_model`)
  - `artifacts/predictions/{run_id}_{model}_predictions.csv` — blended mu
  - `artifacts/weights/{run_id}_{model}_weights.csv` — portfolio weights
  - `artifacts/metrics/{run_id}_{model}_metrics.csv` — evaluation metrics
  - `artifacts/runs/{run_id}_manifest.json` — full run manifest (config + metrics + asset list)
- All pipeline parameters flow from `config/pipeline.yaml` — no hardcoded values

### Stage 3 — COMPLETE
- `src/api/main.py` — FastAPI app factory, CORS, all routers registered
- `src/api/deps.py` — shared helpers: `resolve_run()`, artifact path resolvers
- `src/api/schemas/requests.py` — `PipelineRunRequest` with stage validation
- `src/api/schemas/responses.py` — all response models (Health, Assets, Predictions, Weights, Frontier, Metrics, Runs)
- `src/api/routers/health.py` — `GET /health`
- `src/api/routers/assets.py` — `GET /assets`
- `src/api/routers/predictions.py` — `GET /predictions`
- `src/api/routers/pipeline.py` — `POST /pipeline/run`, `GET /pipeline/runs`, `GET /pipeline/runs/{run_id}`, `GET /pipeline/stages`
- `src/api/routers/portfolio.py` — `GET /portfolio/weights`, `GET /portfolio/frontier`
- `src/api/routers/metrics.py` — `GET /metrics/portfolio`, `GET /metrics/model`
- Swagger docs at `/docs`, ReDoc at `/redoc`
- `requirements.txt` — added `fastapi`, `uvicorn[standard]`

### Stage 4 — COMPLETE
- `src/persistence/__init__.py` + `src/persistence/database.py` — SQLite persistence layer
  - `init_db()` — creates `artifacts/pipeline_runs.db` with `runs` table (CREATE IF NOT EXISTS)
  - `upsert_run(manifest)` — INSERT OR UPDATE from manifest dict (idempotent)
  - `get_run(run_id)` — single record lookup
  - `list_runs(status, limit)` — filtered, paginated query (newest first)
  - `compare_runs(run_ids)` — side-by-side retrieval preserving input order
  - `sync_from_manifests()` — scans `artifacts/runs/*.json` and upserts all into DB
- `src/pipeline/stages.py:stage_export_artifacts` — now calls `upsert_run()` after manifest write
- `src/api/main.py` — lifespan handler: `init_db()` + `sync_from_manifests()` on startup
- `src/api/routers/history.py` — 4 new endpoints:
  - `GET /history/runs` — DB-backed list with `?status=` and `?limit=` filters
  - `GET /history/runs/compare` — side-by-side `?run_ids=id1,id2,...`
  - `GET /history/runs/{run_id}` — single record from DB
  - `POST /history/sync` — rebuild DB from manifest files

### Stage 5 — COMPLETE
- `Dockerfile` — rewritten: `python:3.10-slim`, no Node.js/Claude Code, uses `requirements-api.txt`, creates artifact dirs, `CMD uvicorn`
- `docker-compose.yml` — rewritten: no deprecated `version`, healthcheck on `/health`, `restart: unless-stopped`, volumes for `artifacts/` and `config/`
- `requirements-api.txt` — lean API-only deps (9 packages): excludes `torch`, `matplotlib`, `dash`, `plotly`, `ipykernel`, `seaborn`, `tikzplotlib` (notebook-only or legacy)
- `.dockerignore` — expanded: excludes `.git/`, `notebooks/`, `outputs/`, `artifacts/`, `tests/`, `article_official/`, bytecode, IDE files

Image size reduction: `torch` alone is ~2 GB; the API image installs only what the pipeline + FastAPI needs.

### Stage 8 — COMPLETE
- `config/scheduler.yaml` — scheduler config: `enabled`, `trigger` (interval/cron), `interval_hours`, `cron`, `max_instances`, `misfire_grace_seconds`, log dir
- `src/scheduler/__init__.py` — package init
- `src/scheduler/jobs.py` — `run_scheduled_pipeline()`: generates timestamped run_id (microsecond precision), writes per-run log to `artifacts/logs/{run_id}.log`, calls `run_pipeline()`, logs stage progress and final metrics, propagates failures
- `src/scheduler/scheduler.py` — `build_scheduler()`: reads config, creates APScheduler `BackgroundScheduler` with `IntervalTrigger` or `CronTrigger`, registers `pipeline_retrain` job; `get_scheduler()` singleton getter
- `src/api/routers/scheduler.py` — 4 endpoints:
  - `GET /scheduler/status` — enabled flag, running state, job list with next_run_time
  - `POST /scheduler/trigger` — manual one-shot pipeline trigger via BackgroundTasks
  - `GET /scheduler/logs` — list log files in `artifacts/logs/` with metadata
  - `GET /scheduler/logs/{run_id}` — return full log content (404 if missing)
- `src/api/main.py` — lifespan extended: `build_scheduler()` + `sched.start()` on startup, `sched.shutdown(wait=False)` on teardown
- `requirements.txt` + `requirements-api.txt` — added `apscheduler>=3.10`
- `tests/unit/test_scheduler.py` — 10 tests: scheduler construction (interval/cron triggers, disabled=None), job execution (log file created, run_id in content, failure raises, exception propagates)
- `tests/integration/test_scheduler_api.py` — 18 tests: all 4 endpoints, schema validation, 404 on unknown log
- **142 total tests passing** (22 smoke + 71 unit + 53+18 integration)

### Stage 7 — COMPLETE
- `.github/workflows/ci.yml` — 3-job pipeline: `lint → test → docker`
  - **lint**: `ruff check src/ tests/` (excludes legacy files)
  - **test**: matrix Python 3.10 + 3.11; installs `requirements-api.txt` + `requirements-dev.txt` (no torch); runs smoke → unit → integration
  - **docker**: `docker/build-push-action` with GitHub Actions cache (`type=gha`); only on `push` events, not PRs
- `pyproject.toml` — fixed `requires-python = ">=3.10"`, `target-version = "py310"`, added `ruff.exclude` for legacy files
- `requirements-dev.txt` — added `ruff>=0.4`
- Fixed lint errors in non-legacy files: removed unused imports from `evaluation.py`, `export.py`, `runner.py`, `test_pipeline.py`, `test_markowitz.py`, `test_sharpe.py`

### Stage 6 — COMPLETE
- `tests/conftest.py` — shared fixtures: `price_df`, `return_df`, `small_return_df`, `pipeline_cfg`, `minimal_context`
- `tests/unit/test_returns.py` — 17 tests: `compute_returns` (shape, values, edge cases), `ajustar_risk_free` (round-trip all freqs), `converter_periodo`
- `tests/unit/test_markowitz.py` — 17 tests: `portfolio_return`, `portfolio_volatility`, `minimize_volatility`, `solve_markowitz` (lambda sweep, max_weight)
- `tests/unit/test_sharpe.py` — 8 tests: `maximize_sharpe` (feasibility, bounds, dominance)
- `tests/unit/test_features.py` — 19 tests: `build_lag_features` (shape, no-look-ahead, NaN-free), `make_walk_forward_splits` (count, expansion, edge cases)
- `tests/integration/test_api.py` — 34 tests: all endpoints (health, assets, predictions, pipeline, portfolio, metrics, history), error codes, schema validation
- `tests/integration/test_pipeline.py` — 19 tests: per-stage correctness, full pipeline with synthetic data, artifact files, manifest JSON, metrics finite
- Fixed `stage_export_artifacts`: manifest now written AFTER `context.status = "completed"` (was written with `status='running'` before)
- `requirements-dev.txt` — `httpx` required by FastAPI TestClient (already listed)
- **136 total tests passing** (22 smoke + 61 unit + 53 integration)

**Legacy files (preserved for notebook compatibility — do not modify or delete):**
- `src/data/loader.py` — original loader, still used by notebooks
- `src/data/asset_selection.py` — original selection, still used by notebooks
- `src/models/lr.py` — original Ridge + MLP implementation
- `src/models/rnn.py` — original LSTM/RNN (PyTorch)
- `src/utils/portfolio_utils.py` — wrappers around optimization functions, used only by notebooks/visualization
- `src/utils/visualization.py` — notebook visualization utilities, not used by the main pipeline
- `notebooks/` — all notebooks

### Stage 9 — COMPLETE
- Deployed to Fly.io (app: `dlfinance-api`, region: `gru` / São Paulo)
- `fly.toml` — region `gru`, persistent volume `dl_artifacts` at `/app/artifacts`, HTTPS enforced, health check on `/health` every 30s, `min_machines_running = 1`
- `entrypoint.sh` — recreates artifact subdirectories at container start (volume mount shadows Dockerfile-created dirs)
- `.github/workflows/ci.yml` — extended with `deploy` job: `flyctl deploy --remote-only` on push to `main` only (requires `FLY_API_TOKEN` secret, `production` environment)

### Stage 10 — COMPLETE
- `README.md` — complete rewrite in English: results table, ML methodology, architecture overview, API endpoint reference, local execution, deployment, CI/CD, technology stack, MLOps practices
- `ARCHITECTURE.md` — updated: all modules marked implemented, full data flow diagram, updated directory tree, deployment details added

### Post-Stage 10 — Multi-model pipeline (uncommitted, current branch: dev)
- `config/pipeline.yaml` — `models.default` replaced by `models.enabled: ["ridge", "mlp"]`; every run now trains and compares all listed models
- `src/models/metrics.py` — new module: finance-specific ML diagnostics (`compute_model_metrics`): IC, ICIR, Spearman IC, hit rate, MAE, MSE, R²
- `src/pipeline/context.py` — all result fields (blended_mu, weights, metrics, portfolio_returns) are now `dict[str, ...]` keyed by strategy name; new field `model_metrics`
- `src/pipeline/stages.py` — `stage_train_models` trains all enabled models + computes walk-forward OOS diagnostics per model; `stage_optimize_portfolio` adds `"markowitz"` as an automatic classical baseline (uses historical means, no ML); `stage_evaluate` and `stage_export_artifacts` produce separate artifacts per strategy
- `src/utils/export.py` — new functions: `save_model_metrics`, `save_equity_curve`
- `src/pipeline/registry.py` — `_primary_metrics()` helper handles both multi-model and legacy manifest formats
- `src/api/deps.py` — added `_pick_model()`, `model_metrics_path()`; `resolve_run()` and `latest_completed_run()` now accept optional `model` param
- `src/api/schemas/responses.py` — new schemas: `ModelComparisonItem`, `PortfolioCompareResponse`, `EquityCurvePoint`, `EquityCurveResponse`; `ModelMetricsResponse` now has real metric fields (ic, icir, hit_rate, etc.) replacing the old `note` field
- `src/api/routers/portfolio.py` — all endpoints accept `?model=`; new endpoints: `GET /portfolio/equity-curve`, `GET /portfolio/compare`
- `src/api/routers/metrics.py` — `GET /metrics/portfolio` accepts `?model=`; `GET /metrics/model` now returns real walk-forward diagnostic metrics
- `src/persistence/database.py` — `_manifest_to_row()` updated to handle multi-model manifest format
- Artifact pattern extended: `{run_id}_{model}_model_metrics.csv` (ML diagnostics), `{run_id}_{model}_equity_curve.csv`
- Manifest now includes `models` (list of all strategies) and `primary_model` (highest Sharpe)
- `LEARNING_TRAIL.md`, `UNDERSTANDING.md` — project documentation files (untracked); do not delete
- `tests/unit/test_model_metrics.py` — unit tests for `compute_model_metrics`

---

## Key Conventions

### Config
- **No hardcoded parameters** anywhere in `src/`. All values (dates, tickers, frequencies, hyperparameters) come from `config/*.yaml`.
- Always load config via `src.utils.config_loader.get_config("pipeline")` — never open YAML files directly.
- `config/pipeline.yaml` is the single source of truth for pipeline parameters.
- Models to train are listed under `models.enabled` (list), not `models.default` (legacy, single model).

### Code style
- **Type hints** on all function signatures.
- **Google-style docstrings** on all public functions.
- **No short selling:** portfolio weights constrained to [0, 1], sum to 1.
- **Walk-forward only:** never use future data when training or predicting.
- **Model blending alpha:** default 0.3 — keep conservative to prevent ML overfitting.

### Imports
- New code imports from `src.ingestion`, `src.features`, `src.models`, `src.optimization`, `src.pipeline`, `src.api`, `src.utils`.
- Do not import from `src.data.*` in new modules — that package is legacy.

### Asset selection
- Prefer `method="stable_corr_pairs"` for long-term portfolios (set in `config/pipeline.yaml`).
- The correct function call is `select_assets(method="stable_corr_pairs")` — note: old code used `"stable_pairs"` (wrong name, now fixed).
- Always pass pre-downloaded `prices=` to `select_assets()` when prices are already in memory — avoids a redundant network call.

### Artifacts
- All pipeline outputs go to `artifacts/` subdirectories.
- Persist via `src/utils/export.py` — no ad-hoc CSV writes.
- `artifacts/runs/` stores execution metadata (timestamp, config snapshot, status).

### Optimization
- Solver: SLSQP via `scipy.optimize.minimize` — keep constraints explicit (bounds + equality).
- Risk-free rate: 15% p.a. (SELIC 2025); convert with `ajustar_risk_free(0.15, freq=...)`.

### Tests
- Smoke tests: plain `assert`-based, no pytest, runnable with `python tests/smoke_test_data_layer.py`.
- Unit tests: go in `tests/unit/`, use pytest.
- Integration tests: go in `tests/integration/`, may require network or full pipeline.
- Never mock the data download in integration tests — use real network calls or pre-saved fixtures.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke tests (no network needed)
python tests/smoke_test_data_layer.py

# Run unit tests
pytest tests/unit/ -v

# Run legacy notebook pipeline
jupyter notebook notebooks/00-compare_models.ipynb
```

Docker:
```bash
docker compose up        # starts API on :8000
docker build -t dl-finance .
```
