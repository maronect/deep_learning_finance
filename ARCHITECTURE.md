# Architecture Reference

Technical map of the repository. See CLAUDE.md for conventions and ROADMAP.md for the evolution plan.

---

## Data Flow

```
config/pipeline.yaml
        |
        v
run_data_ingestion()              [src/ingestion/__init__.py]
  +-- load_prices()               [src/ingestion/downloader.py]      -> yfinance
  +-- validate / fill             [src/ingestion/validators.py]
  +-- compute_returns()           [src/features/returns.py]
  +-- select_assets()             [src/features/asset_selection.py]
        |
        v
PipelineContext                   [src/pipeline/context.py]
  +-- build_lag_features()        [src/features/lag_features.py]
  +-- make_walk_forward_splits()  [src/features/lag_features.py]
  +-- train() / predict()         [src/models/ridge.py | mlp.py]
  +-- blend_predictions()         [src/models/blending.py]
  +-- solve_markowitz()           [src/optimization/markowitz.py]
  +-- maximize_sharpe()           [src/optimization/sharpe.py]
  +-- evaluate()                  [src/optimization/evaluation.py]
        |
        v
artifacts/                        [src/utils/export.py]
  +-- data/*.csv
  +-- models/*.joblib
  +-- predictions/*.csv
  +-- metrics/*.csv
  +-- weights/*.csv
  +-- runs/*.json (manifest)
        |
        v
src/persistence/database.py       upsert_run() -> artifacts/pipeline_runs.db
        |
        v
src/api/                          FastAPI (uvicorn)
  GET  /health
  GET  /assets
  GET  /predictions
  POST /pipeline/run
  GET  /pipeline/runs
  GET  /portfolio/weights
  GET  /portfolio/frontier
  GET  /metrics/portfolio
  GET  /metrics/model
  GET  /history/runs
  GET  /scheduler/status
  POST /scheduler/trigger
```

---

## Storage Layer

Artifacts have two possible homes: the local filesystem (default) and an
optional AWS S3 bucket. The active backend is decided at runtime by
`src/utils/storage.py`, based solely on environment variables — never on YAML
config, so credentials stay out of version control and config snapshots.

### When S3 is used vs the local filesystem

- **S3 is the primary layer** when both `AWS_ACCESS_KEY_ID` and `AWS_S3_BUCKET`
  are set (`is_s3_enabled()` returns True). On write, every artifact is saved
  locally **and** mirrored to S3. On read, the API serves the local copy if it
  exists, otherwise it downloads the object from S3 to `/tmp`.
- **The local filesystem is the fallback** when either variable is missing. The
  pipeline then behaves exactly as before S3 was added. This keeps tests and
  local development working without any AWS credentials, and is why the fallback
  exists: the system must run end-to-end with zero cloud configuration.

### Environment variables that activate S3

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` | Yes | — | Access key; one of the two activation flags |
| `AWS_S3_BUCKET` | Yes | — | Bucket name; one of the two activation flags |
| `AWS_SECRET_ACCESS_KEY` | Yes (in practice) | — | Secret key for the credentials |
| `AWS_DEFAULT_REGION` | No | `sa-east-1` | Bucket region (São Paulo by default) |

### Bucket layout

Objects mirror the local `artifacts/` tree one-to-one. The S3 key is the local
path with the leading `artifacts/` prefix removed (`s3_key_from_path()`):

```
artifacts/weights/20260501_ridge_weights.csv  ->  weights/20260501_ridge_weights.csv
artifacts/runs/20260501_manifest.json         ->  runs/20260501_manifest.json
artifacts/data/20260501_returns.csv           ->  data/20260501_returns.csv
```

So a bucket holds the same `data/`, `models/`, `predictions/`, `metrics/`,
`weights/`, and `runs/` folders found on disk.

### Why this matters for deployment

On Fly.io the container can be recreated with an empty local volume. Because
manifests live in S3, `registry.list_runs()` / `load_run_manifest()` and
`database.sync_from_manifests()` fall back to S3 when the local `runs/`
directory is empty, letting the API rebuild its run history and serve artifacts
without any local state.

---

## Module Map

| Module | Status | Responsibility |
|---|---|---|
| `src/ingestion/__init__.py` | Implemented | `DataLayerResult` dataclass + `run_data_ingestion()` entry point |
| `src/ingestion/downloader.py` | Implemented | `load_prices()`, `load_prices_from_config()` via yfinance |
| `src/ingestion/validators.py` | Implemented | `drop_empty_rows()`, `filter_by_coverage()`, `fill_missing_prices()`, `validate_not_empty()` |
| `src/features/returns.py` | Implemented | `compute_returns()`, `ajustar_risk_free()`, `converter_periodo()` |
| `src/features/asset_selection.py` | Implemented | `select_assets()` with 4 strategies + `get_correlation_matrix()` |
| `src/features/lag_features.py` | Implemented | `build_lag_features()`, `make_walk_forward_splits()` |
| `src/models/base.py` | Implemented | `BaseReturnModel` ABC — shared interface for all models |
| `src/models/ridge.py` | Implemented | `RidgeReturnModel` — walk-forward Ridge Regression |
| `src/models/mlp.py` | Implemented | `MLPReturnModel` — walk-forward MLP (sklearn) |
| `src/models/blending.py` | Implemented | `blend_predictions()`, `blend_from_config()` |
| `src/optimization/markowitz.py` | Implemented | `portfolio_return()`, `portfolio_volatility()`, `solve_markowitz()` |
| `src/optimization/sharpe.py` | Implemented | `maximize_sharpe()` via SLSQP |
| `src/optimization/evaluation.py` | Implemented | portfolio metrics, efficient frontier computation |
| `src/pipeline/context.py` | Implemented | `PipelineContext` dataclass — shared state between stages |
| `src/pipeline/stages.py` | Implemented | 8 stage functions: ingest -> returns -> select -> features -> train -> predict -> optimize -> export |
| `src/pipeline/runner.py` | Implemented | `run_pipeline()`, `STAGE_REGISTRY`, `STAGE_ORDER` |
| `src/pipeline/registry.py` | Implemented | `list_runs()`, `load_run_manifest()`, `compare_runs()` |
| `src/persistence/database.py` | Implemented | `init_db()`, `upsert_run()`, `get_run()`, `list_runs()`, `sync_from_manifests()` |
| `src/api/main.py` | Implemented | FastAPI factory, lifespan, CORS, router registration |
| `src/api/deps.py` | Implemented | `resolve_run()`, artifact path resolvers |
| `src/api/routers/` | Implemented | health, assets, predictions, pipeline, portfolio, metrics, history, scheduler |
| `src/api/schemas/` | Implemented | Pydantic request and response models |
| `src/scheduler/jobs.py` | Implemented | `run_scheduled_pipeline()` — timestamped run_id, per-run log file |
| `src/scheduler/scheduler.py` | Implemented | `build_scheduler()`, `get_scheduler()` — APScheduler singleton |
| `src/utils/config_loader.py` | Implemented | `get_config(name)` — loads `config/<name>.yaml` |
| `src/utils/export.py` | Implemented | `save_returns()`, `save_features()`, `save_model()`, `load_model()`; mirrors writes to S3 when enabled |
| `src/utils/storage.py` | Implemented | Optional AWS S3 backend: `is_s3_enabled()`, `get_s3_client()`, `s3_upload/download/read_bytes/list()`, `s3_key_from_path()` |
| `src/data/` | Legacy | Kept for notebook compatibility — do not import in new code |
| `src/models/lr.py` | Legacy | Original Ridge + MLP — used by notebooks only |
| `src/models/rnn.py` | Legacy | LSTM/RNN (PyTorch) — not integrated into the pipeline |

---

## Directory Tree

```
deep_learning_finance/
|
+-- config/
|   +-- pipeline.yaml       # data (tickers, dates, frequency), asset selection, features,
|   |                        #   models, optimization, evaluation, artifacts
|   +-- models.yaml         # Ridge alpha, MLP hidden layers / neurons / activation
|   +-- optimization.yaml   # solver, risk-free rate, frontier point count
|   +-- api.yaml            # FastAPI host/port, CORS origins, Swagger metadata
|   +-- scheduler.yaml      # enabled flag, trigger (interval|cron), interval_hours, cron expr
|
+-- src/
|   +-- ingestion/
|   |   +-- __init__.py     # DataLayerResult + run_data_ingestion()
|   |   +-- downloader.py   # yfinance wrapper
|   |   +-- validators.py   # data quality checks and repair
|   |
|   +-- features/
|   |   +-- returns.py
|   |   +-- asset_selection.py
|   |   +-- lag_features.py
|   |
|   +-- models/
|   |   +-- base.py
|   |   +-- ridge.py
|   |   +-- mlp.py
|   |   +-- blending.py
|   |   +-- lr.py          # LEGACY
|   |   +-- rnn.py         # LEGACY
|   |
|   +-- optimization/
|   |   +-- markowitz.py
|   |   +-- sharpe.py
|   |   +-- evaluation.py
|   |
|   +-- pipeline/
|   |   +-- __init__.py
|   |   +-- context.py
|   |   +-- stages.py
|   |   +-- runner.py
|   |   +-- registry.py
|   |
|   +-- api/
|   |   +-- main.py
|   |   +-- deps.py
|   |   +-- routers/
|   |   |   +-- health.py
|   |   |   +-- assets.py
|   |   |   +-- predictions.py
|   |   |   +-- pipeline.py
|   |   |   +-- portfolio.py
|   |   |   +-- metrics.py
|   |   |   +-- history.py
|   |   |   +-- scheduler.py
|   |   +-- schemas/
|   |       +-- requests.py
|   |       +-- responses.py
|   |
|   +-- persistence/
|   |   +-- __init__.py
|   |   +-- database.py
|   |
|   +-- scheduler/
|   |   +-- __init__.py
|   |   +-- jobs.py
|   |   +-- scheduler.py
|   |
|   +-- utils/
|       +-- __init__.py
|       +-- config_loader.py
|       +-- export.py
|       +-- portfolio_utils.py  # LEGACY
|       +-- visualization.py    # LEGACY
|
+-- src/data/                   # LEGACY — notebook compatibility only
|   +-- loader.py
|   +-- asset_selection.py
|
+-- artifacts/                  # not committed to git
|   +-- data/
|   +-- models/
|   +-- predictions/
|   +-- metrics/
|   +-- weights/
|   +-- runs/
|   +-- logs/
|   +-- pipeline_runs.db
|
+-- tests/
|   +-- conftest.py
|   +-- smoke_test_data_layer.py   # 22 tests
|   +-- unit/                      # 61 tests
|   |   +-- test_returns.py
|   |   +-- test_features.py
|   |   +-- test_markowitz.py
|   |   +-- test_sharpe.py
|   |   +-- test_scheduler.py
|   +-- integration/               # 71 tests
|       +-- test_api.py
|       +-- test_pipeline.py
|       +-- test_scheduler_api.py
|
+-- notebooks/                     # exploratory — legacy
+-- article_official/article.tex   # academic article (LaTeX)
+-- .github/workflows/ci.yml       # lint -> test -> docker -> deploy
+-- Dockerfile                     # python:3.10-slim, requirements-api.txt
+-- docker-compose.yml             # local: API on :8000
+-- fly.toml                       # Fly.io: region gru, persistent volume, health check
+-- entrypoint.sh                  # creates artifact subdirs on container start
+-- pyproject.toml                 # project metadata, pytest, ruff
+-- requirements-api.txt           # runtime deps (no torch/matplotlib/seaborn)
+-- requirements-dev.txt           # pytest, httpx, ruff
+-- requirements.txt               # full deps including notebooks
```

---

## Config Structure

### `config/pipeline.yaml`

```yaml
data:
  tickers: []              # B3 ticker symbols (e.g. ["PETR4.SA", "VALE3.SA"])
  start_date: "2010-01-01"
  end_date:   "2025-12-31"
  frequency:  "monthly"   # daily | weekly | monthly

asset_selection:
  method:            "stable_corr_pairs"   # 4 options available
  n_assets:          10
  min_data_coverage: 0.85

features:
  lag_window:  24    # past periods used as lag features (t-1 ... t-lag_window)
  min_history: 36    # minimum periods required per asset

models:
  default:     "ridge"   # ridge | mlp
  blend_alpha: 0.3       # weight of ML in: final_mu = alpha * ml + (1-alpha) * hist
  train_ratio: 0.7

optimization:
  risk_free_rate: 0.15   # annual (SELIC 2025)
  frequency:      "monthly"
  solver:         "SLSQP"
  weight_bounds:  [0.0, 1.0]   # no short selling

artifacts:
  base_dir:        "artifacts"
  data_dir:        "artifacts/data"
  models_dir:      "artifacts/models"
  predictions_dir: "artifacts/predictions"
  metrics_dir:     "artifacts/metrics"
  weights_dir:     "artifacts/weights"
  runs_dir:        "artifacts/runs"
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| All parameters in `config/*.yaml` | Reproducibility — changing behavior never requires touching source code |
| `DataLayerResult` dataclass | Typed contract between ingestion and pipeline; avoids passing bare DataFrames |
| `PipelineContext` passed between stages | Shared mutable state avoids re-reading artifacts between adjacent stages |
| `blend_alpha = 0.3` default | Conservative — captures ML signal while limiting exposure to overfitting |
| Walk-forward splits only | Strict no-look-ahead: model at time t sees only t-1 and earlier |
| `artifacts/` excluded from git | Outputs are reproducible from config + code; no binary blobs in version control |
| `requirements-api.txt` separate from `requirements.txt` | Docker image excludes torch (~2 GB) and matplotlib — reduces image size significantly |
| `entrypoint.sh` recreates artifact dirs | Fly.io volume mount shadows Dockerfile-created dirs; script runs before uvicorn |
| SQLite for run history | Zero-dependency persistence; sufficient for the query patterns (list, single, compare) |
| S3 activated by env vars, not YAML | Credentials never enter version control or config snapshots; the same image runs locally (no creds) and in the cloud (creds injected) |
| S3 mirrors local writes, reads fall back to S3 | Keeps a fast local copy while surviving container recreation with an empty volume; local-only mode stays fully functional |
| `src/data/` kept as legacy package | Notebooks still import from it; deleting breaks `00-compare_models.ipynb` |
