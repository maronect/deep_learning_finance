# Deep Learning Finance

A Machine Learning Engineering application for portfolio optimization on Brazilian stocks (B3).
The system collects financial data, trains ML models to predict expected returns, optimizes portfolios
using the Markowitz framework, and exposes results through a REST API — running as a continuously
updated, containerized, deployed service.

---

## Results

Backtested on 10 Brazilian equities (B3), monthly frequency, 2010–2025:

| Strategy | Sharpe Ratio | Annualized Return | Annualized Volatility | Cumulative Return |
|---|---|---|---|---|
| Classic Markowitz | 0.543 | 33.16% | 27.48% | 4.41x |
| **Markowitz + Ridge Regression** | **0.591** | **35.73%** | 28.56% | **4.86x** |
| Markowitz + MLP | 0.572 | 34.33% | 27.66% | 4.63x |

Ridge Regression improves Sharpe by ~9% over the classic baseline. Blending ML predictions
conservatively (alpha = 0.3) keeps the benefit while limiting overfitting risk.

---

## Architecture

```
config/pipeline.yaml              # single source of truth for all parameters
        |
        v
src/ingestion/                    # download + validate prices (yfinance)
        |
        v
src/features/                     # returns, asset selection, lag feature matrix
        |
        v
src/models/                       # Ridge / MLP walk-forward training + blending
        |
        v
src/optimization/                 # Markowitz + max-Sharpe (SLSQP)
        |
        v
artifacts/                        # persisted outputs (data, models, weights, metrics)
        |
        v
src/persistence/                  # SQLite run history
        |
        v
src/api/                          # FastAPI — exposes all results as REST endpoints
        |
        v
src/scheduler/                    # APScheduler — periodic retraining
```

The pipeline can be run end-to-end or stage by stage. Every parameter (tickers, dates, model
hyperparameters, risk-free rate) is defined in `config/*.yaml` — no hardcoded values in source.

---

## API Endpoints

The API is deployed at `https://dlfinance-api.fly.dev`. Interactive docs: `/docs` (Swagger), `/redoc`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/assets` | List assets used in the latest run |
| GET | `/predictions` | Expected return predictions per asset |
| POST | `/pipeline/run` | Trigger a full or partial pipeline run |
| GET | `/pipeline/runs` | List all pipeline runs |
| GET | `/pipeline/runs/{run_id}` | Single run manifest |
| GET | `/pipeline/stages` | Available pipeline stages |
| GET | `/portfolio/weights` | Optimized portfolio weights |
| GET | `/portfolio/frontier` | Efficient frontier data points |
| GET | `/metrics/portfolio` | Sharpe, return, volatility, cumulative return |
| GET | `/metrics/model` | Model evaluation metrics (MAE, R²) |
| GET | `/history/runs` | DB-backed run list (filterable by status) |
| GET | `/history/runs/{run_id}` | Single run from DB |
| GET | `/history/runs/compare` | Side-by-side comparison of multiple runs |
| POST | `/history/sync` | Rebuild DB from manifest files |
| GET | `/scheduler/status` | Scheduler state and next run time |
| POST | `/scheduler/trigger` | Manual one-shot pipeline trigger |
| GET | `/scheduler/logs` | List execution log files |
| GET | `/scheduler/logs/{run_id}` | Full log for a specific run |

---

## ML Methodology

### Data

- **Source**: yfinance (B3 tickers)
- **Period**: 2010–2025 (15 years)
- **Frequency**: monthly returns computed from adjusted closing prices
- **Asset selection**: `stable_corr_pairs` — selects the N assets with the most stable
  pairwise correlations over time, reducing covariance estimation noise

### Feature Engineering

- **Lag features**: returns at lags t-1 through t-24 (configurable via `features.lag_window`)
- **Walk-forward splits**: training window expands month by month — no data leakage

### Models

- **Ridge Regression** (`src/models/ridge.py`): L2-regularized linear model, alpha from `config/models.yaml`
- **MLP** (`src/models/mlp.py`): two-hidden-layer neural network (sklearn), walk-forward retrained
- **Blending** (`src/models/blending.py`): `final_mu = alpha * ml_pred + (1 - alpha) * hist_mean`
  with default alpha = 0.3 — conservative weight keeps ML signal without full exposure to overfitting

### Portfolio Optimization

- **Formulation**: Markowitz mean-variance with no short selling (weights in [0, 1], sum to 1)
- **Objective**: maximize Sharpe Ratio — `(mu_p - rf) / sigma_p`
- **Solver**: SLSQP via `scipy.optimize.minimize`
- **Risk-free rate**: 15% p.a. (SELIC 2025), converted to the pipeline frequency

---

## Project Structure

```
deep_learning_finance/
|
+-- config/
|   +-- pipeline.yaml         # data dates, tickers, asset selection, features, optimization
|   +-- models.yaml           # Ridge alpha, MLP layers/neurons
|   +-- optimization.yaml     # solver, risk-free rate, frontier points
|   +-- api.yaml              # FastAPI host/port, CORS, Swagger metadata
|   +-- scheduler.yaml        # APScheduler trigger, interval, cron
|
+-- src/
|   +-- ingestion/            # download and validate raw market data
|   |   +-- downloader.py     # load_prices() via yfinance
|   |   +-- validators.py     # coverage checks, forward-fill, integrity assertions
|   |
|   +-- features/             # feature engineering
|   |   +-- returns.py        # compute_returns(), ajustar_risk_free(), converter_periodo()
|   |   +-- asset_selection.py# select_assets() with 4 strategies
|   |   +-- lag_features.py   # build_lag_features(), make_walk_forward_splits()
|   |
|   +-- models/               # ML model training and prediction
|   |   +-- base.py           # BaseReturnModel ABC
|   |   +-- ridge.py          # RidgeReturnModel — walk-forward Ridge
|   |   +-- mlp.py            # MLPReturnModel — walk-forward MLP
|   |   +-- blending.py       # blend_predictions(), blend_from_config()
|   |
|   +-- optimization/         # portfolio optimization
|   |   +-- markowitz.py      # portfolio_return(), portfolio_volatility(), solve_markowitz()
|   |   +-- sharpe.py         # maximize_sharpe()
|   |   +-- evaluation.py     # portfolio metrics + efficient frontier
|   |
|   +-- pipeline/             # orchestration
|   |   +-- runner.py         # run_pipeline(), STAGE_REGISTRY, STAGE_ORDER
|   |   +-- stages.py         # 8 stage functions: ingest -> export
|   |   +-- context.py        # PipelineContext dataclass (shared state)
|   |   +-- registry.py       # list_runs(), load_run_manifest(), compare_runs()
|   |
|   +-- api/                  # FastAPI serving layer
|   |   +-- main.py           # app factory, lifespan, CORS, router registration
|   |   +-- deps.py           # shared helpers: resolve_run(), artifact path resolvers
|   |   +-- routers/          # one file per resource group
|   |   +-- schemas/          # Pydantic request and response models
|   |
|   +-- persistence/          # SQLite run history
|   |   +-- database.py       # init_db(), upsert_run(), get_run(), list_runs(), sync_from_manifests()
|   |
|   +-- scheduler/            # periodic retraining
|   |   +-- jobs.py           # run_scheduled_pipeline() — generates run_id, writes log
|   |   +-- scheduler.py      # build_scheduler(), get_scheduler() (APScheduler)
|   |
|   +-- utils/
|       +-- config_loader.py  # get_config(name) — loads config/<name>.yaml
|       +-- export.py         # save/load artifacts (models, CSVs)
|
+-- artifacts/                # pipeline outputs (not committed to git)
|   +-- data/                 # processed returns and feature datasets
|   +-- models/               # serialized model parameters (.joblib)
|   +-- predictions/          # expected return vectors per run
|   +-- metrics/              # model (MAE, R2) and portfolio metrics
|   +-- weights/              # optimized portfolio weights
|   +-- runs/                 # run manifests (JSON): config snapshot + metrics + status
|   +-- logs/                 # scheduler execution logs
|
+-- tests/
|   +-- conftest.py           # shared fixtures: price_df, pipeline_cfg, minimal_context
|   +-- smoke_test_data_layer.py  # 22 assert-based smoke tests (no pytest)
|   +-- unit/                 # pytest unit tests per module (61 tests)
|   +-- integration/          # end-to-end API and pipeline tests (71 tests)
|
+-- notebooks/                # exploratory analysis — legacy, not part of the pipeline
|
+-- .github/workflows/ci.yml  # CI/CD: lint -> test -> docker build -> deploy to Fly.io
+-- Dockerfile                # python:3.10-slim, requirements-api.txt only
+-- docker-compose.yml        # local orchestration: API on :8000
+-- fly.toml                  # Fly.io deployment: region gru, persistent volume, health check
+-- pyproject.toml            # project metadata, pytest config, ruff config
+-- requirements-api.txt      # lean runtime deps (no torch/matplotlib)
+-- requirements-dev.txt      # pytest, httpx, ruff
```

---

## Local Execution

### Prerequisites

```bash
pip install -r requirements.txt   # full deps including torch and matplotlib
# or
pip install -r requirements-api.txt  # API + pipeline only (no notebooks)
```

### Run the pipeline

```bash
python -m src.pipeline.runner
```

### Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker (recommended for API)

```bash
docker compose up        # starts API on :8000
docker build -t dl-finance .
```

### Tests

```bash
python tests/smoke_test_data_layer.py   # 22 smoke tests, no pytest
pytest tests/unit/ -v                   # 61 unit tests
pytest tests/integration/ -v           # 71 integration tests
```

---

## Deployment

The application is deployed on [Fly.io](https://fly.io) — region `gru` (São Paulo).

- **App**: `dlfinance-api`
- **Persistent volume**: `/app/artifacts` mounted as `dl_artifacts` — pipeline outputs survive deploys
- **Health check**: `GET /health` every 30 seconds
- **HTTPS**: enforced by Fly.io
- **Auto-deploy**: every push to `main` triggers the CI pipeline (lint → test → docker → deploy)

Manual deploy:

```bash
fly deploy
```

---

## CI/CD Pipeline

Every push or pull request to `main` or `dev` runs:

1. **Lint** — `ruff check src/ tests/`
2. **Test** — smoke + unit + integration on Python 3.10 and 3.11
3. **Docker build** — verifies the image builds cleanly (push events only)
4. **Deploy** — `flyctl deploy --remote-only` (push to `main` only, requires `FLY_API_TOKEN` secret)

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data source | yfinance |
| ML models | scikit-learn (Ridge, MLP) |
| Optimization | scipy (SLSQP) |
| API framework | FastAPI + Uvicorn |
| Schema validation | Pydantic v2 |
| Persistence | SQLite (via stdlib `sqlite3`) |
| Scheduling | APScheduler 3.x |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Fly.io |
| Linting | Ruff |
| Testing | pytest + httpx |

---

## MLOps Practices Demonstrated

- **Reproducibility**: all parameters in `config/*.yaml`, no hardcoded values in source
- **Artifact traceability**: every run writes a manifest JSON with config snapshot, metrics, and status
- **Run history**: SQLite layer enables cross-run comparison without reading raw files
- **Walk-forward validation**: strict temporal split at every training step — no data leakage
- **Automated retraining**: APScheduler triggers the full pipeline on a configurable interval
- **CI/CD**: lint + test + build + deploy on every push to main
- **Containerization**: lean Docker image (~200 MB) excluding notebook and research deps
- **Persistent storage**: Fly.io volume ensures artifacts survive container restarts and deploys

---

## Academic Background

This project originated as a quantitative research study comparing three portfolio optimization
strategies on Brazilian stocks. The research results are documented in `article_official/article.tex`.
The codebase was subsequently refactored and extended into a production ML Engineering application
following the ten-stage roadmap in `ROADMAP.md`.

---

*Educational and research project. Results do not constitute investment advice.*
