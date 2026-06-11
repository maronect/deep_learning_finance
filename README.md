# Deep Learning Finance

A Machine Learning Engineering application for portfolio optimization on Brazilian
stocks (B3). It collects market data, trains ML models to predict expected returns,
optimizes portfolios with the Markowitz framework, and serves every result through a
REST API — running as a continuously retrained, containerized, cloud-deployed service.

**Live demo**

- Dashboard: https://dlfinance-api.fly.dev/
- Interactive API docs (Swagger): https://dlfinance-api.fly.dev/docs

The project began as a quantitative research study and was refactored into a
production-grade ML system: config-driven pipeline, walk-forward validation, run
history, automated retraining, S3-backed artifacts, CI/CD, and live deployment.

---

## Architecture

```
config/*.yaml                     # single source of truth for all parameters
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
src/utils/storage.py              # artifact persistence: AWS S3 (prod) or local FS
        |
        v
src/persistence/                  # SQLite run history (rebuilt from manifests on boot)
        |
        v
src/api/                          # FastAPI — exposes all results as REST endpoints
        |
        v
src/scheduler/                    # APScheduler — periodic automated retraining
```

The pipeline runs end-to-end or stage by stage. Every parameter (tickers, dates, model
hyperparameters, risk-free rate) lives in `config/*.yaml` — no hardcoded values in source.

---

## API Endpoints

Deployed at `https://dlfinance-api.fly.dev`. Docs: `/docs` (Swagger), `/redoc` (ReDoc).
The root path `/` serves an interactive portfolio dashboard.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check + number of runs available |
| GET | `/assets` | Assets used in the latest run |
| GET | `/predictions` | Expected return predictions per asset |
| POST | `/pipeline/run` | Trigger a full or partial pipeline run |
| GET | `/pipeline/runs` | List all pipeline runs |
| GET | `/pipeline/runs/{run_id}` | Single run manifest |
| GET | `/pipeline/stages` | Available pipeline stages |
| GET | `/portfolio/weights` | Optimized portfolio weights (`?model=`) |
| GET | `/portfolio/frontier` | Efficient frontier data points |
| GET | `/portfolio/equity-curve` | Cumulative equity curve for a strategy |
| GET | `/portfolio/compare` | Side-by-side comparison of all strategies |
| GET | `/metrics/portfolio` | Sharpe, return, volatility, cumulative return (`?model=`) |
| GET | `/metrics/model` | ML diagnostics: IC, ICIR, hit rate, MAE, R² |
| GET | `/history/runs` | DB-backed run list (filterable by status) |
| GET | `/history/runs/{run_id}` | Single run from DB |
| GET | `/history/runs/compare` | Side-by-side comparison of multiple runs |
| POST | `/history/sync` | Rebuild DB from manifest files |
| GET | `/scheduler/status` | Scheduler state and next run time |
| POST | `/scheduler/trigger` | Manual one-shot pipeline trigger |
| GET | `/scheduler/logs` | List execution log files |
| GET | `/scheduler/logs/{run_id}` | Full log for a specific run |

Endpoints that accept `?model=` return results for a specific strategy (`ridge`, `mlp`,
or `markowitz`); without it, the run's best strategy is used.

---

## ML Methodology

### Data
- **Source**: yfinance (B3 tickers)
- **Period**: 2010–today (+15 years)
- **Frequency**: monthly returns from adjusted closing prices (logarithmic throughout)
- **Asset selection**: `stable_corr_pairs` — picks the assets with the most stable
  pairwise correlations over time, reducing covariance estimation noise

### Feature engineering
- **Lag features**: past returns up to a configurable lag window (`features.lag_window`)
- **Walk-forward splits**: the training window expands one period at a time — no leakage

### Models (trained and compared every run)
- **Ridge Regression** (`src/models/ridge.py`): L2-regularized linear model
- **MLP** (`src/models/mlp.py`): two-hidden-layer neural network (scikit-learn)
- **Markowitz baseline** (`src/optimization/`): classical mean-variance using historical
  means — the reference both ML strategies are measured against
- **Blending** (`src/models/blending.py`): `final_mu = alpha * ml_pred + (1 - alpha) * hist_mean`,
  default `alpha = 0.3` — keeps ML signal without full exposure to overfitting

### Portfolio optimization
- **Formulation**: Markowitz mean-variance, no short selling (weights in [0, 1], sum to 1)
- **Objective**: maximize Sharpe ratio — `(mu_p - rf) / sigma_p`
- **Solver**: SLSQP via `scipy.optimize.minimize`
- **Risk-free rate**: 15% p.a. (SELIC 2025), converted to the pipeline frequency

### Model diagnostics
Beyond portfolio metrics, each model is scored with finance-specific walk-forward
diagnostics (`src/models/metrics.py`): Information Coefficient (IC), ICIR, Spearman IC,
hit rate, MAE, MSE, and R².

---

## Storage

Artifacts (returns, models, predictions, weights, metrics, run manifests) are persisted
through `src/utils/storage.py`, which selects a backend automatically:

- **AWS S3** when the `AWS_*` environment variables are set — used in production. The S3
  bucket is the single source of truth; the deployed app is stateless and rebuilds its
  run history from S3 manifests on every boot.
- **Local filesystem** otherwise — keeps tests and credential-free local development working.

S3 keys mirror the local `artifacts/` layout, so the same code paths serve both backends.

---

## Project Structure

```
deep_learning_finance/
|
+-- config/                       # YAML config — single source of truth
|   +-- pipeline.yaml             # dates, tickers, asset selection, features, enabled models
|   +-- models.yaml               # Ridge alpha, MLP layers/neurons
|   +-- optimization.yaml         # solver, risk-free rate, frontier points
|   +-- api.yaml                  # FastAPI host/port, CORS, Swagger metadata
|   +-- scheduler.yaml            # APScheduler trigger, interval, cron
|
+-- src/
|   +-- ingestion/                # download + validate market data (yfinance)
|   +-- features/                 # returns, asset selection, lag feature matrix
|   +-- models/                   # Ridge / MLP / blending / model diagnostics
|   +-- optimization/             # Markowitz, max-Sharpe (SLSQP), portfolio metrics
|   +-- pipeline/                 # orchestration: runner, stages, context, registry
|   +-- api/                      # FastAPI: app factory, routers, Pydantic schemas
|   +-- persistence/              # SQLite run history (rebuilt from manifests on boot)
|   +-- scheduler/                # APScheduler periodic retraining
|   +-- utils/                    # config loader, artifact export, S3 storage layer
|
+-- artifacts/                    # pipeline outputs (S3 in prod; gitignored locally)
|   +-- data/ models/ predictions/ metrics/ weights/ runs/ logs/
|
+-- tests/
|   +-- smoke_test_data_layer.py  # assert-based smoke tests (no pytest)
|   +-- unit/                     # pytest unit tests per module
|   +-- integration/              # end-to-end API and pipeline tests
|
+-- .github/workflows/ci.yml      # CI/CD: lint -> test -> docker build -> deploy to Fly.io
+-- Dockerfile                    # python:3.10-slim, requirements-api.txt only
+-- docker-compose.yml            # local orchestration: API on :8000
+-- fly.toml                      # Fly.io deployment: region gru, stateless (S3-backed)
+-- pyproject.toml                # project metadata, pytest + ruff config
+-- requirements-api.txt          # lean runtime deps (no torch/matplotlib)
+-- requirements-dev.txt          # pytest, httpx, ruff
```

---

## Local Execution

### Install
```bash
pip install -r requirements.txt       # full deps (includes torch/matplotlib for notebooks)
# or
pip install -r requirements-api.txt    # API + pipeline only
```

### Run the pipeline
```bash
python -m src.pipeline.runner
```

### Run the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker (recommended for the API)
```bash
docker compose up        # starts API on :8000
```

### Tests
```bash
python tests/smoke_test_data_layer.py   # smoke tests, no pytest
pytest tests/unit/ -v                    # unit tests
pytest tests/integration/ -v            # integration tests
```

---

## Deployment

Deployed on [Fly.io](https://fly.io) — region `gru` (São Paulo).

- **App**: `dlfinance-api`
- **Storage**: AWS S3 (`sa-east-1`) — the app is stateless; artifacts live in the bucket
- **Health check**: `GET /health` every 30 seconds
- **HTTPS**: enforced by Fly.io
- **Auto-deploy**: every push to `main` triggers CI (lint → test → docker → deploy)

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
4. **Deploy** — `flyctl deploy --remote-only` (push to `main` only)

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
| Run history | SQLite (stdlib `sqlite3`) |
| Artifact storage | AWS S3 (boto3) with local fallback |
| Scheduling | APScheduler 3.x |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Fly.io |
| Linting | Ruff |
| Testing | pytest + httpx |

---

## MLOps Practices Demonstrated

- **Reproducibility**: all parameters in `config/*.yaml`, no hardcoded values in source
- **Artifact traceability**: every run writes a manifest (config snapshot, metrics, status)
- **Run history**: SQLite layer enables cross-run comparison without reading raw files
- **Walk-forward validation**: strict temporal split at every training step — no leakage
- **Multi-model comparison**: Ridge, MLP, and the Markowitz baseline scored every run
- **Automated retraining**: APScheduler triggers the full pipeline on a configurable schedule
- **Cloud-native storage**: S3-backed artifacts; the deployed service is fully stateless
- **CI/CD**: lint + test + build + deploy on every push to `main`
- **Containerization**: lean Docker image (~250 MB) excluding notebook/research deps

---

## Academic Background

This project originated as a quantitative research study comparing three portfolio
optimization strategies on Brazilian stocks; the results are documented in
`article_official/article.tex`. The codebase was subsequently refactored and extended
into a production ML Engineering application.

---

*Educational and research project. Results do not constitute investment advice.*
