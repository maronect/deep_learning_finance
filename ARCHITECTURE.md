# Architecture Reference

Technical map of the repository. See CLAUDE.md for conventions and ROADMAP.md for the evolution plan.

---

## Directory Tree

```
deep_learning_finance/
│
├── config/                              # YAML config — single source of truth for all parameters
│   ├── pipeline.yaml                    # data dates/tickers, asset selection, features, optimization, artifacts
│   ├── models.yaml                      # ML hyperparameters: Ridge α, MLP layers/neurons, RNN hidden size
│   ├── optimization.yaml                # Markowitz solver settings, risk-free rate, frontier points
│   └── api.yaml                         # FastAPI host/port, CORS, Swagger metadata
│
├── src/
│   │
│   ├── ingestion/                       # [X] IMPLEMENTED — Stage 1
│   │   ├── __init__.py                  # DataLayerResult dataclass + run_data_ingestion() entry point
│   │   ├── downloader.py                # load_prices(), load_prices_from_config() via yfinance
│   │   └── validators.py                # drop_empty_rows(), filter_by_coverage(), fill_missing_prices(), validate_not_empty()
│   │
│   ├── features/                        # [X] IMPLEMENTED — Stage 1 (lag_features stub)
│   │   ├── __init__.py
│   │   ├── returns.py                   # compute_returns(), ajustar_risk_free(), converter_periodo()
│   │   ├── asset_selection.py           # 4 strategies + select_assets() + select_assets_from_config() + get_correlation_matrix()
│   │   └── lag_features.py              # [ ] STUB — walk-forward lag matrix builder (next: Stage 2)
│   │
│   ├── models/                          # [ ] STUBS — to be implemented in Stage 2
│   │   ├── base.py                      # Abstract base class / interface for all models
│   │   ├── ridge.py                     # Ridge Regression wrapper (config-driven, walk-forward)
│   │   ├── mlp.py                       # MLP wrapper (config-driven, walk-forward)
│   │   ├── blending.py                  # final_mu = alpha * ml + (1-alpha) * hist_mean
│   │   ├── lr.py                        # ! LEGACY ! — original Ridge + MLP, used by notebooks
│   │   └── rnn.py                       # ! LEGACY ! — LSTM/RNN (PyTorch), not yet integrated
│   │
│   ├── optimization/                    # ! LEGACY ! — not yet refactored to new conventions
│   │   ├── markowitz.py                 # Markowitz formulation: μ, Σ, SLSQP constraints
│   │   ├── sharpe.py                    # Maximize Sharpe Ratio: max (μ_p - rf) / σ_p
│   │   └── evaluation.py               # Sharpe, annualized return, volatility, cumulative return
│   │
│   ├── pipeline/                        # [ ] STUBS — to be implemented in Stage 2
│   │   ├── __init__.py
│   │   ├── runner.py                    # Entry point: run all or selected stages
│   │   ├── stages.py                    # ingest → returns → select → features → train → predict → optimize → evaluate → export
│   │   └── context.py                   # PipelineContext dataclass: carries config + intermediate state
│   │
│   ├── api/                             # [ ] STUBS — to be implemented in Stage 3
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI app factory, CORS middleware, router registration
│   │   ├── routers/
│   │   │   ├── health.py                # GET /health
│   │   │   ├── assets.py                # GET /assets
│   │   │   ├── predictions.py           # GET /predictions
│   │   │   ├── pipeline.py              # POST /pipeline/run
│   │   │   ├── portfolio.py             # GET /portfolio/weights  GET /portfolio/frontier
│   │   │   └── metrics.py              # GET /metrics/model  GET /metrics/portfolio
│   │   └── schemas/
│   │       ├── requests.py              # Pydantic input schemas
│   │       └── responses.py             # Pydantic output schemas
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py             # [X] get_config(name) — loads config/<name>.yaml
│       ├── visualization.py             # ! LEGACY ! — 5 comparative charts (matplotlib/seaborn)
│       ├── export.py                    # ! LEGACY ! — save metrics/weights/predictions to CSV
│       └── portfolio_utils.py           # ! LEGACY ! — shared portfolio helpers
│
├── src/data/                            # ! LEGACY ! PACKAGE — kept for notebook compatibility only
│   ├── loader.py                        # Original load_prices() + compute_returns() — DO NOT import in new code
│   └── asset_selection.py              # Original select_assets() — DO NOT import in new code
│
├── artifacts/                           # Pipeline outputs — not committed to git
│   ├── data/                            # Processed return DataFrames
│   ├── models/                          # Serialised model parameters
│   ├── predictions/                     # Expected return vectors per run
│   ├── metrics/                         # Model evaluation (MAE, R²) and portfolio metrics
│   ├── weights/                         # Optimised portfolio weights per run
│   └── runs/                            # Execution logs: timestamp, config snapshot, status
│
├── tests/
│   ├── conftest.py                      # Shared pytest fixtures (synthetic data, mock configs)
│   ├── smoke_test_data_layer.py         # [X] 22 assert-based smoke tests — run without pytest
│   ├── unit/                            # [ ] STUBS — pytest unit tests per module
│   │   ├── test_returns.py
│   │   ├── test_features.py
│   │   ├── test_markowitz.py
│   │   └── test_sharpe.py
│   └── integration/                     # [ ] STUBS — end-to-end tests
│       ├── test_pipeline.py
│       └── test_api.py
│
├── notebooks/                           # Exploratory — not part of the pipeline, kept as-is
│   ├── 00-compare_models.ipynb          # Main legacy pipeline — reference results
│   ├── 01-markowitz_optimization.ipynb
│   └── 02-linear_regretion.ipynb
│
├── article_official/article.tex         # Academic article (LaTeX)
├── .github/workflows/ci.yml             # CI: install deps → pytest → docker build
├── Dockerfile                            # Production image
├── docker-compose.yml                   # Local service: API on :8000
├── pyproject.toml                       # Build, pytest, ruff config
├── requirements.txt                     # Production deps (includes pyyaml)
└── requirements-dev.txt                 # Dev deps: pytest, httpx
```

---

## Module Status

| Module | Status | Notes |
|--------|--------|-------|
| `src/ingestion/` | [X] Implemented | Full data download, validation, unified entry point |
| `src/features/returns.py` | [X] Implemented | compute_returns, ajustar_risk_free, converter_periodo |
| `src/features/asset_selection.py` | [X] Implemented | 4 strategies, config-driven |
| `src/features/lag_features.py` | [ ] Stub | Next: Stage 2 |
| `src/utils/config_loader.py` | [X] Implemented | get_config(name) |
| `src/models/*.py` (new) | [ ] Stubs | Next: Stage 2 |
| `src/optimization/*.py` | ! Legacy ! | Works, not yet refactored |
| `src/pipeline/*.py` | [ ] Stubs | Next: Stage 2 |
| `src/api/` | [ ] Stubs | Stage 3 |
| `tests/smoke_test_data_layer.py` | [X] Implemented | 22 tests, no pytest required |
| `tests/unit/` | [ ] Stubs | Stage 6 |
| `tests/integration/` | [ ] Stubs | Stage 6 |

---

## Data Flow (target — Stages 1–3)

```
config/pipeline.yaml
        │
        ▼
run_data_ingestion()          [src/ingestion/__init__.py]
  ├── load_prices()           [src/ingestion/downloader.py]   → yfinance
  ├── compute_returns()       [src/features/returns.py]
  └── select_assets()         [src/features/asset_selection.py]
        │
        ▼
PipelineContext               [src/pipeline/context.py]
  ├── build_lag_features()    [src/features/lag_features.py]
  ├── train() / predict()     [src/models/ridge.py | mlp.py | rnn.py]
  ├── blend()                 [src/models/blending.py]
  ├── optimize()              [src/optimization/markowitz.py + sharpe.py]
  └── evaluate()              [src/optimization/evaluation.py]
        │
        ▼
artifacts/                    [src/utils/export.py]
        │
        ▼
src/api/                      [FastAPI — Stage 3]
  GET  /predictions
  GET  /portfolio/weights
  GET  /portfolio/frontier
  GET  /metrics/portfolio
  POST /pipeline/run
```

---

## Config Structure (`config/pipeline.yaml`)

```yaml
data:
  tickers: [...]          # B3 ticker symbols
  start_date: "2010-01-01"
  end_date:   "2025-12-31"
  frequency:  "monthly"

asset_selection:
  method:            "stable_corr_pairs"
  n_assets:          10
  min_data_coverage: 0.85

features:
  lag_window:  24
  min_history: 36

models:
  default:     "ridge"
  blend_alpha: 0.3
  train_ratio: 0.7

optimization:
  risk_free_rate: 0.15
  frequency:      "monthly"
  solver:         "SLSQP"
  weight_bounds:  [0.0, 1.0]

evaluation:
  metrics: [sharpe_ratio, annualized_return, annualized_volatility, cumulative_return]

artifacts:
  base_dir:         "artifacts"
  data_dir:         "artifacts/data"
  models_dir:       "artifacts/models"
  predictions_dir:  "artifacts/predictions"
  metrics_dir:      "artifacts/metrics"
  weights_dir:      "artifacts/weights"
  runs_dir:         "artifacts/runs"
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `DataLayerResult` dataclass in `src/ingestion/__init__.py` | Typed contract between ingestion and pipeline; avoids passing bare DataFrames |
| `select_assets(prices=...)` optional param | Prevents double-downloading when prices are already in memory |
| `src/data/` kept as legacy | Notebooks still import from it; deleting would break `00-compare_models.ipynb` |
| `artifacts/` not in git | Outputs are reproducible from config + code; no binary blobs in version control |
| `config/` outside `src/` | Config is operational, not code — can change without touching source |
| Stubs with docstrings only | Scaffolding lets the codebase compile and import cleanly before implementation begins |
