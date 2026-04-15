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

Full evolution plan: see **ROADMAP.md** (10 stages from research to deployed API).
Technical architecture reference: see **ARCHITECTURE.md**.

---

## Current Stage: Stage 3 — API implementation (next)

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

### Still stub (not yet implemented):
- `src/api/main.py`, all routers, all schemas (Stage 3)
- `tests/unit/` and `tests/integration/` test files (Stage 6)

**Legacy files (preserved for notebook compatibility — do not modify or delete):**
- `src/data/loader.py` — original loader, still used by notebooks
- `src/data/asset_selection.py` — original selection, still used by notebooks
- `src/models/lr.py` — original Ridge + MLP implementation
- `src/models/rnn.py` — original LSTM/RNN (PyTorch)
- `notebooks/` — all notebooks

---

## Key Conventions

### Config
- **No hardcoded parameters** anywhere in `src/`. All values (dates, tickers, frequencies, hyperparameters) come from `config/*.yaml`.
- Always load config via `src.utils.config_loader.get_config("pipeline")` — never open YAML files directly.
- `config/pipeline.yaml` is the single source of truth for pipeline parameters.

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
