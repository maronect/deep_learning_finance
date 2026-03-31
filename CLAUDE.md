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

## Current Stage: Stage 1 — Refactoring (in progress)

**Completed in Stage 1:**
- New directory structure created (`config/`, `src/ingestion/`, `src/features/`, `src/pipeline/`, `src/api/`, `artifacts/`, `tests/`)
- YAML-based config system: `config/pipeline.yaml`, `models.yaml`, `optimization.yaml`, `api.yaml`
- `src/utils/config_loader.py` — implemented, all modules must use this to read config
- `src/ingestion/` — **fully implemented**: `downloader.py`, `validators.py`, `__init__.py` (with `DataLayerResult` + `run_data_ingestion()`)
- `src/features/returns.py` — **fully implemented**: `compute_returns`, `ajustar_risk_free`, `converter_periodo`
- `src/features/asset_selection.py` — **fully implemented**: all 4 strategies + `select_assets()` + `select_assets_from_config()`
- `tests/smoke_test_data_layer.py` — 22 assert-based smoke tests for the data layer
- CI workflow (`.github/workflows/ci.yml`), `docker-compose.yml`, `pyproject.toml`

**Still stub (docstring only — not yet implemented):**
- `src/features/lag_features.py`
- `src/models/base.py`, `ridge.py`, `mlp.py`, `blending.py`
- `src/pipeline/runner.py`, `stages.py`, `context.py`
- `src/api/main.py`, all routers, all schemas
- `tests/unit/` and `tests/integration/` test files

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
