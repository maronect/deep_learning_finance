# CLAUDE.md

## Roadmap
Full plan is in ROADMAP.md. Current stage: Etapa 1 — reorganização e 
refatoração da estrutura do projeto. See ROADMAP.md for details on 
upcoming stages.

## Language
- Conversations can be in Portuguese
- All code, comments, docstrings, and git commits must be in English
- You may respond to me in Portuguese

## Project Overview
Quantitative research project comparing three portfolio optimization strategies for Brazilian stocks (B3, 2010–2025): Classic Markowitz, Markowitz + Ridge Regression, and Markowitz + MLP. All approaches are based on Modern Portfolio Theory and evaluated by Sharpe Ratio. Ridge Regression achieved the best result (Sharpe 0.591 vs 0.543 classic).

## Architecture

| Path | Role |
|------|------|
| `src/data/loader.py` | Download prices via yfinance, compute returns at multiple frequencies |
| `src/data/asset_selection.py` | Select uncorrelated assets from 35+ Brazilian stocks (4 strategies) |
| `src/models/lr.py` | Ridge Regression and MLP return predictions (scikit-learn) |
| `src/models/rnn.py` | LSTM/RNN model in PyTorch — implemented but not yet integrated |
| `src/optimization/markowitz.py` | Markowitz formulation: return, volatility, solver |
| `src/optimization/sharpe.py` | Maximize Sharpe Ratio via SLSQP |
| `src/optimization/evaluation.py` | Portfolio metrics: Sharpe, annualized return, volatility, cumulative return |
| `src/utils/visualization.py` | 5 comparative charts (matplotlib/seaborn) |
| `src/utils/export.py` | Export metrics, weights, predictions to CSV |
| `notebooks/00-compare_models.ipynb` | Main pipeline — runs the full comparison |
| `outputs/` | Generated charts (PNG), metrics and weights (CSV) |

## ML Model

**Inputs:** Monthly return time series for each asset; lag features created from past 24 months (t-1 … t-24).

**Output:** Predicted mean return (μ) per asset for the next period.

**Algorithms:**
- `Ridge` — L2-regularized linear regression (default α=1.0)
- `LinearRegression` — unregularized baseline
- `MLPRegressor` — 2 hidden layers × 50 neurons, relu activation

**Training:** Walk-forward validation (70% train / 30% test). No data leakage — each prediction uses only past observations.

**Blending:** `final_μ = 0.3 × model_prediction + 0.7 × historical_mean` to prevent extreme predictions.

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run main pipeline
jupyter notebook notebooks/00-compare_models.ipynb
```

Or with Docker:
```bash
docker build -t dl-finance .
docker run -p 8888:8888 dl-finance
```

Results are saved to `outputs/charts/` and `outputs/models/`.

## Key Conventions

- **No short selling:** portfolio weights constrained to [0, 1], sum to 1.
- **Risk-free rate:** 15% p.a. (SELIC 2025); use `ajustar_risk_free()` to convert to other frequencies.
- **Walk-forward only:** never use future data when training or predicting — use `evaluate_models_monthly()`.
- **Model blending alpha:** default 0.3; keep it conservative to avoid overfitting to ML predictions.
- **Asset selection:** prefer `select_assets(method="stable_pairs")` for long-term portfolios.
- **Optimization solver:** SLSQP via `scipy.optimize.minimize` — keep constraints explicit (bounds + eq constraint).
- **Outputs:** always save metrics and weights via `src/utils/export.py` functions, not ad-hoc CSV writes.
