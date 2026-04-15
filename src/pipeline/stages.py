"""
Defines each pipeline stage as an independent, composable function.

Stages in order: ingest → compute_returns → build_features →
train_models → predict_returns → optimize_portfolio → evaluate → export_artifacts.
Each stage reads from and writes to the PipelineContext.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.context import PipelineContext


def stage_ingest(context: PipelineContext) -> None:
    """Download prices, compute returns, and select assets.

    Delegates entirely to run_data_ingestion(), which reads all parameters
    from config/pipeline.yaml internally.

    Args:
        context: Shared pipeline state.
            Writes: prices, returns_full, selected_assets.
    """
    from src.ingestion import run_data_ingestion

    result = run_data_ingestion()
    context.prices = result.prices
    context.returns_full = result.returns
    context.selected_assets = result.selected_assets


def stage_compute_returns(context: PipelineContext) -> None:
    """Filter the full return matrix to the selected assets.

    Args:
        context: Shared pipeline state.
            Reads: returns_full, selected_assets.
            Writes: returns_selected.

    Raises:
        ValueError: If stage_ingest has not been run.
    """
    if context.returns_full is None or context.selected_assets is None:
        raise ValueError("stage_ingest must run before stage_compute_returns.")
    context.returns_selected = context.returns_full[context.selected_assets].copy()


def stage_build_features(context: PipelineContext) -> None:
    """Build lag feature matrix and walk-forward splits from selected asset returns.

    Args:
        context: Shared pipeline state.
            Reads: returns_selected, pipeline_cfg.
            Writes: feature_matrix, target_matrix, walk_forward_splits.

    Raises:
        ValueError: If stage_compute_returns has not been run.
    """
    from src.features.lag_features import build_lag_features, make_walk_forward_splits

    if context.returns_selected is None:
        raise ValueError("stage_compute_returns must run before stage_build_features.")

    feat_cfg = context.pipeline_cfg["features"]
    model_cfg = context.pipeline_cfg["models"]

    lag_window: int = feat_cfg["lag_window"]
    min_train_size: int = feat_cfg["min_history"]
    train_ratio: float = model_cfg["train_ratio"]

    X, y = build_lag_features(context.returns_selected, lag_window=lag_window)
    splits = make_walk_forward_splits(
        n_samples=len(X),
        train_ratio=train_ratio,
        min_train_size=min_train_size,
    )

    context.feature_matrix = X
    context.target_matrix = y
    context.walk_forward_splits = splits


def stage_train_models(context: PipelineContext) -> None:
    """Instantiate and train the configured ML model on all available data.

    The model type is read from pipeline_cfg:models:default. Training on the
    full dataset produces the best possible forward-looking predictions.
    Walk-forward diagnostic metrics are handled in stage_evaluate if needed.

    The trained model is stored as a transient attribute (_trained_model) on
    the context to avoid forcing serialization of sklearn objects.

    Args:
        context: Shared pipeline state.
            Reads: feature_matrix, target_matrix, pipeline_cfg.
            Writes: context._trained_model (transient attribute).

    Raises:
        ValueError: If stage_build_features has not been run or the model name is unknown.
    """
    from src.models.mlp import MLPReturnModel
    from src.models.ridge import RidgeReturnModel

    if context.feature_matrix is None or context.target_matrix is None:
        raise ValueError("stage_build_features must run before stage_train_models.")

    model_name: str = context.pipeline_cfg["models"].get("default", "ridge")
    model_map = {"ridge": RidgeReturnModel, "mlp": MLPReturnModel}

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(model_map.keys())}."
        )

    model = model_map[model_name]()
    model.fit(context.feature_matrix, context.target_matrix)
    context._trained_model = model  # type: ignore[attr-defined]


def stage_predict_returns(context: PipelineContext) -> None:
    """Generate ML predictions and compute historical means, then blend them.

    Calls predict_expected_returns() on the trained model using the last available
    feature row. Historical means are computed from the training portion of
    returns_selected (aligned to the last walk-forward split's training window).

    Args:
        context: Shared pipeline state.
            Reads: _trained_model, feature_matrix, walk_forward_splits, returns_selected.
            Writes: ml_predictions, historical_means, blended_mu.

    Raises:
        ValueError: If required prior stages have not been run.
    """
    from src.models.blending import blend_from_config

    model = getattr(context, "_trained_model", None)
    if model is None:
        raise ValueError("stage_train_models must run before stage_predict_returns.")
    if context.walk_forward_splits is None or context.returns_selected is None:
        raise ValueError("stage_build_features must run before stage_predict_returns.")

    ml_preds = model.predict_expected_returns(context.feature_matrix)

    # Historical means from the training portion of the last walk-forward split.
    # walk_forward_splits indices are positional into feature_matrix.
    # feature_matrix starts at returns_selected.index[lag_window], so add the offset.
    last_split = context.walk_forward_splits[-1]
    train_end_feat_idx = last_split["train_end"]

    feature_start_loc: int = context.returns_selected.index.get_loc(  # type: ignore[assignment]
        context.feature_matrix.index[0]
    )
    hist_end_loc = feature_start_loc + train_end_feat_idx
    hist_means = context.returns_selected.iloc[:hist_end_loc].mean()

    context.ml_predictions = ml_preds
    context.historical_means = hist_means
    context.blended_mu = blend_from_config(ml_preds, hist_means)


def stage_optimize_portfolio(context: PipelineContext) -> None:
    """Optimize portfolio weights to maximize Sharpe Ratio using blended expected returns.

    Computes the covariance matrix from the training window only (no future data).
    Falls back to equal weights if the optimizer fails to converge.

    Args:
        context: Shared pipeline state.
            Reads: blended_mu, returns_selected, walk_forward_splits,
                   feature_matrix, pipeline_cfg.
            Writes: cov_matrix, weights, weights_series.

    Raises:
        ValueError: If required prior stages have not been run.
    """
    from src.features.returns import ajustar_risk_free
    from src.optimization.sharpe import maximize_sharpe

    if context.blended_mu is None:
        raise ValueError("stage_predict_returns must run before stage_optimize_portfolio.")
    if context.walk_forward_splits is None or context.feature_matrix is None:
        raise ValueError("stage_build_features must run before stage_optimize_portfolio.")

    opt_cfg = context.pipeline_cfg["optimization"]
    rf_annual: float = opt_cfg["risk_free_rate"]
    freq: str = opt_cfg["frequency"]
    rf_period: float = ajustar_risk_free(rf_annual, freq=freq)

    # Slice training returns: returns_selected up to the last training split boundary.
    # walk_forward_splits indices are positional into feature_matrix, not returns_selected.
    last_split = context.walk_forward_splits[-1]
    train_end_feat_idx: int = last_split["train_end"]

    feature_start_loc: int = context.returns_selected.index.get_loc(  # type: ignore[assignment]
        context.feature_matrix.index[0]
    )
    hist_end_loc = feature_start_loc + train_end_feat_idx
    returns_train = context.returns_selected.iloc[:hist_end_loc]

    cov = returns_train.cov()

    n = len(context.selected_assets)
    weights = maximize_sharpe(
        context.blended_mu.values,
        cov.values,
        risk_free_rate=rf_period,
    )

    if weights is None:
        warnings.warn(
            "Sharpe optimization failed to converge. Using equal weights as fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        weights = np.full(n, 1.0 / n)

    context.cov_matrix = cov
    context.weights = weights
    context.weights_series = pd.Series(
        weights, index=context.selected_assets, name="weights"
    )


def stage_evaluate(context: PipelineContext) -> None:
    """Compute portfolio performance metrics on the out-of-sample test period.

    The out-of-sample period consists of all test indices across all walk-forward
    splits. Portfolio returns are computed as returns_selected.dot(weights).

    Args:
        context: Shared pipeline state.
            Reads: weights, returns_selected, walk_forward_splits,
                   feature_matrix, pipeline_cfg.
            Writes: portfolio_returns, metrics.

    Raises:
        ValueError: If required prior stages have not been run.
    """
    from src.optimization.evaluation import evaluate_portfolio

    if context.weights is None:
        raise ValueError("stage_optimize_portfolio must run before stage_evaluate.")
    if context.walk_forward_splits is None or context.feature_matrix is None:
        raise ValueError("stage_build_features must run before stage_evaluate.")

    opt_cfg = context.pipeline_cfg["optimization"]
    rf_annual: float = opt_cfg["risk_free_rate"]
    freq: str = opt_cfg["frequency"]

    _periods_map: dict[str, int] = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "annually": 1,
    }
    periods_per_year: int = _periods_map.get(freq, 12)

    # Collect all out-of-sample test periods.
    feature_start_loc: int = context.returns_selected.index.get_loc(  # type: ignore[assignment]
        context.feature_matrix.index[0]
    )
    test_abs_locs = [
        feature_start_loc + s["test_idx"] for s in context.walk_forward_splits
    ]
    returns_test = context.returns_selected.iloc[test_abs_locs]
    portfolio_returns = returns_test.dot(context.weights)

    model_name: str = context.pipeline_cfg["models"].get("default", "ridge")
    metrics_series = evaluate_portfolio(
        portfolio_returns,
        risk_free_rate_annual=rf_annual,
        periods_per_year=periods_per_year,
        model_name=model_name,
    )

    context.portfolio_returns = portfolio_returns
    context.metrics = metrics_series.to_dict()


def stage_export_artifacts(context: PipelineContext) -> None:
    """Persist all pipeline outputs to the artifacts/ directory.

    Saves the following artifacts (Stage 2 full set):
    - Processed historical returns (returns_selected)
    - Feature matrix and target matrix
    - Trained model (joblib)
    - Predicted expected returns (blended_mu)
    - Optimized portfolio weights
    - Evaluation metrics
    - Run manifest (JSON)

    All paths are derived from config/pipeline.yaml:artifacts and the run_id.

    Args:
        context: Shared pipeline state.
            Reads: returns_selected, feature_matrix, target_matrix, _trained_model,
                   blended_mu, weights, selected_assets, metrics, run_id, pipeline_cfg.
            Writes: artifacts_written, status.

    Raises:
        ValueError: If required prior stages have not been run.
    """
    from src.utils.export import (
        save_features,
        save_model,
        save_portfolio_metrics,
        save_portfolio_weights,
        save_predicted_returns,
        save_returns,
    )

    if context.weights is None or context.metrics is None:
        raise ValueError(
            "stage_optimize_portfolio and stage_evaluate must run before stage_export_artifacts."
        )

    art_cfg = context.pipeline_cfg["artifacts"]
    run_id = context.run_id
    model_name: str = context.pipeline_cfg["models"].get("default", "ridge")

    # Build artifact paths
    returns_path = f"{art_cfg['data_dir']}/{run_id}_returns.csv"
    X_path = f"{art_cfg['data_dir']}/{run_id}_features.csv"
    y_path = f"{art_cfg['data_dir']}/{run_id}_targets.csv"
    model_path = f"{art_cfg['models_dir']}/{run_id}_{model_name}.joblib"
    preds_path = f"{art_cfg['predictions_dir']}/{run_id}_{model_name}_predictions.csv"
    weights_path = f"{art_cfg['weights_dir']}/{run_id}_{model_name}_weights.csv"
    metrics_path = f"{art_cfg['metrics_dir']}/{run_id}_{model_name}_metrics.csv"
    manifest_path = f"{art_cfg['runs_dir']}/{run_id}_manifest.json"

    # Persist intermediate data artifacts
    if context.returns_selected is not None:
        save_returns(context.returns_selected, returns_path)

    if context.feature_matrix is not None and context.target_matrix is not None:
        save_features(context.feature_matrix, context.target_matrix, X_path, y_path)

    # Persist trained model
    trained_model = getattr(context, "_trained_model", None)
    if trained_model is not None:
        save_model(trained_model, model_path)

    # Persist final pipeline outputs
    save_predicted_returns(context.blended_mu, model_name, preds_path)
    save_portfolio_weights(context.weights, context.selected_assets, model_name, weights_path)
    save_portfolio_metrics(context.metrics, metrics_path)

    # Persist run manifest (includes metrics for registry)
    manifest = context.to_run_manifest()
    manifest["metrics"] = context.metrics
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    context.artifacts_written = [
        returns_path, X_path, y_path, model_path,
        preds_path, weights_path, metrics_path, manifest_path,
    ]
    context.status = "completed"
