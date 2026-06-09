"""
Defines each pipeline stage as an independent, composable function.

Stages in order: ingest → compute_returns → build_features →
train_models → predict_returns → optimize_portfolio → evaluate → export_artifacts.
Each stage reads from and writes to the PipelineContext.

Every run trains all models listed in pipeline_cfg:models:enabled and also
adds "markowitz" (classical, no ML) as an automatic baseline. All multi-model
fields in PipelineContext are dicts keyed by model name.
"""
from __future__ import annotations

import warnings

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
    context.pipeline_cfg["data"]["end_date"] = result.config["data"]["end_date"]


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
    """Train all enabled ML models and compute walk-forward OOS diagnostics.

    Reads pipeline_cfg:models:enabled to determine which models to train. Each
    model is trained on the full dataset for the best possible forward-looking
    predictions. A walk-forward loop then trains fresh models per split and
    evaluates them out-of-sample to produce diagnostic metrics (IC, ICIR, etc.)
    without data leakage.

    The classical Markowitz baseline ("markowitz") requires no training and is
    added automatically during stage_optimize_portfolio.

    Args:
        context: Shared pipeline state.
            Reads: feature_matrix, target_matrix, walk_forward_splits, pipeline_cfg.
            Writes: context._trained_models (transient), context.model_metrics.

    Raises:
        ValueError: If required prior stages have not run or an unknown model name
            appears in models.enabled.
    """
    from src.models.metrics import compute_model_metrics
    from src.models.mlp import MLPReturnModel
    from src.models.ridge import RidgeReturnModel

    if context.feature_matrix is None or context.target_matrix is None:
        raise ValueError("stage_build_features must run before stage_train_models.")
    if context.walk_forward_splits is None:
        raise ValueError("stage_build_features must run before stage_train_models.")

    enabled: list[str] = context.pipeline_cfg["models"].get("enabled", ["ridge"])
    model_map: dict[str, type] = {"ridge": RidgeReturnModel, "mlp": MLPReturnModel}

    unknown = [m for m in enabled if m not in model_map]
    if unknown:
        raise ValueError(
            f"Unknown model(s) in models.enabled: {unknown}. "
            f"Available: {list(model_map.keys())}."
        )

    X = context.feature_matrix
    y = context.target_matrix

    trained_models: dict = {}
    model_metrics: dict[str, dict] = {}
    wf_raw_preds: dict[str, np.ndarray] = {}

    for model_name in enabled:
        model = model_map[model_name]()
        model.fit(X, y)
        trained_models[model_name] = model

        y_pred_list: list[np.ndarray] = []
        y_true_list: list[np.ndarray] = []
        for split in context.walk_forward_splits:
            train_end: int = split["train_end"]
            test_idx: int = split["test_idx"]
            diag_model = model_map[model_name]()
            diag_model.fit(X.iloc[:train_end], y.iloc[:train_end])
            pred_df = diag_model.predict(X.iloc[[test_idx]])
            y_pred_list.append(pred_df.values[0])
            y_true_list.append(y.iloc[test_idx].values)

        y_pred_arr = np.array(y_pred_list)
        y_true_arr = np.array(y_true_list)
        model_metrics[model_name] = compute_model_metrics(y_pred_arr, y_true_arr)
        wf_raw_preds[model_name] = y_pred_arr  # shape (n_splits, n_assets)

    context._trained_models = trained_models  # type: ignore[attr-defined]
    context._wf_raw_predictions = wf_raw_preds  # type: ignore[attr-defined]
    context.model_metrics = model_metrics


def stage_predict_returns(context: PipelineContext) -> None:
    """Generate blended predictions for each enabled ML model.

    For each model in _trained_models, calls predict_expected_returns() on the
    last available feature row, then blends the ML prediction with the historical
    mean. Historical means are computed once from the training window of the last
    walk-forward split and shared across all models.

    Args:
        context: Shared pipeline state.
            Reads: _trained_models, feature_matrix, walk_forward_splits,
                   returns_selected, pipeline_cfg.
            Writes: ml_predictions, historical_means, blended_mu.

    Raises:
        ValueError: If required prior stages have not been run.
    """
    from src.models.blending import blend_from_config

    trained_models = getattr(context, "_trained_models", None)
    if trained_models is None:
        raise ValueError("stage_train_models must run before stage_predict_returns.")
    if context.walk_forward_splits is None or context.returns_selected is None:
        raise ValueError("stage_build_features must run before stage_predict_returns.")

    last_split = context.walk_forward_splits[-1]
    train_end_feat_idx = last_split["train_end"]
    feature_start_loc: int = context.returns_selected.index.get_loc(  # type: ignore[assignment]
        context.feature_matrix.index[0]
    )
    hist_end_loc = feature_start_loc + train_end_feat_idx
    hist_means = context.returns_selected.iloc[:hist_end_loc].mean()
    context.historical_means = hist_means

    ml_predictions: dict[str, pd.Series] = {}
    blended_mu: dict[str, pd.Series] = {}

    for model_name, model in trained_models.items():
        ml_pred = model.predict_expected_returns(context.feature_matrix)
        blended = blend_from_config(ml_pred, hist_means)
        ml_predictions[model_name] = ml_pred
        blended_mu[model_name] = blended

    context.ml_predictions = ml_predictions
    context.blended_mu = blended_mu


def stage_optimize_portfolio(context: PipelineContext) -> None:
    """Optimize portfolio weights for each ML model and for Markowitz classical baseline.

    Runs maximize_sharpe for:
    - Each ML model in blended_mu, using its blended expected returns.
    - "markowitz": the classical Markowitz baseline, using historical_means as mu
      with no ML component.

    Falls back to equal weights if the optimizer fails to converge for any model.

    Args:
        context: Shared pipeline state.
            Reads: blended_mu, historical_means, returns_selected,
                   walk_forward_splits, feature_matrix, pipeline_cfg.
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
    if context.historical_means is None:
        raise ValueError("stage_predict_returns must run before stage_optimize_portfolio.")

    opt_cfg = context.pipeline_cfg["optimization"]
    rf_annual: float = opt_cfg["risk_free_rate"]
    freq: str = opt_cfg["frequency"]
    rf_period: float = ajustar_risk_free(rf_annual, freq=freq)
    weight_bounds: list = opt_cfg.get("weight_bounds", [0.0, 1.0])
    max_weight: float = float(weight_bounds[1])

    last_split = context.walk_forward_splits[-1]
    train_end_feat_idx: int = last_split["train_end"]
    feature_start_loc: int = context.returns_selected.index.get_loc(  # type: ignore[assignment]
        context.feature_matrix.index[0]
    )
    hist_end_loc = feature_start_loc + train_end_feat_idx
    returns_train = context.returns_selected.iloc[:hist_end_loc]
    cov = returns_train.cov()

    n = len(context.selected_assets)

    # Build the full set of mu vectors: all ML models + classical baseline.
    mu_per_model: dict[str, pd.Series] = dict(context.blended_mu)
    mu_per_model["markowitz"] = context.historical_means

    weights: dict[str, np.ndarray] = {}
    weights_series: dict[str, pd.Series] = {}

    for model_name, mu in mu_per_model.items():
        w = maximize_sharpe(mu.values, cov.values, risk_free_rate=rf_period, max_weight=max_weight)
        if w is None:
            warnings.warn(
                f"Sharpe optimization failed for '{model_name}'. Using equal weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            w = np.full(n, 1.0 / n)
        weights[model_name] = w
        weights_series[model_name] = pd.Series(
            w, index=context.selected_assets, name="weights"
        )

    context.cov_matrix = cov
    context.weights = weights
    context.weights_series = weights_series


def stage_evaluate(context: PipelineContext) -> None:
    """Compute portfolio performance metrics using walk-forward per-split optimization.

    For each walk-forward split, portfolio weights are computed using only the
    training data available at that point in time — eliminating the look-ahead
    bias that arises from applying a single final weight vector across all test
    periods. Per-split ML predictions stored by stage_train_models are blended
    with per-split historical means to form the mu vector for each model.

    The markowitz baseline uses the per-split historical mean as mu (no ML).

    Args:
        context: Shared pipeline state.
            Reads: weights, returns_selected, walk_forward_splits, feature_matrix,
                   pipeline_cfg, _wf_raw_predictions (set by stage_train_models).
            Writes: portfolio_returns, metrics.

    Raises:
        ValueError: If required prior stages have not been run, or if walk-forward
            predictions are missing for an ML model.
    """
    from src.features.returns import ajustar_risk_free
    from src.models.blending import blend_predictions
    from src.optimization.evaluation import evaluate_portfolio
    from src.optimization.sharpe import maximize_sharpe

    if context.weights is None:
        raise ValueError("stage_optimize_portfolio must run before stage_evaluate.")
    if context.walk_forward_splits is None or context.feature_matrix is None:
        raise ValueError("stage_build_features must run before stage_evaluate.")

    opt_cfg = context.pipeline_cfg["optimization"]
    rf_annual: float = opt_cfg["risk_free_rate"]
    freq: str = opt_cfg["frequency"]
    rf_period: float = ajustar_risk_free(rf_annual, freq=freq)
    weight_bounds: list = opt_cfg.get("weight_bounds", [0.0, 1.0])
    max_weight: float = float(weight_bounds[1])
    blend_alpha: float = context.pipeline_cfg["models"].get("blend_alpha", 0.3)

    _periods_map: dict[str, int] = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "annually": 1,
    }
    periods_per_year: int = _periods_map.get(freq, 12)
    n: int = len(context.selected_assets)

    feature_start_loc: int = context.returns_selected.index.get_loc(  # type: ignore[assignment]
        context.feature_matrix.index[0]
    )

    wf_raw_preds: dict[str, np.ndarray] = getattr(context, "_wf_raw_predictions", {})
    all_model_names: list[str] = list(context.weights.keys())

    # Validate that per-split predictions exist for every ML model.
    for model_name in all_model_names:
        if model_name != "markowitz" and model_name not in wf_raw_preds:
            raise ValueError(
                f"Walk-forward predictions not found for model '{model_name}'. "
                "stage_train_models must run before stage_evaluate."
            )

    per_model_returns: dict[str, list[float]] = {m: [] for m in all_model_names}
    test_dates: list = []

    for split_i, split in enumerate(context.walk_forward_splits):
        train_end: int = split["train_end"]
        test_idx: int = split["test_idx"]

        # Use only training data available at this split — no future information.
        hist_end: int = feature_start_loc + train_end
        returns_train_split = context.returns_selected.iloc[:hist_end]
        cov_split = returns_train_split.cov()
        hist_means_split = returns_train_split.mean()

        test_dates.append(context.returns_selected.index[feature_start_loc + test_idx])

        for model_name in all_model_names:
            if model_name == "markowitz":
                mu_split = hist_means_split
            else:
                raw_pred = pd.Series(
                    wf_raw_preds[model_name][split_i],
                    index=context.selected_assets,
                )
                mu_split = blend_predictions(raw_pred, hist_means_split, alpha=blend_alpha)

            w_split = maximize_sharpe(
                mu_split.values,
                cov_split.values,
                risk_free_rate=rf_period,
                max_weight=max_weight,
            )
            if w_split is None:
                w_split = np.full(n, 1.0 / n)

            test_return = float(
                context.returns_selected.iloc[feature_start_loc + test_idx].dot(w_split)
            )
            per_model_returns[model_name].append(test_return)

    test_date_idx = pd.DatetimeIndex(test_dates)
    metrics: dict[str, dict] = {}
    portfolio_returns_all: dict[str, pd.Series] = {}

    for model_name in all_model_names:
        port_returns = pd.Series(per_model_returns[model_name], index=test_date_idx)
        metrics_series = evaluate_portfolio(
            port_returns,
            risk_free_rate_annual=rf_annual,
            periods_per_year=periods_per_year,
            model_name=model_name,
        )
        metrics[model_name] = metrics_series.to_dict()
        portfolio_returns_all[model_name] = port_returns

    context.portfolio_returns = portfolio_returns_all
    context.metrics = metrics


def stage_export_artifacts(context: PipelineContext) -> None:
    """Persist all pipeline outputs to the artifacts/ directory.

    Writes one set of artifacts per model (weights, predictions, metrics,
    equity curve). ML-only artifacts (trained model file, model diagnostics)
    are written only for models that were trained (not for "markowitz").
    The manifest records all model names and the full metrics dict.

    Args:
        context: Shared pipeline state.
            Reads: returns_selected, feature_matrix, target_matrix, _trained_models,
                   blended_mu, historical_means, weights, selected_assets, metrics,
                   model_metrics, portfolio_returns, run_id, pipeline_cfg.
            Writes: artifacts_written, status.

    Raises:
        ValueError: If required prior stages have not been run.
    """
    from src.utils.export import (
        save_equity_curve,
        save_features,
        save_manifest,
        save_model,
        save_model_metrics,
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
    trained_models = getattr(context, "_trained_models", {}) or {}

    artifacts: list[str] = []

    # Shared data artifacts (one per run, not per model).
    returns_path = f"{art_cfg['data_dir']}/{run_id}_returns.csv"
    X_path = f"{art_cfg['data_dir']}/{run_id}_features.csv"
    y_path = f"{art_cfg['data_dir']}/{run_id}_targets.csv"

    if context.returns_selected is not None:
        save_returns(context.returns_selected, returns_path)
        artifacts.append(returns_path)

    if context.feature_matrix is not None and context.target_matrix is not None:
        save_features(context.feature_matrix, context.target_matrix, X_path, y_path)
        artifacts.extend([X_path, y_path])

    # Per-model artifacts.
    for model_name, w in context.weights.items():
        is_ml_model = model_name in trained_models

        # Trained model file (ML models only).
        if is_ml_model:
            model_path = f"{art_cfg['models_dir']}/{run_id}_{model_name}.joblib"
            save_model(trained_models[model_name], model_path)
            artifacts.append(model_path)

        # Predictions: blended mu for ML models, historical means for markowitz.
        if is_ml_model and context.blended_mu and model_name in context.blended_mu:
            mu_series = context.blended_mu[model_name]
        elif model_name == "markowitz" and context.historical_means is not None:
            mu_series = context.historical_means
        else:
            mu_series = None

        if mu_series is not None:
            preds_path = f"{art_cfg['predictions_dir']}/{run_id}_{model_name}_predictions.csv"
            save_predicted_returns(mu_series, model_name, preds_path)
            artifacts.append(preds_path)

        # Portfolio weights.
        weights_path = f"{art_cfg['weights_dir']}/{run_id}_{model_name}_weights.csv"
        save_portfolio_weights(w, context.selected_assets, model_name, weights_path)
        artifacts.append(weights_path)

        # Portfolio metrics.
        metrics_path = f"{art_cfg['metrics_dir']}/{run_id}_{model_name}_metrics.csv"
        save_portfolio_metrics(context.metrics[model_name], metrics_path)
        artifacts.append(metrics_path)

        # Model diagnostic metrics (ML models only).
        if is_ml_model and context.model_metrics and model_name in context.model_metrics:
            model_metrics_path = (
                f"{art_cfg['metrics_dir']}/{run_id}_{model_name}_model_metrics.csv"
            )
            save_model_metrics(context.model_metrics[model_name], model_metrics_path)
            artifacts.append(model_metrics_path)

        # Equity curve.
        if context.portfolio_returns and model_name in context.portfolio_returns:
            equity_path = f"{art_cfg['metrics_dir']}/{run_id}_{model_name}_equity_curve.csv"
            save_equity_curve(context.portfolio_returns[model_name], equity_path)
            artifacts.append(equity_path)

    # Determine the primary model: highest Sharpe across all models.
    primary_model = max(
        context.weights.keys(),
        key=lambda m: (context.metrics[m].get("Sharpe") or float("-inf")),
    )

    manifest_path = f"{art_cfg['runs_dir']}/{run_id}_manifest.json"
    artifacts.append(manifest_path)

    context.artifacts_written = artifacts
    context.status = "completed"

    manifest = context.to_run_manifest()
    manifest["models"] = list(context.weights.keys())
    manifest["primary_model"] = primary_model
    manifest["metrics"] = context.metrics
    save_manifest(manifest, manifest_path)

    from src.persistence.database import upsert_run
    upsert_run(manifest)
