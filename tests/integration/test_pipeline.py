"""
Integration tests for the full pipeline execution.

Runs the pipeline end-to-end with synthetic data, verifying that all stages
complete, artifacts are written to disk, and output metrics are well-formed.

Note: stage_ingest (yfinance download) is not executed here — the context is
pre-populated with synthetic data before calling stage_compute_returns onward.
This is NOT mocking; it is starting the pipeline from a later stage using a
PipelineContext that already has the ingestion outputs filled in.
"""
from __future__ import annotations

import datetime
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.context import PipelineContext
from src.pipeline.runner import STAGE_ORDER, run_pipeline
from src.pipeline.stages import (
    stage_build_features,
    stage_compute_returns,
    stage_evaluate,
    stage_export_artifacts,
    stage_optimize_portfolio,
    stage_predict_returns,
    stage_train_models,
)
from src.utils.config_loader import get_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_returns(n_rows: int = 80, n_cols: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_rows, freq="ME")
    data = rng.normal(0.008, 0.04, size=(n_rows, n_cols))
    tickers = [f"SYNTH{i}.SA" for i in range(n_cols)]
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def synthetic_context(tmp_path) -> PipelineContext:
    """PipelineContext pre-loaded past stage_ingest, with artifacts redirected to tmp_path."""
    pipeline_cfg = get_config("pipeline")
    models_cfg = get_config("models")

    # Override artifact paths to a temp directory so tests do not pollute artifacts/
    for key in pipeline_cfg["artifacts"]:
        sub = pipeline_cfg["artifacts"][key]
        if isinstance(sub, str) and sub.startswith("artifacts"):
            # Replace "artifacts/..." with "<tmp>/..."
            pipeline_cfg["artifacts"][key] = str(
                tmp_path / Path(sub).relative_to("artifacts")
            )

    returns = _synthetic_returns(n_rows=80, n_cols=4)
    assets = list(returns.columns)

    ctx = PipelineContext(
        pipeline_cfg=pipeline_cfg,
        models_cfg=models_cfg,
        run_id=datetime.datetime.now().strftime("ITEST%Y%m%dT%H%M%S"),
    )
    ctx.prices = returns  # prices field not used after ingest; just needs to not be None
    ctx.returns_full = returns
    ctx.selected_assets = assets
    ctx.status = "running"
    return ctx


# ---------------------------------------------------------------------------
# Individual stage tests
# ---------------------------------------------------------------------------

class TestStageComputeReturns:
    def test_produces_returns_selected(self, synthetic_context) -> None:
        stage_compute_returns(synthetic_context)
        assert synthetic_context.returns_selected is not None

    def test_columns_match_selected_assets(self, synthetic_context) -> None:
        stage_compute_returns(synthetic_context)
        assert list(synthetic_context.returns_selected.columns) == synthetic_context.selected_assets

    def test_no_extra_data(self, synthetic_context) -> None:
        stage_compute_returns(synthetic_context)
        assert len(synthetic_context.returns_selected) == len(synthetic_context.returns_full)

    def test_missing_ingestion_raises(self) -> None:
        pipeline_cfg = get_config("pipeline")
        ctx = PipelineContext(
            pipeline_cfg=pipeline_cfg,
            models_cfg=get_config("models"),
            run_id="ERR",
        )
        with pytest.raises(ValueError, match="stage_ingest"):
            stage_compute_returns(ctx)


class TestStageBuildFeatures:
    def _ctx_with_returns(self, synthetic_context) -> PipelineContext:
        stage_compute_returns(synthetic_context)
        return synthetic_context

    def test_produces_feature_matrix(self, synthetic_context) -> None:
        ctx = self._ctx_with_returns(synthetic_context)
        stage_build_features(ctx)
        assert ctx.feature_matrix is not None

    def test_no_nan_in_features(self, synthetic_context) -> None:
        ctx = self._ctx_with_returns(synthetic_context)
        stage_build_features(ctx)
        assert not ctx.feature_matrix.isna().any().any()

    def test_splits_are_non_empty(self, synthetic_context) -> None:
        ctx = self._ctx_with_returns(synthetic_context)
        stage_build_features(ctx)
        assert len(ctx.walk_forward_splits) > 0


# ---------------------------------------------------------------------------
# Full pipeline run (all stages except ingest)
# ---------------------------------------------------------------------------

class TestFullPipelineRun:
    def _run_all(self, synthetic_context) -> PipelineContext:
        ctx = synthetic_context
        stage_compute_returns(ctx)
        stage_build_features(ctx)
        stage_train_models(ctx)
        stage_predict_returns(ctx)
        stage_optimize_portfolio(ctx)
        stage_evaluate(ctx)
        stage_export_artifacts(ctx)
        return ctx

    def test_status_completed(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert ctx.status == "completed"

    def test_weights_sum_to_one(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert ctx.weights is not None
        assert abs(ctx.weights.sum() - 1.0) < 1e-5

    def test_weights_no_short_selling(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert (ctx.weights >= -1e-6).all()

    def test_metrics_contain_required_keys(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        required = {"Sharpe", "Annualized_Return", "Annualized_Volatility", "Cumulative_Return"}
        assert required.issubset(set(ctx.metrics.keys()))

    def test_metrics_are_finite(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        for key in ("Sharpe", "Annualized_Return", "Annualized_Volatility"):
            val = ctx.metrics[key]
            assert np.isfinite(float(val)), f"{key} is not finite: {val}"

    def test_artifacts_written(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert len(ctx.artifacts_written) > 0

    def test_artifact_files_exist(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        for path_str in ctx.artifacts_written:
            assert Path(path_str).exists(), f"Artifact missing: {path_str}"

    def test_manifest_is_valid_json(self, synthetic_context) -> None:
        import json
        ctx = self._run_all(synthetic_context)
        manifest_path = next(
            (p for p in ctx.artifacts_written if p.endswith("_manifest.json")), None
        )
        assert manifest_path is not None
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["run_id"] == ctx.run_id
        assert manifest["status"] == "completed"

    def test_predictions_cover_all_assets(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert set(ctx.blended_mu.index) == set(ctx.selected_assets)

    def test_blended_mu_values_are_finite(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert ctx.blended_mu.apply(np.isfinite).all()


# ---------------------------------------------------------------------------
# run_pipeline() helper with partial stages
# ---------------------------------------------------------------------------

class TestRunPipelineHelper:
    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            run_pipeline(stages=["not_a_stage"])

    def test_stage_order_is_complete(self) -> None:
        expected = [
            "ingest", "compute_returns", "build_features", "train_models",
            "predict_returns", "optimize_portfolio", "evaluate", "export_artifacts",
        ]
        assert STAGE_ORDER == expected
