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
import json
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
import src.persistence.database as _db_module
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


def _make_context(tmp_path, enabled_models: list[str]) -> PipelineContext:
    """Build a PipelineContext pre-loaded past stage_ingest with given enabled models."""
    pipeline_cfg = get_config("pipeline")
    models_cfg = get_config("models")

    for key in pipeline_cfg["artifacts"]:
        sub = pipeline_cfg["artifacts"][key]
        if isinstance(sub, str) and sub.startswith("artifacts"):
            pipeline_cfg["artifacts"][key] = str(
                tmp_path / Path(sub).relative_to("artifacts")
            )

    # Override the enabled models list for this test context.
    pipeline_cfg["models"]["enabled"] = enabled_models

    returns = _synthetic_returns(n_rows=80, n_cols=4)
    assets = list(returns.columns)

    ctx = PipelineContext(
        pipeline_cfg=pipeline_cfg,
        models_cfg=models_cfg,
        run_id=datetime.datetime.now().strftime("ITEST%Y%m%dT%H%M%S"),
    )
    ctx.prices = returns
    ctx.returns_full = returns
    ctx.selected_assets = assets
    ctx.status = "running"
    return ctx


@pytest.fixture
def synthetic_context(tmp_path, monkeypatch) -> PipelineContext:
    """Single-model context (ridge only) for fast stage-level tests."""
    monkeypatch.setattr(_db_module, "_db_path", lambda: tmp_path / "test_runs.db")
    return _make_context(tmp_path, enabled_models=["ridge"])


@pytest.fixture
def synthetic_context_multi(tmp_path, monkeypatch) -> PipelineContext:
    """Multi-model context (ridge + mlp) for multi-model artifact tests."""
    monkeypatch.setattr(_db_module, "_db_path", lambda: tmp_path / "test_runs.db")
    return _make_context(tmp_path, enabled_models=["ridge", "mlp"])


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
# Full pipeline run — single model (fast)
# ---------------------------------------------------------------------------

class TestFullPipelineRun:
    def _run_all(self, ctx: PipelineContext) -> PipelineContext:
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

    def test_weights_dict_keyed_by_model(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert isinstance(ctx.weights, dict)
        assert len(ctx.weights) > 0

    def test_markowitz_always_present(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert "markowitz" in ctx.weights

    def test_all_weights_sum_to_one(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        for model_name, w in ctx.weights.items():
            assert abs(w.sum() - 1.0) < 1e-5, f"Weights for {model_name} do not sum to 1"

    def test_all_weights_no_short_selling(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        for model_name, w in ctx.weights.items():
            assert (w >= -1e-6).all(), f"Short selling detected in {model_name}"

    def test_metrics_dict_keyed_by_model(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert isinstance(ctx.metrics, dict)
        assert "markowitz" in ctx.metrics

    def test_metrics_contain_required_keys(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        required = {"Sharpe", "Annualized_Return", "Annualized_Volatility", "Cumulative_Return"}
        for model_name, m in ctx.metrics.items():
            assert required.issubset(set(m.keys())), f"Missing keys for {model_name}"

    def test_metrics_are_finite(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        for model_name, m in ctx.metrics.items():
            for key in ("Sharpe", "Annualized_Return", "Annualized_Volatility"):
                val = m[key]
                assert np.isfinite(float(val)), f"{key} is not finite for {model_name}: {val}"

    def test_artifacts_written(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert len(ctx.artifacts_written) > 0

    def test_artifact_files_exist(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        for path_str in ctx.artifacts_written:
            assert Path(path_str).exists(), f"Artifact missing: {path_str}"

    def test_manifest_is_valid_json(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        manifest_path = next(
            (p for p in ctx.artifacts_written if p.endswith("_manifest.json")), None
        )
        assert manifest_path is not None
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["run_id"] == ctx.run_id
        assert manifest["status"] == "completed"

    def test_manifest_contains_models_list(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        manifest_path = next(p for p in ctx.artifacts_written if p.endswith("_manifest.json"))
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "models" in manifest
        assert "markowitz" in manifest["models"]

    def test_manifest_contains_primary_model(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        manifest_path = next(p for p in ctx.artifacts_written if p.endswith("_manifest.json"))
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "primary_model" in manifest
        assert manifest["primary_model"] in manifest["models"]

    def test_predictions_cover_all_assets(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        assert ctx.blended_mu is not None
        for model_name, mu in ctx.blended_mu.items():
            assert set(mu.index) == set(ctx.selected_assets)

    def test_blended_mu_values_are_finite(self, synthetic_context) -> None:
        ctx = self._run_all(synthetic_context)
        for model_name, mu in ctx.blended_mu.items():
            assert mu.apply(np.isfinite).all(), f"Non-finite blended_mu for {model_name}"


# ---------------------------------------------------------------------------
# Multi-model artifact coverage
# ---------------------------------------------------------------------------

class TestMultiModelArtifacts:
    def _run_all(self, ctx: PipelineContext) -> PipelineContext:
        stage_compute_returns(ctx)
        stage_build_features(ctx)
        stage_train_models(ctx)
        stage_predict_returns(ctx)
        stage_optimize_portfolio(ctx)
        stage_evaluate(ctx)
        stage_export_artifacts(ctx)
        return ctx

    def test_all_enabled_models_have_weights_artifact(
        self, synthetic_context_multi
    ) -> None:
        ctx = self._run_all(synthetic_context_multi)
        enabled = ctx.pipeline_cfg["models"]["enabled"]
        for model_name in enabled + ["markowitz"]:
            found = any(
                f"{ctx.run_id}_{model_name}_weights.csv" in p
                for p in ctx.artifacts_written
            )
            assert found, f"Weights artifact missing for model '{model_name}'"

    def test_all_enabled_models_have_metrics_artifact(
        self, synthetic_context_multi
    ) -> None:
        ctx = self._run_all(synthetic_context_multi)
        enabled = ctx.pipeline_cfg["models"]["enabled"]
        for model_name in enabled + ["markowitz"]:
            found = any(
                f"{ctx.run_id}_{model_name}_metrics.csv" in p
                for p in ctx.artifacts_written
            )
            assert found, f"Metrics artifact missing for model '{model_name}'"

    def test_ml_models_have_model_metrics_artifact(
        self, synthetic_context_multi
    ) -> None:
        ctx = self._run_all(synthetic_context_multi)
        enabled = ctx.pipeline_cfg["models"]["enabled"]
        for model_name in enabled:
            found = any(
                f"{ctx.run_id}_{model_name}_model_metrics.csv" in p
                for p in ctx.artifacts_written
            )
            assert found, f"Model metrics artifact missing for ML model '{model_name}'"

    def test_markowitz_has_no_model_metrics_artifact(
        self, synthetic_context_multi
    ) -> None:
        ctx = self._run_all(synthetic_context_multi)
        found = any(
            f"{ctx.run_id}_markowitz_model_metrics.csv" in p
            for p in ctx.artifacts_written
        )
        assert not found, "markowitz should not have a model_metrics artifact"

    def test_metrics_dict_includes_all_models(self, synthetic_context_multi) -> None:
        ctx = self._run_all(synthetic_context_multi)
        enabled = ctx.pipeline_cfg["models"]["enabled"]
        for model_name in enabled + ["markowitz"]:
            assert model_name in ctx.metrics, f"'{model_name}' missing from metrics"


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
