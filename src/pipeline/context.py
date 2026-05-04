"""
Defines the PipelineContext dataclass that carries configuration and intermediate
state between pipeline stages.

Holds loaded config, raw data, processed features, trained models, predictions,
and optimized weights — acting as the single shared state object across stages.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineContext:
    """Shared mutable state passed through every pipeline stage.

    Constructed by the runner before execution begins and passed by reference
    to each stage function. Stages read from fields populated by prior stages
    and write their outputs into new fields.

    Multi-model fields (blended_mu, weights, metrics, etc.) are keyed by model
    name. The "markowitz" key is always present after stage_optimize_portfolio
    as the classical baseline (no ML, historical mean as mu).

    Args:
        pipeline_cfg: Loaded content of config/pipeline.yaml.
        models_cfg: Loaded content of config/models.yaml.
        run_id: Unique identifier for this run (ISO timestamp string).
        started_at: Datetime when the context was created.
    """

    # Stage 0: Config (required at construction)
    pipeline_cfg: dict[str, Any]
    models_cfg: dict[str, Any]
    run_id: str
    started_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    # Stage 1: Ingestion outputs
    prices: Optional[pd.DataFrame] = field(default=None)
    returns_full: Optional[pd.DataFrame] = field(default=None)
    selected_assets: Optional[list[str]] = field(default=None)

    # Stage 2: Returns filtered to selected assets
    returns_selected: Optional[pd.DataFrame] = field(default=None)

    # Stage 3: Feature engineering
    feature_matrix: Optional[pd.DataFrame] = field(default=None)
    target_matrix: Optional[pd.DataFrame] = field(default=None)
    walk_forward_splits: Optional[list[dict[str, int]]] = field(default=None)

    # Stage 4: Training and prediction
    # ml_predictions and blended_mu are keyed by ML model name (e.g. "ridge", "mlp").
    # historical_means is a single Series shared by all models.
    ml_predictions: Optional[dict[str, pd.Series]] = field(default=None)
    historical_means: Optional[pd.Series] = field(default=None)
    blended_mu: Optional[dict[str, pd.Series]] = field(default=None)

    # Stage 5: Optimization
    # weights and weights_series are keyed by model name, including "markowitz".
    cov_matrix: Optional[pd.DataFrame] = field(default=None)
    weights: Optional[dict[str, np.ndarray]] = field(default=None)
    weights_series: Optional[dict[str, pd.Series]] = field(default=None)

    # Stage 6: Evaluation
    # Both dicts keyed by model name (including "markowitz").
    portfolio_returns: Optional[dict[str, pd.Series]] = field(default=None)
    metrics: Optional[dict[str, dict[str, Any]]] = field(default=None)
    # model_metrics is only populated for ML models (not "markowitz").
    model_metrics: Optional[dict[str, dict[str, Any]]] = field(default=None)

    # Stage 7: Export
    artifacts_written: list[str] = field(default_factory=list)
    status: str = field(default="initialized")
    error: Optional[str] = field(default=None)

    def to_run_manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the run for artifact logging.

        Returns:
            Dict with run metadata suitable for writing to artifacts/runs/.
        """
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "status": self.status,
            "error": self.error,
            "selected_assets": self.selected_assets,
            "artifacts_written": self.artifacts_written,
            "pipeline_cfg": self.pipeline_cfg,
        }
