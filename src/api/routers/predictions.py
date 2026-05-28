"""
Router for GET /predictions.

Returns the latest predicted expected returns (mu) per asset, produced by
the most recent pipeline run. Reads from artifacts/predictions/.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.deps import predictions_path, resolve_run
from src.api.schemas.responses import PredictionItem, PredictionsResponse

router = APIRouter(tags=["predictions"])


@router.get(
    "/predictions",
    response_model=PredictionsResponse,
    summary="Retrieve predicted expected returns",
)
def get_predictions(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID to retrieve predictions from. Defaults to the latest completed run.",
    ),
    model: Optional[str] = Query(
        default=None,
        description="Model name (e.g. 'ridge', 'mlp', 'markowitz'). "
                    "Defaults to the model with the highest Sharpe in the run.",
    ),
) -> PredictionsResponse:
    """Return the blended expected-return predictions (mu) for each selected asset.

    Reads from the predictions CSV artifact written by stage_export_artifacts.
    For ML models, each value is the blended prediction:
        alpha * ml_pred + (1 - alpha) * hist_mean.
    For the 'markowitz' baseline, values are the raw historical means.

    Args:
        run_id: Optional run identifier. If omitted, the most recent completed run is used.
        model: Optional model name. If omitted, returns the best-Sharpe model for the run.

    Raises:
        404: If no completed runs exist or the requested run has no predictions artifact.
    """
    try:
        resolved_id, resolved_model = resolve_run(run_id, model)
    except (LookupError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    path = predictions_path(resolved_id, resolved_model)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Predictions artifact not found for run_id='{resolved_id}', "
                   f"model='{resolved_model}'. "
                   "Ensure the pipeline ran successfully through stage_export_artifacts.",
        )

    df = pd.read_csv(path)
    items = [
        PredictionItem(
            ticker=row["Ticker"],
            predicted_return=float(row["Predicted_Return"]),
            model=row["Model"],
        )
        for _, row in df.iterrows()
    ]

    return PredictionsResponse(run_id=resolved_id, model=resolved_model, predictions=items)
