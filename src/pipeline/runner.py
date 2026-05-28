"""
Entry point for pipeline execution.

Accepts a list of stage names (or runs all by default) and executes them
in order using a shared PipelineContext. Supports CLI invocation and
programmatic calls from the API layer.
"""
from __future__ import annotations

import datetime
import sys
import traceback
from typing import Callable, Optional

from src.pipeline.context import PipelineContext
from src.pipeline.stages import (
    stage_build_features,
    stage_compute_returns,
    stage_evaluate,
    stage_export_artifacts,
    stage_ingest,
    stage_optimize_portfolio,
    stage_predict_returns,
    stage_train_models,
)
from src.utils.config_loader import get_config

STAGE_REGISTRY: dict[str, Callable[[PipelineContext], None]] = {
    "ingest": stage_ingest,
    "compute_returns": stage_compute_returns,
    "build_features": stage_build_features,
    "train_models": stage_train_models,
    "predict_returns": stage_predict_returns,
    "optimize_portfolio": stage_optimize_portfolio,
    "evaluate": stage_evaluate,
    "export_artifacts": stage_export_artifacts,
}

STAGE_ORDER: list[str] = list(STAGE_REGISTRY.keys())


def run_pipeline(
    stages: Optional[list[str]] = None,
    context: Optional[PipelineContext] = None,
) -> PipelineContext:
    """Execute the full or partial pipeline using a shared PipelineContext.

    Stages are always executed in the canonical order defined by STAGE_ORDER,
    regardless of the order they appear in the stages argument. If stages is
    None, all stages run.

    Args:
        stages: Optional list of stage names to execute. Must be keys in STAGE_REGISTRY.
            If None, all stages run in canonical order.
        context: Optional pre-built PipelineContext. If None, a fresh context is
            created by loading config/pipeline.yaml and config/models.yaml.

    Returns:
        The PipelineContext after all requested stages have completed.

    Raises:
        ValueError: If an unknown stage name is passed in stages.
        RuntimeError: If a stage raises an exception. context.status is set to
            'failed' and context.error contains the traceback before re-raising.

    Example:
        >>> ctx = run_pipeline()
        >>> print(ctx.metrics)

        >>> ctx = run_pipeline(stages=["ingest", "compute_returns", "build_features"])
    """
    if context is None:
        pipeline_cfg = get_config("pipeline")
        models_cfg = get_config("models")
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        context = PipelineContext(
            pipeline_cfg=pipeline_cfg,
            models_cfg=models_cfg,
            run_id=run_id,
        )

    stages_to_run = stages if stages is not None else STAGE_ORDER

    unknown = [s for s in stages_to_run if s not in STAGE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown stage names: {unknown}. Valid stages: {STAGE_ORDER}."
        )

    # Always execute in canonical order even if a subset is requested.
    ordered_stages = [s for s in STAGE_ORDER if s in stages_to_run]

    context.status = "running"
    for stage_name in ordered_stages:
        stage_fn = STAGE_REGISTRY[stage_name]
        try:
            stage_fn(context)
        except Exception as exc:
            tb = traceback.format_exc()
            context.status = "failed"
            context.error = f"Stage '{stage_name}' failed: {exc}\n{tb}"
            raise RuntimeError(context.error) from exc

    return context


def main() -> None:
    """Command-line entry point for running the pipeline.

    Usage:
        python -m src.pipeline.runner
        python -m src.pipeline.runner ingest compute_returns build_features
    """
    stages = sys.argv[1:] if len(sys.argv) > 1 else None
    ctx = run_pipeline(stages=stages)
    print(f"Pipeline completed. Run ID: {ctx.run_id}")
    if ctx.artifacts_written:
        print("Artifacts written:")
        for path in ctx.artifacts_written:
            print(f"  {path}")
    if ctx.metrics:
        import json
        for model_name, model_metrics in ctx.metrics.items():
            numeric = {
                k: round(float(v), 4)
                for k, v in model_metrics.items()
                if isinstance(v, (int, float))
            }
            print(f"Metrics [{model_name}]: {json.dumps(numeric, indent=2)}")


if __name__ == "__main__":
    main()
