"""
Defines each pipeline stage as an independent, composable function.

Stages in order: ingest → compute_returns → select_assets → build_features →
train_models → predict_returns → optimize_portfolio → evaluate → export_artifacts.
Each stage reads from and writes to the PipelineContext.
"""
