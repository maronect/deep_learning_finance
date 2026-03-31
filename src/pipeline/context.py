"""
Defines the PipelineContext dataclass that carries configuration and intermediate
state between pipeline stages.

Holds loaded config, raw data, processed features, trained models, predictions,
and optimized weights — acting as the single shared state object across stages.
"""
