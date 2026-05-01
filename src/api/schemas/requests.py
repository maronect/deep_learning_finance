"""
Pydantic request schemas for all API endpoints.

Validates and documents the structure of incoming request bodies and query parameters.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator

from src.pipeline.runner import STAGE_ORDER


class PipelineRunRequest(BaseModel):
    """Request body for POST /pipeline/run.

    Attributes:
        stages: Optional list of stage names to execute. If None, all stages run
            in canonical order. Must be valid stage names from STAGE_REGISTRY.
    """

    stages: Optional[list[str]] = None

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Ensure all requested stage names are valid."""
        if v is None:
            return v
        invalid = [s for s in v if s not in STAGE_ORDER]
        if invalid:
            raise ValueError(
                f"Unknown stage names: {invalid}. Valid stages: {STAGE_ORDER}."
            )
        return v
