"""
Unit tests for S3 mirroring in src/utils/export.py.

Regression guard: artifacts written by export.py must be uploaded to S3 when
S3 storage is enabled, and must not be uploaded when it is disabled. The run
manifest in particular is the entry point for the S3 fallback in the registry
and database sync, so its upload is verified explicitly.
"""
from __future__ import annotations

import json
from unittest import mock

import src.utils.export as export


def test_save_manifest_uploads_when_s3_enabled(tmp_path):
    """save_manifest writes locally and mirrors to S3 with the correct key."""
    path = tmp_path / "artifacts" / "runs" / "r1_manifest.json"

    with mock.patch.object(export, "is_s3_enabled", return_value=True), \
         mock.patch.object(export, "s3_upload") as upload:
        export.save_manifest({"run_id": "r1"}, str(path))

    assert path.exists()
    assert json.loads(path.read_text())["run_id"] == "r1"
    upload.assert_called_once()
    args, _ = upload.call_args
    assert args[1] == "runs/r1_manifest.json"


def test_save_manifest_skips_upload_when_s3_disabled(tmp_path):
    """save_manifest writes locally and performs no upload when S3 is disabled."""
    path = tmp_path / "artifacts" / "runs" / "r1_manifest.json"

    with mock.patch.object(export, "is_s3_enabled", return_value=False), \
         mock.patch.object(export, "s3_upload") as upload:
        export.save_manifest({"run_id": "r1"}, str(path))

    assert path.exists()
    upload.assert_not_called()


def test_save_returns_uploads_when_s3_enabled(tmp_path):
    """A representative CSV-writing save function also mirrors to S3."""
    import pandas as pd

    path = tmp_path / "artifacts" / "data" / "r1_returns.csv"
    df = pd.DataFrame({"PETR4.SA": [0.01, -0.02]})

    with mock.patch.object(export, "is_s3_enabled", return_value=True), \
         mock.patch.object(export, "s3_upload") as upload:
        export.save_returns(df, str(path))

    assert path.exists()
    upload.assert_called_once()
    args, _ = upload.call_args
    assert args[1] == "data/r1_returns.csv"
