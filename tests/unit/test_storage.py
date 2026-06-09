"""
Unit tests for the S3 storage layer (src/utils/storage.py).

All boto3 interaction is mocked; no real AWS calls are made.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import src.utils.storage as storage


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the cached S3 client around each test to avoid cross-test leakage."""
    storage._s3_client = None
    yield
    storage._s3_client = None


def test_is_s3_enabled_false_when_unset(monkeypatch):
    """is_s3_enabled returns False when no AWS env vars are set."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    assert storage.is_s3_enabled() is False


def test_is_s3_enabled_false_when_partial(monkeypatch):
    """is_s3_enabled returns False when only one of the required vars is set."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    assert storage.is_s3_enabled() is False


def test_is_s3_enabled_true_when_set(monkeypatch):
    """is_s3_enabled returns True when both required vars are set."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")
    assert storage.is_s3_enabled() is True


@pytest.mark.parametrize(
    "local,expected",
    [
        (Path("artifacts/weights/run_weights.csv"), "weights/run_weights.csv"),
        (Path("artifacts/runs/20260501_manifest.json"), "runs/20260501_manifest.json"),
        (Path("/app/artifacts/data/x.csv"), "data/x.csv"),
        (Path("artifacts/metrics/a/b.csv"), "metrics/a/b.csv"),
    ],
)
def test_s3_key_from_path(local, expected):
    """s3_key_from_path strips the artifacts/ prefix for relative and absolute paths."""
    assert storage.s3_key_from_path(local) == expected


def test_s3_upload_calls_put_object(monkeypatch, tmp_path):
    """s3_upload reads the file and calls put_object with bucket, key, and body."""
    monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")
    local = tmp_path / "file.csv"
    local.write_bytes(b"hello")

    mock_client = mock.Mock()
    monkeypatch.setattr(storage, "get_s3_client", lambda: mock_client)

    storage.s3_upload(local, "weights/file.csv")

    mock_client.put_object.assert_called_once()
    _, kwargs = mock_client.put_object.call_args
    assert kwargs["Bucket"] == "my-bucket"
    assert kwargs["Key"] == "weights/file.csv"
    assert kwargs["Body"] == b"hello"


def test_s3_download_returns_false_when_missing(monkeypatch, tmp_path):
    """s3_download returns False and writes nothing when the object does not exist."""
    from botocore.exceptions import ClientError

    monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")
    mock_client = mock.Mock()
    error = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
    )
    mock_client.get_object.side_effect = error
    monkeypatch.setattr(storage, "get_s3_client", lambda: mock_client)

    out = tmp_path / "out.json"
    result = storage.s3_download("runs/missing.json", out)

    assert result is False
    assert not out.exists()


def test_s3_download_writes_file_when_present(monkeypatch, tmp_path):
    """s3_download writes object bytes to disk and returns True when present."""
    monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")
    body = mock.Mock()
    body.read.return_value = b"content"
    mock_client = mock.Mock()
    mock_client.get_object.return_value = {"Body": body}
    monkeypatch.setattr(storage, "get_s3_client", lambda: mock_client)

    out = tmp_path / "nested" / "out.json"
    result = storage.s3_download("runs/x.json", out)

    assert result is True
    assert out.read_bytes() == b"content"


def test_s3_read_bytes_returns_none_when_missing(monkeypatch):
    """s3_read_bytes returns None when the object does not exist."""
    from botocore.exceptions import ClientError

    monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")
    mock_client = mock.Mock()
    error = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
    )
    mock_client.get_object.side_effect = error
    monkeypatch.setattr(storage, "get_s3_client", lambda: mock_client)

    assert storage.s3_read_bytes("runs/missing.json") is None


def test_s3_list_returns_all_keys(monkeypatch):
    """s3_list returns keys across all paginated pages for the given prefix."""
    monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")
    paginator = mock.Mock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "runs/a_manifest.json"}, {"Key": "runs/b_manifest.json"}]},
        {"Contents": [{"Key": "runs/c_manifest.json"}]},
        {},  # page with no Contents
    ]
    mock_client = mock.Mock()
    mock_client.get_paginator.return_value = paginator
    monkeypatch.setattr(storage, "get_s3_client", lambda: mock_client)

    keys = storage.s3_list("runs/")

    assert keys == [
        "runs/a_manifest.json",
        "runs/b_manifest.json",
        "runs/c_manifest.json",
    ]
    paginator.paginate.assert_called_once_with(Bucket="my-bucket", Prefix="runs/")
