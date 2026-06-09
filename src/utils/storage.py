"""
AWS S3 storage layer for pipeline artifacts.

Provides an optional S3 backend for artifact persistence. S3 acts as the primary
storage layer when AWS_ACCESS_KEY_ID and AWS_S3_BUCKET are set as environment
variables; otherwise the system falls back to the local filesystem so tests and
local development without credentials keep working.

Configuration is read exclusively from environment variables, never from YAML:
    AWS_ACCESS_KEY_ID       access key
    AWS_SECRET_ACCESS_KEY   secret key
    AWS_DEFAULT_REGION      region (default: sa-east-1)
    AWS_S3_BUCKET           bucket name

Objects mirror the local artifacts/ layout: the S3 key is the local path with
the leading "artifacts/" prefix removed (e.g. artifacts/weights/x.csv ->
weights/x.csv).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DEFAULT_REGION = "sa-east-1"
_ARTIFACTS_PREFIX = "artifacts"

# Cached boto3 client (singleton). Built lazily on first use.
_s3_client = None


def is_s3_enabled() -> bool:
    """Return True if S3 storage is configured via environment variables.

    Both AWS_ACCESS_KEY_ID and AWS_S3_BUCKET must be set for S3 to be active.
    When either is missing the pipeline uses the local filesystem instead.

    Returns:
        True if S3 should be used as the primary storage layer.
    """
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_S3_BUCKET"))


def _bucket() -> str:
    """Return the configured S3 bucket name from the environment.

    Raises:
        RuntimeError: If AWS_S3_BUCKET is not set.
    """
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise RuntimeError("AWS_S3_BUCKET is not set; S3 storage is not configured.")
    return bucket


def get_s3_client():
    """Return a cached boto3 S3 client built from environment credentials.

    The client is created once and reused (singleton) to avoid re-establishing
    configuration on every call. boto3 is imported lazily so the dependency is
    only required when S3 is actually used.

    Returns:
        A boto3 S3 client instance.
    """
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", _DEFAULT_REGION),
        )
    return _s3_client


def s3_key_from_path(local_path: Path) -> str:
    """Convert a local artifact path into an S3 key.

    Removes the leading "artifacts/" prefix so objects mirror the local layout.
    Works for both relative and absolute paths (e.g. /app/artifacts/...).

    Args:
        local_path: Local filesystem path (e.g. artifacts/weights/run_weights.csv).

    Returns:
        The S3 key relative to the bucket root (e.g. weights/run_weights.csv).
    """
    parts = Path(local_path).parts
    if _ARTIFACTS_PREFIX in parts:
        idx = parts.index(_ARTIFACTS_PREFIX)
        parts = parts[idx + 1:]
    return "/".join(parts)


def s3_upload(local_path: Path, s3_key: str) -> None:
    """Upload a local file to S3.

    Args:
        local_path: Path to the local file to upload.
        s3_key: Destination key within the bucket (e.g. weights/run_weights.csv).
    """
    client = get_s3_client()
    with open(local_path, "rb") as f:
        client.put_object(Bucket=_bucket(), Key=s3_key, Body=f.read())


def s3_read_bytes(s3_key: str) -> Optional[bytes]:
    """Read an S3 object directly into memory without writing to disk.

    Args:
        s3_key: Key within the bucket.

    Returns:
        The object bytes, or None if the object does not exist.
    """
    client = get_s3_client()
    try:
        response = client.get_object(Bucket=_bucket(), Key=s3_key)
        return response["Body"].read()
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a not-found
        if _is_not_found(exc):
            return None
        raise


def s3_download(s3_key: str, local_path: Path) -> bool:
    """Download an S3 object to a local file.

    Parent directories of the destination are created automatically.

    Args:
        s3_key: Source key within the bucket.
        local_path: Destination local path.

    Returns:
        True if the object was found and downloaded, False if it does not exist.
    """
    data = s3_read_bytes(s3_key)
    if data is None:
        return False
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    return True


def s3_list(prefix: str) -> list[str]:
    """List all object keys in the bucket under a given prefix.

    Uses a paginator so prefixes with more than 1000 objects are fully listed.

    Args:
        prefix: Key prefix to filter by (e.g. "runs/").

    Returns:
        List of matching object keys. Empty if none match.
    """
    client = get_s3_client()
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _is_not_found(exc: Exception) -> bool:
    """Return True if an S3 exception represents a missing object or bucket.

    Handles both the typed boto3 client exceptions (e.g. NoSuchKey) and the
    generic botocore ClientError, which carries the error code in its response.

    Args:
        exc: The exception raised by a boto3 S3 call.

    Returns:
        True if the exception means the object does not exist.
    """
    not_found_codes = {"NoSuchKey", "NoSuchBucket", "404"}
    if type(exc).__name__ in not_found_codes:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        if code in not_found_codes:
            return True
    return False
