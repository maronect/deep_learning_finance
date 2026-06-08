#!/bin/sh
# Ensure artifact subdirectories exist on the container's ephemeral disk.
# In production there is no persistent volume: S3 is the source of truth and
# these dirs only hold transient writes (e.g. the rebuilt SQLite DB and any
# files downloaded from S3 to /tmp). They are recreated on every start.
set -e

mkdir -p \
    /app/artifacts/data \
    /app/artifacts/models \
    /app/artifacts/predictions \
    /app/artifacts/metrics \
    /app/artifacts/weights \
    /app/artifacts/runs \
    /app/artifacts/logs

exec "$@"
