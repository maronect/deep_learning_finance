#!/bin/sh
# Ensure artifact subdirectories exist inside the mounted Fly volume.
# The Dockerfile creates these dirs during build, but the volume mount at
# /app/artifacts shadows them — so we recreate them at container start.
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
