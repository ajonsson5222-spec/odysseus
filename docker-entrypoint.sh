#!/bin/sh
set -e

# Create data directories if they don't exist
mkdir -p "${DATA_DIR:-/app/data}/uploads" "${DATA_DIR:-/app/data}/documents" 2>/dev/null || true

# Ensure read/write permissions for the storage folder regardless of host mount UID
chmod -R 777 "${DATA_DIR:-/app/data}" 2>/dev/null || true

exec "$@"
