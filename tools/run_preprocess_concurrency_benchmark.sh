#!/usr/bin/env bash

# Compatibility entry point for the isolated VLM-request preparation benchmark.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x $PYTHON ]]; then
  echo "Virtual environment not found: $PYTHON" >&2
  exit 1
fi

exec "$PYTHON" tools/benchmark_preprocess_concurrency.py "$@"
