#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYSIS_SCRIPT="$REPO_ROOT/analyse/collect_experiment_results.py"

if [[ "${1:-}" == "--print-repo-root" ]]; then
  printf '%s\n' "$REPO_ROOT"
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 2
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: $PYTHON_BIN" >&2
  echo "Set PYTHON=/path/to/python and rerun." >&2
  exit 2
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$ANALYSIS_SCRIPT" "$@"
