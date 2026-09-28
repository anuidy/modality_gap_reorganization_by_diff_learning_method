#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

# Explicit --runs is required; no training starts unless --execute is supplied.
exec python scripts/training/run_independent.py "$@"
