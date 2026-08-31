#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

runs=(
  clip_standard
  clip_count_matched_mixed
  vista_standard
  vista_count_matched_mixed
  beit3_standard
  beit3_count_matched_mixed
  albef_itc_only
  albef_full
)

for run_id in "${runs[@]}"; do
  python scripts/training/train.py --run "$run_id" --validate-only
done

for run_id in "${runs[@]}"; do
  python -u scripts/training/train.py --run "$run_id"
done
