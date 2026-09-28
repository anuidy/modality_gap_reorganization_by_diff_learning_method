#!/usr/bin/env python
"""Audit every ``m0 -> p001`` delta against the frozen M0 baseline.

The baseline (``outputs/metrics/m0/<model>/<probe>/m0_*_metrics.json``, anchored
by ``data/metadata/m0_sha256.txt``) is the origin of every delta in this project,
so each ``m0 -> p001`` transition must satisfy, key by key::

    delta == p001 point metric - M0 baseline metric

This script checks that for the whole flat schema of every run/probe, plus the
two things that can silently narrow a delta: a missing flat key, or a baseline
file that does not match the anchor list.

Read-only by design. The baseline is frozen data and transitions are written by
the evaluator, so a mismatch is always a signal to investigate rather than
something to auto-repair.

Usage:
    python -u scripts/analysis/audit_m0_transitions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.artifact import sha256_file  # noqa: E402
from metrics.representation_metrics import FLAT_METRIC_KEYS  # noqa: E402

TRAJECTORY_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "trajectory"
M0_METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics" / "m0"
ANCHOR_LIST = PROJECT_ROOT / "data" / "metadata" / "m0_sha256.txt"
MODEL_ARTIFACTS = {
    "clip": "openai_clip_vit_l14",
    "vista": "vista_base_stage1",
    "beit3": "beit3_base_itc_patch16_224",
    "albef": "albef_14m_pretrained",
}
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=None, help="Subset of run directory names (default: all).")
    return parser.parse_args()


def anchor_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in ANCHOR_LIST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        entries[relative.strip()] = digest
    return entries


def main() -> int:
    args = parse_args()
    anchors = anchor_entries()
    runs = sorted(args.runs) if args.runs else sorted(p.name for p in TRAJECTORY_METRICS.iterdir() if p.is_dir())
    failures: list[str] = []
    checked = 0

    for run in runs:
        index_path = TRAJECTORY_METRICS / run / "trajectory_index.json"
        if not index_path.is_file():
            continue
        model = json.loads(index_path.read_text(encoding="utf-8"))["model_name"]
        artifact_model = MODEL_ARTIFACTS[model]
        for probe_directory in sorted(p for p in (TRAJECTORY_METRICS / run).iterdir() if p.is_dir()):
            probe = probe_directory.name
            transition_path = probe_directory / "transitions" / "m0_to_p001.json"
            if not transition_path.is_file():
                continue
            baseline_path = M0_METRICS_ROOT / artifact_model / probe / f"m0_{artifact_model}_{probe}_metrics.json"
            point_path = probe_directory / "points" / f"trajectory_{run}_p001_{probe}_metrics.json"
            delta = json.loads(transition_path.read_text(encoding="utf-8"))["point_metric_delta_target_minus_source"]
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            point = json.loads(point_path.read_text(encoding="utf-8"))
            checked += 1

            missing = [key for key in FLAT_METRIC_KEYS if key not in delta]
            if missing:
                failures.append(f"{run}/{probe}: delta is missing {missing}")
            for key in FLAT_METRIC_KEYS:
                if key not in delta:
                    continue
                expected = float(point[key]) - float(baseline[key])
                if abs(expected - float(delta[key])) > TOLERANCE * max(1.0, abs(expected)):
                    failures.append(f"{run}/{probe}: {key} delta={delta[key]} expected={expected}")

            recorded = anchors.get(str(baseline_path.relative_to(PROJECT_ROOT)))
            if recorded is None:
                failures.append(f"{run}/{probe}: baseline {baseline_path.name} is not in {ANCHOR_LIST.name}")
            elif recorded != sha256_file(baseline_path):
                failures.append(f"{run}/{probe}: baseline {baseline_path.name} does not match the anchor")

    for failure in failures:
        print(f"MISMATCH {failure}")
    print(f"audited {checked} m0 -> p001 transitions; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
