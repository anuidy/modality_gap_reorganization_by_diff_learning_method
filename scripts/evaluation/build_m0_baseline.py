#!/usr/bin/env python
"""Build the M0 baseline of one model on one probe.

Every delta in this project is measured from M0, so a probe needs an M0 baseline
before its checkpoints can be compared: the raw embedding artifact plus the
metric document and geometry reference under ``outputs/metrics/m0/``.

    outputs/embeddings/m0/<model_artifact>/<probe>/m0_*.npz|.json   (input)
    outputs/metrics/m0/<model_artifact>/<probe>/m0_*_metrics.json   (written)
    outputs/metrics/m0/<model_artifact>/<probe>/m0_*_geometry_reference.npz

The embedding export comes from ``extract_m0.py``; this script turns it into the
baseline. It **refuses to overwrite an existing baseline** unless ``--force`` is
given: the shipped baseline is frozen data (recomputation drifts in the last
digits, ~1e-15, and is anchored by ``data/metadata/m0_sha256.txt``), so a rebuild
is only ever a deliberate new baseline, never a refresh.

Usage:
    python -u scripts/evaluation/build_m0_baseline.py \
        --model clip --probe-manifest data/splits/coco_2017_val_probe_v1.json
    python -u scripts/evaluation/build_m0_baseline.py --model clip vista beit3 albef \
        --probe-manifest data/splits/coco_2017_val_probe_v1.json --write-anchor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import load_probe_manifest  # noqa: E402
from embeddings.artifact import load_embedding_artifact, sha256_file  # noqa: E402
from evaluation.trajectory import (  # noqa: E402
    MODEL_ARTIFACT_NAMES,
    REPRESENTATION_BOUNDARY,
    write_json_atomic,
)
from metrics.representation_metrics import (  # noqa: E402
    compute_point_metrics,
    intra_modal_geometry_preservation,
    l2_normalize,
    save_geometry_state,
)


M0_EMBEDDINGS_ROOT = PROJECT_ROOT / "outputs" / "embeddings" / "m0"
M0_METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics" / "m0"
PAIR_INDEX_ROOT = PROJECT_ROOT / "data" / "metadata" / "geometry_references"
ANCHOR_LIST = PROJECT_ROOT / "data" / "metadata" / "m0_sha256.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", required=True, choices=sorted(MODEL_ARTIFACT_NAMES))
    parser.add_argument("--probe-manifest", type=Path, required=True, help="Probe manifest in the canonical schema.")
    parser.add_argument("--dataset", default=None, help="Dataset label for the observation (default: probe name).")
    parser.add_argument("--split", default=None, help="Split label for the observation (default: 'unknown').")
    parser.add_argument("--score-block-size", type=int, default=512)
    parser.add_argument("--write-anchor", action="store_true", help="Rewrite data/metadata/m0_sha256.txt afterwards.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing baseline.")
    return parser.parse_args()


def build_one(model: str, manifest, args: argparse.Namespace) -> tuple[Path, Path]:
    artifact_model = MODEL_ARTIFACT_NAMES[model]
    stem = f"m0_{artifact_model}_{manifest.name}"
    embedding_directory = M0_EMBEDDINGS_ROOT / artifact_model / manifest.name
    metrics_directory = M0_METRICS_ROOT / artifact_model / manifest.name
    artifact_path = embedding_directory / f"{stem}.npz"
    metadata_path = embedding_directory / f"{stem}.json"
    metrics_path = metrics_directory / f"{stem}_metrics.json"
    geometry_path = metrics_directory / f"{stem}_geometry_reference.npz"

    for path in (artifact_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"M0 embedding export is required first (see scripts/evaluation/extract_m0.py): {path}"
            )
    if not args.force:
        for path in (metrics_path, geometry_path):
            if path.exists():
                raise FileExistsError(
                    f"An M0 baseline already exists at {path}. It is frozen data; pass --force only "
                    "when deliberately starting a new baseline."
                )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("probe_manifest_sha256") != manifest.sha256:
        raise ValueError(
            f"{artifact_path}: the embedding was exported for probe manifest "
            f"{metadata.get('probe_manifest_sha256')}, not {manifest.sha256}."
        )
    artifact_sha = sha256_file(artifact_path)
    if metadata.get("artifact", {}).get("sha256") != artifact_sha:
        raise ValueError(f"{artifact_path}: artifact SHA-256 does not match its metadata.")

    sample_ids, image_raw, text_raw = load_embedding_artifact(artifact_path)
    identity = {
        "model": model,
        "training_regime": "m0",
        "checkpoint": str(metadata.get("checkpoint", "")),
        "global_step": 0,
        "training_progress": 0.0,
        "dataset": args.dataset or manifest.name,
        "split": args.split or "unknown",
        "sample_count": int(len(sample_ids)),
    }
    metrics = compute_point_metrics(
        image_raw,
        text_raw,
        score_block_size=args.score_block_size,
        representation_boundary=REPRESENTATION_BOUNDARY.get(model, "unknown"),
        raw_available=True,
        identity=identity,
    )

    # The geometry reference defines the pair set every later comparison uses; a
    # probe's pair index is created here if this is its first baseline.
    pair_index_path = PAIR_INDEX_ROOT / f"{manifest.name}_{manifest.sha256[:12]}_upper_triangle_pairs_v1.npz"
    save_geometry_state(
        l2_normalize(image_raw),
        l2_normalize(text_raw),
        manifest.sha256,
        pair_index_path,
        geometry_path,
    )
    # M0 is the reference, so its own intra-modal preservation is the reference
    # compared with itself (rho = 1 by construction, measured rather than assumed).
    preservation = intra_modal_geometry_preservation(geometry_path, geometry_path)
    metrics["intra_geometry_image"] = preservation["image"]
    metrics["intra_geometry_text"] = preservation["text"]
    auxiliary = metrics.setdefault("auxiliary_metrics", {})
    auxiliary.update(preservation["auxiliary"])
    auxiliary["intra_geometry_reference"] = "m0"
    auxiliary["m0_row_self_comparison"] = True
    metrics["artifact"] = {
        "path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "sha256": artifact_sha,
        "sample_count": int(len(sample_ids)),
        "embedding_dim": int(image_raw.shape[1]),
    }
    metrics["snapshot"] = {
        "label": "m0",
        "optimizer_step": 0,
        "progress_fraction": 0.0,
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "branch": "m0",
    }
    metrics["m0_baseline"] = {
        "role": "baseline",
        "embedding_artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
        "embedding_artifact_sha256": artifact_sha,
        "geometry_reference": str(geometry_path.relative_to(PROJECT_ROOT)),
        "geometry_reference_sha256": sha256_file(geometry_path),
        "probe_manifest_sha256": manifest.sha256,
    }
    write_json_atomic(metrics_path, metrics)
    print(f"built {model}/{manifest.name}: {metrics_path.relative_to(PROJECT_ROOT)}")
    return metrics_path, geometry_path


def write_anchor() -> int:
    """Rewrite the M0 integrity anchor over every baseline on disk."""

    entries: list[tuple[str, str]] = []
    for model_directory in sorted(p for p in M0_EMBEDDINGS_ROOT.iterdir() if p.is_dir()):
        for probe_directory in sorted(p for p in model_directory.iterdir() if p.is_dir()):
            artifact_model = model_directory.name
            stem = f"m0_{artifact_model}_{probe_directory.name}"
            metrics_directory = M0_METRICS_ROOT / artifact_model / probe_directory.name
            for path in (
                probe_directory / f"{stem}.npz",
                probe_directory / f"{stem}.json",
                metrics_directory / f"{stem}_metrics.json",
                metrics_directory / f"{stem}_geometry_reference.npz",
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete M0 baseline: {path}")
                entries.append((sha256_file(path), str(path.relative_to(PROJECT_ROOT))))
    ANCHOR_LIST.write_text("".join(f"{digest}  {name}\n" for digest, name in entries), encoding="utf-8")
    print(f"anchor rewritten: {ANCHOR_LIST.relative_to(PROJECT_ROOT)} ({len(entries)} files)")
    return len(entries)


def main() -> int:
    args = parse_args()
    manifest = load_probe_manifest(args.probe_manifest.resolve(), PROJECT_ROOT)
    print(f"probe {manifest.name}: {len(manifest.samples)} samples, sha256 {manifest.sha256[:12]}")
    for model in args.model:
        build_one(model, manifest, args)
    if args.write_anchor:
        write_anchor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
