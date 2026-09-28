"""Validate the frozen M0 baseline: embeddings, metrics and geometry reference.

The M0 baseline is the immutable origin of every delta in this project. Its
integrity anchor is ``data/metadata/m0_sha256.txt`` (byte-level), and this script
checks the semantics that the anchor cannot: sample-ID order, embedding shape and
dtype, probe-manifest identity, metric self-consistency, and that the geometry
reference points at the probe's fixed pair index.

The geometry reference was produced by the original Windows export, so the path
it stores internally is a Windows path; it is relocated onto the copied file.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import load_coco_manifest, load_lcs_manifest  # noqa: E402
from embeddings.artifact import load_embedding_artifact  # noqa: E402
from metrics.representation_metrics import GEOMETRY_PAIR_COUNT, KNN_K  # noqa: E402


MODELS = {
    "openai_clip_vit_l14": 768,
    "vista_base_stage1": 768,
    "albef_14m_pretrained": 256,
    "beit3_base_itc_patch16_224": 768,
}
PROBES = ("coco_2017_val_5k", "lcs_558k_in_domain_10k")


def assert_finite_json(value: Any, context: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_json(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{context}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite metric value at {context}.")


def manifest_for_probe(probe_directory: str):
    if probe_directory == "coco_2017_val_5k":
        return load_coco_manifest(PROJECT_ROOT / "data/splits/coco_2017_val_probe_v1.json", PROJECT_ROOT)
    if probe_directory == "lcs_558k_in_domain_10k":
        return load_lcs_manifest(PROJECT_ROOT, PROJECT_ROOT / "data/splits/lcs_558k_in_domain_probe_v1.json")
    raise ValueError(f"Unknown probe directory: {probe_directory}")


def relocate_stored_path(value: str, expected: Path) -> Path:
    """Map a path recorded by the original export onto the copied file."""

    expected_parts = expected.relative_to(PROJECT_ROOT).parts
    stored_parts = PureWindowsPath(value).parts
    if stored_parts[-len(expected_parts):] != expected_parts:
        raise ValueError(f"Stored path {value!r} does not match {expected}.")
    return expected


def pair_index_path(probe_directory: str, manifest_sha256: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "geometry_references"
        / f"{probe_directory}_{manifest_sha256[:12]}_upper_triangle_pairs_v1.npz"
    )


def validate_model_probe(model_name: str, probe_directory: str) -> None:
    expected_dimension = MODELS[model_name]
    manifest = manifest_for_probe(probe_directory)
    embedding_directory = PROJECT_ROOT / "outputs/embeddings/m0" / model_name / probe_directory
    metrics_directory = PROJECT_ROOT / "outputs/metrics/m0" / model_name / probe_directory
    stem = next(embedding_directory.glob("*.npz")).stem

    artifact_path = embedding_directory / f"{stem}.npz"
    metadata_path = embedding_directory / f"{stem}.json"
    metrics_path = metrics_directory / f"{stem}_metrics.json"
    geometry_path = metrics_directory / f"{stem}_geometry_reference.npz"
    for path in (artifact_path, metadata_path, metrics_path, geometry_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sample_ids, image_embeddings, text_embeddings = load_embedding_artifact(artifact_path)
    expected_ids = [sample.sample_id for sample in manifest.samples]
    if sample_ids.tolist() != expected_ids:
        raise ValueError(f"Sample-ID order mismatch: {model_name}/{probe_directory}")
    expected_shape = (len(expected_ids), expected_dimension)
    if image_embeddings.shape != expected_shape or text_embeddings.shape != expected_shape:
        raise ValueError(f"Embedding shape mismatch: {model_name}/{probe_directory}")
    if image_embeddings.dtype != np.float32 or text_embeddings.dtype != np.float32:
        raise ValueError(f"Embedding dtype mismatch: {model_name}/{probe_directory}")
    if not np.isfinite(image_embeddings).all() or not np.isfinite(text_embeddings).all():
        raise ValueError(f"Non-finite embedding: {model_name}/{probe_directory}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["probe_manifest_sha256"] != manifest.sha256:
        raise ValueError(f"Manifest hash mismatch: {model_name}/{probe_directory}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert_finite_json(metrics)
    expected_pairs = len(expected_ids) * (len(expected_ids) - 1)
    for direction in ("image_query", "text_query"):
        if metrics["score_gap"][direction]["candidate_pair_count"] != expected_pairs:
            raise ValueError(f"Score-gap pair mismatch ({direction}): {model_name}/{probe_directory}")
    if metrics.get("m0_baseline", {}).get("probe_manifest_sha256") != manifest.sha256:
        raise ValueError(f"Baseline block does not match the probe: {model_name}/{probe_directory}")

    with np.load(geometry_path, allow_pickle=False) as state:
        if len(state["image_pair_cosines"]) != min(GEOMETRY_PAIR_COUNT, len(expected_ids) * (len(expected_ids) - 1) // 2):
            raise ValueError(f"Geometry reference pair count mismatch: {model_name}/{probe_directory}")
        if str(state["manifest_sha256"]) != manifest.sha256:
            raise ValueError(f"Geometry reference manifest mismatch: {model_name}/{probe_directory}")
        if int(state["knn_k"]) != KNN_K:
            raise ValueError(f"Geometry reference kNN mismatch: {model_name}/{probe_directory}")
        stored_pair_index = str(state["pair_index_path"])

    resolved = relocate_stored_path(stored_pair_index, pair_index_path(probe_directory, manifest.sha256))
    if not resolved.is_file():
        raise ValueError(f"Pair index missing: {resolved}")
    print(f"OK {model_name}/{probe_directory}: {expected_shape}")


def main() -> None:
    validated = 0
    for model_name in MODELS:
        for probe_directory in PROBES:
            validate_model_probe(model_name, probe_directory)
            validated += 1
    print(f"Validated {validated} model-probe M0 outputs.")


if __name__ == "__main__":
    main()
