from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import load_coco_manifest, load_lcs_manifest  # noqa: E402
from embeddings.artifact import load_embedding_artifact  # noqa: E402


MODELS = {
    "openai_clip_vit_l14": 768,
    "vista_base_stage1": 768,
    "albef_14m_pretrained": 256,
    "beit3_base_itc_patch16_224": 768,
}


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


def main() -> None:
    validated = 0
    for model_name, expected_dimension in MODELS.items():
        for probe_directory in ("coco_2017_val_5k", "lcs_558k_in_domain_10k"):
            manifest = manifest_for_probe(probe_directory)
            embedding_directory = PROJECT_ROOT / "outputs/embeddings/m0" / model_name / probe_directory
            artifact_path = next(embedding_directory.glob("*.npz"))
            metadata_path = next(embedding_directory.glob("*.json"))
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
            metrics_path = next((PROJECT_ROOT / "outputs/metrics/m0" / model_name / probe_directory).glob("*_six_metrics.json"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert_finite_json(metrics)
            expected_pairs = len(expected_ids) * (len(expected_ids) - 1)
            if metrics["score_gap"]["image_query"]["candidate_pair_count"] != expected_pairs:
                raise ValueError(f"Image score-pair mismatch: {model_name}/{probe_directory}")
            if metrics["score_gap"]["text_query"]["candidate_pair_count"] != expected_pairs:
                raise ValueError(f"Text score-pair mismatch: {model_name}/{probe_directory}")
            geometry = metrics["intra_modal_geometry_preservation"]
            if geometry["pair_count"] != 1_000_000 or not Path(geometry["reference_path"]).is_file():
                raise ValueError(f"Geometry reference mismatch: {model_name}/{probe_directory}")
            print(f"OK {model_name}/{probe_directory}: {expected_shape}")
            validated += 1
    print(f"Validated {validated} model-probe M0 outputs.")


if __name__ == "__main__":
    main()
