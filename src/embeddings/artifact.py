from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_embedding_artifact(
    output_directory: Path,
    stem: str,
    sample_ids: list[str],
    image_embeddings_raw: np.ndarray,
    text_embeddings_raw: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    if image_embeddings_raw.shape != text_embeddings_raw.shape:
        raise ValueError("Image and text embeddings must share shape [N, D].")
    if image_embeddings_raw.shape[0] != len(sample_ids):
        raise ValueError("Sample-ID count does not match embedding rows.")
    if image_embeddings_raw.dtype != np.float32 or text_embeddings_raw.dtype != np.float32:
        raise TypeError("Artifacts must store float32 raw embeddings.")

    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = output_directory / f"{stem}.npz"
    metadata_path = output_directory / f"{stem}.json"
    np.savez(
        artifact_path,
        sample_ids=np.asarray(sample_ids, dtype=str),
        image_embeddings_raw=image_embeddings_raw,
        text_embeddings_raw=text_embeddings_raw,
    )
    completed_metadata = {
        **metadata,
        "artifact": {
            "path": artifact_path.name,
            "sha256": sha256_file(artifact_path),
            "format": "npz",
            "stored_fields": ["sample_ids", "image_embeddings_raw", "text_embeddings_raw"],
            "stored_dtype": "float32",
            "pre_l2_normalized": True,
            "normalized_embeddings_stored": False,
        },
    }
    metadata_path.write_text(
        json.dumps(completed_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return artifact_path, metadata_path


def load_embedding_artifact(artifact_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(artifact_path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"]
        image_embeddings = payload["image_embeddings_raw"].astype(np.float32, copy=False)
        text_embeddings = payload["text_embeddings_raw"].astype(np.float32, copy=False)
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("Artifact image/text shapes differ.")
    return sample_ids, image_embeddings, text_embeddings
