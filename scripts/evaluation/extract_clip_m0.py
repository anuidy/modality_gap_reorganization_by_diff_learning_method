from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import (  # noqa: E402
    ProbeManifest,
    load_coco_manifest,
    load_lcs_manifest,
    validate_manifest_images,
)
from embeddings.artifact import save_embedding_artifact  # noqa: E402
from models.clip_openai import OpenAIClipViTL14Adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw CLIP M0 embeddings for a fixed probe.")
    parser.add_argument("--probe", choices=("coco", "lcs"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_manifest(probe: str) -> ProbeManifest:
    if probe == "coco":
        return load_coco_manifest(
            PROJECT_ROOT / "data/splits/coco_2017_val_probe_v1.json", PROJECT_ROOT
        )
    return load_lcs_manifest(
        PROJECT_ROOT, PROJECT_ROOT / "data/splits/lcs_558k_in_domain_probe_v1.json"
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    manifest = load_manifest(args.probe)
    validate_manifest_images(manifest)
    checkpoint = PROJECT_ROOT / "data/raw/models/clip_sf/ViT-L-14.pt"
    adapter = OpenAIClipViTL14Adapter(checkpoint=checkpoint, device=args.device)

    image_batches: list[np.ndarray] = []
    text_batches: list[np.ndarray] = []
    total = len(manifest.samples)
    for start in range(0, total, args.batch_size):
        batch = manifest.samples[start : start + args.batch_size]
        images = load_images([sample.image_path for sample in batch])
        image_batches.append(adapter.encode_image(images).numpy())
        text_batches.append(adapter.encode_text([sample.text for sample in batch]).numpy())
        print(f"{min(start + len(batch), total)}/{total}", flush=True)

    image_embeddings = np.concatenate(image_batches, axis=0).astype(np.float32, copy=False)
    text_embeddings = np.concatenate(text_batches, axis=0).astype(np.float32, copy=False)
    output_dir = PROJECT_ROOT / "outputs/embeddings/m0/openai_clip_vit_l14" / manifest.name
    stem = f"m0_openai_clip_vit_l14_{manifest.name}"
    metadata = {
        **adapter.metadata(),
        "probe_name": manifest.name,
        "probe_manifest_path": str(manifest.path.relative_to(PROJECT_ROOT)),
        "probe_manifest_sha256": manifest.sha256,
        "sample_count": total,
        "semantic_instance_rule": "one image-text pair per manifest sample",
        "device": str(adapter.device),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": package_version("pillow"),
            "openai_clip": package_version("clip"),
        },
    }
    artifact_path, metadata_path = save_embedding_artifact(
        output_directory=output_dir,
        stem=stem,
        sample_ids=[sample.sample_id for sample in manifest.samples],
        image_embeddings_raw=image_embeddings,
        text_embeddings_raw=text_embeddings,
        metadata=metadata,
    )
    print(f"artifact={artifact_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
