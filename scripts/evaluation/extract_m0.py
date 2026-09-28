from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import ProbeManifest, load_coco_manifest, load_lcs_manifest, validate_manifest_images  # noqa: E402
from embeddings.artifact import save_embedding_artifact  # noqa: E402
from evaluation.embedding_export import export_raw_embeddings, runtime_metadata  # noqa: E402
from model_adapters.factory import create_m0_adapter  # noqa: E402


def load_manifest(probe: str) -> ProbeManifest:
    if probe == "coco":
        return load_coco_manifest(PROJECT_ROOT / "data/splits/coco_2017_val_probe_v1.json", PROJECT_ROOT)
    return load_lcs_manifest(PROJECT_ROOT, PROJECT_ROOT / "data/splits/lcs_558k_in_domain_probe_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export M0 raw embeddings for any configured model adapter.")
    parser.add_argument("--model", choices=("clip", "vista", "albef", "beit3"), required=True)
    parser.add_argument("--probe", choices=("coco", "lcs"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    manifest = load_manifest(args.probe)
    validate_manifest_images(manifest)
    adapter = create_m0_adapter(args.model, PROJECT_ROOT, args.device)
    image_embeddings, text_embeddings = export_raw_embeddings(
        adapter,
        manifest,
        args.batch_size,
        progress=lambda completed, total: print(f"{completed}/{total}", flush=True),
    )
    adapter_metadata = adapter.metadata()
    output_dir = PROJECT_ROOT / "outputs/embeddings/m0" / adapter_metadata["model_name"] / manifest.name
    stem = f"m0_{adapter_metadata['model_name']}_{manifest.name}"
    metadata = {
        **adapter_metadata,
        "probe_name": manifest.name,
        "probe_manifest_path": str(manifest.path.relative_to(PROJECT_ROOT)),
        "probe_manifest_sha256": manifest.sha256,
        "sample_count": len(manifest.samples),
        "semantic_instance_rule": "one image-text pair per manifest sample",
        "device": args.device,
        "runtime": runtime_metadata(),
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
