from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.artifact import load_embedding_artifact  # noqa: E402
from metrics.six_metrics import compute_six_metrics, save_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the locked six-metric protocol from one M0 artifact.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--score-block-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids, image_embeddings, text_embeddings = load_embedding_artifact(args.artifact)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    manifest_sha256 = metadata["probe_manifest_sha256"]
    probe_name = metadata["probe_name"]
    model_name = metadata["model_name"]
    pair_index_path = (
        PROJECT_ROOT
        / "data/metadata/geometry_references"
        / f"{probe_name}_{manifest_sha256[:12]}_upper_triangle_pairs_v1.npz"
    )
    output_dir = PROJECT_ROOT / "outputs/metrics/m0" / model_name / probe_name
    geometry_reference_path = output_dir / f"{args.artifact.stem}_geometry_reference.npz"
    metrics = compute_six_metrics(
        image_embeddings_raw=image_embeddings,
        text_embeddings_raw=text_embeddings,
        manifest_sha256=manifest_sha256,
        pair_index_path=pair_index_path,
        geometry_reference_path=geometry_reference_path,
        score_block_size=args.score_block_size,
    )
    metrics["artifact"] = {
        "path": str(args.artifact),
        "metadata_path": str(args.metadata),
        "sample_count": int(len(sample_ids)),
        "embedding_dim": int(image_embeddings.shape[1]),
    }
    output_path = output_dir / f"{args.artifact.stem}_six_metrics.json"
    save_metrics(output_path, metrics)
    print(f"metrics={output_path}")


if __name__ == "__main__":
    main()
