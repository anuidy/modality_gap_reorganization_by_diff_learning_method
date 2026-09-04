from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import (  # noqa: E402
    ProbeManifest,
    load_coco_manifest,
    load_lcs_manifest,
    validate_manifest_images,
)
from embeddings.artifact import (  # noqa: E402
    load_embedding_artifact,
    save_embedding_artifact,
    sha256_file,
)
from evaluation.embedding_export import export_raw_embeddings, runtime_metadata  # noqa: E402
from evaluation.trajectory import (  # noqa: E402
    MODEL_ARTIFACT_NAMES,
    PointOutput,
    SnapshotSpec,
    compute_adjacent_transition,
    compute_point_output,
    load_completed_run_manifest,
    load_snapshot_into_adapter,
    mark_snapshot_evaluated,
    resolve_trajectory_snapshots,
    write_json_atomic,
)
from metrics.six_metrics import compute_six_metrics, save_metrics  # noqa: E402
from models.factory import create_m0_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw embeddings and adjacent geometry transitions for one completed branch."
    )
    parser.add_argument("--run", required=True, help="Training run_id, including a future Full GCL run.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-block-size", type=int, default=512)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolved_config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _probe_manifests(run_manifest: dict[str, Any]) -> tuple[ProbeManifest, ProbeManifest]:
    config = run_manifest["config"]
    coco = load_coco_manifest(_resolved_config_path(config["coco_probe_manifest"]), PROJECT_ROOT)
    lcs = load_lcs_manifest(
        PROJECT_ROOT,
        _resolved_config_path(config["lcs_probe_manifest"]),
        image_root=_resolved_config_path(config["image_root"]),
    )
    expected = {
        "coco": config["coco_probe_manifest_sha256"],
        "lcs": config["lcs_probe_manifest_sha256"],
    }
    if coco.sha256 != expected["coco"] or lcs.sha256 != expected["lcs"]:
        raise ValueError("Run checkpoint and current probe manifests have different SHA-256 values.")
    validate_manifest_images(coco)
    validate_manifest_images(lcs)
    return coco, lcs


def _valid_existing_embedding(
    artifact_path: Path,
    metadata_path: Path,
    snapshot: SnapshotSpec,
    manifest: ProbeManifest,
) -> bool:
    if not artifact_path.exists() and not metadata_path.exists():
        return False
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise FileExistsError("Trajectory embedding output is incomplete; refusing to overwrite it.")
    metadata = _read_json(metadata_path)
    if metadata.get("probe_manifest_sha256") != manifest.sha256:
        raise ValueError("Existing trajectory artifact uses a different probe manifest.")
    if metadata.get("training_snapshot", {}).get("checkpoint_sha256") != snapshot.artifact_sha256:
        raise ValueError("Existing trajectory artifact belongs to a different checkpoint.")
    if metadata.get("artifact", {}).get("sha256") != sha256_file(artifact_path):
        raise ValueError("Existing trajectory embedding artifact failed SHA-256 validation.")
    return True


def _export_snapshot_probe(
    adapter: Any,
    adapter_metadata: dict[str, Any],
    snapshot: SnapshotSpec,
    manifest: ProbeManifest,
    run_manifest: dict[str, Any],
    batch_size: int,
    device: str,
) -> tuple[Path, Path]:
    output_directory = (
        PROJECT_ROOT
        / "outputs"
        / "embeddings"
        / "trajectory"
        / run_manifest["config"]["run_id"]
        / snapshot.label
        / manifest.name
    )
    stem = f"trajectory_{run_manifest['config']['run_id']}_{snapshot.label}_{manifest.name}"
    artifact_path = output_directory / f"{stem}.npz"
    metadata_path = output_directory / f"{stem}.json"
    if _valid_existing_embedding(artifact_path, metadata_path, snapshot, manifest):
        return artifact_path, metadata_path

    image_embeddings, text_embeddings = export_raw_embeddings(
        adapter,
        manifest,
        batch_size,
        progress=lambda completed, total: print(
            f"{snapshot.label} {manifest.name}: {completed}/{total}", flush=True
        ),
    )
    initialization = {
        "path": adapter_metadata["checkpoint"],
        "sha256": adapter_metadata["checkpoint_sha256"],
        "stage": adapter_metadata["checkpoint_stage"],
    }
    metadata = {
        **adapter_metadata,
        "checkpoint": str(snapshot.checkpoint_path),
        "checkpoint_sha256": snapshot.artifact_sha256,
        "checkpoint_stage": f"trajectory_{snapshot.label}",
        "initialization_checkpoint": initialization,
        "run_id": run_manifest["config"]["run_id"],
        "branch": run_manifest["config"]["branch"],
        "optimizer_step": snapshot.optimizer_step,
        "progress_fraction": snapshot.progress_fraction,
        "training_config_sha256": run_manifest["config_sha256"],
        "probe_name": manifest.name,
        "probe_manifest_path": str(manifest.path),
        "probe_manifest_sha256": manifest.sha256,
        "sample_count": len(manifest.samples),
        "semantic_instance_rule": "one image-text pair per manifest sample",
        "device": device,
        "runtime": runtime_metadata(),
        "training_snapshot": {
            "kind": snapshot.checkpoint_kind,
            "path": str(snapshot.checkpoint_path),
            "checkpoint_sha256": snapshot.artifact_sha256,
            "metadata_path": str(snapshot.metadata_path),
        },
    }
    return save_embedding_artifact(
        output_directory=output_directory,
        stem=stem,
        sample_ids=[sample.sample_id for sample in manifest.samples],
        image_embeddings_raw=image_embeddings,
        text_embeddings_raw=text_embeddings,
        metadata=metadata,
    )


def _point_output(
    run_id: str,
    snapshot: SnapshotSpec,
    manifest: ProbeManifest,
    artifact_path: Path,
    metadata_path: Path,
    score_block_size: int,
) -> PointOutput:
    output_directory = (
        PROJECT_ROOT / "outputs" / "metrics" / "trajectory" / run_id / manifest.name / "points"
    )
    stem = f"trajectory_{run_id}_{snapshot.label}_{manifest.name}"
    metrics_path = output_directory / f"{stem}_metrics.json"
    geometry_state_path = output_directory / f"{stem}_geometry_state.npz"
    pair_index_path = (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "geometry_references"
        / f"{manifest.name}_{manifest.sha256[:12]}_upper_triangle_pairs_v1.npz"
    )
    if metrics_path.exists() != geometry_state_path.exists():
        raise FileExistsError("Trajectory metric output is incomplete; refusing to overwrite it.")
    if metrics_path.is_file() and geometry_state_path.is_file():
        metrics = _read_json(metrics_path)
        if metrics.get("artifact", {}).get("sha256") != sha256_file(artifact_path):
            raise ValueError("Existing point metrics belong to a different embedding artifact.")
        return PointOutput(
            label=snapshot.label,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            metrics_path=metrics_path,
            geometry_state_path=geometry_state_path,
        )
    return compute_point_output(
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        geometry_state_path=geometry_state_path,
        pair_index_path=pair_index_path,
        score_block_size=score_block_size,
        snapshot={
            "label": snapshot.label,
            "optimizer_step": snapshot.optimizer_step,
            "progress_fraction": snapshot.progress_fraction,
            "checkpoint_sha256": snapshot.artifact_sha256,
        },
    )


def _m0_point(model_name: str, manifest: ProbeManifest, score_block_size: int) -> PointOutput:
    artifact_model_name = MODEL_ARTIFACT_NAMES[model_name]
    stem = f"m0_{artifact_model_name}_{manifest.name}"
    embedding_directory = PROJECT_ROOT / "outputs" / "embeddings" / "m0" / artifact_model_name / manifest.name
    metrics_directory = PROJECT_ROOT / "outputs" / "metrics" / "m0" / artifact_model_name / manifest.name
    artifact_path = embedding_directory / f"{stem}.npz"
    metadata_path = embedding_directory / f"{stem}.json"
    metrics_path = metrics_directory / f"{stem}_six_metrics.json"
    geometry_state_path = metrics_directory / f"{stem}_geometry_reference.npz"
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"M0 raw embedding artifact is required before trajectory comparison: {artifact_path}"
        )
    metadata = _read_json(metadata_path)
    if metadata["probe_manifest_sha256"] != manifest.sha256:
        raise ValueError("M0 embedding and trajectory run use different probe manifests.")
    if metadata.get("artifact", {}).get("sha256") != sha256_file(artifact_path):
        raise ValueError("M0 embedding artifact failed SHA-256 validation.")
    if not metrics_path.is_file() or not geometry_state_path.is_file():
        sample_ids, image_embeddings, text_embeddings = load_embedding_artifact(artifact_path)
        pair_index_path = (
            PROJECT_ROOT
            / "data"
            / "metadata"
            / "geometry_references"
            / f"{manifest.name}_{manifest.sha256[:12]}_upper_triangle_pairs_v1.npz"
        )
        metrics = compute_six_metrics(
            image_embeddings,
            text_embeddings,
            manifest.sha256,
            pair_index_path,
            geometry_state_path,
            score_block_size=score_block_size,
        )
        metrics["artifact"] = {
            "path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "sha256": metadata["artifact"]["sha256"],
            "sample_count": int(len(sample_ids)),
            "embedding_dim": int(image_embeddings.shape[1]),
        }
        save_metrics(metrics_path, metrics)
    return PointOutput(
        label="m0",
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        geometry_state_path=geometry_state_path,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.score_block_size <= 0:
        raise ValueError("--batch-size and --score-block-size must be positive.")
    run_directory = PROJECT_ROOT / "outputs" / "training" / args.run
    run_manifest = load_completed_run_manifest(run_directory)
    snapshots = resolve_trajectory_snapshots(run_directory, run_manifest)
    model_name = str(run_manifest["config"]["model_name"])
    if model_name not in MODEL_ARTIFACT_NAMES:
        raise ValueError(f"Unsupported trajectory model: {model_name}")
    manifests = _probe_manifests(run_manifest)

    adapter = create_m0_adapter(model_name, PROJECT_ROOT, args.device)
    adapter_metadata = adapter.metadata()
    if adapter_metadata["checkpoint_sha256"] != run_manifest["config"]["checkpoint_sha256"]:
        raise ValueError("Evaluation adapter M0 checkpoint differs from the training initialization.")

    outputs_by_probe: dict[str, list[PointOutput]] = {
        manifest.name: [_m0_point(model_name, manifest, args.score_block_size)]
        for manifest in manifests
    }
    snapshot_results: dict[str, list[dict[str, Any]]] = {snapshot.label: [] for snapshot in snapshots}
    for snapshot in snapshots:
        load_snapshot_into_adapter(adapter, snapshot, run_manifest)
        for manifest in manifests:
            artifact_path, metadata_path = _export_snapshot_probe(
                adapter,
                adapter_metadata,
                snapshot,
                manifest,
                run_manifest,
                args.batch_size,
                args.device,
            )
            point = _point_output(
                args.run,
                snapshot,
                manifest,
                artifact_path,
                metadata_path,
                args.score_block_size,
            )
            outputs_by_probe[manifest.name].append(point)
            snapshot_results[snapshot.label].append(
                {
                    "probe_name": manifest.name,
                    "embedding_artifact": str(artifact_path),
                    "embedding_sha256": sha256_file(artifact_path),
                    "metrics": str(point.metrics_path),
                    "geometry_state": str(point.geometry_state_path),
                }
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    transition_records: list[dict[str, Any]] = []
    for manifest in manifests:
        sequence = outputs_by_probe[manifest.name]
        transition_directory = (
            PROJECT_ROOT
            / "outputs"
            / "metrics"
            / "trajectory"
            / args.run
            / manifest.name
            / "transitions"
        )
        for source, target in zip(sequence, sequence[1:]):
            output_path = transition_directory / f"{source.label}_to_{target.label}.json"
            transition = compute_adjacent_transition(source, target, output_path)
            transition_records.append(
                {
                    "probe_name": manifest.name,
                    "source": source.label,
                    "target": target.label,
                    "path": str(output_path),
                    "image_spearman": transition["intra_modal_geometry_preservation"]["image"]["spearman"],
                    "text_spearman": transition["intra_modal_geometry_preservation"]["text"]["spearman"],
                }
            )

    for snapshot in snapshots:
        mark_snapshot_evaluated(
            snapshot,
            snapshot_results[snapshot.label],
            run_directory=run_directory,
        )
    index_path = (
        PROJECT_ROOT / "outputs" / "metrics" / "trajectory" / args.run / "trajectory_index.json"
    )
    write_json_atomic(
        index_path,
        {
            "schema_version": 1,
            "run_id": args.run,
            "model_name": model_name,
            "branch": run_manifest["config"]["branch"],
            "config_sha256": run_manifest["config_sha256"],
            "sequence": ["m0", *[snapshot.label for snapshot in snapshots]],
            "raw_embeddings_stored": True,
            "normalized_embeddings_stored": False,
            "transitions": transition_records,
        },
    )
    run_manifest["trajectory_evaluation"] = {
        "status": "complete",
        "index_path": str(index_path),
        "sequence": ["m0", *[snapshot.label for snapshot in snapshots]],
    }
    write_json_atomic(run_directory / "run_manifest.json", run_manifest)
    print(f"trajectory_index={index_path}")


if __name__ == "__main__":
    main()
