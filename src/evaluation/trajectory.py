from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embeddings.artifact import load_embedding_artifact, sha256_file
from metrics.six_metrics import (
    compare_geometry_states,
    compute_point_metrics,
    floating_metric_deltas,
    l2_normalize,
    save_geometry_state,
    save_metrics,
)
from models.base import EmbeddingAdapter


MODEL_ARTIFACT_NAMES = {
    "clip": "openai_clip_vit_l14",
    "vista": "vista_base_stage1",
    "beit3": "beit3_base_itc_patch16_224",
    "albef": "albef_14m_pretrained",
}


@dataclass(frozen=True)
class SnapshotSpec:
    label: str
    optimizer_step: int
    progress_fraction: float
    checkpoint_kind: str
    checkpoint_path: Path
    metadata_path: Path
    artifact_sha256: str


@dataclass(frozen=True)
class PointOutput:
    label: str
    artifact_path: Path
    metadata_path: Path
    metrics_path: Path
    geometry_state_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_completed_run_manifest(run_directory: Path) -> dict[str, Any]:
    manifest = _read_json(run_directory / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Trajectory evaluation requires a completed training branch.")
    if not isinstance(manifest.get("checkpoint_policy"), dict):
        raise ValueError("Run manifest does not contain a checkpoint policy.")
    return manifest


def _inside_directory(path: Path, directory: Path) -> Path:
    resolved = path.resolve()
    root = directory.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Checkpoint path escapes its run directory: {resolved}")
    return resolved


def _validate_snapshot_provenance(
    spec_payload: dict[str, Any],
    run_manifest: dict[str, Any],
    optimizer_step: int,
    progress_fraction: float,
) -> None:
    provenance = spec_payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Checkpoint metadata is missing provenance.")
    config = run_manifest["config"]
    expected = {
        "run_id": config["run_id"],
        "model_name": config["model_name"],
        "branch": config["branch"],
        "optimizer_step": optimizer_step,
        "m0_checkpoint_sha256": config["checkpoint_sha256"],
        "config_sha256": run_manifest["config_sha256"],
    }
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError(f"Checkpoint provenance mismatch for {name}.")
    if float(provenance.get("progress_fraction", -1.0)) != progress_fraction:
        raise ValueError("Checkpoint provenance mismatch for progress_fraction.")
    expected_probes = {
        "lcs": config.get("lcs_probe_manifest_sha256"),
        "coco": config.get("coco_probe_manifest_sha256"),
    }
    if provenance.get("probe_manifests") != expected_probes:
        raise ValueError("Checkpoint provenance uses different probe manifests.")


def resolve_trajectory_snapshots(
    run_directory: Path,
    run_manifest: dict[str, Any],
) -> tuple[SnapshotSpec, ...]:
    points = run_manifest["checkpoint_policy"].get("trajectory_points")
    if not isinstance(points, list) or not points:
        raise ValueError("Run manifest contains no trajectory points.")
    snapshots: list[SnapshotSpec] = []
    for point in points:
        label = str(point["label"])
        optimizer_step = int(point["optimizer_step"])
        progress_fraction = float(point["progress_fraction"])
        if progress_fraction == 1.0:
            metadata_path = run_directory / "checkpoints" / "final.json"
            metadata = _read_json(metadata_path)
            checkpoint_kind = "full_resume"
            checkpoint_path = run_directory / str(metadata["path"])
        else:
            metadata_path = (
                run_directory
                / "checkpoints"
                / "trajectory"
                / f"step_{optimizer_step:08d}_{label}_model.json"
            )
            metadata = _read_json(metadata_path)
            checkpoint_kind = "trajectory_model"
            checkpoint_path = metadata_path.parent / str(metadata["path"])
        checkpoint_path = _inside_directory(checkpoint_path, run_directory)
        if metadata.get("checkpoint_kind") not in {
            checkpoint_kind,
            "final_full_resume" if progress_fraction == 1.0 else checkpoint_kind,
        }:
            raise ValueError(f"Unexpected checkpoint metadata kind for {label}.")
        _validate_snapshot_provenance(
            metadata,
            run_manifest,
            optimizer_step,
            progress_fraction,
        )
        snapshots.append(
            SnapshotSpec(
                label=label,
                optimizer_step=optimizer_step,
                progress_fraction=progress_fraction,
                checkpoint_kind=checkpoint_kind,
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                artifact_sha256=str(metadata["artifact_sha256"]),
            )
        )
    return tuple(snapshots)


def load_snapshot_into_adapter(
    adapter: EmbeddingAdapter,
    snapshot: SnapshotSpec,
    run_manifest: dict[str, Any],
) -> None:
    actual_sha256 = sha256_file(snapshot.checkpoint_path)
    if actual_sha256 != snapshot.artifact_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch for {snapshot.label}.")
    payload = torch.load(
        snapshot.checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if payload.get("schema_version") != 2:
        raise ValueError("Unsupported trajectory checkpoint schema.")
    if payload.get("checkpoint_kind") != snapshot.checkpoint_kind:
        raise ValueError("Checkpoint payload kind does not match its trajectory metadata.")
    if payload.get("run_id") != run_manifest["config"]["run_id"]:
        raise ValueError("Checkpoint payload belongs to another run.")
    if payload.get("config_sha256") != run_manifest["config_sha256"]:
        raise ValueError("Checkpoint payload uses a different training configuration.")
    backend_state = payload.get("model")
    if not isinstance(backend_state, dict) or not backend_state:
        raise ValueError("Checkpoint does not contain a model state dictionary.")
    if any(not str(name).startswith("model.") for name in backend_state):
        raise ValueError("Training checkpoint contains state outside the backend model namespace.")
    model_state = {str(name).removeprefix("model."): value for name, value in backend_state.items()}
    model = getattr(adapter, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Embedding adapter does not expose its underlying torch model.")
    model.load_state_dict(model_state, strict=True)
    model.eval()


def compute_point_output(
    artifact_path: Path,
    metadata_path: Path,
    metrics_path: Path,
    geometry_state_path: Path,
    pair_index_path: Path,
    score_block_size: int,
    snapshot: dict[str, Any],
) -> PointOutput:
    sample_ids, image_embeddings, text_embeddings = load_embedding_artifact(artifact_path)
    metadata = _read_json(metadata_path)
    if metadata.get("artifact", {}).get("sha256") != sha256_file(artifact_path):
        raise ValueError("Embedding artifact SHA-256 does not match its metadata.")
    metrics = compute_point_metrics(
        image_embeddings,
        text_embeddings,
        score_block_size=score_block_size,
    )
    metrics["intra_modal_geometry_state"] = save_geometry_state(
        l2_normalize(image_embeddings),
        l2_normalize(text_embeddings),
        str(metadata["probe_manifest_sha256"]),
        pair_index_path,
        geometry_state_path,
    )
    metrics["artifact"] = {
        "path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "sha256": metadata["artifact"]["sha256"],
        "sample_count": int(len(sample_ids)),
        "embedding_dim": int(image_embeddings.shape[1]),
    }
    metrics["snapshot"] = snapshot
    save_metrics(metrics_path, metrics)
    return PointOutput(
        label=str(snapshot["label"]),
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        geometry_state_path=geometry_state_path,
    )


def compute_adjacent_transition(
    source: PointOutput,
    target: PointOutput,
    output_path: Path,
) -> dict[str, Any]:
    source_ids, _, _ = load_embedding_artifact(source.artifact_path)
    target_ids, _, _ = load_embedding_artifact(target.artifact_path)
    if not np.array_equal(source_ids, target_ids):
        raise ValueError("Adjacent trajectory points use different sample-ID order.")
    source_metadata = _read_json(source.metadata_path)
    target_metadata = _read_json(target.metadata_path)
    if source_metadata["probe_manifest_sha256"] != target_metadata["probe_manifest_sha256"]:
        raise ValueError("Adjacent trajectory points use different probe manifests.")
    source_metrics = _read_json(source.metrics_path)
    target_metrics = _read_json(target.metrics_path)
    transition = {
        "schema_version": 1,
        "source": {
            "label": source.label,
            "artifact_path": str(source.artifact_path),
            "artifact_sha256": source_metadata["artifact"]["sha256"],
        },
        "target": {
            "label": target.label,
            "artifact_path": str(target.artifact_path),
            "artifact_sha256": target_metadata["artifact"]["sha256"],
        },
        "probe_name": source_metadata["probe_name"],
        "probe_manifest_sha256": source_metadata["probe_manifest_sha256"],
        "point_metric_delta_target_minus_source": floating_metric_deltas(
            source_metrics,
            target_metrics,
        ),
        "intra_modal_geometry_preservation": compare_geometry_states(
            source.geometry_state_path,
            target.geometry_state_path,
        ),
    }
    write_json_atomic(output_path, transition)
    return transition


def mark_snapshot_evaluated(
    snapshot: SnapshotSpec,
    results: list[dict[str, Any]],
    run_directory: Path | None = None,
) -> None:
    metadata = _read_json(snapshot.metadata_path)
    provenance = metadata["provenance"]
    provenance["evaluation"] = {
        "status": "complete",
        "results": results,
    }
    write_json_atomic(snapshot.metadata_path, metadata)
    if run_directory is not None:
        index_path = run_directory / "checkpoint_index.jsonl"
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "trajectory_evaluation_completed",
                        "label": snapshot.label,
                        "optimizer_step": snapshot.optimizer_step,
                        "checkpoint_sha256": snapshot.artifact_sha256,
                        "results": results,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
