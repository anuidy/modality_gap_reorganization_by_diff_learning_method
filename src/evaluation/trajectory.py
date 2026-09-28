from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embeddings.artifact import load_embedding_artifact, sha256_file
from metrics.representation_metrics import (
    compare_geometry_states,
    compute_point_metrics,
    floating_metric_deltas,
    intra_modal_geometry_preservation,
    l2_normalize,
    save_geometry_state,
    save_metrics,
)
from model_adapters.base import EmbeddingAdapter


MODEL_ARTIFACT_NAMES = {
    "clip": "openai_clip_vit_l14",
    "vista": "vista_base_stage1",
    "beit3": "beit3_base_itc_patch16_224",
    "albef": "albef_14m_pretrained",
}

# Where the exported embedding sits relative to the model's own L2 normalization.
# Measured 2026-09-11 on the stored M0 artifacts: every model's exported rows have
# norms far from 1, so all four expose a genuine pre-L2 encoder output. VISTA uses
# the native encoder instantiated with normlized=False.
REPRESENTATION_BOUNDARY = {
    "clip": "encoder_output_pre_l2",
    "beit3": "retrieval_head_output_pre_l2",
    "vista": "native_encoder_output_normlized_false",
    "albef": "encoder_output_pre_l2",
}

# Keys every ``m0 -> checkpoint`` delta must carry. ``floating_metric_deltas``
# walks the *source* keys and keeps those the target also has, so a delta is
# bounded by the narrower side. The frozen M0 baseline under outputs/metrics/m0/
# carries the full metric schema; a delta missing these keys therefore did not
# come from the baseline.
M0_REQUIRED_DELTA_KEYS = (
    "norm_imbalance",
    "anisotropy_image",
    "anisotropy_text",
    "anisotropy_gap",
    "cross_modal_alignment",
)

# Only these models define an additive e_IT = e_I + e_T in the experiment; VISTA
# uses its native joint encoder through encode_multimodal, and ALBEF
# deliberately has no artificial IT.
ADDITIVE_IT_MODELS = frozenset({"clip", "beit3"})

PROBE_DATASET = {
    "coco_2017_val_5k": "coco_2017_val",
    "lcs_558k_in_domain_10k": "lcs_558k",
}
PROBE_SPLIT = {
    "coco_2017_val_5k": "val",
    "lcs_558k_in_domain_10k": "in_domain",
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


def load_evaluation_run_manifest(run_directory: Path, *, allow_running: bool = False) -> dict[str, Any]:
    """Read evaluation provenance; live snapshots require explicit opt-in."""
    manifest = _read_json(run_directory / "run_manifest.json")
    if allow_running and manifest.get("status") == "running":
        if not isinstance(manifest.get("checkpoint_policy"), dict):
            raise ValueError("Run manifest does not contain a checkpoint policy.")
        return manifest
    if manifest.get("status") not in {"complete", "paused"}:
        raise ValueError("Evaluate only complete or paused training tasks.")
    if not isinstance(manifest.get("checkpoint_policy"), dict):
        raise ValueError("Run manifest does not contain a checkpoint policy.")
    if type(manifest.get("completed_steps")) is not int or manifest["completed_steps"] <= 0:
        raise ValueError("Stopped task has no valid completed-step count.")
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
    *,
    labels: tuple[str, ...] | None = None,
    allow_running: bool = False,
) -> tuple[SnapshotSpec, ...]:
    points = run_manifest["checkpoint_policy"].get("trajectory_points")
    if not isinstance(points, list) or not points:
        raise ValueError("Run manifest contains no trajectory points.")
    if labels is not None:
        known = {str(point["label"]) for point in points}
        if not labels or len(set(labels)) != len(labels) or not set(labels) <= known:
            raise ValueError("Requested trajectory labels must be unique, nonempty, and declared.")
        points = [point for point in points if str(point["label"]) in labels]
        if not (allow_running and run_manifest.get("status") == "running") and any(int(point["optimizer_step"]) > run_manifest.get("completed_steps", -1) for point in points):
            raise ValueError("Requested checkpoint is beyond the task's completed steps.")
    snapshots: list[SnapshotSpec] = []
    for point in points:
        label = str(point["label"])
        optimizer_step = int(point["optimizer_step"])
        progress_fraction = float(point["progress_fraction"])
        final_step = run_manifest["config"].get("max_steps")
        # Historical minimal manifests used total-budget fractions and omitted max_steps.
        is_final = optimizer_step == int(final_step) if final_step is not None else progress_fraction == 1.0
        if is_final:
            metadata_path = run_directory / "checkpoints" / "final.json"
            metadata = _read_json(metadata_path)
            checkpoint_kind = (
                "trajectory_model" if metadata.get("checkpoint_kind") in {"trajectory_model", "final_trajectory_model"}
                else "full_resume"
            )
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
        if allow_running and not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if metadata.get("checkpoint_kind") not in {
            checkpoint_kind,
            ("final_full_resume" if checkpoint_kind == "full_resume" else "final_trajectory_model")
            if is_final else checkpoint_kind,
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
    *,
    model_name: str | None = None,
    m0_geometry_state_path: Path | None = None,
) -> PointOutput:
    sample_ids, image_embeddings, text_embeddings = load_embedding_artifact(artifact_path)
    metadata = _read_json(metadata_path)
    if metadata.get("artifact", {}).get("sha256") != sha256_file(artifact_path):
        raise ValueError("Embedding artifact SHA-256 does not match its metadata.")
    resolved_model = model_name or str(metadata.get("model_name", ""))
    identity = {
        "model": resolved_model or None,
        "training_regime": snapshot.get("branch") or snapshot.get("label"),
        "checkpoint": str(artifact_path),
        "global_step": snapshot.get("optimizer_step"),
        "training_progress": snapshot.get("progress_fraction"),
        "dataset": PROBE_DATASET.get(str(metadata.get("probe_name")), str(metadata.get("probe_name"))),
        "split": PROBE_SPLIT.get(str(metadata.get("probe_name")), "unknown"),
        "sample_count": int(len(sample_ids)),
    }
    metrics = compute_point_metrics(
        image_embeddings,
        text_embeddings,
        score_block_size=score_block_size,
        representation_boundary=REPRESENTATION_BOUNDARY.get(resolved_model, "unknown"),
        raw_available=True,
        identity=identity,
    )
    metrics["intra_modal_geometry_state"] = save_geometry_state(
        l2_normalize(image_embeddings),
        l2_normalize(text_embeddings),
        str(metadata["probe_manifest_sha256"]),
        pair_index_path,
        geometry_state_path,
    )
    # §6: intra-modal geometry preservation is M0 vs this checkpoint, computed on
    # the shared fixed pair indices. Adjacent-point drift stays available as the
    # separate transition records.
    if m0_geometry_state_path is not None:
        preservation = intra_modal_geometry_preservation(m0_geometry_state_path, geometry_state_path)
        metrics["intra_geometry_image"] = preservation["image"]
        metrics["intra_geometry_text"] = preservation["text"]
        metrics.setdefault("auxiliary_metrics", {}).update(preservation["auxiliary"])
        metrics["auxiliary_metrics"]["intra_geometry_reference"] = "m0"
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
    delta = floating_metric_deltas(source_metrics, target_metrics)
    if source.label == "m0":
        missing = [key for key in M0_REQUIRED_DELTA_KEYS if key not in delta]
        if missing:
            raise ValueError(
                "An m0 -> checkpoint delta must be computed from the frozen M0 baseline under "
                f"outputs/metrics/m0/, but the source metrics {source.metrics_path} cannot supply "
                f"{missing}."
            )
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
        "point_metric_delta_target_minus_source": delta,
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
