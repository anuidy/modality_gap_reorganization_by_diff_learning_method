from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training.config import RunConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_formal_data_identity(config: RunConfig, train_sha256: str) -> dict[str, Any]:
    """Fail before training if any controlled manifest was changed or exchanged."""

    formal_values = (
        config.dataset_name,
        config.dataset_spec,
        config.split_lock,
        config.split_lock_sha256,
        config.train_manifest_sha256,
        config.validation_manifest,
        config.validation_manifest_sha256,
        config.validation_sample_count,
        config.lcs_probe_manifest,
        config.lcs_probe_manifest_sha256,
        config.coco_probe_manifest,
        config.coco_probe_manifest_sha256,
    )
    if all(value is None for value in formal_values):
        return {"mode": "unlocked_test_fixture"}
    if any(value is None for value in formal_values):
        raise ValueError("Formal runs require the complete locked dataset identity.")

    assert config.dataset_spec is not None
    assert config.split_lock is not None
    assert config.validation_manifest is not None
    assert config.validation_sample_count is not None
    assert config.lcs_probe_manifest is not None
    assert config.coco_probe_manifest is not None
    if config.dataset_name != "lcs_558k":
        raise ValueError("Formal training must use the locked LCS-558K dataset.")
    if not config.dataset_spec.is_file():
        raise FileNotFoundError(config.dataset_spec)

    identities = {
        "split_lock": (config.split_lock, config.split_lock_sha256),
        "validation": (config.validation_manifest, config.validation_manifest_sha256),
        "lcs_probe": (config.lcs_probe_manifest, config.lcs_probe_manifest_sha256),
        "coco_probe": (config.coco_probe_manifest, config.coco_probe_manifest_sha256),
    }
    actual_hashes: dict[str, str] = {}
    for name, (path, expected_sha256) in identities.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"Formal {name} SHA-256 mismatch: expected {expected_sha256}, found {actual}."
            )
        actual_hashes[name] = actual
    if train_sha256 != config.train_manifest_sha256:
        raise ValueError(
            "Formal LCS Train manifest SHA-256 mismatch: "
            f"expected {config.train_manifest_sha256}, found {train_sha256}."
        )

    with config.split_lock.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("dataset") != "lcs_558k" or lock.get("invariants", {}).get(
        "branch_resampling_forbidden"
    ) is not True:
        raise ValueError("The LCS split lock does not enforce the formal experiment invariants.")
    expected_splits = {
        "train": (train_sha256, "parameter_updates_only"),
        "validation": (
            actual_hashes["validation"],
            "training_monitoring_only_no_checkpoint_selection",
        ),
        "in_domain_probe": (actual_hashes["lcs_probe"], "representation_metrics_only"),
    }
    for name, (expected_sha256, expected_role) in expected_splits.items():
        entry = lock.get("splits", {}).get(name, {})
        if entry.get("sha256") != expected_sha256 or entry.get("role") != expected_role:
            raise ValueError(f"LCS split lock has an invalid {name} identity or role.")
    if (
        lock.get("splits", {}).get("validation", {}).get("sample_count")
        != config.validation_sample_count
    ):
        raise ValueError("LCS split lock Validation sample count mismatch.")

    controlled_paths = {
        config.train_manifest.resolve(),
        config.validation_manifest.resolve(),
        config.lcs_probe_manifest.resolve(),
        config.coco_probe_manifest.resolve(),
    }
    if len(controlled_paths) != 4:
        raise ValueError("Train, Validation, LCS Probe, and COCO Probe must be distinct manifests.")
    return {
        "mode": "formal_locked_lcs_558k",
        "dataset_name": config.dataset_name,
        "split_lock_sha256": actual_hashes["split_lock"],
        "train_manifest_sha256": train_sha256,
        "validation_manifest_sha256": actual_hashes["validation"],
        "lcs_probe_manifest_sha256": actual_hashes["lcs_probe"],
        "coco_probe_manifest_sha256": actual_hashes["coco_probe"],
        "validation_used_for_checkpoint_selection": False,
        "probes_used_for_training_or_checkpoint_selection": False,
    }
