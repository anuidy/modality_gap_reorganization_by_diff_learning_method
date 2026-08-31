from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.formal_data import (  # noqa: E402
    prepare_lcs_splits,
    sha256_file,
    validate_coco_probe,
    validate_lcs_splits,
    verify_lcs_images,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "formal_datasets.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and verify the immutable LCS/COCO formal experiment manifests."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Refuse to create missing LCS manifests and only validate an existing locked split.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Also check all 558,128 LCS files and all 5,000 COCO files on disk.",
    )
    return parser.parse_args()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _payload(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported formal dataset configuration.")
    return payload


def main() -> None:
    args = _arguments()
    config = _payload(args.config)
    lcs = config["lcs_558k"]
    split = lcs["split"]
    lcs_probe = lcs["in_domain_probe"]
    split_lock_path = _path(split["split_lock"])
    split_lock_sha256 = sha256_file(split_lock_path)
    if split_lock_sha256 != split["split_lock_sha256"]:
        raise ValueError(
            "Versioned LCS split-lock SHA-256 mismatch: "
            f"expected {split['split_lock_sha256']}, found {split_lock_sha256}."
        )
    shared = {
        "project_root": PROJECT_ROOT,
        "annotation_path": _path(lcs["source_annotation"]),
        "probe_manifest_path": _path(lcs_probe["manifest"]),
        "train_manifest_path": _path(split["train_manifest"]),
        "validation_manifest_path": _path(split["validation_manifest"]),
        "lock_path": split_lock_path,
        "expected_source_sha256": lcs["source_annotation_sha256"],
        "expected_source_count": int(lcs["source_sample_count"]),
        "expected_probe_count": int(lcs_probe["sample_count"]),
    }
    if args.validate_only:
        lock = validate_lcs_splits(
            **shared,
            expected_probe_seed=int(lcs_probe["seed"]),
            expected_validation_count=int(split["validation_count"]),
        )
    else:
        lock = prepare_lcs_splits(
            **shared,
            probe_seed=int(lcs_probe["seed"]),
            validation_count=int(split["validation_count"]),
            validation_seed=int(split["validation_seed"]),
        )

    coco = config["coco_2017_val"]
    coco_probe = coco["external_probe"]
    coco_summary = validate_coco_probe(
        project_root=PROJECT_ROOT,
        captions_path=_path(coco["captions_annotation"]),
        manifest_path=_path(coco_probe["manifest"]),
        expected_captions_sha256=coco["captions_annotation_sha256"],
        expected_manifest_sha256=coco_probe["manifest_sha256"],
        expected_count=int(coco_probe["sample_count"]),
        verify_images=args.verify_images,
    )

    image_summary = None
    if args.verify_images:
        image_summary = verify_lcs_images(
            train_manifest_path=_path(split["train_manifest"]),
            validation_manifest_path=_path(split["validation_manifest"]),
            training_image_root=_path(lcs["training_image_root"]),
            frozen_probe_manifest_path=_path(lcs_probe["manifest"]),
            probe_image_root=_path(lcs["probe_image_root"]),
        )

    result = {
        "status": "valid",
        "lcs": {
            "split_lock": split["split_lock"],
            "split_lock_sha256": sha256_file(_path(split["split_lock"])),
            "splits": lock["splits"],
            "images": image_summary,
        },
        "coco": coco_summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
