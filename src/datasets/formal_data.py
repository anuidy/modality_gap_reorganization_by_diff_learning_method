from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LCS_DATASET_NAME = "lcs_558k"
LCS_SOURCE_COUNT = 558_128
LCS_PROBE_COUNT = 10_000
LCS_VALIDATION_COUNT = 8_000
LCS_TRAIN_COUNT = 540_128
LCS_SPLIT_SEED = 20_260_825
COCO_PROBE_COUNT = 5_000


@dataclass(frozen=True)
class ManifestSummary:
    path: Path
    sha256: str
    sample_count: int
    source_indices: frozenset[int]
    sample_ids: frozenset[str]
    semantic_ids: frozenset[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assistant_caption(record: Mapping[str, Any]) -> str:
    for message in record.get("conversations", []):
        if isinstance(message, Mapping) and message.get("from") == "gpt":
            value = message.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(f"LCS record {record.get('id')} has no non-empty assistant caption.")


def _source_record(record: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    source_id = str(record.get("id", "")).strip()
    image = str(record.get("image", "")).strip().replace("\\", "/").lstrip("./")
    if not source_id or not image:
        raise ValueError(f"LCS source record {source_index} has an empty id or image path.")
    if Path(image).is_absolute() or ".." in Path(image).parts:
        raise ValueError(f"LCS source record {source_index} has an unsafe image path: {image}")
    instance_id = f"lcs_558k:{source_id}"
    return {
        "sample_id": instance_id,
        "semantic_id": instance_id,
        "source_index": source_index,
        "image": image,
        "text": _assistant_caption(record),
    }


def load_lcs_source(annotation_path: Path) -> tuple[list[dict[str, Any]], str]:
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    source_sha256 = sha256_file(annotation_path)
    payload = _read_json(annotation_path)
    if not isinstance(payload, list) or not all(isinstance(record, dict) for record in payload):
        raise ValueError("LCS annotation must be a JSON list of objects.")

    records = [_source_record(record, index) for index, record in enumerate(payload)]
    sample_ids = [record["sample_id"] for record in records]
    images = [record["image"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("LCS source annotation contains duplicate ids.")
    if len(images) != len(set(images)):
        raise ValueError("LCS source annotation contains duplicate image paths.")
    return records, source_sha256


def _relative_path(path: Path, project_root: Path) -> str:
    resolved_root = project_root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Formal dataset path is outside project_root: {resolved_path}")
    return resolved_path.relative_to(resolved_root).as_posix()


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_frozen_lcs_probe(
    probe_manifest_path: Path,
    source_records: Sequence[Mapping[str, Any]],
    source_sha256: str,
    expected_count: int,
    expected_seed: int,
) -> tuple[dict[str, Any], frozenset[int]]:
    payload = _read_json(probe_manifest_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("Frozen LCS probe manifest must contain a samples list.")
    if payload.get("source_annotation_sha256") != source_sha256:
        raise ValueError("Frozen LCS probe was created from a different source annotation.")
    sampling = payload.get("sampling", {})
    if sampling.get("seed") != expected_seed or sampling.get("count") != expected_count:
        raise ValueError("Frozen LCS probe seed/count do not match the formal protocol.")
    if len(payload["samples"]) != expected_count:
        raise ValueError(f"Expected {expected_count} frozen LCS probe records.")

    source_indices: set[int] = set()
    probe_ids: set[str] = set()
    for position, record in enumerate(payload["samples"]):
        if not isinstance(record, dict) or not isinstance(record.get("source_index"), int):
            raise ValueError(f"Invalid frozen LCS probe record at position {position}.")
        source_index = record["source_index"]
        if source_index < 0 or source_index >= len(source_records):
            raise ValueError(f"Frozen LCS probe source_index is out of range: {source_index}")
        source = source_records[source_index]
        expected_id = source["sample_id"].split(":", maxsplit=1)[1]
        if (
            str(record.get("id")) != expected_id
            or str(record.get("image", "")).replace("\\", "/").lstrip("./")
            != source["image"]
            or str(record.get("caption", "")).strip() != source["text"]
        ):
            raise ValueError(f"Frozen LCS probe record does not match source_index {source_index}.")
        if source_index in source_indices or expected_id in probe_ids:
            raise ValueError("Frozen LCS probe contains duplicate semantic instances.")
        source_indices.add(source_index)
        probe_ids.add(expected_id)
    return payload, frozenset(source_indices)


def scan_training_manifest(
    path: Path,
    source_records: Sequence[Mapping[str, Any]] | None = None,
) -> ManifestSummary:
    if not path.is_file():
        raise FileNotFoundError(path)
    source_indices: set[int] = set()
    sample_ids: set[str] = set()
    semantic_ids: set[str] = set()
    sample_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Manifest line {line_number} is not a JSON object.")
            sample_id = str(record.get("sample_id", ""))
            semantic_id = str(record.get("semantic_id", ""))
            source_index = record.get("source_index")
            if not sample_id or not semantic_id or not isinstance(source_index, int):
                raise ValueError(f"Manifest line {line_number} lacks its controlled identity fields.")
            if (
                sample_id in sample_ids
                or semantic_id in semantic_ids
                or source_index in source_indices
            ):
                raise ValueError(f"Duplicate identity in manifest at line {line_number}.")
            if source_records is not None:
                if source_index < 0 or source_index >= len(source_records):
                    raise ValueError(f"Manifest source_index out of range at line {line_number}.")
                expected = source_records[source_index]
                for key in ("sample_id", "semantic_id", "source_index", "image", "text"):
                    if record.get(key) != expected[key]:
                        raise ValueError(
                            f"Manifest line {line_number} field {key} differs from the LCS source."
                        )
            sample_ids.add(sample_id)
            semantic_ids.add(semantic_id)
            source_indices.add(source_index)
            sample_count += 1
    return ManifestSummary(
        path=path,
        sha256=sha256_file(path),
        sample_count=sample_count,
        source_indices=frozenset(source_indices),
        sample_ids=frozenset(sample_ids),
        semantic_ids=frozenset(semantic_ids),
    )


def _verify_lock(
    lock: Mapping[str, Any],
    source_sha256: str,
    probe_sha256: str,
    train: ManifestSummary,
    validation: ManifestSummary,
) -> None:
    if lock.get("schema_version") != 1 or lock.get("dataset") != LCS_DATASET_NAME:
        raise ValueError("Unsupported LCS split lock.")
    if lock.get("source", {}).get("sha256") != source_sha256:
        raise ValueError("LCS split lock source SHA-256 mismatch.")
    expected = {
        "train": train,
        "validation": validation,
    }
    for name, summary in expected.items():
        entry = lock.get("splits", {}).get(name, {})
        if entry.get("sha256") != summary.sha256 or entry.get("sample_count") != summary.sample_count:
            raise ValueError(f"LCS split lock does not match the {name} manifest.")
    probe_entry = lock.get("splits", {}).get("in_domain_probe", {})
    if probe_entry.get("sha256") != probe_sha256:
        raise ValueError("LCS split lock does not match the frozen 10K probe.")


def validate_lcs_splits(
    *,
    project_root: Path,
    annotation_path: Path,
    probe_manifest_path: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    lock_path: Path,
    expected_source_sha256: str,
    expected_source_count: int = LCS_SOURCE_COUNT,
    expected_probe_count: int = LCS_PROBE_COUNT,
    expected_probe_seed: int = LCS_SPLIT_SEED,
    expected_validation_count: int = LCS_VALIDATION_COUNT,
) -> dict[str, Any]:
    source_records, source_sha256 = load_lcs_source(annotation_path)
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            f"LCS source SHA-256 mismatch: expected {expected_source_sha256}, found {source_sha256}."
        )
    if len(source_records) != expected_source_count:
        raise ValueError(
            f"Expected {expected_source_count} LCS source records, found {len(source_records)}."
        )
    _, probe_indices = _load_frozen_lcs_probe(
        probe_manifest_path,
        source_records,
        source_sha256,
        expected_probe_count,
        expected_probe_seed,
    )
    train = scan_training_manifest(train_manifest_path, source_records)
    validation = scan_training_manifest(validation_manifest_path, source_records)
    expected_train_count = expected_source_count - expected_probe_count - expected_validation_count
    if train.sample_count != expected_train_count:
        raise ValueError(f"Expected {expected_train_count} LCS training pairs, found {train.sample_count}.")
    if validation.sample_count != expected_validation_count:
        raise ValueError(
            f"Expected {expected_validation_count} LCS validation pairs, found {validation.sample_count}."
        )
    if train.source_indices & validation.source_indices:
        raise ValueError("LCS Train and Validation overlap.")
    if train.source_indices & probe_indices or validation.source_indices & probe_indices:
        raise ValueError("LCS Train/Validation overlaps the frozen 10K Probe.")
    covered = train.source_indices | validation.source_indices | probe_indices
    if covered != frozenset(range(expected_source_count)):
        raise ValueError("LCS Train/Validation/Probe do not exactly partition the source annotation.")

    lock = _read_json(lock_path)
    if not isinstance(lock, dict):
        raise ValueError("LCS split lock must be a JSON object.")
    _verify_lock(lock, source_sha256, sha256_file(probe_manifest_path), train, validation)
    return dict(lock)


def prepare_lcs_splits(
    *,
    project_root: Path,
    annotation_path: Path,
    probe_manifest_path: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    lock_path: Path,
    expected_source_sha256: str,
    expected_source_count: int = LCS_SOURCE_COUNT,
    expected_probe_count: int = LCS_PROBE_COUNT,
    probe_seed: int = LCS_SPLIT_SEED,
    validation_count: int = LCS_VALIDATION_COUNT,
    validation_seed: int = LCS_SPLIT_SEED,
) -> dict[str, Any]:
    """Create immutable LCS Train/Validation manifests around the existing fixed Probe."""

    manifests_exist = train_manifest_path.exists() and validation_manifest_path.exists()
    if lock_path.exists() and manifests_exist:
        return validate_lcs_splits(
            project_root=project_root,
            annotation_path=annotation_path,
            probe_manifest_path=probe_manifest_path,
            train_manifest_path=train_manifest_path,
            validation_manifest_path=validation_manifest_path,
            lock_path=lock_path,
            expected_source_sha256=expected_source_sha256,
            expected_source_count=expected_source_count,
            expected_probe_count=expected_probe_count,
            expected_probe_seed=probe_seed,
            expected_validation_count=validation_count,
        )
    preexisting = [path for path in (train_manifest_path, validation_manifest_path) if path.exists()]
    if preexisting:
        raise FileExistsError(
            "Refusing to overwrite an incomplete formal manifest set: "
            + ", ".join(map(str, preexisting))
        )

    source_records, source_sha256 = load_lcs_source(annotation_path)
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            f"LCS source SHA-256 mismatch: expected {expected_source_sha256}, found {source_sha256}."
        )
    if len(source_records) != expected_source_count:
        raise ValueError(
            f"Expected {expected_source_count} LCS source records, found {len(source_records)}."
        )
    _, probe_indices = _load_frozen_lcs_probe(
        probe_manifest_path,
        source_records,
        source_sha256,
        expected_probe_count,
        probe_seed,
    )
    remaining = [index for index in range(expected_source_count) if index not in probe_indices]
    validation_ranked = sorted(
        remaining,
        key=lambda index: (
            hashlib.sha256(
                f"lcs_558k.validation.v1:{validation_seed}:{index}".encode("ascii")
            ).digest(),
            index,
        ),
    )
    validation_indices = frozenset(validation_ranked[:validation_count])
    train_indices = [index for index in remaining if index not in validation_indices]
    validation_indices_sorted = sorted(validation_indices)

    train_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    validation_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    train_temp = train_manifest_path.with_name(f".{train_manifest_path.name}.{uuid.uuid4().hex}.tmp")
    validation_temp = validation_manifest_path.with_name(
        f".{validation_manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        _write_jsonl(train_temp, (source_records[index] for index in train_indices))
        _write_jsonl(
            validation_temp, (source_records[index] for index in validation_indices_sorted)
        )
        train_sha256 = sha256_file(train_temp)
        validation_sha256 = sha256_file(validation_temp)
        if lock_path.exists():
            frozen_lock = _read_json(lock_path)
            if not isinstance(frozen_lock, dict):
                raise ValueError("The pre-existing LCS split lock must be a JSON object.")
            frozen_splits = frozen_lock.get("splits", {})
            expected_locked_values = {
                "source_sha256": (frozen_lock.get("source", {}).get("sha256"), source_sha256),
                "train_sha256": (frozen_splits.get("train", {}).get("sha256"), train_sha256),
                "train_count": (
                    frozen_splits.get("train", {}).get("sample_count"), len(train_indices)
                ),
                "validation_sha256": (
                    frozen_splits.get("validation", {}).get("sha256"), validation_sha256
                ),
                "validation_count": (
                    frozen_splits.get("validation", {}).get("sample_count"),
                    len(validation_indices_sorted),
                ),
                "probe_sha256": (
                    frozen_splits.get("in_domain_probe", {}).get("sha256"),
                    sha256_file(probe_manifest_path),
                ),
            }
            mismatches = [
                name
                for name, (locked, generated) in expected_locked_values.items()
                if locked != generated
            ]
            if mismatches:
                raise ValueError(
                    "Generated LCS manifests do not match the versioned split lock: "
                    + ", ".join(mismatches)
                )
        train_temp.replace(train_manifest_path)
        validation_temp.replace(validation_manifest_path)
    finally:
        train_temp.unlink(missing_ok=True)
        validation_temp.unlink(missing_ok=True)

    lock = {
        "schema_version": 1,
        "dataset": LCS_DATASET_NAME,
        "source": {
            "annotation": _relative_path(annotation_path, project_root),
            "sha256": source_sha256,
            "sample_count": expected_source_count,
        },
        "split_algorithm": {
            "probe": {
                "status": "preexisting_frozen_manifest",
                "method": "random.sample",
                "seed": probe_seed,
                "count": expected_probe_count,
            },
            "validation": {
                "method": "sha256_rank(lcs_558k.validation.v1:seed:source_index)",
                "seed": validation_seed,
                "count": validation_count,
            },
            "train": {
                "method": "source_order_complement_of_validation_and_probe",
            },
        },
        "splits": {
            "train": {
                "role": "parameter_updates_only",
                "manifest": _relative_path(train_manifest_path, project_root),
                "sha256": train_sha256,
                "sample_count": len(train_indices),
            },
            "validation": {
                "role": "training_monitoring_only_no_checkpoint_selection",
                "manifest": _relative_path(validation_manifest_path, project_root),
                "sha256": validation_sha256,
                "sample_count": len(validation_indices_sorted),
            },
            "in_domain_probe": {
                "role": "representation_metrics_only",
                "manifest": _relative_path(probe_manifest_path, project_root),
                "sha256": sha256_file(probe_manifest_path),
                "sample_count": len(probe_indices),
            },
        },
        "invariants": {
            "pairwise_disjoint": True,
            "exhaustive_over_source": True,
            "branch_resampling_forbidden": True,
            "probe_checkpoint_selection_forbidden": True,
        },
    }
    if not lock_path.exists():
        _write_json_atomic(lock_path, lock)
    return validate_lcs_splits(
        project_root=project_root,
        annotation_path=annotation_path,
        probe_manifest_path=probe_manifest_path,
        train_manifest_path=train_manifest_path,
        validation_manifest_path=validation_manifest_path,
        lock_path=lock_path,
        expected_source_sha256=expected_source_sha256,
        expected_source_count=expected_source_count,
        expected_probe_count=expected_probe_count,
        expected_probe_seed=probe_seed,
        expected_validation_count=validation_count,
    )


def validate_coco_probe(
    *,
    project_root: Path,
    captions_path: Path,
    manifest_path: Path,
    expected_captions_sha256: str,
    expected_manifest_sha256: str,
    expected_count: int = COCO_PROBE_COUNT,
    verify_images: bool = False,
) -> dict[str, Any]:
    if sha256_file(captions_path) != expected_captions_sha256:
        raise ValueError("COCO captions source SHA-256 mismatch.")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Frozen COCO 5K manifest SHA-256 mismatch.")
    captions = _read_json(captions_path)
    manifest = _read_json(manifest_path)
    if not isinstance(captions, dict) or not isinstance(manifest, dict):
        raise ValueError("COCO captions and manifest must be JSON objects.")
    images = {int(image["id"]): image for image in captions.get("images", [])}
    minimum_captions: dict[int, Mapping[str, Any]] = {}
    for annotation in captions.get("annotations", []):
        image_id = int(annotation["image_id"])
        previous = minimum_captions.get(image_id)
        if previous is None or int(annotation["id"]) < int(previous["id"]):
            minimum_captions[image_id] = annotation
    samples = manifest.get("samples", [])
    if len(images) != expected_count or len(samples) != expected_count:
        raise ValueError(f"COCO external probe must contain exactly {expected_count} images/pairs.")
    if manifest.get("caption_selection", {}).get("rule") != "minimum_caption_id_per_image":
        raise ValueError("COCO manifest does not declare the locked minimum-caption-id rule.")

    seen_image_ids: set[int] = set()
    for position, sample in enumerate(samples):
        image_id = int(sample["image_id"])
        if image_id in seen_image_ids or image_id not in images:
            raise ValueError(f"Duplicate or unknown COCO image_id at manifest position {position}.")
        selected = minimum_captions[image_id]
        if int(sample["caption_id"]) != int(selected["id"]):
            raise ValueError(f"COCO image {image_id} does not use its minimum caption_id.")
        if sample["caption"] != selected["caption"]:
            raise ValueError(f"COCO image {image_id} caption text differs from the source annotation.")
        expected_instance_id = f"coco_2017_val:{image_id}"
        if sample.get("sample_id") != expected_instance_id or sample.get("semantic_id") != expected_instance_id:
            raise ValueError(f"COCO image {image_id} has an invalid controlled identity.")
        image_path = (project_root / sample["image_relpath"]).resolve()
        if not image_path.is_relative_to(project_root.resolve()):
            raise ValueError(f"COCO manifest image path escapes project_root: {image_path}")
        if verify_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        seen_image_ids.add(image_id)
    if seen_image_ids != set(images):
        raise ValueError("COCO manifest does not cover every 2017 validation image exactly once.")
    return {
        "dataset": "coco_2017_val",
        "role": "external_representation_probe_only",
        "sample_count": len(samples),
        "manifest_sha256": expected_manifest_sha256,
        "caption_selection": "minimum_caption_id_per_image",
    }


def verify_lcs_images(
    *,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    training_image_root: Path,
    frozen_probe_manifest_path: Path,
    probe_image_root: Path,
) -> dict[str, int]:
    missing_count = 0
    missing_examples: list[Path] = []
    checked = 0
    for manifest_path in (train_manifest_path, validation_manifest_path):
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                checked += 1
                path = training_image_root / record["image"]
                if not path.is_file():
                    missing_count += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(path)
    probe = _read_json(frozen_probe_manifest_path)
    for record in probe["samples"]:
        checked += 1
        path = probe_image_root / record["image"]
        if not path.is_file():
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(path)
    if missing_count:
        examples = ", ".join(str(path) for path in missing_examples)
        raise FileNotFoundError(f"{missing_count} LCS images are missing; examples: {examples}")
    return {"checked": checked, "missing": 0}
