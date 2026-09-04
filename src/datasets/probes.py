from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    semantic_id: str
    image_path: Path
    text: str


@dataclass(frozen=True)
class ProbeManifest:
    name: str
    path: Path
    sha256: str
    samples: tuple[ProbeSample, ...]
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_coco_2017_val_manifest(
    captions_path: Path,
    image_directory: Path,
    destination: Path,
    project_root: Path,
) -> ProbeManifest:
    """Create the locked COCO 5K probe using the smallest caption_id per image."""
    captions = _read_json(captions_path)
    images_by_id = {int(image["id"]): image for image in captions["images"]}
    captions_by_image: dict[int, dict[str, Any]] = {}
    for annotation in captions["annotations"]:
        image_id = int(annotation["image_id"])
        prior = captions_by_image.get(image_id)
        if prior is None or int(annotation["id"]) < int(prior["id"]):
            captions_by_image[image_id] = annotation

    if len(images_by_id) != 5_000 or len(captions_by_image) != 5_000:
        raise ValueError(
            "COCO probe must contain exactly 5,000 validation images and one selected caption each."
        )

    samples: list[dict[str, Any]] = []
    for image_id in sorted(images_by_id):
        image = images_by_id[image_id]
        annotation = captions_by_image[image_id]
        image_path = image_directory / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing COCO probe image: {image_path}")
        samples.append(
            {
                "sample_id": f"coco_2017_val:{image_id}",
                "semantic_id": f"coco_2017_val:{image_id}",
                "image_id": image_id,
                "caption_id": int(annotation["id"]),
                "image_relpath": image_path.relative_to(project_root).as_posix(),
                "caption": annotation["caption"],
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "probe_name": "coco_2017_val_5k",
        "source": {
            "captions_path": captions_path.as_posix(),
            "captions_sha256": sha256_file(captions_path),
            "image_directory": image_directory.as_posix(),
        },
        "caption_selection": {"rule": "minimum_caption_id_per_image"},
        "sample_count": len(samples),
        "samples": samples,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return load_coco_manifest(destination, project_root)


def load_coco_manifest(path: Path, project_root: Path) -> ProbeManifest:
    payload = _read_json(path)
    samples = tuple(
        ProbeSample(
            sample_id=record["sample_id"],
            semantic_id=record["semantic_id"],
            image_path=project_root / record["image_relpath"],
            text=record["caption"],
        )
        for record in payload["samples"]
    )
    if len(samples) != 5_000:
        raise ValueError(f"Expected 5,000 COCO samples, found {len(samples)}.")
    return ProbeManifest(
        name=payload["probe_name"],
        path=path,
        sha256=sha256_file(path),
        samples=samples,
        metadata=payload,
    )


def load_lcs_manifest(
    project_root: Path,
    path: Path,
    image_root: Path | None = None,
) -> ProbeManifest:
    payload = _read_json(path)
    if image_root is None:
        image_root = project_root / "data" / "raw" / "lcs_558k" / "probe_images"
    samples = tuple(
        ProbeSample(
            sample_id=f"lcs_558k:{record['id']}",
            semantic_id=f"lcs_558k:{record['id']}",
            image_path=image_root / record["image"],
            text=record["caption"],
        )
        for record in payload["samples"]
    )
    if len(samples) != 10_000:
        raise ValueError(f"Expected 10,000 LCS samples, found {len(samples)}.")
    return ProbeManifest(
        name="lcs_558k_in_domain_10k",
        path=path,
        sha256=sha256_file(path),
        samples=samples,
        metadata=payload,
    )


def validate_manifest_images(manifest: ProbeManifest) -> None:
    missing = [sample.image_path for sample in manifest.samples if not sample.image_path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"{len(missing)} probe images are missing. Examples: {preview}")
