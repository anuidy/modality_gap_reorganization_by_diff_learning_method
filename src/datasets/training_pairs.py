from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class TrainingPair:
    sample_id: str
    semantic_id: str
    image_path: Path
    text: str


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    semantic_id: str
    image: Image.Image
    text: str


@dataclass(frozen=True)
class RawTrainingBatch:
    sample_ids: tuple[str, ...]
    semantic_ids: tuple[str, ...]
    images: tuple[Image.Image, ...]
    texts: tuple[str, ...]


def _manifest_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Manifest line {line_number} must be a JSON object.")
                yield value
        return

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        records = payload["samples"]
    else:
        raise ValueError("JSON manifest must be a list or an object containing a samples list.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every manifest sample must be a JSON object.")
    yield from records


def _required_text(record: dict[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Manifest record is missing a non-empty field from {names}.")


def load_training_pairs(manifest_path: Path, image_root: Path) -> tuple[TrainingPair, ...]:
    """Load a portable one-positive-per-semantic-instance training manifest."""

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    root = image_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    pairs: list[TrainingPair] = []
    semantic_ids: set[str] = set()
    sample_ids: set[str] = set()
    for index, record in enumerate(_manifest_records(manifest_path)):
        sample_id = _required_text(record, "sample_id")
        semantic_id = _required_text(record, "semantic_id")
        image_value = _required_text(record, "image", "image_relpath")
        text = _required_text(record, "text", "caption")

        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id at record {index}: {sample_id}")
        if semantic_id in semantic_ids:
            raise ValueError(
                f"Duplicate semantic_id at record {index}: {semantic_id}. "
                "Use exactly one image-text positive per semantic instance."
            )
        relative_path = Path(image_value)
        if relative_path.is_absolute():
            raise ValueError("Manifest image paths must be relative to image_root.")
        image_path = (root / relative_path).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"Manifest image path escapes image_root: {image_value}")

        sample_ids.add(sample_id)
        semantic_ids.add(semantic_id)
        pairs.append(
            TrainingPair(
                sample_id=sample_id,
                semantic_id=semantic_id,
                image_path=image_path,
                text=text,
            )
        )

    if len(pairs) < 2:
        raise ValueError("The training manifest must contain at least two unique pairs.")
    return tuple(pairs)


class PairedTrainingDataset(Dataset[TrainingSample]):
    def __init__(self, pairs: Sequence[TrainingPair]) -> None:
        if len(pairs) < 2:
            raise ValueError("PairedTrainingDataset needs at least two samples.")
        self.pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> TrainingSample:
        pair = self.pairs[index]
        with Image.open(pair.image_path) as image:
            rgb = image.convert("RGB").copy()
        return TrainingSample(
            sample_id=pair.sample_id,
            semantic_id=pair.semantic_id,
            image=rgb,
            text=pair.text,
        )


def collate_raw_training_batch(samples: Sequence[TrainingSample]) -> RawTrainingBatch:
    if len(samples) < 2:
        raise ValueError("Every contrastive micro-batch must contain at least two samples.")
    semantic_ids = tuple(sample.semantic_id for sample in samples)
    if len(set(semantic_ids)) != len(semantic_ids):
        raise ValueError("A micro-batch cannot contain duplicate semantic_ids.")
    return RawTrainingBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        semantic_ids=semantic_ids,
        images=tuple(sample.image for sample in samples),
        texts=tuple(sample.text for sample in samples),
    )


class DeterministicEpochSampler(Sampler[int]):
    """Produce the same epoch order for both branches of a controlled pair."""

    def __init__(self, data_source: Sequence[object], seed: int) -> None:
        if len(data_source) < 2:
            raise ValueError("The sampler needs at least two samples.")
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)
