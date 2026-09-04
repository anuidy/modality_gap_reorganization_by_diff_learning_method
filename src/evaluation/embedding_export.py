from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from datasets.probes import ProbeManifest
from models.base import EmbeddingAdapter


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": package_version("pillow"),
    }


def load_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def export_raw_embeddings(
    adapter: EmbeddingAdapter,
    manifest: ProbeManifest,
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    image_batches: list[np.ndarray] = []
    text_batches: list[np.ndarray] = []
    for start in range(0, len(manifest.samples), batch_size):
        batch = manifest.samples[start : start + batch_size]
        image_batches.append(
            adapter.encode_image(load_images([sample.image_path for sample in batch])).numpy()
        )
        text_batches.append(adapter.encode_text([sample.text for sample in batch]).numpy())
        if progress is not None:
            progress(min(start + len(batch), len(manifest.samples)), len(manifest.samples))
    image_embeddings = np.concatenate(image_batches).astype(np.float32, copy=False)
    text_embeddings = np.concatenate(text_batches).astype(np.float32, copy=False)
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("Image/text embedding export shapes differ.")
    return image_embeddings, text_embeddings
