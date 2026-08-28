from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from .base import EmbeddingAdapter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OpenAIClipViTL14Adapter(EmbeddingAdapter):
    """Adapter for the original OpenAI CLIP ViT-L/14 checkpoint and tokenizer."""

    adapter_version = "openai_clip_vit_l14_adapter/v1"

    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        try:
            import clip  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "The official OpenAI CLIP package is required. Install it from "
                "https://github.com/openai/CLIP."
            ) from error

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self._clip = clip
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.checkpoint = checkpoint
        self.model, self.preprocess = clip.load(str(checkpoint), device=str(self.device), jit=False)
        self.model.eval()
        self.embedding_dim = int(self.model.text_projection.shape[1])
        self.input_resolution = int(self.model.visual.input_resolution)

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        image_batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.inference_mode(), self._autocast():
            embeddings = self.model.encode_image(image_batch)
        return embeddings.detach().float().cpu()

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self._clip.tokenize(list(texts), truncate=True).to(self.device)
        with torch.inference_mode(), self._autocast():
            embeddings = self.model.encode_text(tokens)
        return embeddings.detach().float().cpu()

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "openai_clip_vit_l14",
            "adapter_version": self.adapter_version,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "checkpoint_stage": "m0_pretrained",
            "implementation": "openai/CLIP",
            "preprocess_version": "openai_clip_default_resize_center_crop_rgb_normalize/v1",
            "input_resolution": self.input_resolution,
            "embedding_dim": self.embedding_dim,
            "eval_mode": not self.model.training,
            "inference_autocast_dtype": "float16" if self.device.type == "cuda" else None,
        }
