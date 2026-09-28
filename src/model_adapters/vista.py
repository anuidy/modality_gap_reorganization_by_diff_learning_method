from __future__ import annotations

import contextlib
import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from .base import EmbeddingAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISTA_SOURCE_ROOT = PROJECT_ROOT / "third_party" / "flag_embedding" / "research" / "visual_bge"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VistaStage1Adapter(EmbeddingAdapter):
    """Official Visualized-BGE loader configured for raw Stage-1 Z_I/Z_T export."""

    adapter_version = "vista_stage1_adapter/v1"

    def __init__(self, checkpoint: Path, text_backbone: Path, device: str = "cuda") -> None:
        if not checkpoint.is_file() or not text_backbone.is_dir():
            raise FileNotFoundError("VISTA checkpoint or local BGE text backbone is missing.")
        if str(VISTA_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(VISTA_SOURCE_ROOT))
        from visual_bge.modeling import Visualized_BGE

        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.text_backbone = text_backbone
        self.model = Visualized_BGE(
            model_name_bge="bge-base-en-v1.5",
            model_weight=str(checkpoint),
            normlized=False,
            from_pretrained=str(text_backbone),
        )
        self.model.to(self.device).eval()
        self.embedding_dim = 768

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.model.preprocess_val(image) for image in images]).to(self.device)
        with torch.inference_mode(), self._autocast():
            embeddings = self.model.encode_image(batch)
        return embeddings.detach().float().cpu()

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.model.tokenizer(
            list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode(), self._autocast():
            embeddings = self.model.encode_text(tokens)
        return embeddings.detach().float().cpu()

    def encode_multimodal(
        self, images: Sequence[Image.Image], texts: Sequence[str]
    ) -> torch.Tensor:
        """Native joint-encoder output, returned at the pre-L2 boundary.

        This adapter is built with ``normlized=False``, so ``encode_mm`` returns
        its pooled representation before the final ``F.normalize`` — the same
        tensor the training backend normalizes. The temperature that the same
        flag resets to 1.0 is read only inside ``compute_similarity``, which this
        path never calls, so it does not enter the returned vector.

        BGE's position-embedding table has 512 slots and the joint sequence is
        ``[CLS] + image tokens + text tokens``.  The image token count is fixed by
        the vision tower (196 for EVA02-CLIP-B-16 at 224/16), so the text must be
        truncated to ``512 - image_tokens``; using the text-only limit of 512
        would push the position ids past 511 and trigger a CUDA device-side
        assert.  The limit is derived from the vision tower on first use.
        """

        batch = torch.stack([self.model.preprocess_val(image) for image in images]).to(self.device)
        max_text_tokens = 512 - self._image_token_count()
        tokens = self.model.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max(1, max_text_tokens),
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode(), self._autocast():
            embeddings = self.model.encode_mm(batch, tokens)
        return embeddings.detach().float().cpu()

    def _image_token_count(self) -> int:
        """Number of image tokens the vision tower contributes to the joint input."""
        cached = getattr(self, "_joint_image_tokens", None)
        if cached is None:
            probe = Image.new("RGB", (224, 224), color=(0, 0, 0))
            batch = torch.stack([self.model.preprocess_val(probe)]).to(self.device)
            with torch.inference_mode(), self._autocast():
                patches = self.model.img_token_embedding(batch)
            cached = int(patches.shape[1]) - 1  # encode_mm drops the image CLS token
            self._joint_image_tokens = cached
        return cached

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "vista_base_stage1",
            "adapter_version": self.adapter_version,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "checkpoint_stage": "stage1",
            "implementation": "FlagOpen/FlagEmbedding research/visual_bge",
            "text_backbone": str(self.text_backbone),
            "preprocess_version": "official_visualized_bge_preprocess_val/v1",
            "embedding_dim": self.embedding_dim,
            "raw_embedding_mode": "normlized_false",
            "eval_mode": not self.model.training,
        }
