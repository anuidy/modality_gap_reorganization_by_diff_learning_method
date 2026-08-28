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
