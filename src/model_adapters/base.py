from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch
from PIL import Image


class EmbeddingAdapter(ABC):
    """Expose raw, pre-L2 image and text embeddings in one common space."""

    @abstractmethod
    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Return CPU float32 raw embeddings with shape [batch, dimension]."""

    @abstractmethod
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Return CPU float32 raw embeddings with shape [batch, dimension]."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return serializable implementation and preprocessing metadata."""
