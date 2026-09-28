from __future__ import annotations

from pathlib import Path

from .albef import AlbefM0Adapter
from .base import EmbeddingAdapter
from .beit3 import Beit3BaseItcAdapter
from .clip_openai import OpenAIClipViTL14Adapter
from .vista import VistaStage1Adapter


def create_m0_adapter(model_name: str, project_root: Path, device: str) -> EmbeddingAdapter:
    if model_name == "clip":
        return OpenAIClipViTL14Adapter(project_root / "data/raw/models/clip_sf/ViT-L-14.pt", device)
    if model_name == "vista":
        return VistaStage1Adapter(
            project_root / "data/raw/models/vista/BGE_EVA_Token_S1.pth",
            project_root / "data/raw/models/vista/bge-base-en-v1.5",
            device,
        )
    if model_name == "albef":
        return AlbefM0Adapter(
            project_root / "data/raw/models/albef/ALBEF_14M.pth",
            project_root / "data/raw/models/albef/bert-base-uncased",
            device,
        )
    if model_name == "beit3":
        return Beit3BaseItcAdapter(
            project_root / "data/raw/models/beit3/beit3_base_itc_patch16_224.pth",
            project_root / "data/raw/models/beit3/beit3.spm",
            device,
        )
    raise ValueError(f"Unsupported M0 model: {model_name}")
