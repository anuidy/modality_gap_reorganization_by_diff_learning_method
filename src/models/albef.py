from __future__ import annotations

import contextlib
import hashlib
import importlib
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from PIL import Image
from torchvision import transforms

from .base import EmbeddingAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALBEF_SOURCE_ROOT = PROJECT_ROOT / "third_party" / "albef"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pre_caption(caption: str, max_words: int = 30) -> str:
    caption = re.sub(r"([,.''!?\"()*#:;~])", "", caption.lower()).replace("-", " ").replace("/", " ")
    caption = re.sub(r"\s{2,}", " ", caption).rstrip("\n").strip(" ")
    return " ".join(caption.split(" ")[:max_words])


def _official_albef_class() -> type[torch.nn.Module]:
    """Load official ALBEF without permanently shadowing this project's models package."""
    import transformers.file_utils as file_utils
    import transformers.modeling_utils as modeling_utils
    from transformers.pytorch_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )

    # ALBEF pins an older Transformers API. These helpers moved modules but retain
    # the same public behavior required by the official ALBEF xBERT implementation.
    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    modeling_utils.prune_linear_layer = prune_linear_layer
    original_add_code_sample_docstrings = file_utils.add_code_sample_docstrings

    def add_code_sample_docstrings_compat(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("tokenizer_class", None)
        return original_add_code_sample_docstrings(*args, **kwargs)

    file_utils.add_code_sample_docstrings = add_code_sample_docstrings_compat
    saved_modules = {name: module for name, module in sys.modules.items() if name == "models" or name.startswith("models.")}
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(ALBEF_SOURCE_ROOT))
    try:
        module = importlib.import_module("models.model_pretrain")
        return module.ALBEF
    finally:
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path.remove(str(ALBEF_SOURCE_ROOT))


class AlbefM0Adapter(EmbeddingAdapter):
    """Official ALBEF ITC encoder path exporting pre-normalization projections."""

    adapter_version = "albef_m0_itc_adapter/v1"

    def __init__(self, checkpoint: Path, bert_backbone: Path, device: str = "cuda") -> None:
        if not checkpoint.is_file() or not bert_backbone.is_dir():
            raise FileNotFoundError("ALBEF checkpoint or local BERT backbone is missing.")
        from transformers import BertTokenizer

        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.bert_backbone = bert_backbone
        config = yaml.safe_load((ALBEF_SOURCE_ROOT / "configs" / "Pretrain.yaml").read_text(encoding="utf-8"))
        config.update(
            {
                "bert_config": str(ALBEF_SOURCE_ROOT / "configs" / "config_bert.json"),
                "image_res": 256,
                "vision_width": 768,
                "embed_dim": 256,
                "queue_size": 65536,
                "momentum": 0.995,
                "temp": 0.07,
                "distill": True,
            }
        )
        self.tokenizer = BertTokenizer.from_pretrained(str(bert_backbone), local_files_only=True)
        model_class = _official_albef_class()
        self.model = model_class(
            text_encoder=str(bert_backbone), tokenizer=self.tokenizer, config=config, init_deit=False
        )
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = checkpoint_payload["model"]
        load_result = self.model.load_state_dict(state_dict, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected ALBEF checkpoint keys: {load_result.unexpected_keys[:5]}")
        self.missing_keys = list(load_result.missing_keys)
        self.model.to(self.device).eval()
        self.embedding_dim = 256
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.inference_mode(), self._autocast():
            features = self.model.visual_encoder(batch)
            embeddings = self.model.vision_proj(features[:, 0, :])
        return embeddings.detach().float().cpu()

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        captions = [_pre_caption(text, max_words=30) for text in texts]
        tokens = self.tokenizer(
            captions, padding="max_length", truncation=True, max_length=30, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode(), self._autocast():
            features = self.model.text_encoder.bert(
                tokens.input_ids, attention_mask=tokens.attention_mask, return_dict=True, mode="text"
            ).last_hidden_state
            embeddings = self.model.text_proj(features[:, 0, :])
        return embeddings.detach().float().cpu()

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "albef_14m_pretrained",
            "adapter_version": self.adapter_version,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "checkpoint_stage": "m0_pretrained",
            "implementation": "salesforce/ALBEF model_pretrain",
            "text_backbone": str(self.bert_backbone),
            "preprocess_version": "official_albef_test_transform_256_clip_normalize/v1",
            "text_preprocess": "official_albef_pre_caption_max_words_30",
            "embedding_dim": self.embedding_dim,
            "raw_embedding_mode": "itc_projection_pre_l2",
            "missing_checkpoint_keys": self.missing_keys,
            "eval_mode": not self.model.training,
        }
