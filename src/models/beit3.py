from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms

from .base import EmbeddingAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEIT3_SOURCE_ROOT = PROJECT_ROOT / "third_party" / "unilm" / "beit3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Beit3BaseItcAdapter(EmbeddingAdapter):
    """Official BEiT-3 retrieval path exporting head outputs before L2 normalization."""

    adapter_version = "beit3_base_itc_adapter/v1"

    def __init__(self, checkpoint: Path, sentencepiece_model: Path, device: str = "cuda") -> None:
        if not checkpoint.is_file() or not sentencepiece_model.is_file():
            raise FileNotFoundError("BEiT-3 checkpoint or SentencePiece model is missing.")
        if "torch._six" not in sys.modules:
            torch_six = types.ModuleType("torch._six")
            torch_six.inf = math.inf
            sys.modules["torch._six"] = torch_six
        if str(BEIT3_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(BEIT3_SOURCE_ROOT))
        modeling_finetune = importlib.import_module("modeling_finetune")
        from transformers import XLMRobertaTokenizer

        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.sentencepiece_model = sentencepiece_model
        self.model = modeling_finetune.beit3_base_patch16_224_retrieval()
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        load_result = self.model.load_state_dict(checkpoint_payload["model"], strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected BEiT-3 checkpoint keys: {load_result.unexpected_keys[:5]}")
        self.missing_keys = list(load_result.missing_keys)
        self.model.to(self.device).eval()
        self.tokenizer = XLMRobertaTokenizer(str(sentencepiece_model))
        self.embedding_dim = 768
        self.max_text_tokens = 64
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD),
            ]
        )

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.inference_mode(), self._autocast():
            encoder_out = self.model.beit3(
                textual_tokens=None, visual_tokens=batch, text_padding_position=None
            )["encoder_out"]
            embeddings = self.model.vision_head(encoder_out[:, 0, :])
        return embeddings.detach().float().cpu()

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        input_ids = torch.full(
            (len(texts), self.max_text_tokens), self.tokenizer.pad_token_id, dtype=torch.long
        )
        padding_mask = torch.ones((len(texts), self.max_text_tokens), dtype=torch.bool)
        for index, text in enumerate(texts):
            token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            token_ids = token_ids[: self.max_text_tokens - 2]
            sequence = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
            input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            padding_mask[index, : len(sequence)] = False
        with torch.inference_mode(), self._autocast():
            encoder_out = self.model.beit3(
                textual_tokens=input_ids.to(self.device),
                visual_tokens=None,
                text_padding_position=padding_mask.to(self.device),
            )["encoder_out"]
            embeddings = self.model.language_head(encoder_out[:, 0, :])
        return embeddings.detach().float().cpu()

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "beit3_base_itc_patch16_224",
            "adapter_version": self.adapter_version,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "checkpoint_stage": "m0_pretrained_itc",
            "implementation": "microsoft/unilm beit3 retrieval",
            "sentencepiece_model": str(self.sentencepiece_model),
            "preprocess_version": "official_beit3_retrieval_eval_transform_224_inception_normalize/v1",
            "embedding_dim": self.embedding_dim,
            "raw_embedding_mode": "retrieval_head_pre_l2",
            "missing_checkpoint_keys": self.missing_keys,
            "eval_mode": not self.model.training,
        }
