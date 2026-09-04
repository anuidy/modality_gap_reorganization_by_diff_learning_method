from __future__ import annotations

import contextlib
import importlib
import math
import random
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets.training_pairs import RawTrainingBatch
from models.albef import ALBEF_SOURCE_ROOT, _official_albef_class, _pre_caption
from models.beit3 import BEIT3_SOURCE_ROOT
from models.vista import VISTA_SOURCE_ROOT
from objectives.contrastive import (
    ObjectiveResult,
    RelationAudit,
    RepresentationBatch,
    additive_multimodal_embedding,
    count_matched_mixed_objective,
    relation_for_optimizer_step,
    standard_objective,
)
from training.albef_accumulation import DeferredAlbefStateUpdates


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class PreparedBatch:
    semantic_ids: tuple[str, ...]
    images: torch.Tensor
    text_tokens: Any


@dataclass(frozen=True)
class TrainingStepResult:
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    audit: RelationAudit | None


@contextlib.contextmanager
def _seeded_augmentation(seed: int):
    python_state = random.getstate()
    try:
        random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)


def _build_train_transform(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    augmentation: Mapping[str, Any],
) -> Callable[[Any], torch.Tensor]:
    name = str(augmentation["name"])
    horizontal_flip = float(augmentation["horizontal_flip_probability"])
    common_tail: list[Any] = [
        transforms.RandomHorizontalFlip(horizontal_flip),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if name == "random_resized_crop":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(float(augmentation["scale_min"]), float(augmentation["scale_max"])),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                *common_tail,
            ]
        )
    if name == "resize_center_crop":
        return transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                *common_tail,
            ]
        )
    raise ValueError(f"Unsupported augmentation.name: {name}")


def _relation_step(
    semantic_ids: tuple[str, ...],
    image: torch.Tensor,
    text: torch.Tensor,
    logit_scale: float | torch.Tensor,
    branch: str,
    optimizer_step: int,
    joint_encoder: Callable[[], torch.Tensor] | None,
) -> TrainingStepResult:
    image = F.normalize(image, dim=-1)
    text = F.normalize(text, dim=-1)
    if branch == "standard":
        objective = standard_objective(logit_scale)
        representations = {"I": image, "T": text}
    elif branch == "count_matched_mixed":
        relation = relation_for_optimizer_step(optimizer_step)
        representations = {"I": image, "T": text}
        if "IT" in relation:
            multimodal = joint_encoder() if joint_encoder is not None else additive_multimodal_embedding(image, text)
            representations["IT"] = F.normalize(multimodal, dim=-1)
        objective = count_matched_mixed_objective(logit_scale, relation)
    else:
        raise ValueError(f"Unsupported relation branch: {branch}")

    result: ObjectiveResult = objective(
        RepresentationBatch(semantic_ids=semantic_ids, representations=representations)
    )
    metrics = {"loss": result.loss, **{f"loss/{name}": value for name, value in result.directional_losses.items()}}
    return TrainingStepResult(loss=result.loss, metrics=metrics, audit=result.audit)


class TrainingBackend(nn.Module, ABC):
    model_name: str

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.device = device

    @abstractmethod
    def prepare_batch(self, batch: RawTrainingBatch, augmentation_seed: int) -> PreparedBatch:
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch: PreparedBatch, branch: str, optimizer_step: int) -> TrainingStepResult:
        raise NotImplementedError

    def begin_optimizer_step(self, micro_batches: int) -> None:
        """Open one gradient-accumulation window."""
        del micro_batches

    def before_optimizer_step(self) -> None:
        """Flush state derived from all accumulated micro-batches."""

    def after_optimizer_step(self) -> None:
        """Apply model-specific post-update constraints."""

    def no_weight_decay_parameter_names(self) -> set[str]:
        """Return backend-qualified parameter names excluded by the model itself."""

        model = getattr(self, "model", None)
        provider = getattr(model, "no_weight_decay", None)
        if not callable(provider):
            return set()
        model_names = {str(name) for name in (provider() or ())}
        return model_names | {f"model.{name}" for name in model_names}

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }


class ClipTrainingBackend(TrainingBackend):
    model_name = "clip"

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        augmentation: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> None:
        super().__init__(device)
        try:
            import clip  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Install the official OpenAI CLIP package pinned in requirements.txt.") from error
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self._clip = clip
        model, _ = clip.load(str(checkpoint), device="cpu", jit=False)
        self.model = model.float().to(device)
        self.preprocess = _build_train_transform(
            int(options.get("image_size", 224)), CLIP_MEAN, CLIP_STD, augmentation
        )

    def prepare_batch(self, batch: RawTrainingBatch, augmentation_seed: int) -> PreparedBatch:
        with _seeded_augmentation(augmentation_seed):
            images = torch.stack([self.preprocess(image) for image in batch.images])
        tokens = self._clip.tokenize(list(batch.texts), truncate=True)
        return PreparedBatch(
            semantic_ids=batch.semantic_ids,
            images=images.to(self.device, non_blocking=True),
            text_tokens=tokens.to(self.device, non_blocking=True),
        )

    def forward(self, batch: PreparedBatch, branch: str, optimizer_step: int) -> TrainingStepResult:
        image = self.model.encode_image(batch.images)
        text = self.model.encode_text(batch.text_tokens)
        return _relation_step(
            batch.semantic_ids,
            image,
            text,
            self.model.logit_scale.exp(),
            branch,
            optimizer_step,
            joint_encoder=None,
        )

    def after_optimizer_step(self) -> None:
        with torch.no_grad():
            self.model.logit_scale.clamp_(max=math.log(100.0))


class VistaTrainingBackend(TrainingBackend):
    model_name = "vista"

    def __init__(
        self,
        checkpoint: Path,
        text_backbone: Path,
        device: torch.device,
        augmentation: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> None:
        super().__init__(device)
        if not checkpoint.is_file() or not text_backbone.is_dir():
            raise FileNotFoundError("VISTA checkpoint or local BGE text backbone is missing.")
        if str(VISTA_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(VISTA_SOURCE_ROOT))
        from visual_bge.modeling import Visualized_BGE

        self.model = Visualized_BGE(
            model_name_bge="bge-base-en-v1.5",
            model_weight=str(checkpoint),
            normlized=True,
            temperature=float(options.get("temperature", 0.02)),
            from_pretrained=str(text_backbone),
        ).to(device)
        self.preprocess = _build_train_transform(
            int(options.get("image_size", 224)), CLIP_MEAN, CLIP_STD, augmentation
        )
        self.max_text_tokens = int(options.get("max_text_tokens", 64))

    def prepare_batch(self, batch: RawTrainingBatch, augmentation_seed: int) -> PreparedBatch:
        with _seeded_augmentation(augmentation_seed):
            images = torch.stack([self.preprocess(image) for image in batch.images])
        tokens = self.model.tokenizer(
            list(batch.texts),
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        return PreparedBatch(
            semantic_ids=batch.semantic_ids,
            images=images.to(self.device, non_blocking=True),
            text_tokens=tokens.to(self.device),
        )

    def forward(self, batch: PreparedBatch, branch: str, optimizer_step: int) -> TrainingStepResult:
        if branch == "standard":
            relation = "I<->T"
            objective = standard_objective(1.0 / float(self.model.temperature))
        elif branch == "count_matched_mixed":
            relation = relation_for_optimizer_step(optimizer_step)
            objective = count_matched_mixed_objective(
                1.0 / float(self.model.temperature), relation
            )
        else:
            raise ValueError(f"Unsupported relation branch: {branch}")

        required_modalities = {
            "I<->T": ("I", "T"),
            "I<->IT": ("I", "IT"),
            "T<->IT": ("T", "IT"),
        }[relation]
        representations: dict[str, torch.Tensor] = {}
        if "I" in required_modalities:
            representations["I"] = F.normalize(self.model.encode_image(batch.images), dim=-1)
        if "T" in required_modalities:
            representations["T"] = F.normalize(self.model.encode_text(batch.text_tokens), dim=-1)
        if "IT" in required_modalities:
            representations["IT"] = F.normalize(
                self.model.encode_mm(batch.images, batch.text_tokens), dim=-1
            )
        result = objective(
            RepresentationBatch(
                semantic_ids=batch.semantic_ids,
                representations=representations,
            )
        )
        metrics = {
            "loss": result.loss,
            **{f"loss/{name}": value for name, value in result.directional_losses.items()},
        }
        return TrainingStepResult(loss=result.loss, metrics=metrics, audit=result.audit)


def _install_torch_six_compatibility() -> None:
    if "torch._six" not in sys.modules:
        torch_six = types.ModuleType("torch._six")
        torch_six.inf = math.inf
        sys.modules["torch._six"] = torch_six


class Beit3TrainingBackend(TrainingBackend):
    model_name = "beit3"

    def __init__(
        self,
        checkpoint: Path,
        sentencepiece_model: Path,
        device: torch.device,
        augmentation: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> None:
        super().__init__(device)
        if not checkpoint.is_file() or not sentencepiece_model.is_file():
            raise FileNotFoundError("BEiT-3 checkpoint or SentencePiece model is missing.")
        _install_torch_six_compatibility()
        if str(BEIT3_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(BEIT3_SOURCE_ROOT))
        modeling_finetune = importlib.import_module("modeling_finetune")
        from transformers import XLMRobertaTokenizer

        self.model = modeling_finetune.beit3_base_patch16_224_retrieval()
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        load_result = self.model.load_state_dict(checkpoint_payload["model"], strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected BEiT-3 checkpoint keys: {load_result.unexpected_keys[:5]}")
        self.model.to(device)
        self.tokenizer = XLMRobertaTokenizer(str(sentencepiece_model))
        self.max_text_tokens = int(options.get("max_text_tokens", 64))
        self.preprocess = _build_train_transform(
            int(options.get("image_size", 224)),
            tuple(IMAGENET_INCEPTION_MEAN),
            tuple(IMAGENET_INCEPTION_STD),
            augmentation,
        )

    def _tokenize(self, texts: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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
        return input_ids, padding_mask

    def prepare_batch(self, batch: RawTrainingBatch, augmentation_seed: int) -> PreparedBatch:
        with _seeded_augmentation(augmentation_seed):
            images = torch.stack([self.preprocess(image) for image in batch.images])
        input_ids, padding_mask = self._tokenize(batch.texts)
        return PreparedBatch(
            semantic_ids=batch.semantic_ids,
            images=images.to(self.device, non_blocking=True),
            text_tokens=(
                input_ids.to(self.device, non_blocking=True),
                padding_mask.to(self.device, non_blocking=True),
            ),
        )

    def forward(self, batch: PreparedBatch, branch: str, optimizer_step: int) -> TrainingStepResult:
        input_ids, padding_mask = batch.text_tokens
        image, text = self.model(
            image=batch.images,
            text_description=input_ids,
            padding_mask=padding_mask,
            only_infer=True,
        )
        return _relation_step(
            batch.semantic_ids,
            image,
            text,
            self.model.logit_scale.exp(),
            branch,
            optimizer_step,
            joint_encoder=None,
        )

    def after_optimizer_step(self) -> None:
        with torch.no_grad():
            self.model.logit_scale.clamp_(max=math.log(100.0))

    def no_weight_decay_parameter_names(self) -> set[str]:
        names = super().no_weight_decay_parameter_names()
        for name, _ in self.named_parameters():
            if (
                name.endswith("logit_scale")
                or name.endswith("cls_token")
                or name.endswith("mask_token")
                or name.endswith("pos_embed")
                or ".embed_positions." in name
            ):
                names.add(name)
        return names


@torch.no_grad()
def _safe_concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    gathered = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor, async_op=False)
    return torch.cat(gathered, dim=0)


def _device_safe_albef_mask(
    self: nn.Module,
    input_ids: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    targets: torch.Tensor | None = None,
    masked_indices: torch.Tensor | None = None,
    probability_matrix: torch.Tensor | None = None,
):
    del device
    if probability_matrix is None:
        probability_matrix = torch.full(
            input_ids.shape, self.mlm_probability, device=input_ids.device, dtype=torch.float32
        )
    else:
        probability_matrix = probability_matrix.to(input_ids.device)
    if masked_indices is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = masked_indices.to(input_ids.device).bool()
    masked_indices[input_ids == self.tokenizer.pad_token_id] = False
    masked_indices[input_ids == self.tokenizer.cls_token_id] = False
    if targets is not None:
        targets[~masked_indices] = -100

    indices_replaced = (
        torch.bernoulli(torch.full(input_ids.shape, 0.8, device=input_ids.device)).bool()
        & masked_indices
    )
    input_ids[indices_replaced] = self.tokenizer.mask_token_id
    indices_random = (
        torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device)
    input_ids[indices_random] = random_words[indices_random]
    return (input_ids, targets) if targets is not None else input_ids


class AlbefTrainingBackend(TrainingBackend):
    model_name = "albef"

    def __init__(
        self,
        checkpoint: Path,
        text_backbone: Path,
        device: torch.device,
        augmentation: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> None:
        super().__init__(device)
        if not checkpoint.is_file() or not text_backbone.is_dir():
            raise FileNotFoundError("ALBEF checkpoint or local BERT backbone is missing.")
        from transformers import BertTokenizer

        official_config = yaml.safe_load(
            (ALBEF_SOURCE_ROOT / "configs" / "Pretrain.yaml").read_text(encoding="utf-8")
        )
        official_config.update(
            {
                "bert_config": str(ALBEF_SOURCE_ROOT / "configs" / "config_bert.json"),
                "image_res": int(options.get("image_size", 256)),
                "vision_width": 768,
                "embed_dim": 256,
                "queue_size": int(options.get("queue_size", 65536)),
                "momentum": 0.995,
                "temp": 0.07,
                "mlm_probability": float(options.get("mlm_probability", 0.15)),
                "distill": True,
            }
        )
        self.tokenizer = BertTokenizer.from_pretrained(str(text_backbone), local_files_only=True)
        model_class = _official_albef_class()
        self.model = model_class(
            text_encoder=str(text_backbone), tokenizer=self.tokenizer, config=official_config, init_deit=False
        )
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        load_result = self.model.load_state_dict(checkpoint_payload["model"], strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected ALBEF checkpoint keys: {load_result.unexpected_keys[:5]}")
        self.model.__class__.forward.__globals__["concat_all_gather"] = _safe_concat_all_gather
        self.model.mask = types.MethodType(_device_safe_albef_mask, self.model)
        self.model.to(device)
        self._deferred_state_updates = DeferredAlbefStateUpdates()
        self.max_text_tokens = int(options.get("max_text_tokens", 30))
        self.alpha = float(options.get("alpha", 0.4))
        self.alpha_warmup_steps = int(options.get("alpha_warmup_steps") or 0)
        self.preprocess = _build_train_transform(
            int(options.get("image_size", 256)), CLIP_MEAN, CLIP_STD, augmentation
        )

    def begin_optimizer_step(self, micro_batches: int) -> None:
        self._deferred_state_updates.begin(self.model, micro_batches)

    def before_optimizer_step(self) -> None:
        self._deferred_state_updates.flush(
            lambda chunks: torch.cat(chunks, dim=0)
        )

    def after_optimizer_step(self) -> None:
        if self._deferred_state_updates.active:
            raise RuntimeError(
                "ALBEF optimizer step completed before deferred queue updates were flushed."
            )

    def prepare_batch(self, batch: RawTrainingBatch, augmentation_seed: int) -> PreparedBatch:
        with _seeded_augmentation(augmentation_seed):
            images = torch.stack([self.preprocess(image) for image in batch.images])
        captions = [_pre_caption(text, max_words=self.max_text_tokens) for text in batch.texts]
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        return PreparedBatch(
            semantic_ids=batch.semantic_ids,
            images=images.to(self.device, non_blocking=True),
            text_tokens=tokens.to(self.device),
        )

    def _alpha_for_step(self, optimizer_step: int) -> float:
        if self.alpha_warmup_steps <= 0:
            return self.alpha
        return self.alpha * min(1.0, (optimizer_step + 1) / self.alpha_warmup_steps)

    def _native_itc(self, image: torch.Tensor, text: Any, alpha: float) -> torch.Tensor:
        model = self.model
        with torch.no_grad():
            model.temp.clamp_(0.001, 0.5)
        image_embeds = model.visual_encoder(image)
        image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_output = model.text_encoder.bert(
            text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode="text"
        )
        text_feat = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        with torch.no_grad():
            model._momentum_update()
            image_embeds_m = model.visual_encoder_m(image)
            image_feat_m = F.normalize(model.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat(
                [image_feat_m.transpose(0, 1), model.image_queue.clone().detach()], dim=1
            )
            text_output_m = model.text_encoder_m.bert(
                text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode="text"
            )
            text_feat_m = F.normalize(
                model.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1
            )
            text_feat_all = torch.cat(
                [text_feat_m.transpose(0, 1), model.text_queue.clone().detach()], dim=1
            )
            sim_i2t_m = image_feat_m @ text_feat_all / model.temp
            sim_t2i_m = text_feat_m @ image_feat_all / model.temp
            sim_targets = torch.zeros_like(sim_i2t_m)
            sim_targets.fill_diagonal_(1)
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / model.temp
        sim_t2i = text_feat @ image_feat_all / model.temp
        loss_i2t = -(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets).sum(dim=1).mean()
        loss_t2i = -(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets).sum(dim=1).mean()
        model._dequeue_and_enqueue(image_feat_m, text_feat_m)
        return 0.5 * (loss_i2t + loss_t2i)

    def forward(self, batch: PreparedBatch, branch: str, optimizer_step: int) -> TrainingStepResult:
        alpha = self._alpha_for_step(optimizer_step)
        if branch == "itc_only":
            with self._deferred_state_updates.intercept_forward():
                loss_itc = self._native_itc(batch.images, batch.text_tokens, alpha)
            return TrainingStepResult(
                loss=loss_itc,
                metrics={"loss": loss_itc, "loss/ITC": loss_itc},
                audit=None,
            )
        if branch == "full_albef":
            with self._deferred_state_updates.intercept_forward():
                loss_mlm, loss_itc, loss_itm = self.model(batch.images, batch.text_tokens, alpha=alpha)
            loss = loss_itc + loss_itm + loss_mlm
            return TrainingStepResult(
                loss=loss,
                metrics={
                    "loss": loss,
                    "loss/ITC": loss_itc,
                    "loss/ITM": loss_itm,
                    "loss/MLM": loss_mlm,
                },
                audit=None,
            )
        raise ValueError(f"Unsupported ALBEF branch: {branch}")


def create_training_backend(
    model_name: str,
    checkpoint: Path,
    resources: Mapping[str, Path],
    device: torch.device,
    augmentation: Mapping[str, Any],
    options: Mapping[str, Any],
) -> TrainingBackend:
    if model_name == "clip":
        return ClipTrainingBackend(checkpoint, device, augmentation, options)
    if model_name == "vista":
        return VistaTrainingBackend(
            checkpoint, resources["text_backbone"], device, augmentation, options
        )
    if model_name == "beit3":
        return Beit3TrainingBackend(
            checkpoint, resources["sentencepiece_model"], device, augmentation, options
        )
    if model_name == "albef":
        return AlbefTrainingBackend(
            checkpoint, resources["text_backbone"], device, augmentation, options
        )
    raise ValueError(f"Unsupported training model: {model_name}")
