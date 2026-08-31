from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


COUNT_MATCHED_RELATION_CYCLE: tuple[str, ...] = (
    "I<->T",
    "I<->IT",
    "T<->IT",
)


@dataclass(frozen=True)
class RepresentationBatch:
    """One aligned batch of representations for unique semantic instances."""

    semantic_ids: Sequence[str] | torch.Tensor
    representations: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class RelationAudit:
    relation: str
    batch_size: int
    positive_terms: int
    candidates_per_query: int
    negatives_per_query: int


@dataclass(frozen=True)
class ObjectiveResult:
    loss: torch.Tensor
    directional_losses: Mapping[str, torch.Tensor]
    audit: RelationAudit


Objective = Callable[[RepresentationBatch], ObjectiveResult]


def _semantic_id_list(semantic_ids: Sequence[str] | torch.Tensor) -> list[object]:
    if isinstance(semantic_ids, torch.Tensor):
        if semantic_ids.ndim != 1:
            raise ValueError("semantic_ids must be one-dimensional.")
        return semantic_ids.detach().cpu().tolist()
    return list(semantic_ids)


def _validate_relation_batch(
    batch: RepresentationBatch,
    left_name: str,
    right_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        left = batch.representations[left_name]
        right = batch.representations[right_name]
    except KeyError as error:
        raise ValueError(f"Missing representation required by the relation: {error.args[0]}") from error

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Contrastive representations must have shape [batch, dimension].")
    if left.shape != right.shape:
        raise ValueError(
            f"Relation representations must have identical shapes; got {left.shape} and {right.shape}."
        )
    if left.shape[0] < 2:
        raise ValueError("A contrastive batch needs at least two semantic instances.")

    semantic_ids = _semantic_id_list(batch.semantic_ids)
    if len(semantic_ids) != left.shape[0]:
        raise ValueError("semantic_ids length must equal the representation batch size.")
    try:
        unique_count = len(set(semantic_ids))
    except TypeError as error:
        raise ValueError("semantic_ids must contain hashable values.") from error
    if unique_count != len(semantic_ids):
        raise ValueError(
            "Every batch must contain unique semantic_ids so that each query has exactly "
            "one positive and N-1 valid negatives."
        )
    return left, right


def _scale_tensor(logit_scale: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(logit_scale, torch.Tensor):
        if logit_scale.numel() != 1:
            raise ValueError("logit_scale must be scalar.")
        return logit_scale.to(device=reference.device, dtype=torch.float32)
    if logit_scale <= 0:
        raise ValueError("logit_scale must be positive.")
    return torch.tensor(logit_scale, device=reference.device, dtype=torch.float32)


def bidirectional_contrastive(
    batch: RepresentationBatch,
    left_name: str,
    right_name: str,
    logit_scale: float | torch.Tensor,
) -> ObjectiveResult:
    """Compute one count-matched bidirectional in-batch contrastive relation."""

    left, right = _validate_relation_batch(batch, left_name, right_name)
    left_normalized = F.normalize(left.float(), dim=-1)
    right_normalized = F.normalize(right.float(), dim=-1)
    scale = _scale_tensor(logit_scale, left_normalized)
    logits = scale * left_normalized @ right_normalized.transpose(0, 1)
    targets = torch.arange(logits.shape[0], device=logits.device)

    left_to_right = F.cross_entropy(logits, targets)
    right_to_left = F.cross_entropy(logits.transpose(0, 1), targets)
    loss = 0.5 * (left_to_right + right_to_left)
    relation = f"{left_name}<->{right_name}"
    batch_size = int(logits.shape[0])
    return ObjectiveResult(
        loss=loss,
        directional_losses={
            f"{left_name}->{right_name}": left_to_right,
            f"{right_name}->{left_name}": right_to_left,
        },
        audit=RelationAudit(
            relation=relation,
            batch_size=batch_size,
            positive_terms=2 * batch_size,
            candidates_per_query=batch_size,
            negatives_per_query=batch_size - 1,
        ),
    )


def standard_objective(logit_scale: float | torch.Tensor) -> Objective:
    return lambda batch: bidirectional_contrastive(batch, "I", "T", logit_scale)


def count_matched_mixed_objective(
    logit_scale: float | torch.Tensor,
    relation: str,
) -> Objective:
    try:
        left_name, right_name = {
            "I<->T": ("I", "T"),
            "I<->IT": ("I", "IT"),
            "T<->IT": ("T", "IT"),
        }[relation]
    except KeyError as error:
        raise ValueError(f"Unsupported count-matched relation: {relation}") from error
    return lambda batch: bidirectional_contrastive(batch, left_name, right_name, logit_scale)


def relation_for_optimizer_step(optimizer_step: int) -> str:
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative.")
    return COUNT_MATCHED_RELATION_CYCLE[optimizer_step % len(COUNT_MATCHED_RELATION_CYCLE)]


def additive_multimodal_embedding(image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    if image.shape != text.shape:
        raise ValueError("I and T must have identical shapes before additive fusion.")
    return F.normalize(image + text, dim=-1)
