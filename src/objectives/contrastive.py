from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


RELATION_ENDPOINTS = (("I", "T"), ("I", "IT"), ("T", "IT"))
DIRECTION_ORDER = ("I->T", "T->I", "I->IT", "IT->I", "T->IT", "IT->T")
# regime, number of candidate modalities, keep same-instance non-target
BRANCH_DEFINITIONS = {
    "standard": ("standard", 1, False),
    "fixed_2m": ("fixed", 2, False),
    "fixed_3m_fn_off": ("fixed", 3, False),
    "fixed_3m_fn_on": ("fixed", 3, True),
    "mixed_2m": ("mixed", 2, False),
    "mixed_3m_fn_off": ("mixed", 3, False),
    "mixed_3m_fn_on": ("mixed", 3, True),
    "full_3m_fn_off": ("full", 3, False),
    "full_3m_fn_on": ("full", 3, True),
}


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
    masked_candidates_per_query: int = 0
    direction_query_counts: Mapping[str, int] = field(default_factory=dict)
    group_indices: Mapping[str, list[int]] = field(default_factory=dict)
    candidate_pool_size: int = 0


@dataclass(frozen=True)
class ObjectiveResult:
    loss: torch.Tensor
    directional_losses: Mapping[str, torch.Tensor]
    audit: RelationAudit
    debug_metrics: Mapping[str, torch.Tensor] = field(default_factory=dict)


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
    # All formal branches use the same float32 similarity/CE boundary.
    with torch.autocast(device_type=left.device.type, enabled=False):
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
            direction_query_counts={f"{left_name}->{right_name}": batch_size, f"{right_name}->{left_name}": batch_size},
            candidate_pool_size=batch_size,
        ),
    )


def standard_objective(logit_scale: float | torch.Tensor) -> Objective:
    return lambda batch: bidirectional_contrastive(batch, "I", "T", logit_scale)


def additive_multimodal_embedding(image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """Raw sum; normalization belongs only to the contrastive objective."""
    if image.shape != text.shape:
        raise ValueError("I and T must have identical shapes before additive fusion.")
    return image + text


def balanced_relation_groups(sample_count: int, seed: int) -> dict[str, torch.Tensor]:
    """Local CPU generator keeps group sampling independent of all global RNGs."""
    if sample_count < 3 or sample_count % 3:
        raise ValueError("Mixed requires a batch divisible by three.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator)
    return {f"{a}<->{b}": indices for (a, b), indices in zip(
        RELATION_ENDPOINTS, permutation.chunk(3), strict=True
    )}


def training_objective(
    logit_scale: float | torch.Tensor, branch: str, *, group_seed: int = 0,
) -> Objective:
    if branch not in BRANCH_DEFINITIONS:
        raise ValueError(f"Unknown formal training branch: {branch}")
    if branch == "standard":
        return standard_objective(logit_scale)
    regime, pool_modalities, fn_on = BRANCH_DEFINITIONS[branch]

    def objective(batch: RepresentationBatch) -> ObjectiveResult:
        image, text = _validate_relation_batch(batch, "I", "T")
        sample_count = image.shape[0]
        required = ("I", "T") if branch == "fixed_2m" else ("I", "T", "IT")
        for name in required:
            if name not in batch.representations or batch.representations[name].shape != image.shape:
                raise ValueError(f"Missing or incompatible representation: {name}")
        normalized = {name: F.normalize(batch.representations[name].float(), dim=-1) for name in required}
        scale = _scale_tensor(logit_scale, image)
        all_indices = torch.arange(sample_count, device=image.device)
        relations = RELATION_ENDPOINTS[:1] if regime == "fixed" else RELATION_ENDPOINTS
        groups = balanced_relation_groups(sample_count, group_seed) if regime == "mixed" else {}
        losses, sums, counts, debug = {}, [], {}, {}
        for left_name, right_name in relations:
            ids = groups[f"{left_name}<->{right_name}"].to(image.device) if groups else all_indices
            names = ("I", "T", "IT") if pool_modalities == 3 else (left_name, right_name)
            pool = torch.cat([normalized[name] for name in names], dim=0)
            for query_name, target_name in ((left_name, right_name), (right_name, left_name)):
                direction = f"{query_name}->{target_name}"
                # Keep logits/CE in float32 even while the encoders use BF16.
                with torch.autocast(device_type=image.device.type, enabled=False):
                    logits = scale * normalized[query_name][ids] @ pool.T
                    targets = ids + names.index(target_name) * sample_count
                    rows = torch.arange(len(ids), device=image.device)
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    mask[rows, ids + names.index(query_name) * sample_count] = True
                    if pool_modalities == 3 and not fn_on:
                        third = next(name for name in names if name not in (query_name, target_name))
                        mask[rows, ids + names.index(third) * sample_count] = True
                    per_query = F.cross_entropy(logits.masked_fill(mask, -torch.inf), targets, reduction="none")
                sums.append(per_query.sum())
                losses[direction] = per_query.mean()
                counts[direction] = len(ids)
                with torch.no_grad():
                    negatives = mask.clone()
                    negatives[rows, targets] = True
                    pos = logits[rows, targets]
                    hardest = logits.masked_fill(negatives, -torch.inf).max(dim=1).values
                    debug[f"{direction}/positive_cosine_mean"] = (pos / scale).mean()
                    debug[f"{direction}/hardest_negative_cosine_mean"] = (hardest / scale).mean()
                    debug[f"{direction}/positive_minus_hardest_negative_logit_mean"] = (pos - hardest).mean()
        positive_terms = sum(counts.values())
        masked = 2 if pool_modalities == 3 and not fn_on else 1
        pool_size = pool_modalities * sample_count
        return ObjectiveResult(
            loss=torch.stack(sums).sum() / positive_terms,
            directional_losses=losses,
            debug_metrics=debug,
            audit=RelationAudit(
                relation="I<->T" if regime == "fixed" else "I<->T+I<->IT+T<->IT",
                batch_size=sample_count, positive_terms=positive_terms,
                candidates_per_query=pool_size-masked, negatives_per_query=pool_size-masked-1,
                masked_candidates_per_query=masked, candidate_pool_size=pool_size,
                direction_query_counts=counts,
                group_indices={name: ids.tolist() for name, ids in groups.items()},
            ),
        )
    return objective
