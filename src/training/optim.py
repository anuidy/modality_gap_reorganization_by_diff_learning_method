from __future__ import annotations

from collections.abc import Collection
from typing import Any

from torch import nn


_NO_WEIGHT_DECAY_NAME_COMPONENTS = frozenset(
    {
        "class_embedding",
        "cls_token",
        "embed_positions",
        "logit_scale",
        "mask_token",
        "pos_embed",
        "position_embedding",
        "position_embeddings",
        "positional_embedding",
        "relative_position_bias_table",
        "temp",
        "temperature",
    }
)


def build_weight_decay_parameter_groups(
    module: nn.Module,
    weight_decay: float,
    no_weight_decay_names: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Build complete, disjoint AdamW decay and no-decay parameter groups."""

    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    explicit_no_decay = {str(name) for name in no_weight_decay_names}
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    seen_parameter_ids: set[int] = set()

    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            raise RuntimeError(f"Trainable parameter was yielded more than once: {name}")
        seen_parameter_ids.add(parameter_id)

        exclude_from_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or not _NO_WEIGHT_DECAY_NAME_COMPONENTS.isdisjoint(name.split("."))
            or name in explicit_no_decay
        )
        if exclude_from_decay:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    if not seen_parameter_ids:
        raise RuntimeError("The training backend exposed no trainable parameters.")

    groups: list[dict[str, Any]] = []
    if decay_parameters:
        groups.append(
            {
                "group_name": "decay",
                "params": decay_parameters,
                "weight_decay": float(weight_decay),
            }
        )
    if no_decay_parameters:
        groups.append(
            {
                "group_name": "no_decay",
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            }
        )

    grouped_parameter_ids = {
        id(parameter)
        for group in groups
        for parameter in group["params"]
    }
    if grouped_parameter_ids != seen_parameter_ids:
        raise RuntimeError("AdamW parameter grouping omitted or duplicated trainable parameters.")
    return groups
