"""Parameter ownership and path expectations for the locked model backends."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import torch


def parameter_group(model: str, name: str, fusion_layer: int = 6) -> str:
    """Use backend-qualified names; unknown parameters never silently pass."""
    name = name.removeprefix("model.")
    if model == "clip":
        if name == "logit_scale":
            return "temperature"
        if name == "visual.proj":
            return "image_projection"
        if name == "text_projection":
            return "text_projection"
        if name.startswith("visual."):
            return "image_encoder"
        if name.startswith(("transformer.", "token_embedding.", "ln_final.")) or name == "positional_embedding":
            return "text_encoder"
    elif model == "vista":
        if name.startswith("bge_pooler."):
            return "unused_pooler"
        if name == "model_visual.logit_scale" or name.startswith("model_visual.text."):
            return "unused_eva_parameters"
        if name.startswith(("model_visual.visual.head.", "model_visual.visual.norm.", "model_visual.visual.fc_norm.")):
            return "unused_eva_output"
        for prefix, group in (
            ("model_visual.visual.", "eva_visual"),
            ("visual_proj.", "visual_projection"),
            ("bge_embeddings.", "bge_embeddings"),
            ("bge_encoder.", "bge_encoder"),
        ):
            if name.startswith(prefix):
                return group
    elif model == "beit3":
        if name == "logit_scale":
            return "temperature"
        if name == "beit3.vision_embed.mask_token":
            return "unused_mask_token"
        for prefix, group in (
            ("vision_head.", "image_projection"), ("language_head.", "text_projection"),
            ("beit3.vision_embed.", "image_embedding"), ("beit3.text_embed.", "text_embedding"),
            ("beit3.encoder.embed_positions.A.", "image_embedding"),
            ("beit3.encoder.embed_positions.B.", "text_embedding"),
        ):
            if name.startswith(prefix):
                return group
        if name.startswith("beit3.encoder."):
            side = "image" if ".A." in name else "text" if ".B." in name else "shared"
            if ".self_attn." in name:
                return f"{side}_attention"
            if ".ffn." in name:
                return f"{side}_ffn"
            if "layer_norm" in name:
                return f"{side}_norm"
    elif model == "albef":
        for prefix in ("visual_encoder_m", "vision_proj_m", "text_encoder_m", "text_proj_m"):
            if name.startswith(prefix + "."):
                return "momentum/" + prefix
        for prefix, group in (
            ("visual_encoder.", "image_encoder"), ("vision_proj.", "image_projection"),
            ("text_proj.", "text_projection"), ("itm_head.", "itm_head"),
            ("text_encoder.cls.", "mlm_head"),
            ("text_encoder.bert.embeddings.", "text_embeddings"),
            ("text_encoder.bert.pooler.", "unused_pooler"),
        ):
            if name.startswith(prefix):
                return group
        if name == "temp":
            return "temperature"
        match = re.match(r"text_encoder\.bert\.encoder\.layer\.(\d+)\.", name)
        if match:
            if int(match[1]) < fusion_layer:
                return "text_encoder"
            return "cross_attention" if ".crossattention." in name else "fusion_layers"
    return "unclassified"


def expectation(model: str, group: str, component: str) -> str:
    if group == "unclassified":
        return "unclassified"
    if group.startswith("unused_"):
        return "inactive"
    if group.startswith("momentum/"):
        return "frozen"
    if model == "albef":
        if component == "ITC" and group in {"fusion_layers", "cross_attention", "itm_head", "mlm_head"}:
            return "inactive"
        if component == "ITM" and group in {"image_projection", "text_projection", "temperature", "mlm_head"}:
            return "inactive"
        if component == "MLM" and group in {"image_projection", "text_projection", "temperature", "itm_head"}:
            return "inactive"
    return "active"


def required_groups(model: str) -> set[str]:
    if model == "clip":
        return {"image_encoder", "text_encoder", "image_projection", "text_projection", "temperature"}
    if model == "vista":
        return {"eva_visual", "visual_projection", "bge_embeddings", "bge_encoder"}
    if model == "beit3":
        return {"image_embedding", "text_embedding", "image_projection", "text_projection", "temperature",
                "image_attention", "text_attention", "image_ffn", "text_ffn", "image_norm", "text_norm"}
    if model == "albef":
        return {"image_encoder", "text_embeddings", "text_encoder", "image_projection", "text_projection",
                "temperature", "fusion_layers", "cross_attention", "itm_head", "mlm_head",
                "momentum/visual_encoder_m", "momentum/vision_proj_m",
                "momentum/text_encoder_m", "momentum/text_proj_m"}
    raise ValueError(f"Unsupported audit model: {model}")


def inspect_gradients(backend: torch.nn.Module, model: str, component: str) -> dict[str, Any]:
    fusion_layer = 6
    if model == "albef":
        fusion_layer = int(backend.model.text_encoder.config.fusion_layer)
    aliases: dict[int, list[str]] = defaultdict(list)
    for name, parameter in backend.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append(name)
    groups: dict[str, Any] = {}
    for name, parameter in backend.named_parameters():
        # Shared MLM decoder/word embedding weights belong to embeddings once;
        # retain all aliases so ITC does not falsely flag an inactive MLM head.
        group = parameter_group(model, name, fusion_layer)
        entry = groups.setdefault(group, {"expectation": expectation(model, group, component), "parameters": []})
        grad = parameter.grad
        norm = None
        if grad is None:
            state = "none"
        else:
            values = grad.detach().coalesce().values() if grad.is_sparse else grad.detach()
            if not bool(torch.isfinite(values).all()):
                state = "nonfinite"
            elif bool(torch.count_nonzero(values)):
                state = "nonzero"
                norm = float(torch.linalg.vector_norm(values.double()).cpu())
            else:
                state, norm = "zero", 0.0
        entry["parameters"].append({"name": name, "aliases": aliases[id(parameter)],
                                    "numel": parameter.numel(), "requires_grad": parameter.requires_grad,
                                    "gradient": state, "l2_norm": norm})
    for group in required_groups(model) - groups.keys():
        groups[group] = {"expectation": expectation(model, group, component), "parameters": []}
    for entry in groups.values():
        params = entry["parameters"]
        counts = {state: sum(p["gradient"] == state for p in params)
                  for state in ("none", "zero", "nonzero", "nonfinite")}
        expected = entry["expectation"]
        if not params or expected == "unclassified" or counts["nonfinite"]:
            status = "fail"
        elif expected == "active":
            status = "pass" if counts["nonzero"] and all(p["requires_grad"] for p in params) else "fail"
        elif expected == "frozen":
            status = "pass" if counts["none"] == len(params) and not any(p["requires_grad"] for p in params) else "fail"
        else:
            status = "pass" if counts["none"] == len(params) else "fail"
        entry.update(counts=counts, status=status)
    return {"status": "pass" if all(g["status"] == "pass" for g in groups.values()) else "fail",
            "groups": groups,
            "non_parameters": {"temperature": {"status": "not_applicable", "value": float(backend.model.temperature)}}
            if model == "vista" else {}}
