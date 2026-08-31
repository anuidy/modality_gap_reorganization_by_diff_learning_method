from __future__ import annotations

import contextlib
import random
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.training_pairs import RawTrainingBatch
from objectives.contrastive import (
    ObjectiveResult,
    RepresentationBatch,
    additive_multimodal_embedding,
    count_matched_mixed_objective,
    standard_objective,
)
from training.backends import PreparedBatch, TrainingBackend


def _objective_metrics(prefix: str, result: ObjectiveResult) -> dict[str, torch.Tensor]:
    metrics = {f"{prefix}/loss": result.loss}
    metrics.update(
        {f"{prefix}/{direction}": loss for direction, loss in result.directional_losses.items()}
    )
    return metrics


def relation_validation_metrics(
    *,
    semantic_ids: tuple[str, ...],
    image: torch.Tensor,
    text: torch.Tensor,
    logit_scale: float | torch.Tensor,
    branch: str,
    multimodal: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate one common I<->T exam plus optional Mixed-only diagnostics."""

    image = F.normalize(image, dim=-1)
    text = F.normalize(text, dim=-1)
    common = standard_objective(logit_scale)(
        RepresentationBatch(
            semantic_ids=semantic_ids,
            representations={"I": image, "T": text},
        )
    )
    metrics = _objective_metrics("common/I<->T", common)
    if branch == "standard":
        return metrics
    if branch != "count_matched_mixed":
        raise ValueError(f"Unsupported relation validation branch: {branch}")

    if multimodal is None:
        multimodal = additive_multimodal_embedding(image, text)
    representations = {
        "I": image,
        "T": text,
        "IT": F.normalize(multimodal, dim=-1),
    }
    for relation in ("I<->IT", "T<->IT"):
        diagnostic = count_matched_mixed_objective(logit_scale, relation)(
            RepresentationBatch(
                semantic_ids=semantic_ids,
                representations=representations,
            )
        )
        metrics.update(_objective_metrics(f"diagnostic/{relation}", diagnostic))
    return metrics


def _albef_itc_context(
    backend: TrainingBackend,
    batch: PreparedBatch,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ALBEF native ITC formula without momentum/queue/temperature mutation."""

    model = backend.model  # type: ignore[attr-defined]
    image = batch.images
    text = batch.text_tokens
    temperature = model.temp.clamp(0.001, 0.5)

    image_embeds = model.visual_encoder(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)
    image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
    text_output = model.text_encoder.bert(
        text.input_ids,
        attention_mask=text.attention_mask,
        return_dict=True,
        mode="text",
    )
    text_embeds = text_output.last_hidden_state
    text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)

    image_embeds_m = model.visual_encoder_m(image)
    image_feat_m = F.normalize(model.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
    text_output_m = model.text_encoder_m.bert(
        text.input_ids,
        attention_mask=text.attention_mask,
        return_dict=True,
        mode="text",
    )
    text_feat_m = F.normalize(model.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
    image_feat_all = torch.cat(
        [image_feat_m.transpose(0, 1), model.image_queue.detach()], dim=1
    )
    text_feat_all = torch.cat(
        [text_feat_m.transpose(0, 1), model.text_queue.detach()], dim=1
    )

    sim_i2t_m = image_feat_m @ text_feat_all / temperature
    sim_t2i_m = text_feat_m @ image_feat_all / temperature
    sim_targets = torch.zeros_like(sim_i2t_m)
    sim_targets.fill_diagonal_(1)
    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
    sim_i2t = image_feat @ text_feat_all / temperature
    sim_t2i = text_feat @ image_feat_all / temperature
    loss_i2t = -(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets).sum(dim=1).mean()
    loss_t2i = -(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets).sum(dim=1).mean()
    loss_itc = 0.5 * (loss_i2t + loss_t2i)
    return loss_itc, {
        "image_embeds": image_embeds,
        "image_embeds_m": image_embeds_m,
        "image_atts": image_atts,
        "text_embeds": text_embeds,
        "sim_i2t": sim_i2t,
        "sim_t2i": sim_t2i,
    }


def _albef_full_diagnostics(
    backend: TrainingBackend,
    batch: PreparedBatch,
    alpha: float,
    context: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    model = backend.model  # type: ignore[attr-defined]
    image = batch.images
    text = batch.text_tokens
    image_embeds = context["image_embeds"]
    image_embeds_m = context["image_embeds_m"]
    image_atts = context["image_atts"]
    text_embeds = context["text_embeds"]
    sim_i2t = context["sim_i2t"]
    sim_t2i = context["sim_t2i"]
    batch_size = image.size(0)

    output_pos = model.text_encoder.bert(
        encoder_embeds=text_embeds,
        attention_mask=text.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
        mode="fusion",
    )
    weights_i2t = F.softmax(sim_i2t[:, :batch_size], dim=1)
    weights_t2i = F.softmax(sim_t2i[:, :batch_size], dim=1)
    weights_i2t.fill_diagonal_(0)
    weights_t2i.fill_diagonal_(0)
    negative_image_indices = [
        int(torch.multinomial(weights_t2i[index], 1).item())
        for index in range(batch_size)
    ]
    negative_text_indices = [
        int(torch.multinomial(weights_i2t[index], 1).item())
        for index in range(batch_size)
    ]
    image_embeds_neg = torch.stack(
        [image_embeds[index] for index in negative_image_indices], dim=0
    )
    text_embeds_neg = torch.stack(
        [text_embeds[index] for index in negative_text_indices], dim=0
    )
    text_atts_neg = torch.stack(
        [text.attention_mask[index] for index in negative_text_indices], dim=0
    )
    output_neg = model.text_encoder.bert(
        encoder_embeds=torch.cat([text_embeds, text_embeds_neg], dim=0),
        attention_mask=torch.cat([text.attention_mask, text_atts_neg], dim=0),
        encoder_hidden_states=torch.cat([image_embeds_neg, image_embeds], dim=0),
        encoder_attention_mask=torch.cat([image_atts, image_atts], dim=0),
        return_dict=True,
        mode="fusion",
    )
    vl_embeddings = torch.cat(
        [output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0
    )
    itm_logits = model.itm_head(vl_embeddings)
    itm_labels = torch.cat(
        [
            torch.ones(batch_size, dtype=torch.long, device=image.device),
            torch.zeros(2 * batch_size, dtype=torch.long, device=image.device),
        ],
        dim=0,
    )
    loss_itm = F.cross_entropy(itm_logits, itm_labels)

    input_ids = text.input_ids.clone()
    labels = input_ids.clone()
    probability_matrix = torch.full(
        labels.shape, model.mlm_probability, dtype=torch.float32, device=input_ids.device
    )
    input_ids, labels = model.mask(
        input_ids,
        model.text_encoder.config.vocab_size,
        image.device,
        targets=labels,
        probability_matrix=probability_matrix,
    )
    logits_m = model.text_encoder_m(
        input_ids,
        attention_mask=text.attention_mask,
        encoder_hidden_states=image_embeds_m,
        encoder_attention_mask=image_atts,
        return_dict=True,
        return_logits=True,
    )
    mlm_output = model.text_encoder(
        input_ids,
        attention_mask=text.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
        labels=labels,
        soft_labels=F.softmax(logits_m, dim=-1),
        alpha=alpha,
    )
    return loss_itm, mlm_output.loss


def backend_validation_metrics(
    backend: TrainingBackend,
    batch: PreparedBatch,
    branch: str,
    optimizer_step: int,
) -> dict[str, torch.Tensor]:
    custom = getattr(backend, "validation_metrics", None)
    if callable(custom):
        return dict(custom(batch, branch, optimizer_step))

    if backend.model_name == "clip":
        image = backend.model.encode_image(batch.images)  # type: ignore[attr-defined]
        text = backend.model.encode_text(batch.text_tokens)  # type: ignore[attr-defined]
        return relation_validation_metrics(
            semantic_ids=batch.semantic_ids,
            image=image,
            text=text,
            logit_scale=backend.model.logit_scale.exp(),  # type: ignore[attr-defined]
            branch=branch,
        )
    if backend.model_name == "vista":
        model = backend.model  # type: ignore[attr-defined]
        image = model.encode_image(batch.images)
        text = model.encode_text(batch.text_tokens)
        multimodal = (
            model.encode_mm(batch.images, batch.text_tokens)
            if branch == "count_matched_mixed"
            else None
        )
        return relation_validation_metrics(
            semantic_ids=batch.semantic_ids,
            image=image,
            text=text,
            multimodal=multimodal,
            logit_scale=1.0 / float(model.temperature),
            branch=branch,
        )
    if backend.model_name == "beit3":
        input_ids, padding_mask = batch.text_tokens
        image, text = backend.model(  # type: ignore[attr-defined]
            image=batch.images,
            text_description=input_ids,
            padding_mask=padding_mask,
            only_infer=True,
        )
        return relation_validation_metrics(
            semantic_ids=batch.semantic_ids,
            image=image,
            text=text,
            logit_scale=backend.model.logit_scale.exp(),  # type: ignore[attr-defined]
            branch=branch,
        )
    if backend.model_name == "albef":
        alpha = backend._alpha_for_step(optimizer_step)  # type: ignore[attr-defined]
        loss_itc, albef_context = _albef_itc_context(backend, batch, alpha)
        metrics = {"common/loss": loss_itc, "common/ITC": loss_itc}
        if branch == "itc_only":
            return metrics
        if branch != "full_albef":
            raise ValueError(f"Unsupported ALBEF validation branch: {branch}")
        loss_itm, loss_mlm = _albef_full_diagnostics(
            backend, batch, alpha, albef_context
        )
        metrics.update(
            {
                "diagnostic/ITM": loss_itm,
                "diagnostic/MLM": loss_mlm,
                "diagnostic/full_total": loss_itc + loss_itm + loss_mlm,
            }
        )
        return metrics
    raise ValueError(f"Unsupported validation backend: {backend.model_name}")


@contextlib.contextmanager
def _preserve_rng() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _seed_validation(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_validation(
    *,
    backend: TrainingBackend,
    loader: DataLoader[RawTrainingBatch],
    branch: str,
    optimizer_step: int,
    seed: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    was_training = backend.training
    totals: dict[str, float] = {}
    sample_count = 0
    batch_count = 0
    with _preserve_rng():
        backend.eval()
        try:
            with torch.inference_mode():
                for batch_index, raw_batch in enumerate(loader):
                    fixed_seed = seed + 5_000_003 + batch_index
                    _seed_validation(fixed_seed)
                    prepared = backend.prepare_batch(raw_batch, fixed_seed)
                    with _autocast(device, precision):
                        metrics = backend_validation_metrics(
                            backend, prepared, branch, optimizer_step
                        )
                    weight = len(raw_batch.semantic_ids)
                    for name, value in metrics.items():
                        scalar = float(value.detach().float().cpu())
                        if not np.isfinite(scalar):
                            raise FloatingPointError(
                                f"Non-finite Validation metric {name} at step {optimizer_step}."
                            )
                        totals[name] = totals.get(name, 0.0) + scalar * weight
                    sample_count += weight
                    batch_count += 1
        finally:
            backend.train(was_training)
    return {
        "completed_steps": optimizer_step,
        "sample_count": sample_count,
        "batch_count": batch_count,
        "metrics": {name: value / sample_count for name, value in totals.items()},
        "policy": {
            "common_exam": "I<->T" if backend.model_name != "albef" else "native_ITC_read_only",
            "branch_diagnostics": branch in {"count_matched_mixed", "full_albef"},
            "checkpoint_selection": False,
            "parameter_updates": False,
            "fixed_view_and_model_rng_per_batch": True,
        },
    }
