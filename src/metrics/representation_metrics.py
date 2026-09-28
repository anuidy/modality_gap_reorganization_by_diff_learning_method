"""Representation-metric protocol for probe evaluation.

Raw (pre-L2) embeddings are the primary representation-analysis space and L2
rows are the auxiliary diagnostic space. Per metric:

    raw + l2   Centroid Gap, Covariance Gap, Effective Rank
    l2/cosine  Cross-modal Alignment, Intra-modal Geometry Preservation,
               Score Gap, Anisotropy
    raw only   Norm Imbalance (L2 normalization removes norm magnitude)

Score Gap is a score-level statistic, not representation geometry; its entries
are marked with ``category = score_level``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


GEOMETRY_SEED = 20_260_825
GEOMETRY_PAIR_COUNT = 1_000_000
KNN_K = 10

# Flat keys of the observation schema, in schema order. `compute_point_metrics`
# emits them for every observation (M0 baseline, checkpoint points), so any
# consumer that needs "the numbers" works from this one list.
FLAT_METRIC_KEYS = (
    "centroid_gap_raw",
    "centroid_gap_l2",
    "covariance_gap_raw",
    "covariance_gap_l2",
    "effective_rank_image_raw",
    "effective_rank_text_raw",
    "effective_rank_image_l2",
    "effective_rank_text_l2",
    "cross_modal_alignment",
    "matched_pair_cosine_mean",
    "matched_pair_cosine_std",
    "score_gap_image",
    "score_gap_text",
    "score_gap_mean",
    "image_norm_mean",
    "image_norm_std",
    "text_norm_mean",
    "text_norm_std",
    "image_text_norm_ratio",
    "norm_imbalance",
    "anisotropy_image",
    "anisotropy_text",
    "anisotropy_gap",
    "intra_geometry_image",
    "intra_geometry_text",
)

# Numerical guards. They are recorded in the observation metadata so a reader can
# reproduce exactly which protection was applied.
COVARIANCE_GAP_EPSILON = 1e-12
EFFECTIVE_RANK_EPSILON = 1e-12
ANISOTROPY_BLOCK_SIZE = 512

SCORE_GAP_CATEGORY = "score_level"
REPRESENTATION_CATEGORY = "representation_geometry"


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot L2-normalize embeddings containing zero-norm rows.")
    return (embeddings / norms).astype(np.float32, copy=False)


def _covariance(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] < 2:
        raise ValueError("Sample covariance requires at least two embedding rows.")
    return np.cov(embeddings.astype(np.float64, copy=False), rowvar=False, ddof=1)


def centroid_gap(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> float:
    return float(np.linalg.norm(image_embeddings.mean(axis=0) - text_embeddings.mean(axis=0)))


def covariance_gap(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    epsilon: float = COVARIANCE_GAP_EPSILON,
) -> float:
    """Relative Frobenius covariance gap ||S_I - S_T||_F / (||S_I||_F + ||S_T||_F)."""

    image_covariance = _covariance(image_embeddings)
    text_covariance = _covariance(text_embeddings)
    denominator = float(
        np.linalg.norm(image_covariance, ord="fro") + np.linalg.norm(text_covariance, ord="fro")
    )
    # Numerical protection only: the definition is unchanged, a degenerate
    # denominator is floored instead of producing inf/nan.
    denominator = max(denominator, epsilon)
    return float(np.linalg.norm(image_covariance - text_covariance, ord="fro") / denominator)


def effective_rank(embeddings: np.ndarray, epsilon: float = EFFECTIVE_RANK_EPSILON) -> float:
    """exp(entropy of the normalized covariance eigenvalues) for one modality."""

    eigenvalues = np.linalg.eigvalsh(_covariance(embeddings))
    # Negative eigenvalues are floating-point noise; clamp instead of failing.
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    total = float(eigenvalues.sum())
    if total <= epsilon:
        return 0.0
    probabilities = eigenvalues[eigenvalues > epsilon] / total
    # Entropy over p > epsilon only; the skipped mass is bounded by dim*epsilon.
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.exp(entropy))


def cross_modal_alignment(image_embeddings_l2: np.ndarray, text_embeddings_l2: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(image_embeddings_l2 - text_embeddings_l2, axis=1)
    return {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances, ddof=1)),
        "q25": float(np.quantile(distances, 0.25)),
        "median": float(np.quantile(distances, 0.5)),
        "q75": float(np.quantile(distances, 0.75)),
    }


def matched_pair_cosine_summary(
    image_embeddings_l2: np.ndarray,
    text_embeddings_l2: np.ndarray,
) -> dict[str, float]:
    """Auxiliary statistics of the matched-pair cosine (the alignment driver)."""

    cosines = np.sum(image_embeddings_l2 * text_embeddings_l2, axis=1, dtype=np.float64)
    return {
        "mean": float(cosines.mean()),
        "std": float(cosines.std(ddof=1)) if cosines.size > 1 else 0.0,
    }


def norm_imbalance(
    image_embeddings_raw: np.ndarray,
    text_embeddings_raw: np.ndarray,
) -> dict[str, float]:
    """Raw-embedding norm statistics and the norm imbalance between modalities.

    ``norm_imbalance = |log(E||e_I|| / E||e_T||)|`` and ``image_text_norm_ratio``
    keeps the direction that the absolute value removes. Only raw embeddings are
    meaningful here: after L2 normalization every row norm is 1.
    """

    image_norms = np.linalg.norm(image_embeddings_raw, axis=1)
    text_norms = np.linalg.norm(text_embeddings_raw, axis=1)
    image_norm_mean = float(image_norms.mean(dtype=np.float64))
    text_norm_mean = float(text_norms.mean(dtype=np.float64))
    if image_norm_mean <= 0 or text_norm_mean <= 0:
        raise ValueError("Norm imbalance requires strictly positive mean norms.")
    ratio = image_norm_mean / text_norm_mean
    return {
        "image_norm_mean": image_norm_mean,
        "image_norm_std": float(image_norms.std(ddof=1, dtype=np.float64)) if image_norms.size > 1 else 0.0,
        "text_norm_mean": text_norm_mean,
        "text_norm_std": float(text_norms.std(ddof=1, dtype=np.float64)) if text_norms.size > 1 else 0.0,
        "image_text_norm_ratio": ratio,
        "norm_imbalance": float(abs(np.log(ratio))),
    }


def anisotropy(embeddings_l2: np.ndarray, block_size: int = ANISOTROPY_BLOCK_SIZE) -> float:
    """E_{i != j}[cos(e_i, e_j)] for one modality, computed blockwise.

    Cosine already normalizes direction, so the raw and L2 spaces give the same
    value; taking L2 rows here only makes that explicit. The diagonal (self
    cosine, always 1) is excluded from both the sum and the count.
    """

    sample_count = embeddings_l2.shape[0]
    if sample_count < 2:
        raise ValueError("Anisotropy requires at least two samples.")
    if block_size <= 0:
        raise ValueError("Anisotropy block size must be positive.")
    total = 0.0
    count = 0
    for start in range(0, sample_count, block_size):
        end = min(start + block_size, sample_count)
        similarities = embeddings_l2[start:end] @ embeddings_l2.T
        rows = np.arange(end - start)
        similarities[rows, np.arange(start, end)] = 0.0
        total += float(similarities.sum(dtype=np.float64))
        count += (end - start) * sample_count - (end - start)
    return total / count


def intra_modal_geometry_preservation(
    reference_state_path: Path,
    target_state_path: Path,
) -> dict[str, Any]:
    """Spearman rho of pairwise cosine structure between the M0 reference and one checkpoint.

    Both states are stored on the same fixed upper-triangle pair indices keyed by
    the probe manifest SHA-256, so pair correspondence cannot drift with batch or
    chunk order. Neighbor overlap is auxiliary, not one of the eight metrics.
    """

    comparison = compare_geometry_states(reference_state_path, target_state_path)
    return {
        "image": comparison["image"]["spearman"],
        "text": comparison["text"]["spearman"],
        "auxiliary": {
            "image_neighbor_overlap_k10": comparison["image"]["neighbor_overlap"],
            "text_neighbor_overlap_k10": comparison["text"]["neighbor_overlap"],
            "protocol": comparison["protocol"],
        },
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean(dtype=np.float64)), "std": float(values.std(dtype=np.float64, ddof=1))}


def _directional_score_gap(
    query_embeddings: np.ndarray,
    same_modality_candidates: np.ndarray,
    cross_modality_candidates: np.ndarray,
    block_size: int,
) -> dict[str, Any]:
    """Exact all-j!=i score statistics without allocating a full N x N matrix."""
    sample_count = query_embeddings.shape[0]
    total_pairs = sample_count * (sample_count - 1)
    same_scores = np.empty(total_pairs, dtype=np.float32)
    cross_scores = np.empty(total_pairs, dtype=np.float32)
    offset = 0
    directional_bias_sum = 0.0

    for start in range(0, sample_count, block_size):
        end = min(start + block_size, sample_count)
        within_block = query_embeddings[start:end] @ same_modality_candidates.T
        cross_block = query_embeddings[start:end] @ cross_modality_candidates.T
        valid = np.ones(within_block.shape, dtype=bool)
        valid[np.arange(end - start), np.arange(start, end)] = False
        count = int(valid.sum())
        same_values = within_block[valid]
        cross_values = cross_block[valid]
        same_scores[offset : offset + count] = same_values
        cross_scores[offset : offset + count] = cross_values
        directional_bias_sum += float(np.sum(same_values - cross_values, dtype=np.float64))
        offset += count

    if offset != total_pairs:
        raise RuntimeError("Score-gap candidate count did not match N*(N-1).")
    same_summary = _distribution_summary(same_scores)
    cross_summary = _distribution_summary(cross_scores)
    same_scores.sort()
    cross_scores.sort()
    np.subtract(same_scores, cross_scores, out=same_scores)
    np.abs(same_scores, out=same_scores)
    wasserstein_1 = float(same_scores.mean(dtype=np.float64))
    return {
        "wasserstein_1": wasserstein_1,
        "directional_bias": directional_bias_sum / total_pairs,
        "same_modality_distribution": same_summary,
        "cross_modality_distribution": cross_summary,
        "candidate_pair_count": total_pairs,
    }


def score_gap(image_embeddings_l2: np.ndarray, text_embeddings_l2: np.ndarray, block_size: int = 512) -> dict[str, Any]:
    image_query = _directional_score_gap(
        image_embeddings_l2, image_embeddings_l2, text_embeddings_l2, block_size
    )
    text_query = _directional_score_gap(
        text_embeddings_l2, text_embeddings_l2, image_embeddings_l2, block_size
    )
    return {
        "overall_wasserstein_1": (image_query["wasserstein_1"] + text_query["wasserstein_1"]) / 2,
        "image_query": image_query,
        "text_query": text_query,
    }


def _floyd_sample(total: int, count: int, seed: int) -> np.ndarray:
    """Sample unique integers without allocating the full upper-triangle population."""
    if count > total:
        raise ValueError("Cannot sample more pair indices than the upper triangle contains.")
    generator = np.random.default_rng(seed)
    selected: set[int] = set()
    for value in range(total - count, total):
        draw = int(generator.integers(0, value + 1))
        selected.add(value if draw in selected else draw)
    return np.fromiter(sorted(selected), dtype=np.int64, count=count)


def fixed_upper_triangle_pairs(sample_count: int, count: int = GEOMETRY_PAIR_COUNT, seed: int = GEOMETRY_SEED) -> tuple[np.ndarray, np.ndarray]:
    total = sample_count * (sample_count - 1) // 2
    flat_indices = _floyd_sample(total, count, seed)
    row_starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(np.arange(sample_count - 1, 0, -1, dtype=np.int64)))
    )
    row_indices = np.searchsorted(row_starts, flat_indices, side="right") - 1
    column_indices = row_indices + 1 + (flat_indices - row_starts[row_indices])
    return row_indices.astype(np.int32), column_indices.astype(np.int32)


def _knn_indices(embeddings_l2: np.ndarray, k: int = KNN_K, block_size: int = 512) -> np.ndarray:
    sample_count = embeddings_l2.shape[0]
    if sample_count <= k:
        raise ValueError("kNN requires more samples than k.")
    neighbors = np.empty((sample_count, k), dtype=np.int32)
    for start in range(0, sample_count, block_size):
        end = min(start + block_size, sample_count)
        similarities = embeddings_l2[start:end] @ embeddings_l2.T
        similarities[np.arange(end - start), np.arange(start, end)] = -np.inf
        candidate_indices = np.argpartition(similarities, kth=sample_count - k, axis=1)[:, -k:]
        candidate_scores = np.take_along_axis(similarities, candidate_indices, axis=1)
        ordering = np.argsort(-candidate_scores, axis=1, kind="stable")
        neighbors[start:end] = np.take_along_axis(candidate_indices, ordering, axis=1)
    return neighbors


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(source: np.ndarray, target: np.ndarray) -> float:
    if source.shape != target.shape or source.ndim != 1:
        raise ValueError("Spearman inputs must be same-shaped one-dimensional arrays.")
    source_ranks = _average_ranks(source)
    target_ranks = _average_ranks(target)
    source_centered = source_ranks - source_ranks.mean()
    target_centered = target_ranks - target_ranks.mean()
    denominator = np.linalg.norm(source_centered) * np.linalg.norm(target_centered)
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for a constant rank vector.")
    return float(np.dot(source_centered, target_centered) / denominator)


def neighbor_overlap(source_neighbors: np.ndarray, target_neighbors: np.ndarray) -> float:
    if source_neighbors.shape != target_neighbors.shape or source_neighbors.ndim != 2:
        raise ValueError("Neighbor arrays must share shape [N, K].")
    k = source_neighbors.shape[1]
    if k <= 0:
        raise ValueError("Neighbor overlap requires K > 0.")
    overlap_counts = np.fromiter(
        (
            len(set(source_row.tolist()).intersection(target_row.tolist()))
            for source_row, target_row in zip(source_neighbors, target_neighbors, strict=True)
        ),
        dtype=np.float64,
        count=source_neighbors.shape[0],
    )
    return float(np.mean(overlap_counts / k))


def compare_geometry_states(source_path: Path, target_path: Path) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as source, np.load(
        target_path, allow_pickle=False
    ) as target:
        source_manifest = str(source["manifest_sha256"])
        target_manifest = str(target["manifest_sha256"])
        if source_manifest != target_manifest:
            raise ValueError("Geometry states belong to different probe manifests.")
        source_k = int(source["knn_k"])
        target_k = int(target["knn_k"])
        if source_k != target_k:
            raise ValueError("Geometry states use different kNN neighborhood sizes.")
        return {
            "protocol": {
                "pair_similarity": "cosine_on_fixed_upper_triangle_pairs",
                "pair_correlation": "spearman_with_average_tie_ranks",
                "knn": f"neighbor_overlap_at_{source_k}",
            },
            "manifest_sha256": source_manifest,
            "image": {
                "spearman": spearman_correlation(
                    source["image_pair_cosines"], target["image_pair_cosines"]
                ),
                "neighbor_overlap": neighbor_overlap(
                    source["image_knn_indices"], target["image_knn_indices"]
                ),
            },
            "text": {
                "spearman": spearman_correlation(
                    source["text_pair_cosines"], target["text_pair_cosines"]
                ),
                "neighbor_overlap": neighbor_overlap(
                    source["text_knn_indices"], target["text_knn_indices"]
                ),
            },
        }


def floating_metric_deltas(source: Any, target: Any) -> Any:
    """Return target-minus-source deltas for matching floating metric leaves."""

    if isinstance(source, dict) and isinstance(target, dict):
        result = {
            key: floating_metric_deltas(source[key], target[key])
            for key in source
            if key in target
        }
        return {key: value for key, value in result.items() if value not in ({}, None)}
    if isinstance(source, float) and isinstance(target, float):
        return target - source
    return None


def save_geometry_state(
    image_embeddings_l2: np.ndarray,
    text_embeddings_l2: np.ndarray,
    manifest_sha256: str,
    pair_index_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    """Persist pair cosines and kNN indices for later checkpoint transitions."""

    sample_count = image_embeddings_l2.shape[0]
    pair_index_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if pair_index_path.exists():
        with np.load(pair_index_path, allow_pickle=False) as existing:
            row_indices = existing["row_indices"]
            column_indices = existing["column_indices"]
            saved_manifest = str(existing["manifest_sha256"])
        if saved_manifest != manifest_sha256:
            raise ValueError("Existing geometry pair indices belong to a different probe manifest.")
    else:
        pair_count = min(GEOMETRY_PAIR_COUNT, sample_count * (sample_count - 1) // 2)
        row_indices, column_indices = fixed_upper_triangle_pairs(sample_count, pair_count)
        np.savez(
            pair_index_path,
            row_indices=row_indices,
            column_indices=column_indices,
            manifest_sha256=np.asarray(manifest_sha256),
            seed=np.asarray(GEOMETRY_SEED, dtype=np.int64),
        )

    image_pair_cosines = np.sum(
        image_embeddings_l2[row_indices] * image_embeddings_l2[column_indices], axis=1
    )
    text_pair_cosines = np.sum(
        text_embeddings_l2[row_indices] * text_embeddings_l2[column_indices], axis=1
    )
    np.savez(
        state_path,
        image_pair_cosines=image_pair_cosines.astype(np.float32),
        text_pair_cosines=text_pair_cosines.astype(np.float32),
        image_knn_indices=_knn_indices(image_embeddings_l2),
        text_knn_indices=_knn_indices(text_embeddings_l2),
        pair_index_path=np.asarray(str(pair_index_path)),
        manifest_sha256=np.asarray(manifest_sha256),
        knn_k=np.asarray(KNN_K, dtype=np.int64),
    )
    return {
        "status": "geometry_state_saved",
        "pair_count": int(len(row_indices)),
        "pair_seed": GEOMETRY_SEED,
        "knn_metric": "neighbor_overlap_at_10",
        "pair_index_path": str(pair_index_path),
        "state_path": str(state_path),
    }


def compute_point_metrics(
    image_embeddings_raw: np.ndarray,
    text_embeddings_raw: np.ndarray,
    score_block_size: int = 512,
    *,
    representation_boundary: str = "raw_encoder_output",
    raw_available: bool = True,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Eight-metric observation for one checkpoint x probe pair.

    Raw rows are the primary representation space (centroid gap, covariance gap,
    effective rank, norm imbalance); L2 rows are the auxiliary diagnostic space.
    Alignment, anisotropy and score gap live in the cosine geometry, and norm
    imbalance is never computed on L2 rows because their norms are all 1.
    """

    if image_embeddings_raw.shape != text_embeddings_raw.shape:
        raise ValueError("Image/text embedding shape mismatch.")
    image_embeddings_l2 = l2_normalize(image_embeddings_raw)
    text_embeddings_l2 = l2_normalize(text_embeddings_raw)

    centroid_raw = centroid_gap(image_embeddings_raw, text_embeddings_raw)
    centroid_l2 = centroid_gap(image_embeddings_l2, text_embeddings_l2)
    covariance_raw = covariance_gap(image_embeddings_raw, text_embeddings_raw)
    covariance_l2 = covariance_gap(image_embeddings_l2, text_embeddings_l2)
    rank_raw = {
        "image": effective_rank(image_embeddings_raw),
        "text": effective_rank(text_embeddings_raw),
    }
    rank_l2 = {
        "image": effective_rank(image_embeddings_l2),
        "text": effective_rank(text_embeddings_l2),
    }
    alignment = cross_modal_alignment(image_embeddings_l2, text_embeddings_l2)
    pair_cosines = matched_pair_cosine_summary(image_embeddings_l2, text_embeddings_l2)
    scores = score_gap(image_embeddings_l2, text_embeddings_l2, block_size=score_block_size)
    norms = norm_imbalance(image_embeddings_raw, text_embeddings_raw)
    anisotropy_image = anisotropy(image_embeddings_l2, block_size=score_block_size)
    anisotropy_text = anisotropy(text_embeddings_l2, block_size=score_block_size)
    anisotropy_gap = float(abs(anisotropy_image - anisotropy_text))

    observation: dict[str, Any] = {
        "metric_protocol": {
            "covariance": "centered_sample_covariance_ddof_1",
            "covariance_gap": "relative_frobenius_norm_difference",
            "covariance_gap_epsilon": COVARIANCE_GAP_EPSILON,
            "effective_rank": "exp_entropy_of_normalized_covariance_eigenvalues",
            "effective_rank_epsilon": EFFECTIVE_RANK_EPSILON,
            "anisotropy": "mean_off_diagonal_cosine",
            "anisotropy_block_size": score_block_size,
            "raw_embedding_primary": True,
            "normalized_embedding_auxiliary": True,
            "alignment": "paired_l2_distance_after_row_l2_normalization",
            "norm_imbalance": "abs_log_of_mean_raw_row_norm_ratio",
            "intra_geometry": "spearman_of_fixed_upper_triangle_pair_cosines_against_m0",
            "geometry": {
                "pair_count": GEOMETRY_PAIR_COUNT,
                "pair_seed": GEOMETRY_SEED,
                "knn": "neighbor_overlap_at_10",
            },
            "score_gap": "all_nonpaired_semantic_instances_j_not_equal_i_blockwise",
            "categories": {
                "centroid_gap": REPRESENTATION_CATEGORY,
                "covariance_gap": REPRESENTATION_CATEGORY,
                "effective_rank": REPRESENTATION_CATEGORY,
                "cross_modal_alignment": REPRESENTATION_CATEGORY,
                "intra_modal_geometry_preservation": REPRESENTATION_CATEGORY,
                "anisotropy": REPRESENTATION_CATEGORY,
                "norm_imbalance": REPRESENTATION_CATEGORY,
                "score_gap": SCORE_GAP_CATEGORY,
            },
        },
        "representation_boundary": representation_boundary,
        "raw_available": bool(raw_available),
        # Flat schema: one scalar per metric, as consumed by the trajectory plots.
        "centroid_gap_raw": centroid_raw,
        "centroid_gap_l2": centroid_l2,
        "covariance_gap_raw": covariance_raw,
        "covariance_gap_l2": covariance_l2,
        "effective_rank_image_raw": rank_raw["image"],
        "effective_rank_text_raw": rank_raw["text"],
        "effective_rank_image_l2": rank_l2["image"],
        "effective_rank_text_l2": rank_l2["text"],
        "cross_modal_alignment": alignment["mean"],
        "matched_pair_cosine_mean": pair_cosines["mean"],
        "matched_pair_cosine_std": pair_cosines["std"],
        "score_gap_image": scores["image_query"]["wasserstein_1"],
        "score_gap_text": scores["text_query"]["wasserstein_1"],
        "score_gap_mean": scores["overall_wasserstein_1"],
        **norms,
        "anisotropy_image": anisotropy_image,
        "anisotropy_text": anisotropy_text,
        "anisotropy_gap": anisotropy_gap,
        # Legacy nested keys: existing artifacts, validators and delta helpers
        # read these, so both schemas are emitted.
        "centroid_gap": {"raw": centroid_raw, "l2_normalized": centroid_l2},
        "covariance_gap": {"raw": covariance_raw, "l2_normalized": covariance_l2},
        "effective_rank": {"raw": rank_raw, "l2_normalized": rank_l2},
        "cross_modal_alignment_distribution": dict(alignment),
        "score_gap": {**scores, "category": SCORE_GAP_CATEGORY},
        "auxiliary_metrics": {
            "cross_modal_alignment_distribution": dict(alignment),
            "matched_pair_cosine": dict(pair_cosines),
            "score_gap_details": {
                "image_query": scores["image_query"],
                "text_query": scores["text_query"],
            },
            "anisotropy_gap": anisotropy_gap,
            "anisotropy_space": "cosine_geometry_raw_equals_l2",
            "intra_geometry_reference": "m0",
            "score_gap_category": SCORE_GAP_CATEGORY,
        },
    }
    if identity is not None:
        observation.update(dict(identity))
    return observation


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
