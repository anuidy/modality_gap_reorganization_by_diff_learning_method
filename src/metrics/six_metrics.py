from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


GEOMETRY_SEED = 20_260_825
GEOMETRY_PAIR_COUNT = 1_000_000
KNN_K = 10


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


def covariance_gap(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> float:
    image_covariance = _covariance(image_embeddings)
    text_covariance = _covariance(text_embeddings)
    denominator = np.linalg.norm(image_covariance, ord="fro") + np.linalg.norm(text_covariance, ord="fro")
    if denominator == 0:
        raise ValueError("Covariance-gap denominator is zero.")
    return float(np.linalg.norm(image_covariance - text_covariance, ord="fro") / denominator)


def effective_rank(embeddings: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(_covariance(embeddings))
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    total = float(eigenvalues.sum())
    if total == 0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0] / total
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


def save_m0_geometry_reference(
    image_embeddings_l2: np.ndarray,
    text_embeddings_l2: np.ndarray,
    manifest_sha256: str,
    pair_index_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    sample_count = image_embeddings_l2.shape[0]
    pair_index_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
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

    image_pair_cosines = np.sum(image_embeddings_l2[row_indices] * image_embeddings_l2[column_indices], axis=1)
    text_pair_cosines = np.sum(text_embeddings_l2[row_indices] * text_embeddings_l2[column_indices], axis=1)
    np.savez(
        reference_path,
        image_pair_cosines=image_pair_cosines.astype(np.float32),
        text_pair_cosines=text_pair_cosines.astype(np.float32),
        image_knn_indices=_knn_indices(image_embeddings_l2),
        text_knn_indices=_knn_indices(text_embeddings_l2),
        pair_index_path=np.asarray(str(pair_index_path)),
        manifest_sha256=np.asarray(manifest_sha256),
        knn_k=np.asarray(KNN_K, dtype=np.int64),
    )
    return {
        "status": "m0_reference_saved",
        "pair_count": int(len(row_indices)),
        "pair_seed": GEOMETRY_SEED,
        "knn_metric": "neighbor_overlap_at_10",
        "pair_index_path": str(pair_index_path),
        "reference_path": str(reference_path),
    }


def compute_six_metrics(
    image_embeddings_raw: np.ndarray,
    text_embeddings_raw: np.ndarray,
    manifest_sha256: str,
    pair_index_path: Path,
    geometry_reference_path: Path,
    score_block_size: int = 512,
) -> dict[str, Any]:
    if image_embeddings_raw.shape != text_embeddings_raw.shape:
        raise ValueError("Image/text embedding shape mismatch.")
    image_embeddings_l2 = l2_normalize(image_embeddings_raw)
    text_embeddings_l2 = l2_normalize(text_embeddings_raw)
    return {
        "metric_protocol": {
            "covariance": "centered_sample_covariance_ddof_1",
            "raw_embedding_primary": True,
            "normalized_embedding_auxiliary": True,
            "alignment": "paired_l2_distance_after_row_l2_normalization",
            "geometry": {
                "pair_count": GEOMETRY_PAIR_COUNT,
                "pair_seed": GEOMETRY_SEED,
                "knn": "neighbor_overlap_at_10",
            },
            "score_gap": "all_nonpaired_semantic_instances_j_not_equal_i_blockwise",
        },
        "centroid_gap": {
            "raw": centroid_gap(image_embeddings_raw, text_embeddings_raw),
            "l2_normalized": centroid_gap(image_embeddings_l2, text_embeddings_l2),
        },
        "covariance_gap": {
            "raw": covariance_gap(image_embeddings_raw, text_embeddings_raw),
            "l2_normalized": covariance_gap(image_embeddings_l2, text_embeddings_l2),
        },
        "effective_rank": {
            "raw": {"image": effective_rank(image_embeddings_raw), "text": effective_rank(text_embeddings_raw)},
            "l2_normalized": {
                "image": effective_rank(image_embeddings_l2),
                "text": effective_rank(text_embeddings_l2),
            },
        },
        "cross_modal_alignment": cross_modal_alignment(image_embeddings_l2, text_embeddings_l2),
        "intra_modal_geometry_preservation": save_m0_geometry_reference(
            image_embeddings_l2,
            text_embeddings_l2,
            manifest_sha256,
            pair_index_path,
            geometry_reference_path,
        ),
        "score_gap": score_gap(image_embeddings_l2, text_embeddings_l2, block_size=score_block_size),
    }


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
