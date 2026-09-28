"""Exact full-gallery cosine ranks, mean calibration and modality-oracle diagnostics."""
import numpy as np
import torch

from .common import normalized


def validate_relevance(relevance, count, allow_empty=False):
    result = []
    for indices in relevance:
        row = np.asarray(indices, dtype=np.int64)
        if row.ndim != 1 or (not len(row) and not allow_empty) or len(np.unique(row)) != len(row):
            raise ValueError('Each query needs a nonempty unique relevant-candidate list')
        if np.any(row < 0) or np.any(row >= count): raise ValueError('Relevant candidate outside gallery')
        result.append(row)
    return result


def fit_calibration(queries, candidates, modalities, relevance, tasks, calibration_ids, test_ids,
                    candidate_block=8192, vectors_are_normalized=False):
    """Average per-query negative cosine means for each task/candidate modality.

    Linearity avoids forming Q x C: dot(q, mean(nonrelevant candidates)) is the
    exact mathematical background mean. Statistics accumulate in float64.
    """
    if candidate_block < 1: raise ValueError('candidate_block must be positive')
    queries = np.asarray(queries, np.float32) if vectors_are_normalized else normalized(queries)
    modalities = np.asarray(modalities, dtype=np.int8)
    if len(queries) != len(tasks) or len(queries) != len(calibration_ids): raise ValueError('Calibration row mismatch')
    if len(set(calibration_ids)) != len(calibration_ids) or set(calibration_ids) & set(test_ids):
        raise ValueError('Calibration/test query identities overlap or repeat')
    if modalities.shape != (len(candidates),) or not set(np.unique(modalities)) <= {0, 1, 2}:
        raise ValueError('Invalid candidate modalities')
    relevant = validate_relevance(relevance, len(candidates), allow_empty=True)
    if len(relevant) != len(queries): raise ValueError('Calibration relevance count mismatch')
    sums = np.zeros((3, candidates.shape[1]), np.float64); counts = np.bincount(modalities, minlength=3)
    if np.any(counts == 0): raise ValueError('Global calibration requires all three candidate modalities')
    for start in range(0, len(candidates), candidate_block):
        block = candidates[start:start + candidate_block]
        rows = (np.asarray(block, np.float32) if vectors_are_normalized else normalized(block)).astype(np.float64)
        mode = modalities[start:start + candidate_block]
        for m in range(3): sums[m] += rows[mode == m].sum(axis=0)
    totals = {}; query_counts = {}
    for q, positives, task in zip(queries, relevant, tasks, strict=True):
        block = candidates[positives]
        positive = (np.asarray(block, np.float32) if vectors_are_normalized else normalized(block)).astype(np.float64)
        offsets = []
        for m in range(3):
            selected = modalities[positives] == m
            denominator = int(counts[m]) - int(selected.sum())
            if denominator <= 0: raise ValueError('No nonrelevant candidates for a calibration modality')
            offsets.append(float(np.dot(q.astype(np.float64), (sums[m] - positive[selected].sum(axis=0)) / denominator)))
        totals[task] = totals.get(task, np.zeros(3, np.float64)) + offsets
        query_counts[task] = query_counts.get(task, 0) + 1
    return {'method': 'task_by_candidate_modality_negative_mean',
            'weighting': 'equal_query_weight_within_task; all_nonrelevant_candidates_within_modality',
            'candidate_modality_counts': counts.tolist(), 'query_counts': query_counts,
            'offsets': {task: (value / query_counts[task]).tolist() for task, value in totals.items()}}


@torch.inference_mode()
def exact_global_ranks(queries, candidates, modalities, relevance, tasks, calibration=None,
                       oracle=True, query_block=32, candidate_block=8192, device='cpu', vectors_are_normalized=False):
    """Two identical scoring passes retain exact best-positive ranks with bounded memory."""
    if query_block < 1 or candidate_block < 1: raise ValueError('Invalid scoring block size')
    modalities = np.asarray(modalities, dtype=np.int8)
    if modalities.shape != (len(candidates),) or not set(np.unique(modalities)) <= {0, 1, 2}:
        raise ValueError('Invalid candidate modality array')
    positives = validate_relevance(relevance, len(candidates))
    if len(queries) != len(positives) or len(tasks) != len(queries): raise ValueError('Query metadata count mismatch')
    views = ['raw'] + (['calibrated'] if calibration is not None else []) + (['oracle'] if oracle else [])
    result = {view: {key: np.zeros(len(queries), np.int64) for key in
              ['ranks', 'ties', 'top1_candidate', 'best_relevant_candidate', 'candidate_count']} for view in views}
    counts = np.bincount(modalities, minlength=3)
    for qstart in range(0, len(queries), query_block):
        qend = min(qstart + query_block, len(queries)); n = qend - qstart
        query_rows = queries[qstart:qend]
        q = torch.as_tensor(np.array(query_rows, copy=True) if vectors_are_normalized else normalized(query_rows), device=device)
        allowed = np.zeros((n, 3), bool)
        for i, ids in enumerate(positives[qstart:qend]): allowed[i, np.unique(modalities[ids])] = True
        offsets = np.zeros((n, 3), np.float32)
        if calibration is not None:
            for i, task in enumerate(tasks[qstart:qend]):
                if task not in calibration['offsets']: raise ValueError('Missing calibration task: ' + task)
                offsets[i] = calibration['offsets'][task]
            if not np.isfinite(offsets).all(): raise ValueError('Nonfinite calibration offsets')
        positive_score = {v: np.full(n, -np.inf, np.float32) for v in views}
        positive_index = {v: np.full(n, len(candidates), np.int64) for v in views}
        top_score = {v: np.full(n, -np.inf, np.float32) for v in views}
        top_index = {v: np.full(n, len(candidates), np.int64) for v in views}
        # Identical block boundaries in both passes keep floating-point comparisons consistent.
        for scoring_pass in (0, 1):
            for cstart in range(0, len(candidates), candidate_block):
                cend = min(cstart + candidate_block, len(candidates))
                candidate_rows = candidates[cstart:cend]
                vectors = torch.as_tensor(np.array(candidate_rows, copy=True) if vectors_are_normalized else normalized(candidate_rows), device=device)
                base = (q @ vectors.T).float().cpu().numpy()
                modes = modalities[cstart:cend]; indices = np.arange(cstart, cend)
                for view in views:
                    scores = base - offsets[:, modes] if view == 'calibrated' else base
                    if view == 'oracle': scores = np.where(allowed[:, modes], scores, -np.inf)
                    for i in range(n):
                        row = scores[i]
                        if scoring_pass == 0:
                            best = int(np.argmax(row)); value = row[best]; idx = cstart + best
                            if value > top_score[view][i] or (value == top_score[view][i] and idx < top_index[view][i]):
                                top_score[view][i] = value; top_index[view][i] = idx
                            gold = positives[qstart + i]
                            local = gold[(gold >= cstart) & (gold < cend)]
                            if len(local):
                                values = row[local - cstart]; value = values.max(); idx = int(local[values == value].min())
                                if value > positive_score[view][i] or (value == positive_score[view][i] and idx < positive_index[view][i]):
                                    positive_score[view][i] = value; positive_index[view][i] = idx
                        else:
                            value = positive_score[view][i]; idx = positive_index[view][i]
                            result[view]['ranks'][qstart + i] += np.count_nonzero((row > value) | ((row == value) & (indices < idx)))
                            result[view]['ties'][qstart + i] += np.count_nonzero(row == value)
        for view in views:
            if not np.isfinite(positive_score[view]).all(): raise ValueError('Missing relevant score')
            result[view]['ranks'][qstart:qend] += 1
            result[view]['top1_candidate'][qstart:qend] = top_index[view]
            result[view]['best_relevant_candidate'][qstart:qend] = positive_index[view]
            result[view]['candidate_count'][qstart:qend] = allowed @ counts if view == 'oracle' else len(candidates)
    return result


def summarize_global(rows, tasks):
    tasks = np.asarray(tasks); result = {}
    for view, arrays in rows.items():
        groups = {}
        for task in sorted(set(tasks)):
            ranks = arrays['ranks'][tasks == task]
            groups[str(task)] = {**{f'R@{k}': float(np.mean(ranks <= k)) for k in (1, 5, 10)},
                                 'queries': len(ranks), 'median_rank': float(np.median(ranks))}
        result[view] = {'tasks': groups,
                       'macro': {f'R@{k}': float(np.mean([x[f'R@{k}'] for x in groups.values()])) for k in (1, 5, 10)},
                       'query_weighted': {f'R@{k}': float(np.mean(arrays['ranks'] <= k)) for k in (1, 5, 10)},
                       'recall_definition': 'any_relevant_in_top_k; matches_UniIR_official_evaluator',
                       'tie_policy': 'score_descending_then_canonical_candidate_row'}
    return result
