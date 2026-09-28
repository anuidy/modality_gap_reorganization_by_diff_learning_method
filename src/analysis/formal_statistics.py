"""Confirmed C1/C2 statistics; C3 deliberately exports data without fitting a linkage model."""
from collections import defaultdict
import math
from pathlib import Path
import sys

import numpy as np

_optional = Path(__file__).resolve().parents[2] / '_local/extended_eval_dependencies'
if _optional.is_dir() and str(_optional) not in sys.path: sys.path.append(str(_optional))


def paired_seed_summary(treatment, reference, expected_seeds=(42, 43, 44), confidence=.95):
    if not 0 < confidence < 1: raise ValueError('Invalid confidence level')
    seeds = list(expected_seeds)
    if len(seeds) != len(set(seeds)) or not seeds: raise ValueError('Expected seeds must be unique and explicit')
    if (set(treatment) | set(reference)) - set(seeds): raise ValueError('Unexpected training seed')
    present = [seed for seed in seeds if seed in treatment and seed in reference]
    differences = {seed: float(treatment[seed]) - float(reference[seed]) for seed in present}
    if any(not math.isfinite(v) for v in differences.values()): raise ValueError('Nonfinite paired observation')
    values = np.asarray(list(differences.values()), np.float64)
    left = np.asarray([treatment[seed] for seed in present], np.float64)
    right = np.asarray([reference[seed] for seed in present], np.float64)
    missing = [seed for seed in seeds if seed not in present]
    result = {'status': 'incomplete' if missing else ('single_seed_descriptive' if len(values) == 1 else 'complete'),
              'difference': 'treatment_minus_reference', 'expected_seeds': seeds, 'paired_seeds': present,
              'missing_seeds': missing, 'per_seed_difference': differences, 'n_training_seeds': len(values),
              'mean_difference': float(values.mean()) if len(values) else None,
              'treatment_mean': float(left.mean()) if len(left) else None,
              'reference_mean': float(right.mean()) if len(right) else None,
              'treatment_sample_std': float(left.std(ddof=1)) if len(left) > 1 else None,
              'reference_sample_std': float(right.std(ddof=1)) if len(right) > 1 else None,
              'sample_std': float(values.std(ddof=1)) if len(values) > 1 else None,
              'confidence_level': confidence, 'ci': None,
              'ci_method': 'two_sided_Student_t_over_paired_training_seed_differences',
              'interpretation': 'small_seed_count_parametric_approximation; queries_and_checkpoints_are_not_independent_seed_replicates'}
    if len(values) >= 2 and not missing:
        from scipy.stats import t
        half = float(t.ppf((1 + confidence) / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
        result['ci'] = [float(values.mean() - half), float(values.mean() + half)]
    return result


def observation_key(row):
    return tuple(row[k] for k in ('model', 'point', 'stage', 'dataset', 'metric', 'comparison_view'))


def contrast_seeds(model, treatment, reference, expected_seeds):
    """Apply the same declared seed42 bridge to ordinary and calibration contrasts."""
    return (42,) if model == 'vista' and 'fixed_2m' in (treatment, reference) else expected_seeds


def c1_comparisons(rows, contrasts, expected_seeds=(42, 43, 44)):
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row['point'] == 'm0' or row['value'] is None: continue
        bucket = groups[observation_key(row)][row['branch']]
        if row['seed'] in bucket: raise ValueError('Duplicate seed observation')
        bucket[row['seed']] = row['value']
    result = []
    for key, branches in sorted(groups.items()):
        for treatment, reference in contrasts:
            if treatment not in branches and reference not in branches: continue
            paired_seeds = contrast_seeds(key[0], treatment, reference, expected_seeds)
            result.append({**dict(zip(('model', 'point', 'stage', 'dataset', 'metric', 'comparison_view'), key)),
                           'treatment': treatment, 'reference': reference,
                           **paired_seed_summary({s: v for s, v in branches.get(treatment, {}).items() if s in paired_seeds},
                                                 {s: v for s, v in branches.get(reference, {}).items() if s in paired_seeds}, paired_seeds)})
    return result


def c2_changes(rows):
    baselines = {}
    for row in rows:
        if row['point'] != 'm0': continue
        key = tuple(row[k] for k in ('model', 'stage', 'dataset', 'metric', 'comparison_view'))
        if key in baselines: raise ValueError('Duplicate M0 baseline')
        baselines[key] = row
    results = []
    for row in rows:
        if row['point'] == 'm0': continue
        key = tuple(row[k] for k in ('model', 'stage', 'dataset', 'metric', 'comparison_view'))
        if key not in baselines: raise ValueError('Missing matching M0 observation')
        baseline = baselines[key]
        defined = row['value'] is not None and baseline['value'] is not None
        delta = float(row['value'] - baseline['value']) if defined else None
        results.append({**row, 'M0': baseline['value'], 'delta_M0': delta,
                        'direction': None if delta is None else ('increase' if delta > 0 else 'decrease' if delta < 0 else 'unchanged'),
                        'M0_artifact': baseline['artifact'], 'normalization': 'direct_difference; no_division_by_M0'})
    return results


def c2_ordering(changes):
    groups = defaultdict(list)
    for row in changes:
        key = tuple(row[k] for k in ('model', 'seed', 'point', 'stage', 'dataset', 'metric', 'comparison_view'))
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        defined = [r for r in rows if r['delta_M0'] is not None]
        ordered = sorted(defined, key=lambda r: (-r['delta_M0'], r['branch']))
        result.append({**dict(zip(('model', 'seed', 'point', 'stage', 'dataset', 'metric', 'comparison_view'), key)),
                       'order': [{'branch': r['branch'], 'delta_M0': r['delta_M0']} for r in ordered],
                       'meaning': 'descending_numeric_change; not_an_automatic_quality_ranking',
                       'undefined_branches': [r['branch'] for r in rows if r['delta_M0'] is None]})
    return result


def calibration_gap_changes(rows, contrasts, expected_seeds=(42, 43, 44)):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        if row['stage'] != 'B2_Global' or row['point'] == 'm0' or row['value'] is None: continue
        view, metric = row['metric'].split('/', 1)
        if view not in ('raw', 'calibrated'): continue
        key = tuple(row[k] for k in ('model', 'point', 'dataset', 'comparison_view')) + (metric,)
        slot = groups[key][row['branch']][row['seed']]
        if view in slot: raise ValueError('Duplicate calibrated/raw observation')
        slot[view] = row['value']
    result = []
    for key, branches in groups.items():
        for treatment, reference in contrasts:
            raw, calibrated = {}, {}
            paired_seeds = contrast_seeds(key[0], treatment, reference, expected_seeds)
            for seed in paired_seeds:
                a = branches.get(treatment, {}).get(seed, {}); b = branches.get(reference, {}).get(seed, {})
                if set(a) == set(b) == {'raw', 'calibrated'}:
                    raw[seed] = a['raw'] - b['raw']; calibrated[seed] = a['calibrated'] - b['calibrated']
            if not raw: continue
            result.append({**dict(zip(('model', 'point', 'dataset', 'comparison_view', 'metric'), key)),
                           'treatment': treatment, 'reference': reference, 'Gap_raw': raw, 'Gap_cal': calibrated,
                           'absolute_gap_change': {s: abs(calibrated[s]) - abs(raw[s]) for s in raw},
                           'Gap_cal_minus_Gap_raw': paired_seed_summary(calibrated, raw, paired_seeds),
                           'interpretation': 'signed_gap_change_and_absolute_gap_change_are_distinct; no_causal_claim'})
    return result
