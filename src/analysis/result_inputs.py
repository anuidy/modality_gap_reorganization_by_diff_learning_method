"""Validate declared A/B coverage and preserve artifact identity in C tables."""
import json
import math
from pathlib import Path

from embeddings.artifact import sha256_file
from evaluation.protocol import identity_hash
from evaluation.extended.common import verified

PROBES = {'lcs_558k_in_domain_10k': 'lcs', 'coco_2017_val_4407': 'coco',
          'coco_2017_val_probe_excluding_karpathy': 'coco', 'coco_2017_val_5k_excluding_karpathy': 'coco'}
AXES = ('model', 'branch', 'seed', 'point', 'stage', 'dataset')


def scalar_rows(payload, prefix=''):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {'identity', 'geometry_protocol', 'norm_protocol', 'queries', 'samples', 'count',
                       'image_candidates', 'text_candidates', 'candidate_count', 'tasks'}: continue
            yield from scalar_rows(value, prefix + '/' + key if prefix else key)
    elif payload is None or (isinstance(payload, (int, float)) and not isinstance(payload, bool)):
        if payload is not None and not math.isfinite(payload): raise ValueError('Nonfinite analysis input metric')
        yield prefix, payload


def identify(summary):
    identity = summary['identity']; stage = identity['stage']
    if stage == 'A0-A6':
        checkpoint = identity['embeddings']['checkpoint']; view = identity['embeddings']['view']
        probe = view['probe_name']
        dataset = PROBES.get(probe)
        if dataset is None:
            if probe.startswith('coco_') and '4407' in probe: dataset = 'coco'
            else: raise ValueError('Unknown A probe identity: ' + probe)
        comparison = {'view': view, 'settings': identity['settings'],
                      'code_sha256': identity['code_sha256'], 'runtime': identity['embeddings']['runtime']}
        metrics = {k: v for k, v in summary.items() if k in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6')}
        stage = 'A'
    else:
        checkpoint = identity['checkpoint']
        if stage == 'B1': dataset = identity['dataset']['dataset']
        elif stage == 'B_Local': dataset = identity['dataset']['task']
        elif stage == 'B3': dataset = 'imagenet'
        elif stage == 'B2_Global': dataset = 'mbeir_global'
        else: return None
        comparison = {k: v for k, v in identity.items() if k not in
                      {'checkpoint', 'calibration_id', 'vector_indices'}}
        # Calibrators and vector contents vary by checkpoint; their definitions must stay fixed.
        metrics = summary['metrics']
        if stage == 'B2_Global':
            metrics = {view: {**{task: value for task, value in item['tasks'].items()},
                             'macro': item['macro'], 'query_weighted': item['query_weighted']} for view, item in metrics.items()}
    key = {'model': checkpoint['model'], 'branch': checkpoint['branch'], 'seed': checkpoint['seed'],
           'point': checkpoint['point'], 'stage': stage, 'dataset': dataset}
    return key, checkpoint, comparison, metrics


def collect_scope(scope, root):
    required = {tuple(row[k] for k in AXES) for row in scope['required']}
    if len(required) != len(scope['required']) or not required: raise ValueError('Scope requirements must be nonempty and unique')
    selected = {}; observations = []; excluded_reference = []
    for folder in scope['result_roots']:
        base = Path(folder); base = base if base.is_absolute() else root / base
        for path in sorted(base.rglob('summary.json')):
            summary = json.loads(path.read_text(encoding='utf-8'))
            if summary.get('status') != 'complete' or 'identity' not in summary: continue
            found = identify(summary)
            if found is None: continue
            key, checkpoint, comparison, metrics = found
            if checkpoint.get('analysis_role') == 'reference_only':
                excluded_reference.append(str(path))
                continue
            lookup = tuple(key[k] for k in AXES)
            if lookup not in required: continue
            if not verified(path.parent, summary['identity']): continue
            if lookup in selected:
                if sha256_file(path) == selected[lookup]['summary_sha256']: continue
                raise ValueError('Multiple incompatible results for a declared observation: ' + str(lookup))
            artifact = {'path': str(path), 'summary_sha256': sha256_file(path),
                        'complete_sha256': sha256_file(path.parent / 'complete.json')}
            selected[lookup] = artifact
            for metric, value in scalar_rows(metrics):
                observations.append({**key, 'step': checkpoint.get('step'), 'checkpoint_sha256': checkpoint['checkpoint_sha256'],
                                     'metric': metric, 'value': value, 'comparison_view': identity_hash(comparison),
                                     'artifact': artifact})
    missing = [dict(zip(AXES, key)) for key in required - set(selected)]
    return observations, {'status': 'complete' if not missing else 'incomplete', 'required': len(required),
                          'found': len(selected), 'missing': sorted(missing, key=lambda x: json.dumps(x, sort_keys=True)),
                          'artifacts': list(selected.values()), 'excluded_reference_only': excluded_reference}


def build_scope(training_config, seeds, result_roots):
    from evaluation.protocol import MMEB_TASKS
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= {42, 43, 44}:
        raise ValueError('Select unique confirmed training seeds')
    states = []; models = set()
    datasets = [('A', 'lcs'), ('A', 'coco'), ('B1', 'coco'), ('B1', 'flickr30k')]
    datasets += [('B_Local', name) for name in MMEB_TASKS] + [('B3', 'imagenet'), ('B2_Global', 'mbeir_global')]
    for name, run in training_config['runs'].items():
        if run['model'] not in ('clip', 'beit3', 'vista'): continue
        models.add(run['model'])
        for seed in seeds:
            # Full multi-seed matrix explicitly retains VISTA Fixed-2M only as seed42 bridge.
            if run['model'] == 'vista' and run['branch'] == 'fixed_2m' and seed != 42: continue
            for point in ['p001', 'p005', 'p020', 'p050', 'p100']:
                states.append({'model': run['model'], 'branch': run['branch'], 'seed': seed, 'point': point})
    states += [{'model': model, 'branch': None, 'seed': None, 'point': 'm0'} for model in sorted(models)]
    return {'schema_version': 1, 'seeds': seeds, 'result_roots': result_roots,
            'required': [{**state, 'stage': stage, 'dataset': dataset} for state in states for stage, dataset in datasets],
            'C3': 'data_only_no_association_or_significance', 'requires_all_A_B': True}


def validate_full_analysis_scope(scope):
    from evaluation.protocol import MMEB_TASKS
    expected = {('A', 'lcs'), ('A', 'coco'), ('B1', 'coco'), ('B1', 'flickr30k'),
                ('B3', 'imagenet'), ('B2_Global', 'mbeir_global')} | {('B_Local', t) for t in MMEB_TASKS}
    states = {}
    for row in scope['required']:
        key = tuple(row[k] for k in ('model', 'branch', 'seed', 'point'))
        states.setdefault(key, set()).add((row['stage'], row['dataset']))
    if not states or any(coverage != expected for coverage in states.values()):
        raise ValueError('Every declared model state must include all A/B1/Local/B3/Global views before C')
    models = {state[0] for state in states}
    if any((m, None, None, 'm0') not in states for m in models): raise ValueError('C scope requires model M0 baselines')
