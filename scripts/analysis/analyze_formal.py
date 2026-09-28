"""Build C coverage plans or analyze fully completed A/B results; no automatic early C execution."""
import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from analysis.result_inputs import build_scope, collect_scope, validate_full_analysis_scope
from analysis.formal_statistics import c1_comparisons, c2_changes, c2_ordering, calibration_gap_changes
from embeddings.artifact import sha256_file
from evaluation.trajectory import write_json_atomic
from evaluation.extended.common import finish

CONTRASTS = [('fixed_2m', 'standard'), ('mixed_2m', 'fixed_2m'),
             ('fixed_3m_fn_off', 'fixed_3m_fn_on'), ('mixed_3m_fn_off', 'mixed_3m_fn_on'),
             ('full_3m_fn_off', 'full_3m_fn_on'), ('mixed_3m_fn_off', 'fixed_3m_fn_off'),
             ('mixed_3m_fn_on', 'fixed_3m_fn_on'), ('full_3m_fn_off', 'mixed_3m_fn_off'),
             ('full_3m_fn_on', 'mixed_3m_fn_on')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/evaluation/extended_bc.yaml')
    parser.add_argument('--prepare-scope', action='store_true')
    parser.add_argument('--training-config', type=Path, default=ROOT / 'configs/training/formal_single_seed.yaml')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--result-roots', nargs='+', default=['outputs/evaluation/formal_full_v1', 'outputs/evaluation/extended_v1'])
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/analysis/formal_C')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding='utf-8'))['c']
    expected = {'execution': 'only_after_declared_A_B_scopes_complete', 'confidence_level': .95,
                'c1': 'paired_seed_differences_mean_sample_std_student_t',
                'c2': 'current_minus_same_model_M0', 'c3': 'linkage_data_only_no_association_or_significance'}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError('C settings differ from the confirmed analysis protocol')
    if args.prepare_scope:
        if args.execute or args.scope.exists(): raise ValueError('Scope preparation refuses execution/overwrite')
        config = yaml.safe_load(args.training_config.read_text(encoding='utf-8'))
        write_json_atomic(args.scope, build_scope(config, args.seeds or settings['expected_seeds'], args.result_roots))
        print(json.dumps({'status': 'scope_prepared', 'path': str(args.scope)})); return
    scope = json.loads(args.scope.read_text(encoding='utf-8'))
    if not scope.get('requires_all_A_B'): raise ValueError('C execution must require the declared complete A/B scope')
    validate_full_analysis_scope(scope)
    rows, coverage = collect_scope(scope, ROOT)
    print(json.dumps({'status': coverage['status'], 'required': coverage['required'], 'found': coverage['found'], 'execute': args.execute}))
    if not args.execute: return
    if coverage['status'] != 'complete': raise RuntimeError('A/B results are incomplete; C analysis has not run')
    if args.output.exists(): raise FileExistsError('Use a new analysis output directory')
    args.output.mkdir(parents=True)
    changes = c2_changes(rows)
    write_json_atomic(args.output / 'C1.json', {'comparisons': c1_comparisons(rows, CONTRASTS, tuple(scope['seeds'])),
                      'calibration_gap_changes': calibration_gap_changes(rows, CONTRASTS, tuple(scope['seeds']))})
    write_json_atomic(args.output / 'C2.json', {'changes': changes, 'ordering': c2_ordering(changes)})
    with (args.output / 'C3_linkage_data.jsonl').open('w', encoding='utf-8') as handle:
        for row in changes: handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json_atomic(args.output / 'coverage.json', coverage)
    identity = {'scope_sha256': sha256_file(args.scope), 'analysis_config_sha256': sha256_file(args.config), 'C1': 'paired_seed_t95',
                'C2': 'same_model_M0_difference', 'C3': 'data_only_no_statistical_fitting',
                'sources': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in (ROOT / 'src/analysis').glob('*.py')}}
    identity['entrypoint_sha256'] = sha256_file(Path(__file__))
    finish(args.output, identity, ['C1.json', 'C2.json', 'C3_linkage_data.jsonl', 'coverage.json'])
    print(json.dumps({'status': 'complete', 'output': str(args.output)}))


if __name__ == '__main__': main()
