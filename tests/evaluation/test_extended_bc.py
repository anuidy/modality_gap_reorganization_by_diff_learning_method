import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.extended.classification import class_prototypes, classify_vectors
from evaluation.extended.common import normalized, finish, encoder_dependencies
from evaluation.extended.global_retrieval import fit_calibration, exact_global_ranks, summarize_global
from evaluation.extended.vector_store import encode_shards, ShardedVectors
from evaluation.extended.mbeir_data import build_index, MBeirIndex
from analysis.formal_statistics import paired_seed_summary, c1_comparisons, c2_changes, calibration_gap_changes
from analysis.result_inputs import collect_scope, build_scope
from embeddings.artifact import sha256_file


class ClassificationTests(unittest.TestCase):
    def test_normalize_before_average_and_preserve_duplicate_official_names(self):
        vectors = np.array([[10., 0.], [0., 1.]], np.float32)
        result = class_prototypes(['bird', 'bird'], ['{} one', '{} two'], lambda text: vectors)
        np.testing.assert_allclose(result, np.tile([2**-.5, 2**-.5], (2, 1)), rtol=1e-6)
        self.assertFalse(np.allclose(result[0], normalized(vectors.mean(0)[None])[0]))

    def test_classification_tail_and_ties(self):
        result, top = classify_vectors(np.eye(3), np.eye(3), [0, 1, 2])
        self.assertEqual(result['top1'], 1.)
        _, tied = classify_vectors(np.ones((2, 3)), np.ones((7, 3)), [0, 6])
        np.testing.assert_array_equal(tied, np.tile(np.arange(5), (2, 1)))

    def test_official_template_inventory(self):
        data = json.loads((ROOT / 'configs/evaluation/imagenet_clip_80.json').read_text())
        self.assertEqual(len(data['templates']), 80); self.assertEqual(len(data['class_names']), 1000)
        self.assertEqual(data['class_names'][0], 'tench')


class EncoderDependencyTests(unittest.TestCase):
    def test_tokenizer_and_official_source_changes_invalidate_dependency_identity(self):
        import importlib.util
        from types import SimpleNamespace
        from evaluation.protocol import identity_hash
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); official = root / 'third_party/unilm/beit3'; official.mkdir(parents=True)
            source = official / 'modeling_finetune.py'; source.write_text('class Model: pass\n')
            tokenizer = root / 'beit3.spm'; tokenizer.write_bytes(b'original vocabulary')
            name = '_bc_review_fixture_model'
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
                adapter = SimpleNamespace(model=module.Model(), sentencepiece_model=tokenizer)
                initial = encoder_dependencies(root, 'beit3', adapter)
                self.assertIn('tokenizer/beit3.spm', initial['files'])
                tokenizer.write_bytes(b'different vocabulary')
                changed = encoder_dependencies(root, 'beit3', adapter)
                self.assertNotEqual(identity_hash(initial), identity_hash(changed))
                source.write_text('class Model: pass\n# official implementation change\n')
                self.assertNotEqual(identity_hash(changed), identity_hash(encoder_dependencies(root, 'beit3', adapter)))
            finally: sys.modules.pop(name, None)


class GlobalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.c = rng.normal(size=(13, 5)).astype(np.float32)
        self.q = rng.normal(size=(5, 5)).astype(np.float32)
        self.m = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.gold = [[0, 1], [4], [5, 8], [9, 11], [2]]
        self.tasks = ['a', 'a', 'b', 'b', 'b']

    def test_calibration_matches_explicit_nonrelevant_background_means(self):
        fitted = fit_calibration(self.q, self.c, self.m, self.gold, self.tasks,
                                 ['v' + str(i) for i in range(5)], ['test'], candidate_block=4)
        score = normalized(self.q).astype(np.float64) @ normalized(self.c).astype(np.float64).T
        for task in ('a', 'b'):
            for mode in range(3):
                values = [score[i, (self.m == mode) & ~np.isin(np.arange(13), self.gold[i])].mean()
                          for i in range(5) if self.tasks[i] == task]
                self.assertAlmostEqual(fitted['offsets'][task][mode], np.mean(values), places=7)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit_calibration(self.q, self.c, self.m, self.gold, self.tasks, list('abcde'), ['b'])

    def test_raw_calibrated_oracle_against_dense_independent_sort(self):
        calibration = {'offsets': {'a': [.1, -.2, .3], 'b': [0., .3, -.1]}}
        rows = exact_global_ranks(self.q, self.c, self.m, self.gold, self.tasks, calibration,
                                  query_block=2, candidate_block=4)
        score = normalized(self.q) @ normalized(self.c).T
        for view in ('raw', 'calibrated', 'oracle'):
            expected = []
            for i in range(5):
                values = score[i].copy()
                if view == 'calibrated': values -= np.asarray(calibration['offsets'][self.tasks[i]], np.float32)[self.m]
                if view == 'oracle': values[~np.isin(self.m, self.m[self.gold[i]])] = -np.inf
                order = np.argsort(-values, kind='stable')
                expected.append(min(int(np.flatnonzero(order == gold)[0]) + 1 for gold in self.gold[i]))
            np.testing.assert_array_equal(rows[view]['ranks'], expected)
        self.assertEqual(rows['oracle']['candidate_count'][0], np.count_nonzero(np.isin(self.m, [0, 1])))
        summary = summarize_global(rows, self.tasks)
        self.assertEqual(summary['raw']['tasks']['a']['queries'], 2)
        other = exact_global_ranks(self.q, self.c, self.m, self.gold, self.tasks, calibration, query_block=3, candidate_block=8)
        for view in rows: np.testing.assert_array_equal(rows[view]['ranks'], other[view]['ranks'])

    def test_ties_empty_calibration_positives_and_invalid_relevance(self):
        rows = exact_global_ranks(np.ones((2, 2)), np.ones((6, 2)), [0, 1, 2, 0, 1, 2], [[4], [5]], ['a', 'a'])
        np.testing.assert_array_equal(rows['raw']['ranks'], [5, 6])
        np.testing.assert_array_equal(rows['oracle']['ranks'], [2, 2])
        fit = fit_calibration(np.ones((1, 2)), np.ones((6, 2)), [0, 1, 2, 0, 1, 2], [[]], ['a'], ['val'], ['test'])
        np.testing.assert_allclose(fit['offsets']['a'], [1, 1, 1], atol=1e-6)
        with self.assertRaises(ValueError): exact_global_ranks(self.q, self.c, self.m, [[]] * 5, self.tasks)


class VectorStoreTests(unittest.TestCase):
    def test_resume_preserves_completed_shards_and_refuses_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = [{'id': str(i)} for i in range(11)]; calls = []
            def interrupted(batch):
                calls.append(batch[0]['id'])
                if int(batch[0]['id']) >= 4: raise RuntimeError('interrupted')
                return np.array([[int(x['id']) + 1, 1.] for x in batch], np.float32)
            with self.assertRaises(RuntimeError): encode_shards(rows, 11, interrupted, root, {'source': 'test'}, 2, 4)
            calls.clear()
            def encode(batch):
                calls.append(batch[0]['id']); return np.array([[int(x['id']) + 1, 1.] for x in batch], np.float32)
            store = encode_shards(rows, 11, encode, root, {'source': 'test'}, 2, 4)
            self.assertEqual(calls[0], '4'); self.assertEqual(store.shape, (11, 2))
            np.testing.assert_allclose(store[2:9], normalized(np.array([[i + 1, 1.] for i in range(2, 9)])))
            np.testing.assert_allclose(store[[10, 0]], store[:][[10, 0]])
            with self.assertRaises(ValueError): encode_shards(rows[::-1], 11, encode, root, {'source': 'test'}, 2, 4)
            # Close Windows mmap handles before intentionally corrupting this temporary fixture.
            for value in store.maps: value._mmap.close()
            (root / 'shard_000000.npy').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): ShardedVectors(root)


class StatisticsTests(unittest.TestCase):
    def test_paired_seed_t_interval_and_single_seed_limit(self):
        result = paired_seed_summary({42: 2, 43: 4, 44: 6}, {42: 1, 43: 2, 44: 3})
        self.assertEqual(result['mean_difference'], 2.)
        self.assertEqual(result['sample_std'], 1.)
        np.testing.assert_allclose(result['ci'], [-.4841377117, 4.4841377117], atol=1e-8)
        single = paired_seed_summary({42: 2}, {42: 1}, [42])
        self.assertIsNone(single['ci']); self.assertIsNone(single['sample_std'])
        self.assertEqual(paired_seed_summary({42: 2}, {42: 1})['status'], 'incomplete')

    def test_delta_M0_does_not_divide_by_zero_and_preserves_provenance(self):
        base = dict(model='clip', stage='B1', dataset='coco', metric='mR', comparison_view='same', artifact={'id': 'x'})
        rows = [{**base, 'point': 'm0', 'branch': None, 'seed': None, 'value': 0.},
                {**base, 'point': 'p100', 'branch': 'standard', 'seed': 42, 'value': .2}]
        changes = c2_changes(rows)
        self.assertEqual(changes[0]['delta_M0'], .2)
        self.assertEqual(changes[0]['direction'], 'increase')
        with self.assertRaisesRegex(ValueError, 'Missing matching M0'): c2_changes(rows[1:])

    def test_scope_requires_B3_and_global_and_counts_the_seed42_bridge(self):
        import yaml
        config = yaml.safe_load((ROOT / 'configs/training/formal_single_seed.yaml').read_text())
        scope = build_scope(config, [42], ['outputs/evaluation/formal_full_v1'])
        self.assertEqual(len(scope['required']), 138 * 18)
        multi = build_scope(config, [42, 43, 44], ['unused'])
        self.assertEqual(len(multi['required']), (79 * 5 + 3) * 18)
        with tempfile.TemporaryDirectory() as tmp:
            _, status = collect_scope(scope, Path(tmp))
            self.assertEqual(status['status'], 'incomplete'); self.assertEqual(status['found'], 0)

    def test_calibration_contrast_keeps_signed_and_absolute_gap_changes_distinct(self):
        base = dict(model='clip', point='p100', stage='B2_Global', dataset='mbeir_global', comparison_view='v', seed=42)
        rows = [{**base, 'branch': branch, 'metric': view + '/task/R@1', 'value': value}
                for branch, view, value in [('A', 'raw', .2), ('B', 'raw', .6),
                                            ('A', 'calibrated', .5), ('B', 'calibrated', .6)]]
        result = calibration_gap_changes(rows, [('A', 'B')], [42])[0]
        self.assertAlmostEqual(result['Gap_cal_minus_Gap_raw']['mean_difference'], .3)
        self.assertAlmostEqual(result['absolute_gap_change'][42], -.3)

    def test_vista_seed42_bridge_applies_to_both_C1_and_calibration_gaps(self):
        base = dict(model='vista', point='p100', stage='B2_Global', dataset='mbeir_global', comparison_view='v')
        rows = []
        for branch in ('standard', 'fixed_2m', 'mixed_2m'):
            for seed in ([42] if branch == 'fixed_2m' else [42, 43, 44]):
                for view, value in [('raw', .2), ('calibrated', .4)]:
                    rows.append({**base, 'branch': branch, 'seed': seed, 'metric': view + '/task/R@1', 'value': value})
        contrasts = [('fixed_2m', 'standard'), ('mixed_2m', 'fixed_2m'), ('mixed_2m', 'standard')]
        ordinary = c1_comparisons(rows, contrasts)
        calibrated = calibration_gap_changes(rows, contrasts)
        for record in ordinary + calibrated:
            bridge = 'fixed_2m' in (record['treatment'], record['reference'])
            summary = record.get('Gap_cal_minus_Gap_raw', record)
            self.assertEqual(summary['expected_seeds'], [42] if bridge else [42, 43, 44])
            self.assertEqual(summary['missing_seeds'], [])
            self.assertEqual(summary['status'], 'single_seed_descriptive' if bridge else 'complete')
            if bridge: self.assertIsNone(summary['ci'])
        clip_rows = [{**row, 'model': 'clip'} for row in rows]
        clip = calibration_gap_changes(clip_rows, [contrasts[0]])[0]['Gap_cal_minus_Gap_raw']
        self.assertEqual(clip['status'], 'incomplete')
        self.assertEqual(clip['missing_seeds'], [43, 44])


class MBeirDataTests(unittest.TestCase):
    def test_official_records_relevance_hashes_and_heldout_boundaries(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'raw'; source.mkdir()
            (source / 'images').mkdir(); Image.new('RGB', (2, 2)).save(source / 'images/a.jpg')
            candidates = [dict(did=str(i), modality=m, txt=t, img_path=p) for i, (m, t, p) in enumerate([
                ('image', None, 'images/a.jpg'), ('text', 'first', None), ('image,text', 'third', 'images/a.jpg')])]
            pool = source / 'cand_pool/global/mbeir_union_test_cand_pool.jsonl'; pool.parent.mkdir(parents=True)
            pool.write_text(''.join(json.dumps(r) + '\n' for r in candidates))
            for split in ('val', 'test'):
                query = source / f'query/{split}/mbeir_toy_task0_{split}.jsonl'; query.parent.mkdir(parents=True)
                query.write_text(json.dumps({'qid': split, 'query_modality': 'text', 'query_txt': split + ' input',
                                            'query_img_path': None, 'pos_cand_list': [{'did': '1'}]}) + '\n')
                label = source / f'qrels/{split}/mbeir_toy_task0_{split}_qrels.txt'; label.parent.mkdir(parents=True)
                label.write_text(f'{split} 0 1 1 0\n')
            lock = {p.relative_to(source).as_posix(): sha256_file(p) for p in source.rglob('*')
                    if p.is_file() and p.suffix in ('.jsonl', '.txt')}
            prepared = root / 'prepared'; build_index(source, prepared, lock, minimum_candidates=3)
            index = MBeirIndex(prepared)
            self.assertEqual(index.count('candidates'), 3)
            self.assertEqual(next(index.records('test'))['positives'], [1])
            self.assertEqual(next(index.records('val'))['id'], 'mbeir_toy_task0/val')
            index.validate_images(list(index.records()))
            (source / 'images/a.jpg').write_bytes(b'changed')
            with self.assertRaises(ValueError): index.validate_images(list(index.records()))
            index.close()

    def test_global_runner_end_to_end_and_completed_cache_reuse(self):
        from PIL import Image
        from types import SimpleNamespace
        from evaluation.extended.global_runner import run_global
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'raw'; (source / 'images').mkdir(parents=True)
            Image.new('RGB', (2, 2)).save(source / 'images/a.jpg')
            pool = source / 'cand_pool/global/mbeir_union_test_cand_pool.jsonl'; pool.parent.mkdir(parents=True)
            rows = []
            for i in range(6):
                mode = i % 3
                rows.append({'did': str(i), 'modality': ['image', 'text', 'image,text'][mode],
                             'txt': 'candidate ' + str(i) if mode else None,
                             'img_path': 'images/a.jpg' if mode != 1 else None})
            pool.write_text(''.join(json.dumps(x) + '\n' for x in rows))
            for split in ('val', 'test'):
                path = source / f'query/{split}/mbeir_toy_task0_{split}.jsonl'; path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'qid': split, 'query_modality': 'text', 'query_txt': split,
                                            'query_img_path': None, 'pos_cand_list': ['1']}) + '\n')
                qrel = source / f'qrels/{split}/mbeir_toy_task0_{split}_qrels.txt'; qrel.parent.mkdir(parents=True)
                qrel.write_text(f'{split} 0 1 1 0\n')
            lock = {p.relative_to(source).as_posix(): sha256_file(p) for p in source.rglob('*')
                    if p.is_file() and p.suffix in ('.jsonl', '.txt')}
            prepared = root / 'prepared'; build_index(source, prepared, lock, minimum_candidates=6)
            class FakeEncoder:
                checkpoint = dict(model='clip', point='m0', branch=None, seed=None, step=0, checkpoint_sha256='fixture')
                runner = SimpleNamespace(runtime_identity={'device': 'cpu', 'fixture': True})
                encoder = SimpleNamespace(stats={'fixture': True})
                dependencies = {'tokenizer_sha256': 'original'}
                batch_size = 2; device = 'cpu'; calls = 0
                def pairs(self, batch, image_root):
                    self.calls += 1
                    return np.asarray([[1., len(x.get('text') or ''), int(bool(x.get('image')))] for x in batch], np.float32)
            encoder = FakeEncoder()
            spec = {'minimum_candidates': 6, 'shard_rows': 4, 'candidate_block': 3, 'query_block': 1,
                    'calibration': 'task_by_candidate_modality_negative_mean', 'calibration_split': 'official_val_disjoint_from_test',
                    'oracle': 'union_of_all_relevant_answer_modalities', 'text_policy': 'native_text_no_added_instruction'}
            destination = run_global(ROOT, encoder, prepared, root / 'results', spec)
            summary = json.loads((destination / 'summary.json').read_text())
            self.assertEqual(set(summary['metrics']), {'raw', 'calibrated', 'oracle'})
            calls = encoder.calls
            self.assertEqual(run_global(ROOT, encoder, prepared, root / 'results', spec), destination)
            self.assertEqual(encoder.calls, calls)
            encoder.dependencies = {'tokenizer_sha256': 'changed'}
            changed = run_global(ROOT, encoder, prepared, root / 'results', spec)
            self.assertNotEqual(changed, destination)
            self.assertGreater(encoder.calls, calls)
            # Sharded mappings are scoped to the runner and no longer open on return.


class AnalysisInputTests(unittest.TestCase):
    def test_C_entry_point_refuses_execution_until_all_AB_is_complete(self):
        import subprocess
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scope = root / 'scope.json'; output = root / 'C_results'
            config = yaml.safe_load((ROOT / 'configs/training/formal_single_seed.yaml').read_text())
            scope.write_text(json.dumps(build_scope(config, [42], [str(root / 'empty_results')])))
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/analysis/analyze_formal.py'),
                                     '--scope', str(scope), '--output', str(output), '--execute'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('A/B results are incomplete', result.stderr)
            self.assertFalse(output.exists())

    def test_complete_scope_reads_artifacts_and_rejects_changed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); required = []
            for point, branch, seed, value in [('m0', None, None, .3), ('p100', 'standard', 42, .4)]:
                key = dict(model='clip', branch=branch, seed=seed, point=point, stage='B1', dataset='coco')
                required.append(key); directory = root / 'results' / point; directory.mkdir(parents=True)
                identity = {'stage': 'B1', 'checkpoint': {k: key[k] for k in ['model', 'branch', 'seed', 'point']},
                            'dataset': {'dataset': 'coco', 'split': 'test'}, 'preprocess': 'same'}
                identity['checkpoint'].update(step=0 if point == 'm0' else 15003, checkpoint_sha256=point)
                (directory / 'summary.json').write_text(json.dumps({'identity': identity, 'status': 'complete', 'metrics': {'mR': value}}))
                finish(directory, identity, ['summary.json'])
            scope = {'required': required, 'result_roots': ['results']}
            rows, coverage = collect_scope(scope, root)
            self.assertEqual(coverage['status'], 'complete')
            self.assertAlmostEqual(c2_changes(rows)[0]['delta_M0'], .1)
            (root / 'results/p100/summary.json').write_text('{}')
            _, incomplete = collect_scope(scope, root)
            self.assertEqual(incomplete['status'], 'incomplete')


if __name__ == '__main__': unittest.main()
