import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from evaluation.b1_retrieval import bidirectional_retrieval, load_caption_task
from evaluation.trajectory import load_evaluation_run_manifest, resolve_trajectory_snapshots
from scripts.evaluation.run_formal_queue import discover_jobs, POINTS
from embeddings.artifact import sha256_file


class B1RetrievalTests(unittest.TestCase):
    def test_variable_positive_counts_and_both_directions_against_dense_sort(self):
        rng = np.random.default_rng(8)
        images = rng.normal(size=(4, 7)).astype(np.float32)
        texts = rng.normal(size=(13, 7)).astype(np.float32)
        owners = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3])
        result, rows = bidirectional_retrieval(images, texts, owners, query_block=3)
        score = (images / np.linalg.norm(images, axis=1, keepdims=True)) @ (texts / np.linalg.norm(texts, axis=1, keepdims=True)).T
        image_rank = [min(k + 1 for k, t in enumerate(np.argsort(-score[i], kind='stable')) if owners[t] == i) for i in range(4)]
        text_rank = [list(np.argsort(-score[:, t], kind='stable')).index(owners[t]) + 1 for t in range(13)]
        np.testing.assert_array_equal(rows['i2t_ranks'], image_rank)
        np.testing.assert_array_equal(rows['t2i_ranks'], text_rank)
        self.assertEqual(result['I->T']['queries'], 4)
        self.assertEqual(result['T->I']['queries'], 13)
        self.assertAlmostEqual(result['mR'], np.mean([result[d][f'R@{k}'] for d in ['I->T', 'T->I'] for k in [1, 5, 10]]))
        _, other = bidirectional_retrieval(images, texts, owners, query_block=1)
        np.testing.assert_array_equal(rows['i2t_ranks'], other['i2t_ranks'])

    def test_ties_do_not_turn_collapsed_embeddings_into_perfect_retrieval(self):
        result, rows = bidirectional_retrieval(np.ones((3, 2)), np.ones((6, 2)), [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(rows['i2t_ranks'], [1, 3, 5])
        np.testing.assert_array_equal(rows['t2i_ranks'], [1, 1, 2, 2, 3, 3])
        self.assertAlmostEqual(result['I->T']['R@1'], 1 / 3)

    def test_sixth_caption_can_be_the_best_correct_answer(self):
        images = np.eye(2, dtype=np.float32)
        texts = np.array([[0, 1]] * 5 + [[1, 0], [0, 1]], np.float32)
        _, rows = bidirectional_retrieval(images, texts, [0] * 6 + [1])
        self.assertEqual(rows['i2t_ranks'][0], 1)
        self.assertEqual(rows['i2t_best_relevant_caption'][0], 5)

    def test_invalid_relevance_and_nonfinite_vectors_rejected(self):
        for owners in ([0, 2], [0, 0]):
            with self.assertRaises(ValueError): bidirectional_retrieval(np.eye(2), np.eye(2), owners)
        with self.assertRaises(ValueError): bidirectional_retrieval(np.zeros((2, 2)), np.eye(2), [0, 1])


class LiveCheckpointTests(unittest.TestCase):
    def test_running_requires_explicit_opt_in_and_published_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'status': 'running', 'config_sha256': 'cfg', 'config': {
                'run_id': 'clip_standard', 'model_name': 'clip', 'branch': 'standard', 'checkpoint_sha256': 'm0',
                'lcs_probe_manifest_sha256': 'lcs', 'coco_probe_manifest_sha256': 'coco'},
                'checkpoint_policy': {'trajectory_points': [{'label': 'p050', 'optimizer_step': 5, 'progress_fraction': .5}]}}
            (root / 'run_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): load_evaluation_run_manifest(root)
            loaded = load_evaluation_run_manifest(root, allow_running=True)
            with self.assertRaises(FileNotFoundError): resolve_trajectory_snapshots(root, loaded, labels=('p050',), allow_running=True)
            folder = root / 'checkpoints/trajectory'; folder.mkdir(parents=True)
            weight = folder / 'step_00000005_p050_model.pt'; weight.write_bytes(b'fixture')
            metadata = {'path': weight.name, 'checkpoint_kind': 'trajectory_model', 'artifact_sha256': sha256_file(weight),
                        'provenance': {'run_id': 'clip_standard', 'model_name': 'clip', 'branch': 'standard',
                        'optimizer_step': 5, 'progress_fraction': .5, 'm0_checkpoint_sha256': 'm0', 'config_sha256': 'cfg',
                        'probe_manifests': {'lcs': 'lcs', 'coco': 'coco'}}}
            weight.with_suffix('.json').write_text(json.dumps(metadata))
            self.assertEqual(resolve_trajectory_snapshots(root, loaded, labels=('p050',), allow_running=True)[0].optimizer_step, 5)
            metadata['provenance']['run_id'] = 'wrong'
            weight.with_suffix('.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError): resolve_trajectory_snapshots(root, loaded, labels=('p050',), allow_running=True)
            self.assertEqual(json.loads((root / 'run_manifest.json').read_text())['status'], 'running')

    def test_full_queue_waits_for_published_metadata_and_supports_50_100(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); groups = []
            for model in ('clip', 'beit3', 'vista'):
                runs = [f'{model}_branch_{i}' for i in range(9)]
                groups.append({'model': model, 'runs': runs, 'jobs': [
                    {'run': run + '_seed_42', 'output_dir': str(root / 'outputs/training/formal_full_v1' / run / 'seed_42')} for run in runs]})
            plan = {'groups': groups, 'seed': 42, 'max_steps': 15003, 'trajectory_steps': [150, 750, 3001, 7502, 15003]}
            directory = Path(groups[0]['jobs'][0]['output_dir']); directory.mkdir(parents=True)
            manifest = {'status': 'running', 'config': {'run_id': groups[0]['runs'][0] + '_seed_42', 'model_name': 'clip', 'seed': 42},
                        'checkpoint_policy': {'trajectory_points': [dict(label=l, optimizer_step=s) for l, s in zip(POINTS, plan['trajectory_steps'])]}}
            (directory / 'run_manifest.json').write_text(json.dumps(manifest))
            config = root / 'formal.yaml'
            self.assertEqual(len(discover_jobs(plan, config, root)), 5)
            checkpoint = directory / 'checkpoints/trajectory/step_00007502_p050_model.pt'
            checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'fixture')
            self.assertEqual(len(discover_jobs(plan, config, root)), 5)
            checkpoint.with_suffix('.json').write_text(json.dumps({'path': checkpoint.name}))
            self.assertEqual(len(discover_jobs(plan, config, root)), 10)
            final = checkpoint.parent / 'final.pt'; final.write_bytes(b'fixture')
            (directory / 'checkpoints/final.json').write_text(json.dumps({'path': 'checkpoints/trajectory/final.pt'}))
            jobs = discover_jobs(plan, config, root)
            self.assertEqual(len(jobs), 15)
            self.assertEqual(len({j['id'] for j in jobs}), 15)
            self.assertEqual(sum(j['stage'] == 'b1' for j in jobs), 6)


if __name__ == '__main__': unittest.main()
