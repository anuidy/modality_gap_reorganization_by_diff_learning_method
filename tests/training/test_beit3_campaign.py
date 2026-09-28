import contextlib
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.training.test_trajectory_only import fixture
from tests.training.test_formal_runtime import SmallEncoder

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('beit3_campaign', ROOT / 'scripts/training/run_beit3_two_epoch.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
from training.engine import run_training
from training.checkpoint_plan import build_checkpoint_plan
from evaluation.trajectory import resolve_trajectory_snapshots


class Beit3CampaignTest(unittest.TestCase):
    def test_adoption_rejects_unfinished_run_and_keeps_verified_results(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            job=SimpleNamespace(config_run_id='beit3_standard',run_id='beit3_standard_seed_42',output_dir=root)
            campaign.atomic(root/'run_manifest.json', {'status':'running','completed_steps':30006,'config':{'run_id':job.run_id}})
            prior={'completed_training':{'beit3_standard':{'steps':30006}}}
            with self.assertRaisesRegex(ValueError,'incomplete'):
                campaign.adopt_finished_jobs([job],prior)
            campaign.atomic(root/'run_manifest.json', {'status':'complete','completed_steps':30006,'config':{'run_id':job.run_id}})
            folder=root/'checkpoints';folder.mkdir()
            weight=folder/'final.pt';weight.write_bytes(b'full recovery')
            campaign.atomic(folder/'final.json',{'checkpoint_kind':'final_full_resume','path':'checkpoints/final.pt',
                'artifact_sha256':campaign.sha(weight),'provenance':{'optimizer_step':30006}})
            complete,results=campaign.adopt_finished_jobs([job],prior)
            self.assertEqual(set(complete),{'beit3_standard'});self.assertFalse(results)
            weight.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity'):
                campaign.adopt_finished_jobs([job],prior)

    def test_evaluation_waits_for_actual_training_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            self.assertFalse(campaign.training_is_settled(p))
            log = p/'train_metrics.jsonl'
            log.write_text(json.dumps({'completed_steps':3})+'\n')
            self.assertFalse(campaign.training_is_settled(p))
            log.write_text(json.dumps({'completed_steps':10})+'\n'+ '{"partially_written":')
            self.assertTrue(campaign.training_is_settled(p))

    def test_config_and_exact_schedule(self):
        jobs = campaign.protocol(ROOT / 'configs/training/beit3_two_epoch.yaml')
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all(not job.config_run_id.endswith("fn_on") for job in jobs))
        self.assertTrue(all("beit3_"+b in campaign.RUNS for b in ["fixed_3m_fn_off","mixed_3m_fn_off","full_3m_fn_off"]))
        p = build_checkpoint_plan(30006, tuple(i/5 for i in range(1,11)), .5, 4, True, .2, progress_reference_steps=15003)
        self.assertEqual({x.label:x.optimizer_step for x in p.trajectory_points}, campaign.POINTS)
        self.assertEqual(p.resume_steps, {7502, 15003, 22505, 30006})
        self.assertFalse(p.trajectory_points[4].is_final)
        self.assertTrue(p.trajectory_points[-1].is_final)
        self.assertEqual(campaign.expected_evaluations({}), 305)
        self.assertEqual(campaign.expected_evaluations({"continuations":{"beit3_standard":{"completed_steps":15003}}}), 285)
        self.assertEqual(p.validation_steps, set(campaign.POINTS.values()))

    def test_two_epoch_save_resume_and_final_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Fixture has 12 samples/batch 6: 2 updates per data epoch. Ten updates
            # exercise five data epochs, and a recovery crossing epoch boundaries.
            c = replace(fixture(root), max_steps=10, trajectory_progress_fractions=(.2,.4,.6,.8,1.),
                        resume_progress_interval=.5, resume_retention=2, save_resume_checkpoints=True,
                        validation_progress_interval=.2)
            def run(**kwargs):
                with patch('training.engine.create_training_backend', side_effect=lambda **kw: SmallEncoder()), \
                     patch('training.engine.validate_formal_data_identity', return_value={'mode':'test'}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    return run_training(c, 'cpu', **kwargs)
            final = run()
            before = torch.load(final, weights_only=False)
            mid = c.output_dir / 'checkpoints/resume/step_00000005.pt'
            half = torch.load(mid, weights_only=False)
            self.assertTrue({'optimizer','rng','stream','model'} <= half.keys())
            self.assertEqual(len(half['optimizer']['state']), len(before['optimizer']['state']))
            self.assertEqual(half['stream']['epoch'], 2)
            manifest = campaign.read(c.output_dir / 'run_manifest.json')
            snapshots = resolve_trajectory_snapshots(c.output_dir, manifest)
            self.assertEqual([s.label for s in snapshots], ['p020','p040','p060','p080','p100'])
            self.assertEqual([s.checkpoint_kind for s in snapshots], ['trajectory_model']*4+['full_resume'])
            self.assertEqual(len(list((c.output_dir/'checkpoints/trajectory').glob('*.pt'))), 4)
            after = torch.load(run(resume_checkpoint=mid), weights_only=False)
            for name, tensor in before['model'].items():
                torch.testing.assert_close(tensor, after['model'][name], rtol=0, atol=0)
            for k, v in before['optimizer']['state'].items():
                for name in v:
                    torch.testing.assert_close(v[name], after['optimizer']['state'][k][name], rtol=0, atol=0)
            self.assertEqual(before['stream'], after['stream'])

    def test_retirement_requires_all_verified_results_and_preserves_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'outputs/training/beit3_two_epoch_v2/beit3_standard_reference/seed_42'
            folder = directory/'checkpoints/trajectory'
            folder.mkdir(parents=True)
            weight = folder/'step_00003001_p020_model.pt'
            weight.write_bytes(b'model')
            full = directory/'checkpoints/resume/step_00015003.pt'
            full.parent.mkdir();full.write_bytes(b'full recovery')
            meta = {'checkpoint_kind':'trajectory_model','path':weight.name,
                    'artifact_sha256':campaign.sha(weight),'provenance':{'optimizer_step':3001}}
            campaign.atomic(weight.with_suffix('.json'),meta)
            campaign.atomic(directory/'run_manifest.json', {'config':{'run_id':'beit3_standard_seed_42','model_name':'beit3','branch':'standard'}})
            completed = {}
            for stage in campaign.STAGES:
                output = root/'eval'/stage;output.mkdir(parents=True)
                identity = {'checkpoint_sha256':meta['artifact_sha256'],'point':'p020','run_id':'beit3_standard_seed_42'}
                block = {'stage':'A0-A6','embeddings':{'checkpoint':identity}} if stage.startswith('a__') else {'stage':'B1','checkpoint':identity}
                campaign.atomic(output/'summary.json', {'identity':block})
                campaign.atomic(output/'complete.json', {'status':'complete','files':{'summary.json':campaign.sha(output/'summary.json')}})
                outputs = [str(output)]
                if stage == 'b__MMEB12':
                    import shutil
                    for i in range(11):
                        extra = root/'eval'/f'local_{i}'
                        shutil.copytree(output, extra)
                        outputs.append(str(extra))
                log = root/(stage+'.log')
                log.write_text(json.dumps({'status':'complete','outputs':outputs})+'\n')
                completed['beit3_standard__p020__'+stage] = {'log':str(log),'outputs':outputs}
            subset = dict(list(completed.items())[:-1])
            self.assertFalse(campaign.retire(directory,'p020',subset,root))
            self.assertTrue(weight.exists())
            first = root/'eval'/campaign.STAGES[0]/'summary.json'
            original = first.read_bytes();first.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):campaign.retire(directory,'p020',completed,root)
            self.assertTrue(weight.exists())
            first.write_bytes(original)
            self.assertTrue(campaign.retire(directory,'p020',completed,root))
            self.assertFalse(weight.exists());self.assertTrue(full.exists())
            self.assertTrue(weight.with_suffix('.json').exists())
            self.assertFalse(campaign.retire(directory,'p200',completed,root))
            self.assertFalse(campaign.retire(directory,'p020',completed,root))
            with self.assertRaises(ValueError):campaign.retire(root/'outside','p020',completed,root)


if __name__ == '__main__':
    unittest.main()
