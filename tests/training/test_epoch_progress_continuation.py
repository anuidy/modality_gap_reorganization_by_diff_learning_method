import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tests.training.test_trajectory_only import fixture
from tests.training.test_formal_runtime import SmallEncoder
from training.engine import run_training, _validate_recording_only_continuation, BatchStream
from datasets.training_pairs import DeterministicEpochSampler
from evaluation.trajectory import resolve_trajectory_snapshots


class EpochProgressContinuationTest(unittest.TestCase):
    def test_epoch_boundary_resume_does_not_reread_previous_epoch(self):
        class Data(torch.utils.data.Dataset):
            def __init__(self): self.reads=[]
            def __len__(self): return 8
            def __getitem__(self,i): self.reads.append(i);return i
        data=Data();sampler=DeterministicEpochSampler(data,42)
        loader=torch.utils.data.DataLoader(data,batch_size=2,sampler=sampler,drop_last=True,
                                           generator=torch.Generator().manual_seed(123))
        stream=BatchStream(loader,sampler,epoch=0,batch_index=4)
        epoch,index,batch=stream.next()
        self.assertEqual((epoch,index),(1,0))
        expected=DeterministicEpochSampler(data,42);expected.set_epoch(1)
        self.assertEqual(data.reads,list(expected)[:2])
        self.assertEqual(batch.tolist(),data.reads)

    def run_model(self, config, **kwargs):
        with patch('training.engine.create_training_backend', side_effect=lambda **kw: SmallEncoder()), \
             patch('training.engine.validate_formal_data_identity', return_value={'mode':'test'}), \
             contextlib.redirect_stdout(io.StringIO()):
            return run_training(config, 'cpu', **kwargs)

    def test_epoch_checkpoints_and_recording_only_resume_are_numerically_equal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = fixture(root)
            rows = json.loads(old.train_manifest.read_text())['samples']
            rows = [{**rows[i % 12], 'sample_id': str(i), 'semantic_id': str(i)} for i in range(60)]
            old.train_manifest.write_text(json.dumps({'samples':rows}))
            old = replace(old, max_steps=20, save_resume_checkpoints=True,
                          trajectory_progress_fractions=(.2,.4,.6,.8,1.),
                          resume_progress_interval=.5, resume_retention=2)
            reference = torch.load(self.run_model(old), weights_only=False)
            origin = old.output_dir/'checkpoints/resume/step_00000010.pt'
            manifest = old.output_dir/'run_manifest.json'
            original_sha = __import__('hashlib').sha256(origin.read_bytes()).hexdigest()
            new = replace(old, output_dir=root/'epoch_progress', progress_reference_steps=10,
                          trajectory_progress_fractions=tuple(i/5 for i in range(1,11)), resume_retention=4)
            after = torch.load(self.run_model(new, resume_checkpoint=origin, resume_source_manifest=manifest), weights_only=False)
            self.assertEqual(original_sha, __import__('hashlib').sha256(origin.read_bytes()).hexdigest())
            for key,value in reference['model'].items():
                torch.testing.assert_close(value, after['model'][key], rtol=0, atol=0)
            for key,values in reference['optimizer']['state'].items():
                for name,value in values.items():
                    torch.testing.assert_close(value, after['optimizer']['state'][key][name], rtol=0, atol=0)
            self.assertEqual(reference['stream'], after['stream'])
            m = json.loads((new.output_dir/'run_manifest.json').read_text())
            self.assertEqual(m['initial_completed_steps'], 10)
            self.assertTrue(m['resume_origin']['training_controls_unchanged'])
            self.assertEqual(m['checkpoint_policy']['progress_unit'],'epoch')
            snapshots = resolve_trajectory_snapshots(new.output_dir,m,labels=('p100','p120','p140','p160','p180','p200'))
            self.assertEqual(snapshots[0].optimizer_step,10)
            self.assertEqual(snapshots[0].checkpoint_kind,'trajectory_model')
            self.assertEqual(snapshots[-1].checkpoint_kind,'full_resume')
            self.assertEqual(snapshots[-1].progress_fraction,2.)
            self.assertFalse((new.output_dir/'checkpoints/trajectory/step_00000002_p020_model.pt').exists())
            self.assertEqual(sorted(p.name for p in (new.output_dir/'checkpoints/resume').glob('*.pt')),['step_00000015.pt','step_00000020.pt'])
            fresh = replace(new, output_dir=root/'fresh_epoch_progress')
            self.run_model(fresh)
            self.assertEqual(sorted(p.name for p in (fresh.output_dir/'checkpoints/resume').glob('*.pt')),
                             ['step_00000005.pt','step_00000010.pt','step_00000015.pt','step_00000020.pt'])
            m = json.loads((fresh.output_dir/'run_manifest.json').read_text())
            self.assertEqual(len(resolve_trajectory_snapshots(fresh.output_dir,m)),10)

            bad = replace(new, output_dir=root/'bad', learning_rate=0.3)
            payload = torch.load(origin,weights_only=False)
            with self.assertRaisesRegex(ValueError,'training controls'):
                _validate_recording_only_continuation(bad,origin,manifest,payload)
            tampered = dict(payload,config_sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                _validate_recording_only_continuation(replace(new,output_dir=root/'other'),origin,manifest,tampered)


if __name__=='__main__':unittest.main()
