from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tests.training import test_epoch_progress_continuation as continuation_tests
from tests.training.test_trajectory_only import fixture
from tests.training.test_beit3_campaign import campaign
from training.engine import _learning_rate, _validate_recording_only_continuation
from evaluation.trajectory import resolve_trajectory_snapshots


class ThirdEpochExtensionTest(unittest.TestCase):
    def test_lr_history_optimizer_and_stream_survive_completed_budget_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=fixture(root)
            rows=json.loads(old.train_manifest.read_text())['samples']
            old.train_manifest.write_text(json.dumps({'samples':[
                {**rows[i%12],'sample_id':str(i),'semantic_id':str(i)} for i in range(60)]}))
            old=replace(old,max_steps=20,progress_reference_steps=10,save_resume_checkpoints=True,
                        trajectory_progress_fractions=tuple(i/5 for i in range(1,11)),
                        resume_progress_interval=.5,resume_retention=4)
            run=continuation_tests.EpochProgressContinuationTest().run_model
            origin=run(old);source=old.output_dir/'run_manifest.json'
            initial=torch.load(origin,weights_only=False)
            extended=replace(old,max_steps=30,scheduler_decay_steps=20,output_dir=root/'extended',
                             trajectory_progress_fractions=tuple(i/5 for i in range(1,16)),resume_retention=6)
            for step in range(20):self.assertEqual(_learning_rate(old,step),_learning_rate(extended,step))
            for step in range(20,30):self.assertEqual(_learning_rate(extended,step),old.learning_rate*old.min_lr_ratio)
            with self.assertRaisesRegex(ValueError,'training controls'):
                _validate_recording_only_continuation(extended,origin,source,initial)
            with self.assertRaisesRegex(ValueError,'LR decay horizon'):
                _validate_recording_only_continuation(replace(extended,scheduler_decay_steps=30),origin,source,initial,extend_budget=True)
            mid=old.output_dir/'checkpoints/resume/step_00000010.pt'
            with self.assertRaisesRegex(ValueError,'final full checkpoint'):
                _validate_recording_only_continuation(extended,mid,source,torch.load(mid,weights_only=False),extend_budget=True)
            result=torch.load(run(extended,resume_checkpoint=origin,resume_source_manifest=source,
                                  extend_training_budget=True),weights_only=False)
            reference=torch.load(run(replace(extended,output_dir=root/'continuous')),weights_only=False)
            for k,v in result['model'].items():torch.testing.assert_close(v,reference['model'][k],rtol=0,atol=0)
            for k,values in result['optimizer']['state'].items():
                self.assertEqual(int(values['step']),30)
                for name,v in values.items():torch.testing.assert_close(v,reference['optimizer']['state'][k][name],rtol=0,atol=0)
            self.assertEqual(result['stream'],reference['stream'])
            m=json.loads((extended.output_dir/'run_manifest.json').read_text())
            self.assertTrue(m['resume_origin']['budget_extension'])
            self.assertTrue(m['resume_origin']['optimizer_data_model_controls_unchanged'])
            snapshots=resolve_trajectory_snapshots(extended.output_dir,m,labels=('p200','p220','p240','p260','p280','p300'))
            self.assertEqual(snapshots[-1].progress_fraction,3.)
            self.assertEqual(snapshots[-1].checkpoint_kind,'full_resume')
            self.assertEqual(sorted(p.name for p in (extended.output_dir/'checkpoints/resume').glob('*.pt')),['step_00000025.pt','step_00000030.pt'])

            retimed=replace(extended,output_dir=root/'reference_only',scheduler_decay_steps=30)
            run(retimed,resume_checkpoint=origin,resume_source_manifest=source,
                extend_training_budget=True,retime_cosine_for_extension=True)
            m=json.loads((retimed.output_dir/'run_manifest.json').read_text())
            self.assertEqual(m['analysis_role'],'reference_only')
            self.assertTrue(m['resume_origin']['cosine_horizon_retimed'])
            self.assertFalse(m['resume_origin']['past_learning_rate_schedule_preserved'])
            self.assertGreater(m['resume_origin']['resume_learning_rate'],old.learning_rate*old.min_lr_ratio)
            self.assertEqual(m['initial_completed_steps'],20)
            from evaluation.diagnostic_runner import DiagnosticRunner
            runner=DiagnosticRunner.__new__(DiagnosticRunner)
            runner.manifest=m;runner.model='clip'
            runner.snapshots={x.label:x for x in resolve_trajectory_snapshots(retimed.output_dir,m,labels=('p300',))}
            self.assertEqual(runner.checkpoint_identity('p300')['analysis_role'],'reference_only')
            # A later ordinary resume must not silently relabel a reference run as formal.
            mid_reference=retimed.output_dir/'checkpoints/resume/step_00000025.pt'
            self.assertEqual(torch.load(mid_reference,weights_only=False)['provenance']['analysis_role'],'reference_only')
            run(retimed,resume_checkpoint=mid_reference)
            continued=json.loads((retimed.output_dir/'run_manifest.json').read_text())
            self.assertEqual(continued['analysis_role'],'reference_only')

    def test_six_branch_third_epoch_schedule(self):
        try:
            campaign.configure_campaign({'max_steps':45009})
            jobs=campaign.protocol(Path(__file__).resolve().parents[2]/'configs/training/beit3_three_epoch.yaml')
            self.assertEqual(len(jobs),6)
            self.assertEqual(campaign.FINAL_LABEL,'p300')
            self.assertEqual(campaign.RESUME_STEPS,(7502,15003,22505,30006,37508,45009))
            self.assertEqual({k:campaign.POINTS[k] for k in ['p220','p240','p260','p280','p300']},
                             {'p220':33007,'p240':36007,'p260':39008,'p280':42008,'p300':45009})
            plan={'continuations':{r:{'completed_steps':30006} for r in campaign.RUNS}}
            self.assertEqual(campaign.expected_evaluations(plan),185)
            self.assertEqual(campaign.expected_evaluations({'continuations':{'beit3_standard':{'completed_steps':30006}}}),410)
            from training.config import load_run_config
            root=Path(__file__).resolve().parents[2]
            c=load_run_config(root/'configs/training/beit3_three_epoch.yaml','beit3_fixed_2m',root,{'seed':42})
            self.assertGreater(_learning_rate(c,30006),3e-6)
            self.assertAlmostEqual(_learning_rate(c,45008),1e-6)
        finally:
            campaign.configure_campaign({'max_steps':30006})

    def test_reference_results_cannot_satisfy_formal_C_scope(self):
        from analysis.result_inputs import collect_scope
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'results';directory.mkdir()
            identity={'stage':'B1','checkpoint':{'model':'beit3','branch':'standard','seed':42,'point':'p300',
                       'step':45009,'checkpoint_sha256':'reference','analysis_role':'reference_only'},
                      'dataset':{'dataset':'coco'}}
            summary={'status':'complete','identity':identity,'metrics':{'mR':.7}}
            p=directory/'summary.json';p.write_text(json.dumps(summary))
            import hashlib
            (directory/'complete.json').write_text(json.dumps({'status':'complete','identity':identity,
                'files':{'summary.json':hashlib.sha256(p.read_bytes()).hexdigest()}}))
            scope={'result_roots':['results'],'required':[{'model':'beit3','branch':'standard','seed':42,
                   'point':'p300','stage':'B1','dataset':'coco'}]}
            observations,coverage=collect_scope(scope,root)
            self.assertFalse(observations);self.assertEqual(coverage['status'],'incomplete')
            self.assertEqual(coverage['excluded_reference_only'],[str(p)])


if __name__=='__main__':unittest.main()
