import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from evaluation.protocol import ASettings,MMEB_GROUPS,MMEB_TASKS
from evaluation.score_analysis import analyze_scores,fractional_exposure,low_temperature_diagnostic
from evaluation.representation_analysis import norm_dynamics,analyze_representation
from evaluation.retrieval import rank_candidates,aggregate_tasks,clean_text,load_mmeb_task,cached_mmeb_task
from embeddings.artifact import sha256_file
from evaluation.diagnostic_runner import complete_artifact,finish_artifact
from evaluation.trajectory import load_evaluation_run_manifest,resolve_trajectory_snapshots
from evaluation.probe_views import derive_coco_probe
from metrics.representation_metrics import score_gap,l2_normalize


class ScoreAnalysisTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123)
        image=rng.normal(size=(11,5)).astype(np.float32)
        text=rng.normal(size=(11,5)).astype(np.float32)
        self.raw={'I':image,'T':text,'IT':image+text}
        self.settings=ASettings(.1,query_block=4,geometry_pair_count=30)

    def test_dense_oracle_all_directions_and_background_masks(self):
        summary,rows,plots=analyze_scores(self.raw,self.settings,50.)
        names=('I','T','IT');views={m:l2_normalize(v) for m,v in self.raw.items()};n=11
        pool=np.concatenate([views[m] for m in names])
        for qi,q in enumerate(names):
            scores=views[q]@pool.T
            for i in range(n):
                scores[i,qi*n+i]=-np.inf
                wrong=scores[i].copy();wrong[np.arange(3)*n+i]=-np.inf
                p=[scores[i,names.index(t)*n+i] for t in names if t!=q]
                self.assertAlmostEqual(rows[f'A3/{q}/M_all'][i],min(p)-np.max(wrong),places=5)
                self.assertAlmostEqual(rows[f'A3/{q}/M_first'][i],max(p)-np.max(wrong),places=5)
                for t in names:
                    if t==q:continue
                    positive=scores[i,names.index(t)*n+i]
                    self.assertEqual(rows[f'A4/{q}->{t}/rank'][i],1+np.sum(scores[i]>positive))
                    logits=scores[i].astype(np.float64)*10;peak=np.max(logits)
                    ce=peak+np.log(np.exp(logits-peak).sum())-positive*10
                    self.assertAlmostEqual(rows[f'A6/fixed/{q}->{t}/CE'][i],ce,places=5)
            for a,b in [('I','T'),('I','IT'),('T','IT')]:
                for i in range(n):
                    va=np.delete(views[q][i]@views[a].T,i)
                    vb=np.delete(views[q][i]@views[b].T,i)
                    self.assertAlmostEqual(rows[f'A2/{q}/{a}-{b}/W1'][i],np.mean(np.abs(np.sort(va)-np.sort(vb))),places=5)
        self.assertEqual(summary['A1']['I->T']['negative']['count'],110)
        self.assertEqual(plots['A1/I->T/negative_histogram'].sum(),110)

    def test_block_size_and_temperature_do_not_change_cosine_statistics(self):
        one,r1,_=analyze_scores(self.raw,ASettings(.1,query_block=1),50.)
        many,r2,_=analyze_scores(self.raw,ASettings(.2,query_block=20),50.)
        for key in r1:
            if not key.startswith('A6/fixed'):
                np.testing.assert_allclose(r1[key],r2[key],rtol=1e-5,atol=3e-6,err_msg=key)
        self.assertFalse(np.allclose(r1['A6/fixed/I->T/CE'],r2['A6/fixed/I->T/CE']))

    def test_old_pooled_score_gap_is_preserved_and_distinct_from_A2(self):
        summary,rows,_=analyze_scores(self.raw,self.settings,50.)
        original=score_gap(l2_normalize(self.raw['I']),l2_normalize(self.raw['T']),3)
        self.assertAlmostEqual(summary['global_score_gap']['I']['W1'],original['image_query']['wasserstein_1'],places=6)
        self.assertNotAlmostEqual(summary['global_score_gap']['I']['W1'],rows['A2/I/I-T/W1'].mean(),places=4)

    def test_ties_do_not_invent_background_modality_bias(self):
        raw={m:np.ones((8,4),np.float32) for m in ('I','T','IT')}
        summary,rows,_=analyze_scores(raw,ASettings(.1),100.)
        for q in ('I','T','IT'):
            for m in ('I','T','IT'):
                np.testing.assert_allclose(rows[f'A5/{q}/background/k5/{m}/P'],1/3)
                np.testing.assert_allclose(rows[f'A5/{q}/background/k5/{m}/bias'],0.,atol=1e-15)
        np.testing.assert_array_equal(rows['A4/I->T/rank'],1)
        np.testing.assert_array_equal(rows['A4/I->T/pessimistic_rank'],23)

    def test_zero_norm_refused_and_all_tail_queries_present(self):
        _,rows,_=analyze_scores(self.raw,self.settings,50.)
        self.assertTrue(all(len(v)==11 for v in rows.values()))
        bad={k:v.copy() for k,v in self.raw.items()};bad['I'][0]=0
        with self.assertRaisesRegex(ValueError,'Zero-norm'):analyze_scores(bad,self.settings,50.)

    def test_self_comparison_A0_preserves_geometry_and_norm_delta(self):
        scores,_,_=analyze_scores(self.raw,self.settings,50.)
        summary,arrays=analyze_representation({**self.raw,'IT_definition':'raw_sum'},self.raw,scores,self.settings)
        self.assertAlmostEqual(summary['intra_geometry_image'],1.)
        self.assertAlmostEqual(summary['intra_geometry_text'],1.)
        np.testing.assert_array_equal(arrays['norm/delta_out'],0.)
        self.assertIn('D_IT_additive_diagnostic',summary)

    def test_constant_collapse_is_recorded_with_undefined_correlation(self):
        raw={m:np.ones((11,5),np.float32) for m in ('I','T','IT')}
        scores,_,_=analyze_scores(raw,self.settings,50.)
        result,_=analyze_representation(raw,raw,scores,self.settings)
        self.assertEqual(result['effective_rank_image_raw'],0.)
        self.assertIsNone(result['intra_geometry_image'])


class NormAndRetrievalTests(unittest.TestCase):
    def test_norm_ratio_and_outward_direction_are_per_sample(self):
        i0=np.array([[4.,0.],[1.,0.]])
        t0=np.array([[1.,0.],[2.,0.]])
        image=np.array([[8.,0.],[.5,0.]])
        summary,rows=norm_dynamics(image,t0,i0,t0)
        np.testing.assert_allclose(rows['delta_out'],np.log(2.))
        self.assertAlmostEqual(summary['r']['mean'],(np.log(8)+np.log(.25))/2)
        self.assertNotAlmostEqual(summary['r']['mean'],np.log(image[:,0].mean()/t0[:,0].mean()))
        self.assertAlmostEqual(summary['r']['std'],np.std([np.log(8),np.log(.25)],ddof=1))

    def test_native_candidates_duplicates_and_ties(self):
        vectors=np.array([[1,0],[0,1],[1,0],[-1,0]],np.float32)
        queries=np.array([0,1,3]);candidates=np.array([[2,0,1],[0,1,2],[0,2,3]])
        summary,ranks,ties=rank_candidates(vectors,queries,candidates,query_block=2)
        np.testing.assert_array_equal(ranks,[1,2,2]);np.testing.assert_array_equal(ties,[2,2,2])
        self.assertAlmostEqual(summary['R@1'],1/3)

    def test_partial_task_groups_are_not_reported_as_complete(self):
        values={'MSCOCO_t2i':{'R@1':.2,'R@5':.5,'R@10':.7}}
        summary=aggregate_tasks(values)
        self.assertEqual(summary['status'],'incomplete');self.assertNotIn('all_12',summary['groups'])
        self.assertIn('VisDial',MMEB_GROUPS['cross_modal'])
        self.assertIn('OVEN',MMEB_GROUPS['mixed_composite'])

    def test_template_cleanup_keeps_actual_composite_content(self):
        self.assertIsNone(clean_text('Represent the given image.'))
        self.assertEqual(clean_text('Represent the given image with text: <|image_1|> dog'),'Represent the given image with text: dog')

    def test_small_parquet_adapter_preserves_row_candidate_order(self):
        try:import pyarrow as pa;import pyarrow.parquet as pq
        except ImportError:self.skipTest('pyarrow unavailable locally; server test required')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'CIRR').mkdir()
            rows=[{'qry_text':'query','qry_img_path':'','tgt_text':['correct','wrong','correct'],
                   'tgt_img_path':['','','']} for _ in range(2)]
            pq.write_table(pa.Table.from_pylist(rows),root/'CIRR/test-0.parquet')
            task=load_mmeb_task(root,'CIRR',expected_queries=2,expected_candidates=3)
            self.assertEqual(task.candidates[0,0],task.candidates[0,2])
            self.assertEqual(task.pairs[task.candidates[0,0]],('correct',None))

    def test_verified_prepared_cache_reuse_and_source_change_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';(root/'CIRR').mkdir(parents=True)
            source=root/'CIRR/test-0.parquet';source.write_bytes(b'fixture-source')
            legacy=Path(tmp)/'prepared';legacy.mkdir();path=legacy/'CIRR.pt'
            identity={'task':'CIRR','text_mode':'official',
                'legacy_code_sha256':'a145af7453f0c76140ca2ca7a3ef6416a6b177722fc4a388e4edcfb79a00cd4b',
                'sources':{'data/raw/mmeb_eval/CIRR/test-0.parquet':sha256_file(source)}}
            torch.save({'identity':identity,'pairs':[('answer',None)],'queries':torch.zeros(1000,dtype=torch.long),
                        'candidates':torch.zeros((1000,1000),dtype=torch.long)},path)
            path.with_suffix('.json').write_text(json.dumps({**identity,'cache_sha256':sha256_file(path)}))
            task=cached_mmeb_task(root,'CIRR',Path(tmp)/'cache',legacy)
            self.assertEqual(task.candidates.shape,(1000,1000));self.assertEqual(task.pairs,[('answer',None)])
            source.write_bytes(b'changed-source')
            with self.assertRaisesRegex(ValueError,'source/loader'):cached_mmeb_task(root,'CIRR',Path(tmp)/'cache',legacy)


class IdentityAndPartialTests(unittest.TestCase):
    def test_cancelled_full_fn_on_runs_leave_225_evaluation_jobs(self):
        from scripts.evaluation.run_diagnostic_queue import make_jobs,expected_job_count
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan={'jobs':[]}
            branches=['standard','fixed_2m','fixed_3m_fn_off','fixed_3m_fn_on',
                      'mixed_2m','mixed_3m_fn_off','mixed_3m_fn_on','full_3m_fn_off','full_3m_fn_on']
            for branch in branches:
                for model in ('clip','beit3','vista'):
                    directory=root/'outputs/training'/f'{model}_{branch}'/'seed_42';directory.mkdir(parents=True)
                    (directory/'run_manifest.json').write_text(json.dumps({'status':'paused','completed_steps':3001}))
                    plan['jobs'].append({'model':model,'run_id':f'{model}_{branch}_seed_42','output_dir':str(directory)})
            original_ids={j['id'] for j in make_jobs(plan,root)}
            self.assertEqual(expected_job_count(plan),252)
            plan['jobs']=[j for j in plan['jobs'] if '_full_3m_fn_on_' not in j['run_id']]
            jobs=make_jobs(plan,root)
            self.assertEqual(expected_job_count(plan),225)
            self.assertEqual(len(jobs),225)
            self.assertEqual(len({j['id'] for j in jobs}),225)
            self.assertTrue({j['id'] for j in jobs}.issubset(original_ids))
            self.assertFalse(any('_full_3m_fn_on_' in j['id'] for j in jobs))

    def test_queue_covers_all_points_and_shares_M0_without_starting_running_tasks(self):
        from scripts.evaluation.run_diagnostic_queue import make_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan={'jobs':[]}
            for branch in range(9):
                for model in ('clip','beit3','vista'):
                    directory=root/'outputs/training'/f'{model}_{branch}'/'seed_42';directory.mkdir(parents=True)
                    (directory/'run_manifest.json').write_text(json.dumps({'status':'paused','completed_steps':3001}))
                    plan['jobs'].append({'model':model,'run_id':f'{model}_{branch}_seed_42','output_dir':str(directory)})
            jobs=make_jobs(plan,root)
            self.assertEqual(len(jobs),252);self.assertEqual(len({j['id'] for j in jobs}),252)
            self.assertEqual(sum(j['point']=='m0' for j in jobs),9)
            first=Path(plan['jobs'][0]['output_dir'])/'run_manifest.json'
            first.write_text(json.dumps({'status':'running','completed_steps':None}))
            selected=make_jobs(plan,root)
            self.assertFalse(any(j['id'].startswith('clip_0_seed_42') for j in selected))

    def test_artifact_cache_accepts_json_tuple_roundtrip_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'data.txt').write_text('verified')
            identity={'settings':ASettings(.1).identity()}
            finish_artifact(d,identity,['data.txt']);self.assertTrue(complete_artifact(d,identity))
            (d/'data.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):complete_artifact(d,identity)

    def test_partial_checkpoint_selection_does_not_require_future_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);folder=d/'checkpoints/trajectory';folder.mkdir(parents=True)
            cfg={'run_id':'clip_standard_seed_42','model_name':'clip','branch':'standard',
                 'checkpoint_sha256':'m0','lcs_probe_manifest_sha256':'lcs','coco_probe_manifest_sha256':'coco'}
            points=[{'label':'p001','optimizer_step':150,'progress_fraction':.01},
                    {'label':'p100','optimizer_step':15003,'progress_fraction':1.}]
            manifest={'status':'paused','completed_steps':150,'config':cfg,'config_sha256':'cfg',
                      'checkpoint_policy':{'trajectory_points':points}}
            (d/'run_manifest.json').write_text(json.dumps(manifest))
            provenance={'run_id':cfg['run_id'],'model_name':'clip','branch':'standard','optimizer_step':150,
                        'progress_fraction':.01,'m0_checkpoint_sha256':'m0','config_sha256':'cfg',
                        'probe_manifests':{'lcs':'lcs','coco':'coco'}}
            (folder/'step_00000150_p001_model.json').write_text(json.dumps({'checkpoint_kind':'trajectory_model',
                  'path':'step_00000150_p001_model.pt','artifact_sha256':'sha','provenance':provenance}))
            loaded=load_evaluation_run_manifest(d)
            self.assertEqual(len(resolve_trajectory_snapshots(d,loaded,labels=('p001',))),1)
            with self.assertRaisesRegex(ValueError,'beyond'):resolve_trajectory_snapshots(d,loaded,labels=('p100',))
            self.assertEqual(json.loads((d/'run_manifest.json').read_text())['status'],'paused')

    def test_coco_derivation_excludes_ids_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);parent=d/'parent.json';split=d/'split.json';dest=d/'derived.json'
            samples=[{'sample_id':str(i),'semantic_id':str(i),'image_id':i,'image_relpath':str(i)+'.jpg','caption':'x'} for i in [6001,1,6002]]
            parent.write_text(json.dumps({'samples':samples}));original=parent.read_bytes()
            split.write_text(json.dumps({'images':[{'cocoid':i,'split':'test'} for i in range(1,5001)]}))
            result=derive_coco_probe(parent,split,dest,d,expected_removed=1,expected_remaining=2)
            self.assertEqual([s.sample_id for s in result.samples],['6001','6002'])
            self.assertEqual(parent.read_bytes(),original)
            with self.assertRaisesRegex(ValueError,'split conflict'):derive_coco_probe(parent,split,dest,d)


if __name__=='__main__':unittest.main()
