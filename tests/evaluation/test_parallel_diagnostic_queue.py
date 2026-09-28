import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluation.run_adaptive_diagnostic_queue import concurrency_choice,benchmark_jobs,estimated_seconds
from scripts.evaluation.run_diagnostic_queue import make_jobs,expected_job_count


class ParallelSchedulingTests(unittest.TestCase):
    def test_three_workers_require_useful_speedup_and_headroom(self):
        two={'normalized_work_per_second':1.7,'gpu_peak_MiB':9000}
        self.assertEqual(concurrency_choice(two,{'normalized_work_per_second':1.9,'gpu_peak_MiB':13000}),3)
        self.assertEqual(concurrency_choice(two,{'normalized_work_per_second':1.75,'gpu_peak_MiB':13000}),2)
        self.assertEqual(concurrency_choice(two,{'normalized_work_per_second':2.2,'gpu_peak_MiB':20500}),2)

    def test_benchmarks_compare_the_same_workload_type_without_duplicates(self):
        jobs=[{'id':str(i),'model':'clip','stage':'a','probe':'lcs'} for i in range(5)]
        jobs.append({'id':'other','model':'vista','stage':'b','probe':None})
        selected=benchmark_jobs(jobs,5)
        self.assertEqual(len(selected),5)
        self.assertEqual(len({j['id'] for j in selected}),5)
        self.assertEqual({(j['model'],j['stage'],j['probe']) for j in selected},{('clip','a','lcs')})
        self.assertEqual(benchmark_jobs(jobs[:4],5),[])

    def test_cached_jobs_do_not_create_unrealistic_speed_estimates(self):
        job={'model':'clip','stage':'a','probe':'lcs'}
        completed={'clip_standard_seed_42__p020__a__lcs':{'seconds':80},
                   'clip_fixed_2m_seed_42__p020__a__lcs':{'seconds':90},
                   'clip_mixed_2m_seed_42__p020__a__lcs':{'seconds':7},
                   'vista_standard_seed_42__p020__a__lcs':{'seconds':200}}
        self.assertEqual(estimated_seconds(job,completed),85)

    def test_final_selection_covers_198_jobs_without_cancelled_runs(self):
        branches=['standard','fixed_2m','fixed_3m_fn_off','fixed_3m_fn_on','mixed_2m','mixed_3m_fn_off','mixed_3m_fn_on','full_3m_fn_off']
        removed={'vista_mixed_3m_fn_on','beit3_full_3m_fn_off','vista_full_3m_fn_off'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan={'jobs':[]}
            for branch in branches:
                for model in ['clip','beit3','vista']:
                    name=model+'_'+branch
                    if name in removed:continue
                    directory=root/'outputs/training/formal_v1'/name/'seed_42';directory.mkdir(parents=True)
                    (directory/'run_manifest.json').write_text(json.dumps({'status':'paused','completed_steps':3001}))
                    plan['jobs'].append({'run_id':name+'_seed_42','model':model,'output_dir':str(directory)})
            self.assertEqual(len(plan['jobs']),21)
            jobs=make_jobs(plan,root)
            self.assertEqual(len(jobs),198);self.assertEqual(expected_job_count(plan),198)
            self.assertEqual(len({j['id'] for j in jobs}),198)

    def test_restoring_three_runs_adds_only_27_jobs_and_reuses_all_M0(self):
        branches=['standard','fixed_2m','fixed_3m_fn_off','fixed_3m_fn_on','mixed_2m','mixed_3m_fn_off','mixed_3m_fn_on','full_3m_fn_off']
        restored={'vista_mixed_3m_fn_on','beit3_full_3m_fn_off','vista_full_3m_fn_off'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);all_jobs=[]
            for branch in branches:
                for model in ['clip','beit3','vista']:
                    name=model+'_'+branch
                    directory=root/'outputs/training/formal_v1'/name/'seed_42';directory.mkdir(parents=True)
                    (directory/'run_manifest.json').write_text(json.dumps({'status':'paused','completed_steps':3001}))
                    all_jobs.append({'run_id':name+'_seed_42','config_run_id':name,'model':model,'output_dir':str(directory)})
            before={'jobs':[j for j in all_jobs if j['config_run_id'] not in restored]}
            after={'jobs':all_jobs}
            old={j['id'] for j in make_jobs(before,root)};new={j['id'] for j in make_jobs(after,root)}
            self.assertEqual(len(old),198);self.assertEqual(len(new),225)
            self.assertTrue(old.issubset(new));self.assertEqual(expected_job_count(after),225)
            self.assertEqual(len(new-old),27)
            self.assertFalse(any('__m0__' in key for key in new-old))
            for name in restored:self.assertEqual(sum(k.startswith(name+'_seed_42__') for k in new-old),9)


if __name__=='__main__':unittest.main()
