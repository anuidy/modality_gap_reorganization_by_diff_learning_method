"""One evaluation worker during training; measure two/three workers after training."""
from pathlib import Path
import argparse
import collections
import contextlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
from evaluation.protocol import evaluation_worker_lock
from scripts.evaluation.run_diagnostic_queue import atomic,file_sha,make_jobs,expected_job_count,gpu_memory,training_reserve,stop_worker


class MemoryPressure(RuntimeError):pass


def concurrency_choice(two,three):
    """Require a >5% gain in normalized work/s and >4 GiB of GPU headroom."""
    return 3 if three['normalized_work_per_second']>two['normalized_work_per_second']*1.05 and three['gpu_peak_MiB']<20000 else 2


def estimated_seconds(job,completed):
    key=(job['model'],job['stage'],job['probe'] or 'MMEB12');values=[]
    for jid,record in completed.items():
        run,point,stage,probe=jid.split('__')
        if point!='m0' and (run.split('_')[0],stage,probe)==key and record['seconds']>20:values.append(record['seconds'])
    defaults={'clip':{'lcs':85,'coco':56,'MMEB12':180},'beit3':{'lcs':70,'coco':48,'MMEB12':120},
              'vista':{'lcs':110,'coco':74,'MMEB12':167}}
    return statistics.median(values[-6:]) if values else defaults[key[0]][key[2]]


def benchmark_jobs(pending,count):
    # Use one task type/model in both trials, reducing composition bias.
    groups=collections.defaultdict(list)
    for job in pending:groups[(job['model'],job['stage'],job['probe'])].append(job)
    eligible=[jobs for jobs in groups.values() if len(jobs)>=count]
    return max(eligible,key=len)[:count] if eligible else []


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-plan',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();plan=json.loads(args.training_plan.read_text())
    if plan['seed']!=42 or plan['stop_after_step']!=3001 or len(plan['jobs']) not in (21,24):
        raise ValueError('Expected the authorized 21-run or restored 24-run gate')
    expected=expected_job_count(plan)
    if not args.execute:
        print(json.dumps({'expected_jobs':expected,'training_workers':1,'post_training_worker_candidates':[2,3],
                          'per_worker_memory_GiB':3,'gpu_limit_MiB':22500}));return
    if os.name!='posix':raise RuntimeError('This coordinator requires Linux')
    out=ROOT/'outputs/evaluation/ab_diagnostic_v1/queue';state_path=out/'status.json'
    sources=list((ROOT/'src/evaluation').glob('*.py'))+[ROOT/'scripts/evaluation/run_diagnostics.py',
        ROOT/'scripts/evaluation/run_diagnostic_queue.py',Path(__file__),
        ROOT/'scripts/evaluation/run_parallel_diagnostic_worker.py',ROOT/'configs/evaluation/ab_diagnostic.yaml',
        ROOT/'src/metrics/representation_metrics.py']
    hashes={p.relative_to(ROOT).as_posix():file_sha(p) for p in sources}
    identity={'training_plan_sha256':file_sha(args.training_plan),'evaluation_sources':hashes}
    with contextlib.ExitStack() as stack:
        stack.enter_context(evaluation_worker_lock(out))
        # Retain exclusive ownership against the original single-worker CLI;
        # only this coordinator's leased children bypass that global lock.
        stack.enter_context(evaluation_worker_lock(out.parent))
        state=json.loads(state_path.read_text())
        if state['identity']!=identity:raise ValueError('Queue identity changed without an explicit migration')
        lease={'pid':os.getpid(),'token':uuid.uuid4().hex,'sources':hashes}
        atomic(out/'parallel_lease.json',lease)
        children={}
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0',WORLD_SIZE='1')
        def save(**kw):
            state.update(pid=os.getpid(),updated_epoch=time.time(),**kw);atomic(state_path,state)
        def validate_sources():
            if file_sha(args.training_plan)!=identity['training_plan_sha256']:raise RuntimeError('Active plan changed')
            if any(file_sha(ROOT/n)!=h for n,h in hashes.items()):raise RuntimeError('Evaluation implementation changed')
        def stop_all():
            for item in children.values():
                stop_worker(item['process']);item['handle'].close()
            children.clear()
        def interrupted(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)

        def wave(jobs,workers,phase,estimates=None):
            pending=list(jobs);started=time.time();peak=0;finished=[]
            try:
                while pending or children:
                    validate_sources();used,free=gpu_memory();peak=max(peak,used)
                    if children and used>22500:raise MemoryPressure('Whole-GPU memory guard')
                    for jid,item in list(children.items()):
                        code=item['process'].poll()
                        if code is None:continue
                        item['handle'].close();del children[jid]
                        if code:
                            tail=item['log'].read_text(errors='replace')[-16000:].lower()
                            if 'out of memory' in tail:raise MemoryPressure(jid)
                            raise RuntimeError('Evaluation worker failed: '+str(item['log']))
                        state['completed'][jid]={'seconds':time.time()-item['started'],'gpu_peak_MiB':peak,
                                                  'log':str(item['log']),'workers':workers,'phase':phase}
                        finished.append(jid)
                    training=json.loads(args.training_plan.with_name('queue_state.json').read_text())
                    if workers>1 and training['status']!='complete_at_p020':raise RuntimeError('Parallel evaluation requires completed training')
                    if training['status'] not in ('running','complete_at_p020'):raise RuntimeError('Training queue needs attention')
                    while pending and len(children)<workers:
                        used,free=gpu_memory()
                        disk=shutil.disk_usage(ROOT).free
                        if free<5120 or disk<training_reserve(plan)+10*2**30+workers*512*2**20:break
                        job=pending.pop(0);jid=job['id'];attempt=state['attempts'].get(jid,0)+1
                        if attempt>5:raise RuntimeError('Repeated retries: '+jid)
                        state['attempts'][jid]=attempt
                        dispatch=out/'dispatch'/f'{jid}.attempt{attempt}.json'
                        atomic(dispatch,{'job':job,'lease_token':lease['token']})
                        log=out/f'{jid}.attempt{attempt}.log';handle=log.open('x')
                        command=[sys.executable,'-B','-u',str(ROOT/'scripts/evaluation/run_parallel_diagnostic_worker.py'),
                                 '--dispatch',str(dispatch)]
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,
                                               stderr=subprocess.STDOUT,start_new_session=True)
                        children[jid]={'process':child,'handle':handle,'log':log,'started':time.time(),'job':job}
                    running=[{'job':v['job'],'pid':v['process'].pid} for v in children.values()]
                    save(status='running' if children else 'waiting_for_resources',active=running[0]['job'] if running else None,
                         active_workers=running,worker_limit=workers,phase=phase)
                    if pending or children:time.sleep(1)
                elapsed=time.time()-started
                return {'workers':workers,'jobs':finished,'seconds':elapsed,'gpu_peak_MiB':peak,
                        'normalized_work_per_second':sum(estimates[j] for j in finished)/elapsed if estimates else None}
            except BaseException:
                stop_all();raise

        try:
            save(status='running',active=None,active_workers=[],phase='serial_during_training',expected_jobs=expected)
            while True:
                validate_sources();jobs=make_jobs(plan,ROOT);pending=[j for j in jobs if j['id'] not in state['completed']]
                training=json.loads(args.training_plan.with_name('queue_state.json').read_text())
                if not pending:
                    if training['status']=='complete_at_p020':
                        if len(jobs)!=expected or set(state['completed'])!={j['id'] for j in jobs}:raise RuntimeError('Incomplete diagnostic coverage')
                        save(status='complete',active=None,active_workers=[],phase='complete');return
                    if training['status']!='running':raise RuntimeError('Training queue needs attention')
                    save(status='waiting_for_training',active=None,active_workers=[]);time.sleep(10);continue
                if training['status']!='complete_at_p020':
                    try:wave(pending[:1],1,'serial_during_training')
                    except MemoryPressure:
                        save(status='waiting_for_resources',active=None,active_workers=[]);time.sleep(15)
                    continue
                if 'selected_post_training_workers' not in state:
                    trials=benchmark_jobs(pending,5)
                    if trials:
                        estimates={j['id']:estimated_seconds(j,state['completed']) for j in trials}
                        try:
                            two=wave(trials[:2],2,'benchmark_two',estimates)
                            three=wave(trials[2:],3,'benchmark_three',estimates)
                            chosen=concurrency_choice(two,three)
                            state['parallel_benchmarks']={'two':two,'three':three,'comparison':'same model/stage/probe; normalized against serial medians'}
                        except MemoryPressure as error:
                            chosen=1;state['parallel_benchmarks']={'fallback_reason':str(error)}
                        state['selected_post_training_workers']=chosen
                    else:
                        # Avoid extra/repeated inference just to fill a benchmark.
                        state['selected_post_training_workers']=2 if len(pending)>=2 else 1
                        state['parallel_benchmarks']={'decision':'too few comparable remaining jobs for a fair two-vs-three trial; conservative memory-budget choice'}
                    save(status='running',active=None,active_workers=[],phase='parallel_selection_complete');continue
                workers=state['selected_post_training_workers']
                try:wave(pending,workers,'post_training_parallel')
                except MemoryPressure as error:
                    if workers==1:raise
                    state['selected_post_training_workers']=workers-1
                    state.setdefault('memory_fallbacks',[]).append({'from':workers,'to':workers-1,'reason':str(error),'epoch':time.time()})
                    save(status='running',active=None,active_workers=[])
        except BaseException as error:
            stop_all();save(status='needs_attention',error=repr(error),active=None,active_workers=[]);raise


if __name__=='__main__':main()
