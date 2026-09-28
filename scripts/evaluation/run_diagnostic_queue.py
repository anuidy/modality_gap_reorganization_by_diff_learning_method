"""One bounded A/B worker alongside training; discover only completed p020 tasks."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from evaluation.protocol import evaluation_worker_lock,identity_hash


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_job_count(plan):
    # Three shared M0 observations per model; three stages at each of three
    # trajectory points per selected run. Cancelled runs are absent from plan.
    return 3*len({job['model'] for job in plan['jobs']})+9*len(plan['jobs'])


def make_jobs(plan,root):
    jobs=[];m0_seen=set()
    for source in plan['jobs']:
        directory=Path(source['output_dir']);manifest_path=directory/'run_manifest.json'
        if source['model'] not in ('clip','beit3','vista') or not re.fullmatch(r'[a-z0-9_]+',source['run_id']):
            raise ValueError('Invalid diagnostic task identity')
        if not directory.resolve().is_relative_to((root/'outputs/training').resolve()):
            raise ValueError('Training directory is outside the project training tree')
        if not manifest_path.exists():continue
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('status') not in ('complete','paused') or manifest.get('completed_steps',0)<3001:continue
        model=source['model'];points=['p020','p001','p005']
        if model not in m0_seen:points=['m0']+points;m0_seen.add(model)
        for point in points:
            key=model if point=='m0' else source['run_id']
            for stage,probe in [('a','lcs'),('a','coco'),('b',None)]:
                command=[sys.executable,'-B','-u',str(root/'scripts/evaluation/run_diagnostics.py'),
                         '--execute','--run-directory',str(directory),'--points',point,'--stage',stage,'--memory-gib','3']
                if probe:command+=['--probes',probe]
                jobs.append({'id':f'{key}__{point}__{stage}__{probe or "MMEB12"}',
                             'model':model,'point':point,'stage':stage,'probe':probe,'command':command})
    priority={'m0':0,'p020':1,'p001':2,'p005':3}
    return sorted(jobs,key=lambda job:priority[job['point']])


def gpu_memory():
    result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.total',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    used,total=map(float,result.stdout.strip().split(','));return used,total-used


def training_reserve(plan):
    total=0
    for job in plan['jobs']:
        d=Path(job['output_dir']);p=d/'run_manifest.json'
        manifest=json.loads(p.read_text()) if p.exists() else {}
        if manifest.get('status') in ('complete','paused') and manifest.get('completed_steps',0)>=3001:continue
        written=sum(p.stat().st_size for p in (d/'checkpoints/trajectory').glob('*.pt'))
        total+=max(0,job['estimated_output_bytes']-written)
    return total


def stop_worker(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-plan',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();plan=json.loads(args.training_plan.read_text())
    if len(plan['jobs']) not in (24,27) or plan['seed']!=42 or plan['stop_after_step']!=3001:
        raise ValueError('This diagnostic queue is scoped to the original or amended seed-42 p020 gate')
    expected=expected_job_count(plan)
    ready=make_jobs(plan,ROOT)
    if not args.execute:
        print(json.dumps({'ready_jobs':len(ready),'expected_jobs':expected,'order':'shared M0, then p020/p001/p005',
                          'stages':['A0-A6 on LCS10K/COCO4407','MMEB12 Local'],'C':'deferred','Global':'deferred'}));return
    if os.name!='posix':raise RuntimeError('Background GPU queue runs on the Linux experiment server')
    out=ROOT/'outputs/evaluation/ab_diagnostic_v1/queue';state_path=out/'status.json'
    sources=list((ROOT/'src/evaluation').glob('*.py'))+[ROOT/'scripts/evaluation/run_diagnostics.py',
        Path(__file__),ROOT/'configs/evaluation/ab_diagnostic.yaml',ROOT/'src/metrics/representation_metrics.py']
    source_hashes={p.relative_to(ROOT).as_posix():file_sha(p) for p in sources}
    identity={'training_plan_sha256':file_sha(args.training_plan),'evaluation_sources':source_hashes}
    out.mkdir(parents=True,exist_ok=True)
    with evaluation_worker_lock(out):
        state=json.loads(state_path.read_text()) if state_path.exists() else {'identity':identity,'completed':{},'attempts':{}}
        if state['identity']!=identity:raise ValueError('Queue source identity changed; review before restarting')
        child=None
        def interrupted(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
        try:
            while True:
                if any(file_sha(ROOT/n)!=h for n,h in source_hashes.items()):raise RuntimeError('Evaluation code changed while queue was running')
                jobs=make_jobs(plan,ROOT);pending=[job for job in jobs if job['id'] not in state['completed']]
                training=json.loads(args.training_plan.with_name('queue_state.json').read_text())
                if not pending:
                    if training['status']=='complete_at_p020':
                        if set(state['completed'])!={job['id'] for job in jobs} or len(jobs)!=expected:
                            raise RuntimeError('Unexpected diagnostic job coverage')
                        state.update(status='complete',updated_epoch=time.time());atomic(state_path,state);return
                    if training['status']!='running':raise RuntimeError('Training queue stopped before all tasks became available')
                    state.update(status='waiting_for_training',pid=os.getpid(),updated_epoch=time.time());atomic(state_path,state)
                    time.sleep(15);continue
                import shutil
                used,free=gpu_memory();disk=shutil.disk_usage(ROOT).free
                if free<5120 or disk<training_reserve(plan)+10*2**30+512*2**20:
                    state.update(status='waiting_for_resources',gpu_free_MiB=free,disk_free_GiB=disk/2**30,
                                 updated_epoch=time.time(),pid=os.getpid());atomic(state_path,state);time.sleep(15);continue
                job=pending[0];attempt=state['attempts'].get(job['id'],0)+1;state['attempts'][job['id']]=attempt
                logfile=out/(job['id']+f'.attempt{attempt}.log');started=time.time();peak=used;reason=None
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                         PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                         CUDA_VISIBLE_DEVICES='0',WORLD_SIZE='1')
                with logfile.open('x') as log:
                    child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',pid=os.getpid(),active=job,child_pid=child.pid,updated_epoch=time.time());atomic(state_path,state)
                    while child.poll() is None:
                        used,_=gpu_memory();peak=max(peak,used)
                        if used>22500:reason='memory_guard';stop_worker(child);break
                        time.sleep(1)
                if reason:
                    if attempt>=3:raise RuntimeError('Repeated memory guard for '+job['id'])
                    state.update(status='waiting_for_resources',active=None,updated_epoch=time.time());atomic(state_path,state);time.sleep(30);continue
                if child.returncode:raise RuntimeError('Evaluation failed; see '+str(logfile))
                state['completed'][job['id']]={'seconds':time.time()-started,'gpu_peak_MiB':peak,'log':str(logfile)}
                state.update(active=None,updated_epoch=time.time());atomic(state_path,state)
        except BaseException as error:
            stop_worker(child);state.update(status='needs_attention',error=str(error),updated_epoch=time.time());atomic(state_path,state);raise


if __name__=='__main__':main()
