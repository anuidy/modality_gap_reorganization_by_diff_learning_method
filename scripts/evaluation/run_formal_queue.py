"""One GPU evaluation worker alongside formal training; A, B1 and MMEB Local."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.protocol import evaluation_worker_lock

STAGES = [('b1', 'flickr30k'), ('b1', 'coco'), ('a', 'lcs'), ('a', 'coco'), ('b', 'MMEB12')]
POINTS = ['p001', 'p005', 'p020', 'p050', 'p100']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def plan_sources(plan):
    sources = []
    for group in plan['groups']:
        for run, job in zip(group['runs'], group['jobs'], strict=True):
            if job['run'] != f'{run}_seed_42':
                raise ValueError('Unexpected seeded training job identity')
            sources.append({'model': group['model'], 'run': run, 'instance_run': job['run'], 'directory': job['output_dir']})
    if len(sources) != 27 or len({x['run'] for x in sources}) != 27:
        raise ValueError('Expected the prepared 27-task campaign')
    if plan['seed'] != 42 or plan['max_steps'] != 15003:
        raise ValueError('Unexpected campaign controls')
    return sources


def discover_jobs(plan, config_path, root=ROOT):
    sources = plan_sources(plan); jobs = []; baselines = set()
    for source in sources:
        directory = Path(source['directory']).resolve()
        if not directory.is_relative_to((root / 'outputs/training/formal_full_v1').resolve()):
            raise ValueError('Unexpected formal training output path')
        path = directory / 'run_manifest.json'
        if not path.exists(): continue
        manifest = json.loads(path.read_text())
        if manifest['status'] not in ('running', 'complete', 'paused'): continue
        cfg = manifest['config']
        if cfg['run_id'] != source['instance_run'] or cfg['model_name'] != source['model'] or cfg['seed'] != 42:
            raise ValueError('Training manifest identity differs from the queue')
        labels = []
        if source['model'] not in baselines:
            labels.append('m0'); baselines.add(source['model'])
        points = manifest['checkpoint_policy']['trajectory_points']
        if [p['label'] for p in points] != POINTS or [p['optimizer_step'] for p in points] != plan['trajectory_steps']:
            raise ValueError('Unexpected trajectory schedule')
        for point in points:
            label, step = point['label'], point['optimizer_step']
            meta = directory / ('checkpoints/final.json' if label == 'p100' else
                                f'checkpoints/trajectory/step_{step:08d}_{label}_model.json')
            if meta.is_file():
                record = json.loads(meta.read_text())
                weight = (directory if label == 'p100' else meta.parent) / record['path']
                if weight.is_file(): labels.append(label)
        for point in labels:
            for stage, task in STAGES:
                key = source['model'] if point == 'm0' else source['run']
                command = [sys.executable, '-B', '-u', str(root / 'scripts/evaluation/run_diagnostics.py'),
                           '--execute', '--allow-running', '--config', str(config_path),
                           '--run-directory', str(directory), '--points', point, '--stage', stage, '--memory-gib', '3']
                if stage == 'a': command += ['--probes', task]
                elif stage == 'b1': command += ['--tasks', task]
                jobs.append({'id': f'{key}__{point}__{stage}__{task}', 'model': source['model'],
                             'point': point, 'stage': stage, 'task': task, 'command': command})
    return sorted(jobs, key=lambda j: (['m0'] + POINTS).index(j['point']))


def memory():
    output = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10)
    return tuple(float(x.strip()) for x in output.strip().split(','))


def storage_reserve(plan):
    # Five snapshots/run; empirical sizes plus a 5% margin, based on the p020 campaign.
    sizes = {'clip': 1710649943, 'beit3': 888882151, 'vista': 785648557}
    remaining = 0
    for source in plan_sources(plan):
        directory = Path(source['directory']) / 'checkpoints/trajectory'
        written = sum(p.stat().st_size for p in directory.glob('*.pt'))
        remaining += max(0, int(5 * sizes[source['model']] * 1.05) - written)
    return remaining + 12 * 2**30


def stop(child):
    if child is None or child.poll() is not None: return
    os.killpg(child.pid, signal.SIGTERM)
    try: child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL); child.wait()


def verified_outputs(log):
    for line in reversed(log.read_text(errors='replace').splitlines()):
        try: payload = json.loads(line)
        except ValueError: continue
        if payload.get('status') == 'complete' and 'outputs' in payload:
            outputs = payload['outputs']
            if not outputs: raise ValueError('Evaluation returned no artifacts')
            for directory in outputs:
                directory = Path(directory); marker = json.loads((directory / 'complete.json').read_text())
                if marker['status'] != 'complete': raise ValueError('Incomplete evaluation artifact')
                for name, wanted in marker['files'].items():
                    if sha(directory / name) != wanted: raise ValueError('Evaluation artifact integrity failed')
            return outputs
    raise ValueError('Evaluation returned no completion record')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/evaluation/formal_ab.yaml')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); args.config = args.config.resolve()
    plan = json.loads(args.training_plan.read_text()); sources = plan_sources(plan)
    config = yaml.safe_load(args.config.read_text())
    expected = (len({s['model'] for s in sources}) + len(sources) * len(POINTS)) * len(STAGES)
    out = ROOT / config['output_root'] / 'queue'
    files = sorted(set(list((ROOT / 'src/evaluation').glob('*.py')) + list((ROOT / 'src/model_adapters').glob('*.py')) +
                       [ROOT / 'src/metrics/representation_metrics.py', ROOT / 'src/datasets/probes.py',
                        ROOT / 'src/embeddings/artifact.py', ROOT / 'scripts/evaluation/run_diagnostics.py', Path(__file__)]))
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    dataset_hashes = {name: sha(ROOT / spec['directory'] / 'validation_report.json') for name, spec in config['b1']['datasets'].items()}
    identity = {'training_plan_sha256': sha(args.training_plan), 'config_sha256': sha(args.config),
                'sources': hashes, 'b1_data_reports': dataset_hashes}
    ready = discover_jobs(plan, args.config)
    print(json.dumps({'expected_jobs': expected, 'B1_jobs': expected * 2 // 5, 'ready_jobs': len(ready),
                      'workers': 1, 'execute': args.execute}), flush=True)
    if not args.execute: return
    if os.name != 'posix': raise RuntimeError('Queue runs on the Linux experiment server')
    out.mkdir(parents=True, exist_ok=True); path = out / 'status.json'; child = None
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1')
    with evaluation_worker_lock(out):
        state = json.loads(path.read_text()) if path.exists() else {'identity': identity, 'completed': {}, 'attempts': {}}
        if state['identity'] != identity: raise ValueError('Existing queue identity changed')
        def save(**kwargs):
            state.update(pid=os.getpid(), expected_jobs=expected, worker_limit=1, updated_epoch=time.time(), **kwargs)
            atomic(path, state)
        def interrupted(signum, frame): raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            while True:
                if sha(args.training_plan) != identity['training_plan_sha256'] or sha(args.config) != identity['config_sha256']:
                    raise RuntimeError('Frozen plan/config changed')
                if any(sha(ROOT / p) != h for p, h in hashes.items()): raise RuntimeError('Evaluation source changed')
                training = json.loads(args.training_plan.with_name('queue_state.json').read_text())
                if training['status'] not in ('validating_then_training', 'running', 'complete'):
                    raise RuntimeError('Training queue requires attention')
                jobs = discover_jobs(plan, args.config)
                pending = [j for j in jobs if j['id'] not in state['completed']]
                if not pending:
                    if training['status'] == 'complete':
                        if len(state['completed']) != expected or len(jobs) != expected:
                            raise RuntimeError('Incomplete full-trajectory coverage')
                        save(status='complete', active=None); return
                    save(status='waiting_for_training', active=None); time.sleep(15); continue
                used, free = memory(); reserve = storage_reserve(plan)
                if free < 5120 or shutil.disk_usage(ROOT).free < reserve:
                    save(status='waiting_for_resources', active=None, gpu_free_MiB=free); time.sleep(15); continue
                job = pending[0]; attempt = state['attempts'].get(job['id'], 0) + 1
                if attempt > 5: raise RuntimeError('Repeated evaluation failure: ' + job['id'])
                state['attempts'][job['id']] = attempt
                log = out / (job['id'] + f'.attempt{attempt}.log'); started = time.time(); peak = used; pressure = False
                with log.open('x') as handle:
                    child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    save(status='running', active=job, child_pid=child.pid)
                    while child.poll() is None:
                        used, _ = memory(); peak = max(peak, used)
                        if used > 22500:
                            pressure = True; stop(child); break
                        time.sleep(1)
                if pressure or (child.returncode and 'out of memory' in log.read_text(errors='replace').lower()):
                    save(status='waiting_for_resources', active=None); time.sleep(30); continue
                if child.returncode: raise RuntimeError('Evaluation failed: ' + str(log))
                outputs = verified_outputs(log)
                state['completed'][job['id']] = {'seconds': time.time() - started, 'gpu_peak_MiB': peak,
                                                'outputs': outputs, 'log': str(log)}
                save(status='running', active=None)
        except BaseException as error:
            stop(child); save(status='needs_attention', active=None, error=repr(error)); raise


if __name__ == '__main__': main()
