"""Detached active-branch campaign, with bounded evaluation and verified retirement."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from training.config import load_run_config
from training.independent_runs import build_jobs, child_environment, training_run_lock
from run_formal_queue import verified_outputs

BRANCHES = ('standard', 'fixed_2m', 'fixed_3m_fn_off', 'mixed_2m',
            'mixed_3m_fn_off', 'full_3m_fn_off')
RUNS = tuple('beit3_' + b for b in BRANCHES)
POINTS = {'p020': 3001, 'p040': 6001, 'p060': 9002, 'p080': 12002, 'p100': 15003,
          'p120': 18004, 'p140': 21004, 'p160': 24005, 'p180': 27005, 'p200': 30006}
RESUME_STEPS = (7502, 15003, 22505, 30006)
MAX_STEPS = 30006
FINAL_LABEL = 'p200'
PROGRESS_FRACTIONS = tuple(i / 5 for i in range(1, 11))
RETENTION = 4
DECAY_STEPS = None
TRAINING_ROOT = 'outputs/training/beit3_two_epoch_v2'
STAGES = ('b1__flickr30k', 'b1__coco', 'a__lcs', 'a__coco', 'b__MMEB12')
EXPECTED_EVALUATIONS = (1 + len(RUNS) * len(POINTS)) * len(STAGES)


def configure_campaign(plan):
    """The file name is retained for compatibility; the plan selects a 2/3-epoch budget."""
    from training.checkpoint_plan import build_checkpoint_plan
    global MAX_STEPS, FINAL_LABEL, PROGRESS_FRACTIONS, RETENTION, DECAY_STEPS, TRAINING_ROOT
    global POINTS, RESUME_STEPS, EXPECTED_EVALUATIONS
    MAX_STEPS = int(plan['max_steps'])
    if MAX_STEPS not in (30006, 45009):
        raise ValueError('Only the confirmed two-epoch or appended-third-epoch campaigns are supported')
    epochs = MAX_STEPS // 15003
    PROGRESS_FRACTIONS = tuple(i / 5 for i in range(1, epochs * 5 + 1))
    RETENTION = epochs * 2
    DECAY_STEPS = 45009 if epochs == 3 else None
    TRAINING_ROOT = 'outputs/training/beit3_three_epoch_v1' if epochs == 3 else 'outputs/training/beit3_two_epoch_v2'
    resolved = build_checkpoint_plan(MAX_STEPS, PROGRESS_FRACTIONS, .5, RETENTION, True, .2, 15003)
    POINTS = {p.label: p.optimizer_step for p in resolved.trajectory_points}
    RESUME_STEPS = tuple(sorted(resolved.resume_steps))
    FINAL_LABEL = resolved.trajectory_points[-1].label
    EXPECTED_EVALUATIONS = (1 + len(RUNS) * len(POINTS)) * len(STAGES)


def expected_evaluations(plan):
    omitted = sum(sum(step < entry["completed_steps"] for step in POINTS.values())
                  for entry in plan.get("continuations", {}).values())
    return EXPECTED_EVALUATIONS - omitted * len(STAGES)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def protocol(config, root=ROOT):
    jobs = build_jobs(config, RUNS, root, [42])
    for job in jobs:
        c = load_run_config(config, job.config_run_id, root, {'seed': 42})
        assert c.model_name == 'beit3' and c.max_steps == MAX_STEPS
        assert c.micro_batch_size == 36 and c.gradient_accumulation == 1
        assert c.precision == 'bf16' and c.gradient_clip_norm is None
        assert c.trajectory_progress_fractions == PROGRESS_FRACTIONS
        assert c.progress_reference_steps == 15003
        assert c.save_resume_checkpoints and c.resume_progress_interval == .5 and c.resume_retention == RETENTION
        assert c.scheduler_decay_steps == DECAY_STEPS
        assert c.validation_progress_interval == .2
        assert c.warmup_steps == 300 and c.learning_rate == 1e-5
        assert job.output_dir.is_relative_to((root / TRAINING_ROOT).resolve())
    return jobs


def snapshot(directory, label):
    if label == 'm0':
        return None, None
    meta = directory / ('checkpoints/final.json' if label == FINAL_LABEL else
                         f'checkpoints/trajectory/step_{POINTS[label]:08d}_{label}_model.json')
    if not meta.exists():
        return None, None
    m = read(meta)
    path = (directory if label == FINAL_LABEL else meta.parent) / m['path']
    path = path.resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Checkpoint escaped its run directory')
    expected_kind = 'final_full_resume' if label == FINAL_LABEL else 'trajectory_model'
    if m['checkpoint_kind'] != expected_kind or m['provenance']['optimizer_step'] != POINTS[label]:
        raise ValueError('Unexpected checkpoint kind or step')
    return path, m


def discover(jobs, completed):
    ready = []
    baseline = False
    for job in jobs:
        manifest = job.output_dir / 'run_manifest.json'
        if not manifest.exists():
            continue
        m = read(manifest)
        if m['status'] not in ('running', 'complete', 'paused'):
            raise ValueError('Unexpected training status')
        c = m['config']
        if c['run_id'] != job.run_id or c['model_name'] != 'beit3' or c['seed'] != 42 or c['max_steps'] != MAX_STEPS:
            raise ValueError('Run identity differs from campaign')
        if {p['label']: p['optimizer_step'] for p in m['checkpoint_policy']['trajectory_points']} != POINTS:
            raise ValueError('Run schedule differs from campaign')
        labels = []
        if not baseline:
            labels.append('m0')
            baseline = True
        for label in POINTS:
            path, _ = snapshot(job.output_dir, label)
            if path is not None and path.is_file():
                labels.append(label)
        for label in labels:
            for stage in STAGES:
                key = ('beit3' if label == 'm0' else job.config_run_id) + '__' + label + '__' + stage
                if key not in completed:
                    ready.append({'id': key, 'run': job.config_run_id, 'directory': str(job.output_dir),
                                  'point': label, 'stage': stage})
    return ready


def retire(directory, label, completed, root=ROOT):
    """Never delete full-state files; require all five output groups and read-back hashes."""
    if label not in POINTS or label == FINAL_LABEL:
        return False
    directory = Path(directory).resolve()
    allowed = (root / TRAINING_ROOT).resolve()
    if not directory.is_relative_to(allowed):
        raise ValueError('Refusing retirement outside this campaign')
    cfg = read(directory / 'run_manifest.json')['config']
    run = (cfg['model_name'] + '_' + cfg['branch']) if 'model_name' in cfg and 'branch' in cfg else directory.parent.name
    keys = [run + '__' + label + '__' + s for s in STAGES]
    if not all(k in completed for k in keys):
        return False
    path, meta = snapshot(directory, label)
    if path is None:
        return False
    record = directory / 'checkpoints/trajectory' / (label + '.retired.json')
    if not path.exists():
        if not record.exists() or read(record).get('checkpoint_sha256') != meta['artifact_sha256']:
            raise ValueError('Missing checkpoint without an authorized retirement record')
        return False
    if path.parent != (directory / 'checkpoints/trajectory').resolve() or path.suffix != '.pt':
        raise ValueError('Refusing non-trajectory retirement')
    evidence = []
    for key in keys:
        job = completed[key]
        if verified_outputs(Path(job['log'])) != job['outputs']:
            raise ValueError('Evaluation outputs changed before retirement')
        expected_count = 12 if key.endswith('b__MMEB12') else 1
        if len(set(job['outputs'])) != expected_count:
            raise ValueError('Incomplete task coverage before retirement')
        for output in job['outputs']:
            summary = read(Path(output) / 'summary.json')
            identity_block = summary['identity']
            identity = (identity_block['embeddings']['checkpoint'] if identity_block['stage'] == 'A0-A6'
                        else identity_block['checkpoint'])
            if identity['checkpoint_sha256'] != meta['artifact_sha256'] or identity['point'] != label:
                raise ValueError('Evaluation belongs to another checkpoint')
            if identity['run_id'] != read(directory / 'run_manifest.json')['config']['run_id']:
                raise ValueError('Evaluation belongs to another run')
            evidence.append({'path': output, 'complete_sha256': sha(Path(output) / 'complete.json')})
    if sha(path) != meta['artifact_sha256']:
        raise ValueError('Checkpoint hash changed; refusing retirement')
    receipt = {'status': 'verified_pending_unlink', 'checkpoint': str(path),
               'checkpoint_sha256': meta['artifact_sha256'], 'bytes': path.stat().st_size,
               'evaluation_evidence': evidence, 'authorized_policy': 'delete model-only trajectory after A/B1/Local verification'}
    atomic(record, receipt)
    path.unlink()
    receipt.update(status='retired', retired_epoch=time.time())
    atomic(record, receipt)
    return True


def gpu_memory():
    text = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used,memory.free',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    return tuple(float(x.strip()) for x in text.strip().split(','))


def training_is_settled(directory):
    """Input validation can take minutes: wait for real updates, not launch time."""
    path = Path(directory) / 'train_metrics.jsonl'
    if not path.exists():
        return False
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 24000))
        lines = handle.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        return row.get('completed_steps', 0) >= 10
    return False


def stop(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def adopt_finished_jobs(jobs, previous):
    """Accept only finished runs and verified result groups from a retired controller."""
    completed_training = {}
    completed_evaluation = {}
    by_run = {job.config_run_id: job for job in jobs}
    for run, record in previous.get('completed_training', {}).items():
        if run not in by_run:
            continue
        manifest = read(by_run[run].output_dir / 'run_manifest.json')
        if manifest['status'] != 'complete' or manifest['completed_steps'] != MAX_STEPS:
            raise ValueError('Cannot adopt incomplete training: ' + run)
        if manifest['config']['run_id'] != by_run[run].run_id:
            raise ValueError('Adopted run identity mismatch')
        final, meta = snapshot(by_run[run].output_dir, FINAL_LABEL)
        if not final.is_file() or sha(final) != meta['artifact_sha256']:
            raise ValueError('Adopted final checkpoint integrity mismatch')
        completed_training[run] = record
    for key, entry in previous.get('completed_evaluation', {}).items():
        prefix, point, stage, task = key.split('__')
        if prefix != 'beit3' and prefix not in by_run:
            continue
        if (point != 'm0' and point not in POINTS) or stage+'__'+task not in STAGES:
            raise ValueError('Unexpected adopted evaluation identity')
        if verified_outputs(Path(entry['log'])) != entry['outputs']:
            raise ValueError('Adopted evaluation integrity mismatch')
        completed_evaluation[key] = entry
    return completed_training, completed_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--after-control', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    queue = args.queue.resolve()
    plan = read(queue / 'plan.json')
    configure_campaign(plan)
    config, evaluation = queue / 'train.frozen.yaml', queue / 'evaluation.frozen.yaml'
    jobs = protocol(config)
    expected = expected_evaluations(plan)
    assert plan['runs'] == list(RUNS) and plan['expected_evaluations'] == expected
    for run, origin in plan.get('continuations', {}).items():
        assert run in RUNS and origin['completed_steps'] == (30006 if MAX_STEPS == 45009 else 15003)
        if origin.get('retime_cosine_for_extension'):
            assert origin.get('extend_training_budget') and plan.get('run_roles', {}).get(run) == 'reference_only'
        assert sha(origin['checkpoint']) == origin['checkpoint_sha256']
        assert sha(origin['source_manifest']) == origin['source_manifest_sha256']
    for p, wanted in plan['files'].items():
        if sha(ROOT / p) != wanted:
            raise ValueError('Frozen source/config changed: ' + p)
    previous = read(plan['adopted_state']) if plan.get('adopted_state') else {}
    if plan.get('adopted_state') and sha(plan['adopted_state']) != plan['adopted_state_sha256']:
        raise ValueError('Adopted queue state changed')
    prior_training, prior_evaluation = adopt_finished_jobs(jobs, previous)
    pending_jobs = [job for job in jobs if job.config_run_id not in prior_training]
    print(json.dumps({'jobs': len(jobs), 'steps': MAX_STEPS, 'trajectory': POINTS,
                      'resume_steps': list(RESUME_STEPS), 'evaluations': expected, 'execute': args.execute}), flush=True)
    if not args.execute:
        return
    if os.name != 'posix':
        raise RuntimeError('Execute on the Linux experiment server')
    import shutil
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    state = {'pid': os.getpid(), 'session_id': os.getsid(0), 'status': 'waiting_for_previous_campaign',
             'plan_sha256': sha(queue / 'plan.json'), 'completed_training': prior_training, 'completed_evaluation': prior_evaluation,
             'active_training': None, 'active_evaluation': None, 'attempts': {},
             'expected_training': len(jobs), 'expected_evaluations': expected,
             'run_roles': plan.get('run_roles', {}),
             'cancelled_runs': plan.get('cancelled_runs', [])}
    train = worker = None
    train_log = eval_log = None
    active_job = eval_job = None
    def save(**kwargs):
        state.update(updated_epoch=time.time(), **kwargs)
        atomic(queue / 'status.json', state)
    # An exclusive lock prevents a second handoff from ever entering this queue.
    with training_run_lock(queue / 'controller_lock'):
        if (queue / 'status.json').exists():
            raise FileExistsError('Existing campaign: inspect state before an explicit recovery')
        save()
        try:
            while args.after_control:
                previous = read(args.after_control)
                if previous['status'] == 'complete':
                    break
                if previous['status'] == 'needs_attention':
                    raise RuntimeError('Previous training/evaluation requires attention')
                try:
                    os.kill(previous['pid'], 0)
                except ProcessLookupError:
                    raise RuntimeError('Previous stop controller exited before completion')
                time.sleep(10)
            for p, wanted in plan['files'].items():
                if sha(ROOT / p) != wanted:
                    raise ValueError('Source/config changed while waiting: ' + p)
            minimum_free_gib = 60 if MAX_STEPS == 45009 else 115
            if shutil.disk_usage(ROOT).free < minimum_free_gib * 2**30:
                raise RuntimeError(f'Need at least {minimum_free_gib} GiB free before this campaign')
            for job in pending_jobs:
                if job.output_dir.exists() and any(job.output_dir.iterdir()):
                    raise FileExistsError(job.output_dir)
            lock = ROOT / 'outputs/.independent_gpu_locks' / socket.gethostname() / 'gpu_0'
            with training_run_lock(lock):
                occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
                if occupied:
                    raise RuntimeError('GPU still occupied after old campaign drained')
                env = child_environment('0')
                env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                save(status='validating_inputs')
                with (queue / 'preflight.log').open('x') as f:
                    if pending_jobs:
                        subprocess.run(pending_jobs[0].validation_command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=f, stderr=subprocess.STDOUT, check=True)
                next_index = 0
                last_identity_check = 0
                while next_index < len(pending_jobs) or train is not None or worker is not None or len(state['completed_evaluation']) < expected:
                    if time.time() - last_identity_check > 60:
                        for p, wanted in plan['files'].items():
                            if sha(ROOT / p) != wanted:
                                raise ValueError('Frozen source/config changed during campaign: ' + p)
                        last_identity_check = time.time()
                    if train is not None and train.poll() is not None:
                        code = train.returncode
                        train_log.close(); train_log = None
                        if code:
                            raise RuntimeError(f'Training failed: {active_job.run_id}, exit {code}')
                        m = read(active_job.output_dir / 'run_manifest.json')
                        if m['status'] != 'complete' or m['completed_steps'] != MAX_STEPS:
                            raise RuntimeError('Training exited before completion')
                        start_step = plan.get('continuations', {}).get(active_job.config_run_id, {}).get('completed_steps', 0)
                        expected_resume = [f'step_{step:08d}.pt' for step in RESUME_STEPS if step > start_step]
                        if sorted(p.name for p in (active_job.output_dir / 'checkpoints/resume').glob('*.pt')) != expected_resume:
                            raise RuntimeError('Missing a retained full recovery checkpoint')
                        state['completed_training'][active_job.config_run_id] = {'completed_epoch': time.time(), 'steps': MAX_STEPS,
                                                                                'analysis_role': m.get('analysis_role', 'formal')}
                        train = None
                        save(active_training=None)
                    if worker is not None and worker.poll() is not None:
                        eval_log.close(); eval_log = None
                        if worker.returncode:
                            if state.pop('memory_retry', False) or 'out of memory' in Path(eval_job['log']).read_text(errors='replace').lower():
                                state['evaluation_deferred_for_run'] = active_job.config_run_id if train is not None else None
                                save(active_evaluation=None)
                            else:
                                raise RuntimeError('Evaluation failed: ' + eval_job['log'])
                        else:
                            outputs = verified_outputs(Path(eval_job['log']))
                            state['completed_evaluation'][eval_job['id']] = {'outputs': outputs, 'log': eval_job['log']}
                            save(active_evaluation=None)
                            retire(Path(eval_job['directory']), eval_job['point'], state['completed_evaluation'])
                        worker = None
                    # A new training process gets the GPU alone during model/optimizer startup.
                    if train is None and worker is None and next_index < len(pending_jobs):
                        if shutil.disk_usage(ROOT).free < 20 * 2**30:
                            raise RuntimeError('Less than 20 GiB disk reserve; stop before next run')
                        active_job = pending_jobs[next_index]; next_index += 1
                        active_job.output_dir.mkdir(parents=True, exist_ok=True)
                        train_log = (active_job.output_dir / 'launcher.log').open('x')
                        command = list(active_job.command)
                        origin = plan.get('continuations', {}).get(active_job.config_run_id)
                        if origin:
                            command += ['--resume', origin['checkpoint'], '--resume-source-manifest', origin['source_manifest']]
                            if origin.get('extend_training_budget'):
                                command.append('--extend-training-budget')
                            if origin.get('retime_cosine_for_extension'):
                                command.append('--retime-cosine-for-extension')
                        train = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=train_log, stderr=subprocess.STDOUT, start_new_session=True)
                        state['training_started_epoch'] = time.time()
                        save(status='running', active_training={'run': active_job.config_run_id, 'pid': train.pid})
                    if worker is None:
                        ready = discover(jobs, state['completed_evaluation'])
                        settled = train is None or (time.time() - state['training_started_epoch'] >= 120
                            and training_is_settled(active_job.output_dir)
                            and state.get('evaluation_deferred_for_run') != active_job.config_run_id)
                        if ready and settled:
                            used, free = gpu_memory()
                            if free >= 5120 and shutil.disk_usage(ROOT).free >= 15 * 2**30:
                                eval_job = ready[0]
                                attempts = state['attempts'].get(eval_job['id'], 0) + 1
                                if attempts > 5:
                                    raise RuntimeError('Evaluation exceeded five attempts: ' + eval_job['id'])
                                state['attempts'][eval_job['id']] = attempts
                                log = queue / (eval_job['id'] + f'.attempt{attempts}.log')
                                eval_job['log'] = str(log)
                                stage, task = eval_job['stage'].split('__')
                                command = [sys.executable, '-B', '-u', str(ROOT / 'scripts/evaluation/run_diagnostics.py'),
                                           '--execute', '--allow-running', '--config', str(evaluation),
                                           '--run-directory', eval_job['directory'], '--points', eval_job['point'],
                                           '--stage', stage, '--memory-gib', '3']
                                if stage == 'a': command += ['--probes', task]
                                if stage == 'b1': command += ['--tasks', task]
                                eval_env = dict(env, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                                eval_log = log.open('x')
                                worker = subprocess.Popen(command, cwd=ROOT, env=eval_env, stdin=subprocess.DEVNULL,
                                                          stdout=eval_log, stderr=subprocess.STDOUT, start_new_session=True)
                                save(active_evaluation={**eval_job, 'pid': worker.pid})
                    if worker is not None:
                        used, _ = gpu_memory()
                        if used > 22500:
                            state['memory_retry'] = True
                            stop(worker)
                    if len(state['completed_training']) == len(jobs) and worker is None and not discover(jobs, state['completed_evaluation']):
                        if len(state['completed_evaluation']) != expected:
                            raise RuntimeError('Training finished but evaluation coverage is incomplete')
                    time.sleep(2)
                save(status='complete', active_training=None, active_evaluation=None)
        except BaseException as error:
            save(status='needs_attention', error=repr(error))
            raise
        finally:
            stop(worker); stop(train)
            if train_log: train_log.close()
            if eval_log: eval_log.close()


if __name__ == '__main__':
    main()
