"""Explicit B3/Global entry point; never modifies or joins the live A/B1 queue automatically."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['b3', 'global'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/evaluation/extended_bc.yaml')
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--point', choices=['m0', 'p001', 'p005', 'p020', 'p050', 'p100'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ.setdefault(name, '2')
    import yaml
    from evaluation.protocol import evaluation_worker_lock, identity_hash
    from evaluation.trajectory import load_evaluation_run_manifest, resolve_trajectory_snapshots
    from evaluation.extended.common import CheckpointEncoder
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.stage == 'global' and config['global']['minimum_candidates'] < 5_000_000:
        raise ValueError('The production Global entry point requires the full gallery protocol')
    prepared = ROOT / config[args.stage]['prepared_directory']
    print(json.dumps({'stage': args.stage, 'point': args.point, 'data_ready': (prepared / 'complete.json').is_file(),
                      'execute': args.execute, 'C_execution': 'not_triggered'}))
    if not args.execute: return
    if not (prepared / 'complete.json').is_file(): raise FileNotFoundError('Prepare and verify the complete dataset first')
    if args.device in ('cuda', 'cuda:0') and os.name == 'posix':
        queue_state = ROOT / 'outputs/evaluation/formal_full_v1/queue/status.json'
        if queue_state.exists():
            active = json.loads(queue_state.read_text()); pid = active.get('pid')
            proc = Path('/proc') / str(pid) / 'cmdline'
            if proc.exists() and b'run_formal_queue.py' in proc.read_bytes():
                raise RuntimeError('GPU0 already belongs to the live A/B1 queue; schedule extended evaluation separately')
    import torch
    torch.set_num_threads(config['runtime']['torch_threads'])
    output = ROOT / config['output_root']
    # This is separate from the live queue; users must schedule GPU resources before execution.
    manifest = load_evaluation_run_manifest(args.run_directory, allow_running=True)
    checkpoint_sha = manifest['config']['checkpoint_sha256'] if args.point == 'm0' else resolve_trajectory_snapshots(
        args.run_directory, manifest, labels=(args.point,), allow_running=True)[0].artifact_sha256
    job_lock = output / 'locks' / identity_hash({'stage': args.stage, 'checkpoint_sha256': checkpoint_sha})
    with evaluation_worker_lock(job_lock):
        encoder = CheckpointEncoder(ROOT, args.run_directory, args.point, args.device,
                                    config['runtime']['encoder_batch_size'], config['runtime']['memory_gib'])
        try:
            if args.stage == 'b3':
                from evaluation.extended.classification import run_classification
                destination = run_classification(ROOT, encoder, prepared, ROOT / config['b3']['template_file'], output)
            else:
                from evaluation.extended.global_runner import run_global
                destination = run_global(ROOT, encoder, prepared, output, config['global'])
        finally: encoder.close()
    print(json.dumps({'status': 'complete', 'output': str(destination)}))


if __name__ == '__main__': main()
