"""Plan or execute read-only A/B diagnostics on stopped training checkpoints."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--stage',choices=['a','b','b1'],required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/ab_diagnostic.yaml')
    parser.add_argument('--points',nargs='+',default=['m0','p001','p005','p020'])
    parser.add_argument('--probes',nargs='+',choices=['lcs','coco'],default=['lcs','coco'])
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--memory-gib',type=float)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-running',action='store_true',help='Read explicitly requested, atomically published snapshots.')
    args=parser.parse_args()
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from evaluation.protocol import load_protocol,MMEB_TASKS,evaluation_worker_lock
    from evaluation.trajectory import load_evaluation_run_manifest,resolve_trajectory_snapshots
    config,_=load_protocol(args.config,args.stage)
    directory=args.run_directory.resolve();manifest=load_evaluation_run_manifest(directory,allow_running=args.allow_running)
    labels=tuple(p for p in args.points if p!='m0')
    if len(set(args.points))!=len(args.points):raise ValueError('Duplicate checkpoint labels')
    if labels:resolve_trajectory_snapshots(directory,manifest,labels=labels,allow_running=args.allow_running)
    choices=list(config['b1']['datasets']) if args.stage=='b1' else list(MMEB_TASKS)
    tasks=args.tasks or choices
    if any(t not in choices for t in tasks) or len(set(tasks))!=len(tasks):raise ValueError('Invalid/duplicate task selection')
    plan={'stage':args.stage,'run_directory':str(directory),'points':args.points,
          'probes':args.probes if args.stage=='a' else None,'tasks':tasks if args.stage in ('b','b1') else None,
          'execute':args.execute,'C':'deferred','Global':'deferred','training_updates':0}
    print(json.dumps(plan,ensure_ascii=False),flush=True)
    if not args.execute:return
    from evaluation.diagnostic_runner import DiagnosticRunner
    with evaluation_worker_lock(ROOT/config['output_root']):
        runner=DiagnosticRunner(ROOT,args.config,directory,args.stage,args.device,args.batch_size,args.memory_gib,
                                requested_points=args.points,allow_running=args.allow_running)
        if args.stage=='b1':
            from evaluation.b1_retrieval import run_b1
            outputs=run_b1(runner,args.points,tasks)
        else:outputs=runner.run_a(args.points,args.probes) if args.stage=='a' else runner.run_b(args.points,tasks)
    print(json.dumps({'status':'complete','outputs':outputs}),flush=True)


if __name__=='__main__':main()
