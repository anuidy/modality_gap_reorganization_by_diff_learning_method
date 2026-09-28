"""A coordinator-owned worker using the unchanged diagnostic calculations."""
from pathlib import Path
import argparse
import contextlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


@contextlib.contextmanager
def blocking_lock(path):
    import fcntl
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch',type=Path,required=True);args=parser.parse_args()
    queue=ROOT/'outputs/evaluation/ab_diagnostic_v1/queue'
    if not args.dispatch.resolve().is_relative_to((queue/'dispatch').resolve()):raise ValueError('Invalid dispatch path')
    spec=json.loads(args.dispatch.read_text());lease=json.loads((queue/'parallel_lease.json').read_text())
    if lease['pid']!=os.getppid() or spec['lease_token']!=lease['token']:raise RuntimeError('Worker has no live coordinator lease')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'2')
    from evaluation import diagnostic_runner as implementation
    from evaluation.protocol import identity_hash,MMEB_TASKS
    from embeddings.artifact import sha256_file
    for rel,digest in lease['sources'].items():
        if sha256_file(ROOT/rel)!=digest:raise RuntimeError('Evaluation implementation changed: '+rel)
    locks=queue/'shared_resource_locks'
    original_cache=implementation.cached_mmeb_task
    def locked_cache(root,name,*args,**kwargs):
        with blocking_lock(locks/('mmeb_'+name+'.lock')):
            return original_cache(root,name,*args,**kwargs)
    # This process-local wrapper coordinates cache creation only; no numeric code changes.
    implementation.cached_mmeb_task=locked_cache
    class ParallelRunner(implementation.DiagnosticRunner):
        def probe(self,name):
            with blocking_lock(locks/('probe_'+name+'.lock')):return super().probe(name)
        def image_identity(self,paths):
            with blocking_lock(locks/'image_manifest.lock'):return super().image_identity(paths)
        def embeddings(self,point,probe):
            key=identity_hash({'checkpoint':self.checkpoint_identity(point),'probe':probe.name})
            with blocking_lock(locks/('embeddings_'+key+'.lock')):return super().embeddings(point,probe)
    job=spec['job']
    command=job['command'];directory=Path(command[command.index('--run-directory')+1])
    if not directory.resolve().is_relative_to((ROOT/'outputs/training/formal_v1').resolve()):raise ValueError('Unexpected run directory')
    with blocking_lock(locks/('job_'+identity_hash(job['id'])+'.lock')):
        runner=ParallelRunner(ROOT,ROOT/'configs/evaluation/ab_diagnostic.yaml',directory,job['stage'],memory_gib=3)
        if job['stage']=='a':outputs=runner.run_a([job['point']],[job['probe']])
        else:outputs=runner.run_b([job['point']],list(MMEB_TASKS))
    print(json.dumps({'status':'complete','outputs':outputs,'job':job['id'],
                      'execution_wrapper_sha256':sha256_file(Path(__file__))}),flush=True)


if __name__=='__main__':main()
