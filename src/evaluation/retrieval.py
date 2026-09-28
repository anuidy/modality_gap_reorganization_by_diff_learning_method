"""Native MMEB Local retrieval inputs and exact candidate-list ranking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch

from embeddings.artifact import sha256_file
from .protocol import MMEB_TASKS,MMEB_GROUPS,identity_hash


@dataclass
class RetrievalTask:
    name: str
    pairs: list[tuple[str|None,str|None]]
    queries: np.ndarray
    candidates: np.ndarray
    identity: dict


def clean_text(value):
    if value is None:return None
    text=' '.join(value.replace('<|image_1|>',' ').split())
    if text.lower().startswith('represent the given') and ':' not in text:return None
    return text or None


def load_mmeb_task(root:Path,name:str, expected_queries=1000, expected_candidates=1000):
    if name not in MMEB_TASKS:raise ValueError('Unsupported MMEB retrieval task')
    import pyarrow.parquet as pq
    paths=sorted((root/name).glob('test-*.parquet'))
    if not paths:raise FileNotFoundError('No local parquet data for '+name)
    sources={p.relative_to(root).as_posix():sha256_file(p) for p in paths}
    pairs=[];lookup={};queries=[];candidates=[]
    def add(text,image):
        text=clean_text(text);image=image or None
        if image:
            resolved=(root/image).resolve()
            if not resolved.is_relative_to(root.resolve()):raise ValueError('Image path escapes dataset root')
        if text is None and image is None:raise ValueError('Query/candidate lost both modalities')
        key=(text,image)
        if key not in lookup:lookup[key]=len(pairs);pairs.append(key)
        return lookup[key]
    for path in paths:
        table=pq.read_table(path,columns=['qry_text','qry_img_path','tgt_text','tgt_img_path'])
        for batch in table.to_batches(max_chunksize=64):
            for row in batch.to_pylist():
                texts,images=row['tgt_text'],row['tgt_img_path']
                if len(texts)!=expected_candidates or len(images)!=expected_candidates:
                    raise ValueError('Unexpected candidate count for '+name)
                queries.append(add(row['qry_text'],row['qry_img_path']))
                candidates.append([add(t,i) for t,i in zip(texts,images,strict=True)])
    if len(queries)!=expected_queries:raise ValueError('Unexpected query count for '+name)
    identity={'dataset':'MMEB','task':name,'sources':sources,'text_mode':'official',
              'cleanup':'remove_image_placeholder_collapse_whitespace_drop_contentless_represent_templates',
              'relevance':'first_candidate_id; duplicate occurrences of that ID are the same answer',
              'query_count':len(queries),'candidates_per_query':expected_candidates,
              'ordered_inputs_sha256':identity_hash({'pairs':pairs,'queries':queries,'candidates':candidates})}
    return RetrievalTask(name,pairs,np.asarray(queries,np.int64),np.asarray(candidates,np.int64),identity)


@torch.inference_mode()
def rank_candidates(vectors, queries, candidates, query_block=32, device='cpu'):
    value=torch.as_tensor(vectors,dtype=torch.float32,device=device)
    if not torch.isfinite(value).all():raise ValueError('Nonfinite retrieval embedding')
    norms=torch.linalg.vector_norm(value,dim=1,keepdim=True)
    if torch.any(norms<=0):raise ValueError('Zero-norm retrieval embedding')
    value=value/norms;ranks=[];ties=[]
    queries=np.asarray(queries);candidates=np.asarray(candidates)
    if candidates.ndim!=2 or candidates.shape[0]!=len(queries):raise ValueError('Candidate-list shape mismatch')
    if np.any(queries<0) or np.any(candidates<0) or np.any(queries>=len(value)) or np.any(candidates>=len(value)):
        raise ValueError('Candidate/query index out of bounds')
    for start in range(0,len(queries),query_block):
        q=torch.as_tensor(queries[start:start+query_block],device=device)
        c=torch.as_tensor(candidates[start:start+query_block],device=device)
        # Scores use all native candidates; compute blocks do not change the pool.
        scores=(value[q]@value.T).gather(1,c)
        ranks.extend((1+(scores>scores[:,:1]).sum(1)).cpu().tolist())
        ties.extend((scores==scores[:,:1]).sum(1).cpu().tolist())
    ranks=np.asarray(ranks,np.int32)
    return {'R@1':float(np.mean(ranks<=1)),'R@5':float(np.mean(ranks<=5)),
            'R@10':float(np.mean(ranks<=10)),'median_rank':float(np.median(ranks)),
            'queries':len(ranks),'candidates_per_query':int(candidates.shape[1]),
            'tie_policy':'strictly_greater_plus_one','ground_truth':'first_candidate_id'},ranks,np.asarray(ties,np.int32)


def aggregate_tasks(results):
    missing=sorted(set(MMEB_TASKS)-set(results))
    output={'status':'incomplete' if missing else 'complete','missing_tasks':missing,'groups':{}}
    for name,tasks in {'all_12':MMEB_TASKS,**MMEB_GROUPS}.items():
        if not set(tasks)<=set(results):continue
        output['groups'][name]={key:float(np.mean([results[t][key] for t in tasks])) for key in ('R@1','R@5','R@10')}
        output['groups'][name]['tasks']=list(tasks)
    return output


def cached_mmeb_task(root:Path,name:str,cache_root:Path,prepared_cache:Path|None=None):
    """Rebuild on parquet or loader changes; never trust only a cache filename."""
    paths=sorted((root/name).glob('test-*.parquet'))
    if not paths:raise FileNotFoundError('Missing MMEB task: '+name)
    identity={'sources':{p.relative_to(root).as_posix():sha256_file(p) for p in paths},
              'loader_sha256':sha256_file(Path(__file__))}
    path=cache_root/name/(identity_hash(identity)+'.pt');meta=path.with_suffix('.json')
    if path.is_file() and meta.is_file():
        record=json.loads(meta.read_text())
        if record['identity']!=identity or record['sha256']!=sha256_file(path):raise ValueError('MMEB cache identity mismatch')
        value=torch.load(path,map_location='cpu',weights_only=True)
        return RetrievalTask(name,value['pairs'],value['queries'].numpy(),value['candidates'].numpy(),value['identity'])
    legacy_path=prepared_cache/(name+'.pt') if prepared_cache is not None else None
    imported_from=None
    if legacy_path is not None and legacy_path.is_file():
        record=json.loads(legacy_path.with_suffix('.json').read_text())
        pinned_loader='a145af7453f0c76140ca2ca7a3ef6416a6b177722fc4a388e4edcfb79a00cd4b'
        def canonical_sources(sources):
            return {k.removeprefix('data/raw/mmeb_eval/'):v for k,v in sources.items()}
        if (record.get('legacy_code_sha256')!=pinned_loader or record.get('task')!=name
                or record.get('text_mode')!='official'
                or canonical_sources(record.get('sources',{}))!=identity['sources']):
            raise ValueError('Prepared MMEB source/loader identity mismatch')
        if record.get('cache_sha256')!=sha256_file(legacy_path):raise ValueError('Prepared MMEB cache hash mismatch')
        value=torch.load(legacy_path,map_location='cpu',weights_only=True)
        if canonical_sources(value['identity'].get('sources',{}))!=identity['sources'] or value['identity'].get('text_mode')!='official':
            raise ValueError('Prepared MMEB payload identity mismatch')
        q,c=value['queries'].numpy(),value['candidates'].numpy();pairs=value['pairs']
        if q.shape!=(1000,) or c.shape!=(1000,1000):raise ValueError('Prepared MMEB query/candidate dimensions differ')
        semantic={'dataset':'MMEB','task':name,'sources':identity['sources'],'text_mode':'official',
            'cleanup':'remove_image_placeholder_collapse_whitespace_drop_contentless_represent_templates',
            'relevance':'first_candidate_id; duplicate occurrences of that ID are the same answer',
            'query_count':1000,'candidates_per_query':1000,
            'ordered_inputs_sha256':identity_hash({'pairs':pairs,'queries':q.tolist(),'candidates':c.tolist()})}
        task=RetrievalTask(name,pairs,q,c,semantic)
        imported_from={'artifact_sha256':record['cache_sha256'],'loader_sha256':pinned_loader,
                       'protocol':'verified_official_MMEB_prepared_inputs'}
    else:task=load_mmeb_task(root,name)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save({'pairs':task.pairs,'queries':torch.from_numpy(task.queries),'candidates':torch.from_numpy(task.candidates),
                'identity':task.identity},temporary);temporary.replace(path)
    from .trajectory import write_json_atomic
    write_json_atomic(meta,{'identity':identity,'sha256':sha256_file(path),'imported_from':imported_from})
    return task
