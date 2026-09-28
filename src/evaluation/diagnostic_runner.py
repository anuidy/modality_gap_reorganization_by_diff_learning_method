"""A0-A6 and existing MMEB Local tasks for completed trajectory snapshots."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time
import numpy as np
import torch

from datasets.probes import load_lcs_manifest,validate_manifest_images
from embeddings.artifact import sha256_file
from model_adapters.factory import create_m0_adapter
from .trajectory import load_evaluation_run_manifest,resolve_trajectory_snapshots,load_snapshot_into_adapter,write_json_atomic
from .protocol import identity_hash,load_protocol,MODALITIES
from .probe_views import derive_coco_probe
from .gpu_encoding import RawEncoder
from .score_analysis import analyze_scores
from .representation_analysis import analyze_representation
from .retrieval import cached_mmeb_task,rank_candidates,aggregate_tasks


def save_arrays(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as handle:np.savez(handle,**arrays)
    temp.replace(path)


def load_arrays(path):
    with np.load(path,allow_pickle=False) as payload:return {k:payload[k] for k in payload.files}


def complete_artifact(directory,identity):
    marker=directory/'complete.json'
    if not marker.exists():return False
    payload=json.loads(marker.read_text(encoding='utf-8'))
    if identity_hash(payload['identity'])!=identity_hash(identity):raise ValueError('Existing evaluation identity differs')
    for name,wanted in payload['files'].items():
        path=directory/name
        if not path.is_file() or sha256_file(path)!=wanted:raise ValueError('Evaluation artifact hash mismatch: '+str(path))
    return True


def finish_artifact(directory,identity,names):
    write_json_atomic(directory/'complete.json',{'identity':identity,'status':'complete',
        'files':{name:sha256_file(directory/name) for name in names}})


def source_identity(root):
    paths=list((root/'src/evaluation').glob('*.py'))+[root/'src/metrics/representation_metrics.py',
        root/'src/datasets/probes.py',root/'src/embeddings/artifact.py',root/'scripts/evaluation/run_diagnostics.py']
    paths+=list((root/'src/model_adapters').glob('*.py'))
    return {p.relative_to(root).as_posix():sha256_file(p) for p in sorted(paths)}


class DiagnosticRunner:
    def __init__(self,root,config_path,run_directory,stage,device='cuda',batch_size=None,memory_gib=None,
                 requested_points=None,allow_running=False):
        self.root=Path(root).resolve();self.config,self.settings=load_protocol(config_path,stage)
        self.stage=stage;self.run_directory=Path(run_directory).resolve()
        self.manifest=load_evaluation_run_manifest(self.run_directory,allow_running=allow_running)
        self.model=self.manifest['config']['model_name']
        if self.model not in ('clip','beit3','vista'):raise ValueError('Only the three main models are currently enabled')
        self.device=torch.device(device)
        if self.device.type=='cuda' and self.device.index is None:self.device=torch.device('cuda',0)
        self.batch_size=batch_size or self.config['runtime']['encoder_batch_size']
        self.output=self.root/self.config['output_root'];self.code=source_identity(self.root)
        self.runtime_identity={'torch':str(torch.__version__),'numpy':str(np.__version__),
                               'device_type':self.device.type,'encoder_batch_size':self.batch_size}
        self.code_sha=identity_hash(self.code);self.adapter=None
        self.image_hashes={};self.reference={};self.scales={}
        if self.device.type=='cuda':
            cap=memory_gib or self.config['runtime']['gpu_memory_limit_gib']
            torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(self.device).total_memory,self.device)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_num_threads(self.config['runtime']['torch_threads'])
        labels=tuple(p for p in requested_points if p!='m0') if requested_points is not None else tuple(
            p['label'] for p in self.manifest['checkpoint_policy']['trajectory_points']
            if p['optimizer_step']<=self.manifest['completed_steps'])
        self.snapshots={s.label:s for s in resolve_trajectory_snapshots(self.run_directory,self.manifest,
            labels=labels,allow_running=allow_running)} if labels else {}

    def checkpoint_identity(self,point):
        cfg=self.manifest['config']
        if point=='m0':return {'model':self.model,'point':'m0','step':0,'seed':None,'branch':None,
                              'checkpoint_sha256':cfg['checkpoint_sha256']}
        spec=self.snapshots[point]
        identity={'model':self.model,'point':point,'step':spec.optimizer_step,'seed':cfg['seed'],'branch':cfg['branch'],
                  'run_id':cfg['run_id'],'config_sha256':self.manifest['config_sha256'],
                  'checkpoint_sha256':spec.artifact_sha256}
        if self.manifest.get('analysis_role')=='reference_only':
            identity['analysis_role']='reference_only'
            identity['reference_reason']='cosine_horizon_retimed_after_completed_budget'
        return identity

    def model_at(self,point):
        # Instantiate each point from the same validated initialization; no training state is changed.
        if self.adapter is not None:
            del self.adapter;self.adapter=None;gc.collect()
            if self.device.type=='cuda':torch.cuda.empty_cache()
        self.adapter=create_m0_adapter(self.model,self.root,str(self.device))
        expected=self.manifest['config']['checkpoint_sha256']
        if sha256_file(self.adapter.checkpoint)!=expected:raise ValueError('Initialization hash mismatch')
        if point!='m0':load_snapshot_into_adapter(self.adapter,self.snapshots[point],self.manifest)
        self.adapter.model.eval()
        native=(1/float(self.manifest['config']['model_options'].get('temperature',.02))) if self.model=='vista' else float(self.adapter.model.logit_scale.detach().float().exp().item())
        self.scales[point]=native
        return self.adapter

    def encoder(self):
        runtime=self.config['runtime'];cache=self.root/runtime['image_cache_root']
        value=RawEncoder(self.adapter,self.model,self.batch_size,image_cache=cache if self.stage=='b' else None,
                         cache_source_root=self.root/runtime['image_cache_source_root'])
        value.image_hashes=self.image_hashes
        return value

    def probe(self,name):
        cfg=self.config['probes'];frozen=self.manifest['config']
        lcs=self.root/cfg['lcs'];coco=self.root/cfg['coco_parent']
        for path,key in [(lcs,'lcs_probe_manifest_sha256'),(coco,'coco_probe_manifest_sha256')]:
            if sha256_file(path)!=frozen[key]:raise ValueError('Frozen probe hash mismatch')
        if name=='lcs':result=load_lcs_manifest(self.root,lcs,self.root/cfg['lcs_image_root'])
        elif name=='coco':result=derive_coco_probe(coco,self.root/cfg['karpathy'],self.root/cfg['coco_derived'],self.root)
        else:raise ValueError('Unknown probe '+name)
        validate_manifest_images(result);return result

    def image_identity(self,paths):
        records={}
        for path in sorted(set(Path(p).resolve() for p in paths)):
            key=str(path)
            if key not in self.image_hashes:self.image_hashes[key]=sha256_file(path)
            records[path.relative_to(self.root).as_posix()]=self.image_hashes[key]
        digest=identity_hash(records);path=self.output/'inputs/images'/f'{digest}.json'
        if not path.exists():write_json_atomic(path,records)
        return {'image_contents_sha256':digest,'image_count':len(records)}

    def embeddings(self,point,probe):
        images=[s.image_path for s in probe.samples];texts=[s.text for s in probe.samples]
        ids=[s.sample_id for s in probe.samples]
        view={'probe_name':probe.name,'manifest_sha256':probe.sha256,'sample_order_sha256':identity_hash(ids),
              **self.image_identity(images),'preprocess':'gpu_nvjpeg_tensor_bicubic_v1',
              'encoder_precision':'native_adapter_fp16_autocast','stored_dtype':'float32_raw',
              'IT_definition':'vista_native_joint_encoder_pre_l2' if self.model=='vista' else 'raw_sum'}
        identity={'checkpoint':self.checkpoint_identity(point),'view':view,'code_sha256':self.code_sha,
                  'runtime':self.runtime_identity}
        directory=self.output/'embeddings'/identity_hash(identity)
        if complete_artifact(directory,identity):
            raw=load_arrays(directory/'raw.npz');meta=json.loads((directory/'provenance.json').read_text())
            self.scales[point]=meta['native_logit_scale'];return raw,identity
        directory.mkdir(parents=True,exist_ok=True)
        self.model_at(point);encoder=self.encoder()
        try:
            raw=encoder.paired(images,texts);raw['sample_ids']=np.asarray(ids)
            # Additive IT is reconstructible; do not duplicate its disk matrix.
            stored={k:v for k,v in raw.items() if k!='IT' or self.model=='vista'}
            save_arrays(directory/'raw.npz',stored)
            write_json_atomic(directory/'provenance.json',{'identity':identity,'native_logit_scale':self.scales[point],
                'pipeline':encoder.stats,'source_code':self.code,'raw_boundary':'pre_l2','training_updates':0})
            finish_artifact(directory,identity,['raw.npz','provenance.json'])
        finally:encoder.close()
        return raw,identity

    def run_a(self,points,probes=('lcs','coco')):
        outputs=[]
        for probe_name in probes:
            probe=self.probe(probe_name);reference,reference_id=self.embeddings('m0',probe)
            if 'IT' not in reference:reference['IT']=reference['I']+reference['T']
            for point in points:
                started=time.time();raw,embed_id=self.embeddings(point,probe)
                if 'IT' not in raw:raw['IT']=raw['I']+raw['T']
                if not np.array_equal(raw['sample_ids'],reference['sample_ids']):raise ValueError('M0/current sample order mismatch')
                identity={'stage':'A0-A6','embeddings':embed_id,'reference':reference_id,
                          'settings':self.settings.identity(),'code_sha256':self.code_sha}
                directory=self.output/'A'/self.model/point/probe.name/identity_hash(identity)
                if not complete_artifact(directory,identity):
                    directory.mkdir(parents=True,exist_ok=True)
                    scores,rows,plots=analyze_scores({m:raw[m] for m in MODALITIES},self.settings,self.scales[point],str(self.device))
                    representation={**raw,'IT_definition':'native' if self.model=='vista' else 'raw_sum'}
                    a0,more=analyze_representation(representation,reference,scores,self.settings,str(self.device))
                    save_arrays(directory/'per_sample.npz',{'sample_ids':raw['sample_ids'],**rows,**more})
                    save_arrays(directory/'distributions.npz',plots)
                    write_json_atomic(directory/'summary.json',{'identity':identity,'sample_count':len(raw['I']),
                        'A0':a0,**scores,'status':'complete','seconds':time.time()-started,
                        'scope':'A0-A6 at the requested checkpoint; C and Global deferred',
                        'step1_low_temperature_diagnostic':'separate first-batch input required; not inferred from this probe'})
                    finish_artifact(directory,identity,['per_sample.npz','distributions.npz','summary.json'])
                outputs.append(str(directory));print(json.dumps({'A_complete':point,'probe':probe.name,'directory':str(directory)}),flush=True)
        return outputs

    def run_b(self,points,tasks):
        outputs=[];data_root=self.root/self.config['b']['data_root']
        for point in points:
            self.model_at(point);encoder=self.encoder();results={};task_outputs={}
            try:
                for name in tasks:
                    started=time.time()
                    prepared=self.config['b'].get('prepared_cache_root')
                    task=cached_mmeb_task(data_root,name,self.root/'data/processed/evaluation/mmeb',
                                          self.root/prepared if prepared else None)
                    images=[data_root/image for text,image in task.pairs if image]
                    dataset={**task.identity,**self.image_identity(images)}
                    identity={'stage':'B_Local','checkpoint':self.checkpoint_identity(point),'dataset':dataset,
                              'preprocess':'gpu_nvjpeg_tensor_bicubic_v1','precision':'fp16_encoder_fp32_cosine',
                              'fusion':'native_joint' if self.model=='vista' else 'raw_sum_before_l2',
                              'code_sha256':self.code_sha,'runtime':self.runtime_identity,
                              'tie_policy':'strictly_greater_plus_one'}
                    directory=self.output/'B_Local'/self.model/point/name/identity_hash(identity)
                    if not complete_artifact(directory,identity):
                        directory.mkdir(parents=True,exist_ok=True)
                        vectors=encoder.fused(task.pairs,data_root)
                        result,ranks,ties=rank_candidates(vectors,task.queries,task.candidates,32,str(self.device))
                        save_arrays(directory/'per_query.npz',{'query_ids':np.arange(len(ranks)),'ranks':ranks,'ties':ties,
                            'query_pair_ids':task.queries,'relevant_pair_ids':task.candidates[:,0]})
                        write_json_atomic(directory/'summary.json',{'identity':identity,'metrics':result,'seconds':time.time()-started,
                            'status':'complete','pipeline':encoder.stats,'cache_identity':encoder.cache_identity})
                        finish_artifact(directory,identity,['per_query.npz','summary.json'])
                        del vectors
                    results[name]=json.loads((directory/'summary.json').read_text())['metrics'];task_outputs[name]=str(directory)
                    outputs.append(str(directory));print(json.dumps({'B_complete':point,'task':name,'metrics':results[name]}),flush=True)
                group={'checkpoint':self.checkpoint_identity(point),'code_sha256':self.code_sha,
                       **aggregate_tasks(results),'task_outputs':task_outputs}
                group_path=self.output/'B_Local/grouped'/f'{identity_hash(group)}.json'
                write_json_atomic(group_path,group)
            finally:encoder.close()
        return outputs
