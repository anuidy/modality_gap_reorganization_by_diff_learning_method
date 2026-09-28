"""Reproduce additive models' first prepared training batch, without updates."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(_name,'2')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config',type=Path,default=ROOT/'configs/training/train_runs.yaml')
    parser.add_argument('--run',required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    from training.config import load_run_config
    config=load_run_config(args.training_config.resolve(),args.run,ROOT,{'seed':args.seed})
    if config.model_name not in ('clip','beit3'):
        raise ValueError('The raw-sum low-temperature approximation is restricted to CLIP/BEiT-3')
    if config.gradient_accumulation!=1:raise ValueError('This diagnostic covers the confirmed single-microbatch first update')
    print(json.dumps({'scope':'pre_update_step_1','run':config.run_id,'batch':config.micro_batch_size,
                      'execute':args.execute,'optimizer_steps':0}),flush=True)
    if not args.execute:return
    import numpy as np
    import torch
    from datasets.training_pairs import load_training_pairs,PairedTrainingDataset,DeterministicEpochSampler,collate_raw_training_batch
    from training.data_control import validate_formal_data_identity
    from training.backends import create_training_backend
    from training.randomness import stream_seed
    from training.engine import _seed_everything,_seed_model_forward,_autocast,_config_signature
    from embeddings.artifact import sha256_file
    from evaluation.protocol import load_protocol,identity_hash,evaluation_worker_lock
    from evaluation.score_analysis import analyze_scores,low_temperature_diagnostic
    from evaluation.diagnostic_runner import save_arrays,finish_artifact,complete_artifact
    from evaluation.trajectory import write_json_atomic
    output=ROOT/'outputs/evaluation/ab_diagnostic_v1'
    _,settings=load_protocol(ROOT/'configs/evaluation/ab_diagnostic.yaml','a')
    if sha256_file(config.checkpoint)!=config.checkpoint_sha256:raise ValueError('M0 checkpoint hash mismatch')
    train_sha=sha256_file(config.train_manifest)
    data_identity=validate_formal_data_identity(config,train_sha)
    with evaluation_worker_lock(output):
        _seed_everything(config.seed,config.deterministic)
        torch.set_num_threads(2);device=torch.device('cuda',0)
        torch.cuda.set_per_process_memory_fraction(3*2**30/torch.cuda.get_device_properties(0).total_memory,0)
        pairs=load_training_pairs(config.train_manifest,config.image_root);dataset=PairedTrainingDataset(pairs)
        sampler=DeterministicEpochSampler(dataset,stream_seed(config.seed,'data'))
        indices=list(itertools.islice(iter(sampler),config.micro_batch_size))
        batch=collate_raw_training_batch([dataset[i] for i in indices])
        backend=create_training_backend(config.model_name,config.checkpoint,config.resources,device,config.augmentation,config.model_options)
        backend.set_random_seed(config.seed);backend.train()
        augmentation_seed=stream_seed(config.seed,'augmentation',0,0)
        prepared=backend.prepare_batch(batch,augmentation_seed)
        forward_seed=stream_seed(config.seed,'model',0,0);_seed_model_forward(forward_seed)
        with torch.no_grad(),_autocast(device,config.precision):
            if config.model_name=='clip':
                image=backend.model.encode_image(prepared.images);text=backend.model.encode_text(prepared.text_tokens)
            else:
                tokens,padding=prepared.text_tokens
                image=backend.model.vision_head(backend.model.beit3(textual_tokens=None,visual_tokens=prepared.images,text_padding_position=None)['encoder_out'][:,0,:])
                text=backend.model.language_head(backend.model.beit3(textual_tokens=tokens,visual_tokens=None,text_padding_position=padding)['encoder_out'][:,0,:])
            joint=image+text
        raw={name:value.detach().float().cpu().numpy().copy() for name,value in [('I',image),('T',text),('IT',joint)]}
        native=float(backend.model.logit_scale.detach().float().exp().item())
        tensor_sha=hashlib.sha256(prepared.images.detach().float().cpu().numpy().tobytes()).hexdigest()
        identity={'scope':'pre_update_step_1','model':config.model_name,'seed':config.seed,
                  'config_sha256':_config_signature(config),'checkpoint_sha256':config.checkpoint_sha256,
                  'data':data_identity,'sample_ids':list(batch.sample_ids),'indices':indices,
                  'augmentation_seed':augmentation_seed,'model_forward_seed':forward_seed,
                  'prepared_image_tensor_sha256':tensor_sha,'settings':settings.identity(),
                  'code_sha256':sha256_file(Path(__file__))}
        directory=output/'step1'/config.model_name/identity_hash(identity)
        if complete_artifact(directory,identity):print('already_complete',directory);return
        # Only 36 rows: keep the diagnostic statistics on CPU while preserving
        # the training backend's deterministic GPU forward settings above.
        scores,rows,_=analyze_scores(raw,settings,native,'cpu')
        diagnostics={mode:low_temperature_diagnostic(rows,np.linalg.norm(raw['I'],axis=1),np.linalg.norm(raw['T'],axis=1),mode)
                     for mode in ('fixed','native')}
        save_arrays(directory/'raw_and_per_query.npz',{'sample_ids':np.asarray(batch.sample_ids),**raw,**rows})
        write_json_atomic(directory/'summary.json',{'identity':identity,'diagnostics':diagnostics,'A6':scores['A6'],
            'training_mode':True,'forward_autocast':config.precision,'raw_sum_dtype':str(joint.dtype),
            'native_logit_scale':native,'optimizer_steps':0,'backward_calls':0,
            'score_device':'cpu','loss_view':'uniform_A6_3N_pool_not_selected_branch_training_loss',
            'note':'same first-batch data, augmentation and model RNG; fresh M0 train-mode forward with gradients disabled',
            'training_pass_fail':False})
        finish_artifact(directory,identity,['raw_and_per_query.npz','summary.json'])
        print(json.dumps({'status':'complete','directory':str(directory),'diagnostics':diagnostics}),flush=True)


if __name__=='__main__':main()
