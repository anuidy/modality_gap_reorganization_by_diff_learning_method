"""A0 retains the established formulas and adds sample-level norm dynamics."""
from __future__ import annotations

import numpy as np
import torch

from metrics.representation_metrics import (centroid_gap,covariance_gap,effective_rank,l2_normalize,
    cross_modal_alignment,matched_pair_cosine_summary,norm_imbalance,fixed_upper_triangle_pairs,
    spearman_correlation,neighbor_overlap)
from .score_analysis import describe


def norm_dynamics(image, text, reference_image, reference_text):
    arrays=[np.asarray(x,dtype=np.float64) for x in (image,text,reference_image,reference_text)]
    if any(x.shape!=arrays[0].shape for x in arrays):
        raise ValueError('Norm comparison requires aligned current/M0 shapes')
    norms=[np.linalg.norm(x,axis=1) for x in arrays]
    if any(np.any(x<=0) or not np.isfinite(x).all() for x in norms):
        raise ValueError('Zero/nonfinite norms have no defined log ratio')
    image_norm,text_norm,image0,text0=norms
    ratio=np.log(image_norm/text_norm);reference=np.log(image0/text0)
    outward=np.sign(reference)*(ratio-reference)
    per_sample={'image_norm':image_norm,'text_norm':text_norm,'r':ratio,'r_m0':reference,'delta_out':outward}
    return {key:describe(value) for key,value in per_sample.items()},per_sample


@torch.inference_mode()
def geometry_state(raw, pairs, device='cpu', block=64, knn=10):
    value=torch.as_tensor(l2_normalize(np.asarray(raw,dtype=np.float32)),device=device)
    a,b=pairs;n=len(value);k=min(knn,n-1);cosines=np.empty(len(a),np.float32)
    for start in range(0,len(a),4096):
        end=min(len(a),start+4096)
        ia=torch.as_tensor(a[start:end].astype(np.int64),device=device)
        ib=torch.as_tensor(b[start:end].astype(np.int64),device=device)
        cosines[start:end]=(value[ia]*value[ib]).sum(1).cpu().numpy()
    neighbors=np.empty((n,k),np.int32)
    for start in range(0,n,block):
        end=min(n,start+block);scores=value[start:end]@value.T
        scores[torch.arange(end-start,device=device),torch.arange(start,end,device=device)]=-torch.inf
        neighbors[start:end]=torch.topk(scores,k,dim=1).indices.cpu().numpy()
    return {'pair_cosines':cosines,'neighbors':neighbors}


def analyze_representation(raw, reference, score_result, settings, device='cpu', pairs=None):
    image,text=raw['I'],raw['T'];i0,t0=reference['I'],reference['T'];n=len(image)
    if any(np.asarray(x).shape!=np.asarray(image).shape for x in (text,i0,t0)):
        raise ValueError('A0 requires aligned current/M0 representations')
    if pairs is None:
        pairs=fixed_upper_triangle_pairs(n,min(settings.geometry_pair_count,n*(n-1)//2),settings.seed)
    result={};states={};arrays={}
    il,tl=l2_normalize(image),l2_normalize(text)
    for space,i,t in [('raw',image,text),('l2',il,tl)]:
        result['centroid_gap_'+space]=centroid_gap(i,t)
        result['covariance_gap_'+space]=covariance_gap(i,t)
        result['effective_rank_image_'+space]=effective_rank(i)
        result['effective_rank_text_'+space]=effective_rank(t)
    result['cross_modal_alignment']=cross_modal_alignment(il,tl)['mean']
    result['matched_pair_cosine']=matched_pair_cosine_summary(il,tl)
    result['legacy_norm_imbalance_auxiliary']=norm_imbalance(image,text)
    result['norm_dynamics'],norm_arrays=norm_dynamics(image,text,i0,t0)
    arrays.update({'norm/'+k:v for k,v in norm_arrays.items()})
    for name,word in [('I','image'),('T','text')]:
        current=geometry_state(raw[name],pairs,device,settings.query_block)
        initial=current if np.array_equal(raw[name],reference[name]) else geometry_state(reference[name],pairs,device,settings.query_block)
        try:rho=spearman_correlation(initial['pair_cosines'],current['pair_cosines'])
        except ValueError as error:
            if 'constant rank vector' not in str(error):raise
            rho=float('nan')
        result['intra_geometry_'+word]=rho if np.isfinite(rho) else None
        result['neighbor_overlap_'+word]=neighbor_overlap(initial['neighbors'],current['neighbors'])
        result['anisotropy_'+word]=score_result['anisotropy'][name]
        result['score_gap_'+word]=score_result['global_score_gap'][name]['W1']
        states[name]=current
    result['score_gap_mean']=(result['score_gap_image']+result['score_gap_text'])/2
    result['anisotropy_gap']=abs(result['anisotropy_image']-result['anisotropy_text'])
    result['geometry_protocol']={'pairs':len(pairs[0]),'seed':settings.seed,'reference':'M0',
        'constant_spearman':'null/undefined', 'knn':min(10,n-1),'knn_ties':'torch_topk; auxiliary_only'}
    result['norm_protocol']={'ddof':1,'r':'log(per_sample_image_norm/per_sample_text_norm)',
        'delta_out':'sign(r_m0_i)*(r_i-r_m0_i)','aggregation':'after_per_sample_computation'}
    if raw.get('IT_definition')=='raw_sum':
        it=l2_normalize(raw['IT'])
        dit=np.einsum('ij,ij->i',il,it)-np.einsum('ij,ij->i',tl,it)
        arrays['D_IT']=dit;result['D_IT_additive_diagnostic']=describe(dit)
    arrays['geometry_pair_i'],arrays['geometry_pair_j']=pairs
    for m,state in states.items():
        arrays['geometry/'+m+'/pair_cosines']=state['pair_cosines']
        arrays['geometry/'+m+'/neighbors']=state['neighbors']
    return result,arrays
