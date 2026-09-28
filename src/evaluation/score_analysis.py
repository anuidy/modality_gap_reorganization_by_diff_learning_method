"""A1-A6 over the entire probe pool, with bounded GPU score blocks."""
from __future__ import annotations

import itertools
import math
import numpy as np
import torch

from .protocol import ASettings, MODALITIES


def describe(values, ddof=1):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Statistics require nonempty finite values')
    return {'mean':float(values.mean()), 'median':float(np.median(values)),
            'std':float(values.std(ddof=ddof)) if values.size>ddof else 0.,
            'min':float(values.min()), 'max':float(values.max()), 'count':int(values.size)}


def normalized_views(raw, device):
    if set(raw) != set(MODALITIES):
        raise ValueError('A1-A6 require genuine I/T/IT representations')
    views = {}
    shape = np.asarray(raw['I']).shape
    if len(shape)!=2 or shape[0]<2:
        raise ValueError('Expected at least two aligned embedding rows')
    for name in MODALITIES:
        value = torch.as_tensor(np.asarray(raw[name]),dtype=torch.float32,device=device)
        if tuple(value.shape)!=shape or not torch.isfinite(value).all():
            raise ValueError('Representation shapes/values are invalid')
        norms = torch.linalg.vector_norm(value,dim=1,keepdim=True)
        if torch.any(norms<=0):
            raise ValueError('Zero-norm representation has no defined cosine direction')
        views[name] = value/norms
    return views


def fractional_exposure(scores, candidate_modalities, k):
    """Share membership at an exact score tie, avoiding modality-order bias."""
    threshold = torch.topk(scores,k,dim=1).values[:,-1:]
    greater, equal = scores>threshold, scores==threshold
    weight = (k-greater.sum(1)).double()/equal.sum(1).double()
    return torch.stack([((greater & (candidate_modalities==m)).sum(1).double()
                         +weight*(equal & (candidate_modalities==m)).sum(1).double())/k
                        for m in range(3)],dim=1)


@torch.inference_mode()
def analyze_scores(raw, settings: ASettings, native_logit_scale: float, device='cpu'):
    if not math.isfinite(native_logit_scale) or native_logit_scale<=0:
        raise ValueError('Native logit scale must be finite and positive')
    views=normalized_views(raw,device);n=len(views['I'])
    if max(settings.top_ks)>3*(n-1):
        raise ValueError('Top-K exceeds the available background pool')
    per_query={};plots={};summary={'A1':{},'A2':{},'A3':{},'A4':{},'A5':{},'A6':{}}
    total=n*(n-1);rng=np.random.default_rng(settings.seed)
    sample_flat=np.sort(rng.choice(total,min(total,settings.negative_plot_samples),replace=False))
    sample_i=sample_flat//(n-1);j0=sample_flat%(n-1);sample_j=j0+(j0>=sample_i)
    plots['negative_sample_query_indices']=sample_i;plots['negative_sample_candidate_indices']=sample_j
    candidate_mod=torch.arange(3,device=device).repeat_interleave(n)[None,:]
    pooled_gap={};anisotropy={}
    scales={'fixed':1/settings.fixed_temperature,'native':native_logit_scale}

    def save(key, values, start, end, dtype=np.float64):
        if key not in per_query:per_query[key]=np.empty(n,dtype=dtype)
        per_query[key][start:end]=values.detach().cpu().numpy() if isinstance(values,torch.Tensor) else values

    for qi,qname in enumerate(MODALITIES):
        distribution={name:dict(count=0,sum=0.,squares=0.,hist=np.zeros(settings.histogram_bins,dtype=np.int64))
                      for name in MODALITIES if name!=qname}
        # Exact historical pooled W1 remains distinct from A2 per-query W1.
        same_flat=np.empty(total,np.float32) if qname in ('I','T') else None
        cross_flat=np.empty(total,np.float32) if qname in ('I','T') else None
        anisotropy_sum=0.
        for start in range(0,n,settings.query_block):
            end=min(n,start+settings.query_block);ids=torch.arange(start,end,device=device)
            local=torch.arange(end-start,device=device)
            scores={m:views[qname][start:end]@views[m].T for m in MODALITIES}
            valid=torch.ones((end-start,n),dtype=torch.bool,device=device);valid[local,ids]=False
            backgrounds={m:scores[m][valid].reshape(end-start,n-1) for m in MODALITIES}
            positives={m:scores[m][local,ids] for m in MODALITIES if m!=qname}
            anisotropy_sum+=backgrounds[qname].double().sum().item()
            if same_flat is not None:
                cross='T' if qname=='I' else 'I'
                same_flat[start*(n-1):end*(n-1)]=backgrounds[qname].cpu().numpy().reshape(-1)
                cross_flat[start*(n-1):end*(n-1)]=backgrounds[cross].cpu().numpy().reshape(-1)
            sampled=(sample_i>=start)&(sample_i<end)
            for target,positive in positives.items():
                direction=qname+'->'+target;save('A1/'+direction+'/positive',positive,start,end)
                stats=distribution[target];values=backgrounds[target]
                stats['count']+=values.numel();stats['sum']+=values.double().sum().item()
                stats['squares']+=values.double().square().sum().item()
                stats['hist']+=torch.histc(values.clamp(-1,1),settings.histogram_bins,-1,1).long().cpu().numpy()
                key='A1/'+direction+'/negative_sample'
                if key not in plots:plots[key]=np.empty(len(sample_flat),np.float32)
                plots[key][sampled]=scores[target][torch.as_tensor(sample_i[sampled]-start,device=device),
                                                  torch.as_tensor(sample_j[sampled],device=device)].cpu().numpy()
            ordered={m:torch.sort(backgrounds[m],dim=1).values for m in MODALITIES}
            for a,b in itertools.combinations(MODALITIES,2):
                prefix=f'A2/{qname}/{a}-{b}'
                save(prefix+'/W1',(ordered[a]-ordered[b]).abs().double().mean(1),start,end)
                save(prefix+'/signed_shift',backgrounds[a].double().mean(1)-backgrounds[b].double().mean(1),start,end)
            del ordered
            hard_by_mod=torch.stack([backgrounds[m].max(1).values for m in MODALITIES],1)
            h,hard_mod=hard_by_mod.max(1);p=torch.stack(list(positives.values()),1)
            save('A3/'+qname+'/M_first',p.max(1).values-h,start,end)
            save('A3/'+qname+'/M_all',p.min(1).values-h,start,end)
            save('A3/'+qname+'/hardest_modality',hard_mod,start,end,np.int8)
            save('A3/'+qname+'/hardest_modality_ties',(hard_by_mod==h[:,None]).sum(1),start,end,np.int8)
            pool=torch.cat([scores[m] for m in MODALITIES],dim=1)
            pool[local,qi*n+ids]=-torch.inf
            for target,positive in positives.items():
                direction=qname+'->'+target
                save('A4/'+direction+'/rank',1+(pool>positive[:,None]).sum(1),start,end,np.int32)
                save('A4/'+direction+'/pessimistic_rank',(pool>=positive[:,None]).sum(1),start,end,np.int32)
            for mode,scale in scales.items():
                denominator=torch.logsumexp(pool*scale,dim=1)
                for target,positive in positives.items():
                    direction=qname+'->'+target;ce=(denominator-positive*scale).clamp_min(0)
                    save(f'A6/{mode}/{direction}/CE',ce,start,end)
                    save(f'A6/{mode}/{direction}/probability',torch.exp(-ce.double()),start,end)
            bg_pool=torch.cat([backgrounds[m] for m in MODALITIES],dim=1)
            bg_mod=torch.arange(3,device=device).repeat_interleave(n-1)[None,:]
            for mode,values,mods in [('inclusive',pool,candidate_mod),('background',bg_pool,bg_mod)]:
                prior=np.full(3,1/3) if mode=='background' else np.array([n-(m==qi) for m in range(3)])/(3*n-1)
                for k in settings.top_ks:
                    exposure=fractional_exposure(values,mods,k)
                    for mi,m in enumerate(MODALITIES):
                        prefix=f'A5/{qname}/{mode}/k{k}/{m}'
                        save(prefix+'/P',exposure[:,mi],start,end)
                        save(prefix+'/bias',exposure[:,mi]-prior[mi],start,end)
        anisotropy[qname]=anisotropy_sum/total
        for target,stats in distribution.items():
            direction=qname+'->'+target;mean=stats['sum']/total
            variance=max(0.,(stats['squares']-stats['sum']**2/total)/(total-1))
            plots['A1/'+direction+'/negative_histogram']=stats['hist']
            summary['A1'][direction]={'positive':describe(per_query['A1/'+direction+'/positive']),
                'negative':{'mean':mean,'std':math.sqrt(variance),'count':total},
                'negative_sample':describe(plots['A1/'+direction+'/negative_sample']),
                'sampling_scope':'plot sample only; all negative pairs contribute to moments/histograms'}
        if same_flat is not None:
            bias=float(same_flat.mean(dtype=np.float64)-cross_flat.mean(dtype=np.float64))
            same_flat.sort();cross_flat.sort();np.subtract(same_flat,cross_flat,out=same_flat);np.abs(same_flat,out=same_flat)
            pooled_gap[qname]={'W1':float(same_flat.mean(dtype=np.float64)), 'signed_shift':bias,'candidate_pair_count':total}
            del same_flat,cross_flat
    for key,values in per_query.items():
        group=key.split('/')[0]
        if group=='A1':continue
        if key.endswith('/hardest_modality'):
            summary[group][key]={'proportion':{m:float(np.mean(values==i)) for i,m in enumerate(MODALITIES)},
                                 'tie_break':'first of I,T,IT; tie count recorded separately'}
        else:
            summary[group][key]=describe(values)
            if group=='A3' and key.endswith(('/M_first','/M_all')):
                summary[group][key]['negative_fraction']=float(np.mean(values<0))
    summary['global_score_gap']=pooled_gap;summary['anisotropy']=anisotropy
    summary['protocol']=settings.identity();summary['native_logit_scale']=native_logit_scale
    plots['histogram_edges']=np.linspace(-1,1,settings.histogram_bins+1)
    return summary,per_query,plots


def low_temperature_diagnostic(per_query, image_norms, text_norms, mode='native'):
    """Optional algebra check on explicitly supplied first-batch observations."""
    difference=np.abs(per_query[f'A6/{mode}/I->T/CE']-per_query[f'A6/{mode}/T->I/CE'])
    weak=np.where(image_norms<text_norms,per_query[f'A6/{mode}/IT->I/CE'],per_query[f'A6/{mode}/IT->T/CE'])
    return {'abs_directional_difference':describe(difference),'IT_to_weak_CE':describe(weak),
            'residual':describe(weak-difference),'training_pass_fail':False,
            'scope':'diagnostic approximation; caller must establish first-batch identity'}
