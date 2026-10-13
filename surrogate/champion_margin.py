"""Config-driven budget ranking and label-free selection for the margin experiment.

Derived narrowly from budget_champion_ranking.py; original functions are unchanged.
"""
import hashlib
import math
import numpy as np
import torch


def list_indices(ids, stage, task_id, event, step, config):
    if stage != 'replay' or step < 0 or len(set(ids)) != len(ids):
        raise ValueError('invalid list identity')
    result=[]
    for li in range(config['list_count']):
        prefix=f"{config['list_prefix']}:{stage}:{task_id}:{event}:{step}:{li}:"
        order=sorted(range(len(ids)),key=lambda i:(hashlib.sha256((prefix+ids[i]).encode('utf-8')).hexdigest(),ids[i]))
        result.append(order[:min(config['list_size'],len(ids))])
    return result


def objective(embedding,scores,target,raw_y,ids,lists,rho,sigma,config,mode,weight):
    if mode not in ['fixed','gap'] or len(lists)!=config['list_count'] or raw_y.requires_grad or sigma<=0:
        raise ValueError('invalid ranking inputs')
    regression=(scores-target).square().mean();size=embedding.square().sum(-1).mean()
    losses=[];margins=[];true_ranks=[];prediction_ranks=[];ks=[];winners_log=[];competitors=[]
    for indices in lists:
        if not indices or len(indices)!=len(set(indices)):raise ValueError('invalid list')
        k=min(len(indices),max(1,math.ceil(len(indices)*rho)));ks.append(k)
        js=sorted(indices,key=lambda i:ids[i]);peak=raw_y[js].max()
        winners=[i for i in js if abs(float(raw_y[i]-peak))<=config['tie_tolerance']]
        losers=[i for i in js if i not in winners]
        wi=torch.tensor(winners,device=scores.device)[torch.argsort(scores[winners],descending=True,stable=True)[0]]
        champion_score=scores[wi]
        ordered=torch.tensor(js,device=scores.device)[torch.argsort(scores[js],descending=True,stable=True)].tolist()
        prediction_ranks.append(min(ordered.index(i)+1 for i in winners))
        true_ranks.append(1+int((raw_y>peak+config['tie_tolerance']).sum()));winners_log.append(int(wi))
        if len(losers)<k:
            loss=scores.sum()*0.;margin=0.;ci=None
        else:
            ci=torch.tensor(losers,device=scores.device)[torch.argsort(scores[losers],descending=True,stable=True)[k-1]]
            margin=config['margin_cap'] if mode=='fixed' else min(config['margin_cap'],config['gap_factor']*float((raw_y[wi]-raw_y[ci]).detach())/sigma)
            loss=torch.relu(margin+scores[ci]-champion_score).square()
        losses.append(loss);margins.append(margin);competitors.append(None if ci is None else int(ci))
    ranking=torch.stack(losses).mean()
    total=regression+config['size_weight']*size+weight*ranking
    diag=dict(true_label_rank_in_known=true_ranks,predicted_champion_rank_in_list=prediction_ranks,k=ks,
        winner_indices=winners_log,competitor_indices=competitors,margins=margins,margin_min=min(margins),margin_max=max(margins),
        margin_mean=sum(margins)/len(margins),triggered_lists=sum(float(v.detach())>0 for v in losses),list_count=len(lists))
    return dict(total=total,regression=regression,size=size,ranking=ranking),diag


def select_batch(rows,known,seed,batch,config,strategy):
    ids=[r['candidate_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)&set(known):raise ValueError('candidate identity/known exclusion')
    if rows!=sorted(rows,key=lambda r:(-r['score'],r['candidate_id'])):raise ValueError('unstable score order')
    n=config['batch_size']
    if strategy=='greedy':return dict(selected_ids=ids[:n],sources=['greedy']*n,exploration=None)
    if strategy!='mixed' or config['exploration_slots']!=1:raise ValueError('unsupported strategy')
    greedy=ids[:n-config['exploration_slots']];eligible=sorted(set(ids)-set(greedy))
    text=f"{config['exploration_prefix']}:{seed}:{batch}";derived=int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest(),'big')
    generator=np.random.Generator(np.random.PCG64(derived));index=int(generator.integers(len(eligible)))
    canonical=__import__('json').dumps(eligible,sort_keys=True,separators=(',',':')).encode('utf-8')
    return dict(selected_ids=greedy+[eligible[index]],sources=['greedy']*len(greedy)+['exploration'],
        exploration=dict(seed_text=text,derived_seed=str(derived),generator='numpy.Generator(PCG64)',encoding='UTF-8 SHA256, full digest big-endian unsigned integer',
            eligible_count=len(eligible),eligible_ids_sha256=hashlib.sha256(canonical).hexdigest(),sampled_index=index,candidate_id=eligible[index]))
