"""Budget-ranking approximation on revealed labels; same head trains and selects."""
from __future__ import annotations
import hashlib
import math
import torch


def list_indices(ids, stage, task_id, update_event, optimizer_step):
    if stage not in ['offline','replay'] or optimizer_step < 0 or len(set(ids)) != len(ids):
        raise ValueError('invalid deterministic list identity')
    result=[]
    for li in range(16):
        prefix=f'budget_champion_v1:list:{stage}:{task_id}:{update_event}:{optimizer_step}:{li}:'
        order=sorted(range(len(ids)),key=lambda i:(hashlib.sha256((prefix+ids[i]).encode('utf-8')).hexdigest(),ids[i]))
        result.append(order[:min(16,len(ids))])
    return result


def budget_loss(scores, raw_y, ids, lists, rho):
    """Stable tensor order statistics keep both competing scores differentiable."""
    if scores.ndim!=1 or raw_y.shape!=scores.shape or not 0<rho<=1:
        raise ValueError('invalid scores, frozen raw labels or budget fraction')
    if raw_y.requires_grad or len(lists)!=16:
        raise ValueError('labels must be frozen; exactly 16 lists required')
    losses=[];ranks=[];ks=[];winner_counts=[];champion_positions=[];competitor_positions=[]
    for indices in lists:
        m=len(indices)
        if not m or len(set(indices))!=m:
            raise ValueError('each list must contain distinct candidates')
        k=min(m,max(1,math.ceil(m*rho)))
        # Sorting identities first makes stable score ties resolve by candidate ID.
        js=sorted(indices,key=lambda i:ids[i])
        peak=raw_y[js].max()
        winner_mask=(raw_y[js]-peak).abs()<=1e-12
        winners=[i for i,w in zip(js,winner_mask.tolist()) if w]
        losers=[i for i,w in zip(js,winner_mask.tolist()) if not w]
        best_order=torch.argsort(scores[winners],descending=True,stable=True)
        champion=torch.tensor(winners,device=scores.device)[best_order[0]]
        champion_score=scores[champion]
        if len(losers)<k:
            loss=scores.sum()*0.
            competitor=None
        else:
            order=torch.argsort(scores[losers],descending=True,stable=True)
            competitor=torch.tensor(losers,device=scores.device)[order[k-1]]
            loss=torch.relu(.1+scores[competitor]-champion_score).square()
        losses.append(loss)
        # Diagnostics use only D's raw labels; never the unknown candidate pool.
        ranks.append(1+int((raw_y>peak+1e-12).sum()))
        ks.append(k);winner_counts.append(len(winners))
        champion_positions.append(champion.detach())
        competitor_positions.append(None if competitor is None else competitor.detach())
    values=torch.stack(losses)
    return values.mean(),dict(nonzero_fraction=float((values.detach()>0).float().mean()),
        champion_ranks_in_D=ranks,k=ks,winner_counts=winner_counts,
        champion_indices=[int(x) for x in champion_positions],
        competitor_indices=[None if x is None else int(x) for x in competitor_positions])


def objective(embedding,scores,target,raw_y,ids,lists,rho,weight):
    regression=(scores-target).square().mean()
    size=embedding.square().sum(-1).mean()
    ranking,diagnostics=budget_loss(scores,raw_y,ids,lists,rho)
    base=regression+.001*size
    total=base+weight*ranking
    return dict(total=total,regression=regression,size=size,ranking=ranking),diagnostics
