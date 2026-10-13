"""Current support AND query GP guidance; only GP parameters and labels are frozen."""
from __future__ import annotations
import time
import torch
from surrogate.gp_mean_guidance import matern52,solve_cholesky

class SolveMeter:
    """Count actual conditioning forwards separately from optimizer updates."""
    def __init__(self,records=None): self.records=records if records is not None else {}
    def record(self,category,seconds,jitter,attempts):
        r=self.records.setdefault(category,dict(conditionings=0,seconds=0.,cholesky_attempts=0,max_jitter=0.,nonzero_jitter=0))
        r['conditionings']+=1;r['seconds']+=seconds;r['cholesky_attempts']+=len(attempts)
        r['max_jitter']=max(r['max_jitter'],jitter);r['nonzero_jitter']+=jitter>0

def synchronized_mean(support,query,target,se,scale,parameters,meter=None,category='additional'):
    started=time.monotonic()
    if target.requires_grad or se.requires_grad:raise ValueError('labels and SE must be frozen')
    if any(isinstance(v,torch.Tensor) and v.requires_grad for v in parameters.values()):raise ValueError('GP parameters must be frozen')
    pars={k:float(v) for k,v in parameters.items()}
    s=support.to(torch.float64);q=query.to(torch.float64)
    if target.dtype!=torch.float64 or se.dtype!=torch.float64:raise ValueError('original standardized targets and SE must be float64')
    # vector_norm supplies zero derivative at coincident coordinates. No bare sqrt(0).
    covariance=matern52(s,s,pars['lengthscale'],pars['outputscale'])+torch.diag((se/scale).square()+pars['noise'])
    factor,jitter,attempts=solve_cholesky(covariance)
    alpha=torch.cholesky_solve((target-pars['mean'])[:,None],factor).squeeze(-1)
    mu=pars['mean']+matern52(q,s,pars['lengthscale'],pars['outputscale'])@alpha
    if meter is not None:meter.record(category,time.monotonic()-started,jitter,attempts)
    return mu

def guidance(embedding,target64,se,scale,groups,teachers,meter=None,category='additional',query_embedding=None,support_embedding=None):
    if len(groups)!=3 or len(teachers)!=3:raise ValueError('exactly three architecture groups required')
    q=embedding if query_embedding is None else query_embedding
    s=embedding if support_embedding is None else support_embedding
    losses=[]
    for group,teacher in zip(groups,teachers):
        si,qi=group['support_indices'],group['query_indices']
        mu=synchronized_mean(s[si],q[qi],target64[si],se[si],scale,teacher['parameters'],meter,category)
        losses.append((mu-target64[qi]).square())
    joined=torch.cat(losses)
    if len(joined)!=len(embedding):raise ValueError('groups must cover all candidates exactly once')
    return joined.mean()

def objective(e,p,target32,target64,se,scale,groups=None,teachers=None,enabled=True,meter=None):
    regression=(p-target32).square().mean();size=e.square().sum(-1).mean()
    guide=guidance(e,target64,se,scale,groups,teachers,meter,'formal_mapping_forward') if enabled else e.sum()*0.
    total=regression+.001*size+.1*guide if enabled else regression+0.*guide+.001*size
    return dict(total=total,regression=regression,size=size,guidance=guide)

def gradient_probe(model,train,target32,target64,se,scale,groups,teachers,meter=None):
    e,p=model(train['z'],train['hp']);regression=(p-target32).square().mean()
    guide=guidance(e,target64,se,scale,groups,teachers,meter,'probe_full')
    named=list(model.named_parameters());shared=[v for n,v in named if not n.startswith('regressor.')]
    grads=torch.autograd.grad(guide,[e]+[v for _,v in named],retain_graph=True,allow_unused=True)
    ge=grads[0];q=e.detach().clone().requires_grad_(True);s=e.detach().clone().requires_grad_(True)
    split=guidance(e,target64,se,scale,groups,teachers,meter,'probe_split',query_embedding=q,support_embedding=s)
    gq,gs=torch.autograd.grad(split,[q,s]);assert torch.allclose(ge,gq+gs,atol=2e-6,rtol=2e-5)
    qparams=torch.autograd.grad(e,shared,grad_outputs=gq,retain_graph=True)
    sparams=torch.autograd.grad(e,shared,grad_outputs=gs,retain_graph=True)
    regparams=torch.autograd.grad(regression,shared,retain_graph=True)
    mapping=[(n,g) for (n,_),g in zip(named,grads[1:])]
    assert all(g is None for n,g in mapping if n.startswith('regressor.'))
    assert all(g is not None and torch.isfinite(g).all() for n,g in mapping if not n.startswith('regressor.'))
    cat=lambda vs:torch.cat([v.reshape(-1).double() for v in vs])
    full=cat([g for n,g in mapping if not n.startswith('regressor.')]);query=cat(qparams);support=cat(sparams);reg=cat(regparams)
    assert torch.allclose(full,query+support,atol=2e-5,rtol=2e-5)
    norm=lambda v:float(v.double().norm());ng,nr=norm(full),norm(reg)
    cosine=lambda a,b:float(torch.dot(a,b)/(a.norm()*b.norm())) if a.norm()>0 and b.norm()>0 else None
    return dict(guidance_loss=float(guide.detach()),regression_loss=float(regression.detach()),guidance_shared_norm=ng,
      query_shared_norm=norm(query),support_shared_norm=norm(support),regression_shared_norm=nr,weighted_ratio=.1*ng/nr if nr else None,
      cosine=cosine(full,reg),query_regression_cosine=cosine(query,reg),support_regression_cosine=cosine(support,reg),query_support_cosine=cosine(query,support),
      embedding_full_norm=norm(ge),embedding_query_norm=norm(gq),embedding_support_norm=norm(gs),embedding_decomposition_maxdiff=float((ge-gq-gs).abs().max()),
      shared_decomposition_maxdiff=float((full-query-support).abs().max()),translation_gradient_sum_norm=norm(ge.double().sum(0)),
      parameter_gradient_norms={n:None if g is None else norm(g) for n,g in mapping},NA_reason=None if ng and nr else 'zero gradient norm',
      support_and_query_connected=True,head_guidance_gradient_is_None=True,frozen_input_requires_grad=bool(train['z'].requires_grad or train['hp'].requires_grad),
      interpretation='Unpreconditioned gradients, not Adam update directions; full = query + support.')
