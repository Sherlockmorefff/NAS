"""Differentiable mean guidance with immutable detached GP conditioning state."""
from __future__ import annotations
import copy,hashlib,math
import torch

def architecture_groups(ids,architectures):
    ordered=sorted(set(architectures),key=lambda a:(hashlib.sha256(('gp_mean_guidance_v1:'+a).encode()).hexdigest(),a))
    assignment={a:i%3 for i,a in enumerate(ordered)}
    result=[]
    for g in range(3):
        query=[i for i,a in enumerate(architectures) if assignment[a]==g];support=[i for i,a in enumerate(architectures) if assignment[a]!=g]
        if not query or not support:raise ValueError('three nonempty architecture groups required')
        result.append(dict(group=g,query_indices=query,support_indices=support,query_ids=[ids[i] for i in query],support_ids=[ids[i] for i in support],
          query_architectures=sorted({architectures[i] for i in query}),support_architectures=sorted({architectures[i] for i in support})))
    return result

def matern52(x,support,lengthscale,outputscale):
    x=x.to(torch.float64)
    distance=torch.linalg.vector_norm((x[:,None,:]-support[None,:,:])/lengthscale,dim=-1)
    r=math.sqrt(5.)*distance
    return outputscale*(1.+r+r.square()/3.)*torch.exp(-r)

def solve_cholesky(matrix):
    """Match psd_safe_cholesky: no jitter if PD; at most three double retries."""
    identity=torch.eye(len(matrix),dtype=matrix.dtype,device=matrix.device)
    attempts=[]
    for jitter in [0.,1e-6,1e-5,1e-4]:
        factor,info=torch.linalg.cholesky_ex(matrix+jitter*identity)
        attempts.append(dict(jitter=jitter,positive_definite=int(info.max())==0))
        if int(info.max())==0:return factor,jitter,attempts
    raise RuntimeError('Cholesky failed under fixed retry rule')

def condition(support,target,se,scale,parameters,metadata=None):
    support=support.detach().double().clone();target=target.detach().double().clone();se=se.detach().double().clone()
    if support.requires_grad or target.shape!=(len(support),) or se.shape!=target.shape:raise ValueError('invalid detached support')
    pars={k:float(v) for k,v in parameters.items()}
    covariance=matern52(support,support,pars['lengthscale'],pars['outputscale'])+torch.diag((se/scale).square()+pars['noise'])
    factor,jitter,attempts=solve_cholesky(covariance)
    alpha=torch.cholesky_solve((target-pars['mean'])[:,None],factor).squeeze(-1)
    return dict(support=support,target=target,se=se,scale=float(scale),parameters=pars,cholesky=factor.detach(),alpha=alpha.detach(),
                jitter=jitter,cholesky_attempts=attempts,metadata=copy.deepcopy(metadata or {}))

def from_gp(gp,support,target,se,scale,metadata=None):
    for p in gp.parameters():p.requires_grad_(False)
    gp.eval();gp.likelihood.eval()
    pars=dict(mean=float(gp.mean_module.constant.detach()),lengthscale=float(gp.covar_module.base_kernel.lengthscale.detach()),
        outputscale=float(gp.covar_module.outputscale.detach()),noise=float(gp.likelihood.second_noise.detach()))
    return condition(support,target,se,scale,pars,metadata)

def mean(teacher,query):
    p=teacher['parameters']
    return p['mean']+matern52(query,teacher['support'],p['lengthscale'],p['outputscale'])@teacher['alpha']

def posterior(teacher,query):
    p=teacher['parameters'];cross=matern52(query,teacher['support'],p['lengthscale'],p['outputscale'])
    prediction=p['mean']+cross@teacher['alpha']
    v=torch.linalg.solve_triangular(teacher['cholesky'],cross.T,upper=False)
    raw=p['outputscale']-v.square().sum(0)
    if float(raw.min()) < -1e-9:raise ValueError('materially negative posterior variance')
    return prediction,raw.clamp_min(0),raw

def guidance(embedding,target64,groups,teachers):
    if target64.dtype!=torch.float64 or target64.shape!=(len(embedding),):raise ValueError('guidance target must be original standardized float64')
    if len(groups)!=3 or len(teachers)!=3:raise ValueError('three fixed teachers required')
    losses=[]
    for group,teacher in zip(groups,teachers):
        indices=group['query_indices'];losses.append((mean(teacher,embedding[indices])-target64[indices]).square())
    joined=torch.cat(losses)
    if len(joined)!=len(embedding):raise ValueError('guidance must cover each training candidate once')
    return joined.mean()

def objective(embedding,prediction,target32,target64,groups=None,teachers=None,enabled=False):
    regression=(prediction-target32).square().mean();size=embedding.square().sum(-1).mean()
    guide=guidance(embedding,target64,groups,teachers) if enabled else embedding.sum()*0.
    # Identical arithmetic order to the verified R0 when guidance is disabled.
    total=regression+0.*guide+.001*size if not enabled else regression+.001*size+.1*guide
    return dict(total=total,regression=regression,size=size,guidance=guide)

def gradient_probe(model,train,target32,target64,groups,teachers):
    e,p=model(train['z'],train['hp']);loss=(p-target32).square().mean();guide=guidance(e,target64,groups,teachers)
    named=list(model.named_parameters());shared=[p for n,p in named if not n.startswith('regressor.')]
    gg=torch.autograd.grad(guide,[e]+[p for _,p in named],retain_graph=True,allow_unused=True)
    gr=torch.autograd.grad(loss,shared,retain_graph=True)
    mapping=[(n,g) for (n,_),g in zip(named,gg[1:])]
    if any(g is None or not torch.isfinite(g).all() for n,g in mapping if not n.startswith('regressor.')):raise ValueError('guidance does not reach mapping/HP')
    if any(g is not None for n,g in mapping if n.startswith('regressor.')):raise ValueError('guidance unexpectedly depends on linear head')
    vec=torch.cat([g.reshape(-1) for n,g in mapping if not n.startswith('regressor.')]);reg=torch.cat([g.reshape(-1) for g in gr]);ng,nr=float(vec.double().norm()),float(reg.double().norm())
    return dict(guidance_loss=float(guide.detach()),regression_loss=float(loss.detach()),guidance_query_gradient_norm=float(gg[0].double().norm()),
      guidance_shared_norm=ng,regression_shared_norm=nr,weighted_ratio=.1*ng/nr if nr else None,
      cosine=float(torch.dot(vec.double(),reg.double())/(ng*nr)) if ng and nr else None,NA_reason=None if ng and nr else 'zero gradient norm',
      parameter_gradient_norms={n:None if g is None else float(g.double().norm()) for n,g in mapping},
      HP_and_mapping_connected=True,head_guidance_gradient_is_None=True,
      frozen_support_gradients_none=all(not t['support'].requires_grad and not t['alpha'].requires_grad for t in teachers))
