"""Read-only gradient diagnostics and isolated one-step Adam branches."""
from __future__ import annotations

import copy
from pathlib import Path
import time
import torch

from analyse.fuzzy_loss_interaction_core import (
    PERF, adam, capture, components, embedding_cache, load, neighborhood_affinity,
    predict, reload_model, restore_rng, rng_state, save, tree_hash, write,
)

ZERO = 1e-12


def cosine(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float(torch.dot(a, b)/(na*nb)) if min(na, nb) > ZERO else None


def gradients(model, data):
    before = tree_hash(dict(model=model.state_dict(), rng=rng_state(), grads=[p.grad for p in model.parameters()]))
    losses, numbers, _, _ = components(model, data, 1.)
    params = list(model.named_parameters())
    vectors = {}
    for name in ("regression", "neighborhood", "scale"):
        started = time.perf_counter()
        grad = torch.autograd.grad(losses[name], [p for _,p in params], retain_graph=True, allow_unused=True)
        PERF["diagnostic_backward_seconds"] += time.perf_counter()-started
        PERF["diagnostic_backward_calls"] += 1
        vectors[name] = {n: (g if g is not None else torch.zeros_like(p)).detach().reshape(-1).double().clone()
                         for (n,p),g in zip(params,grad)}
        assert all(torch.isfinite(g).all() for g in vectors[name].values())
    assert before == tree_hash(dict(model=model.state_dict(), rng=rng_state(), grads=[p.grad for p in model.parameters()]))
    return vectors, numbers


def flat(vectors, loss, block="shared"):
    def include(name):
        return not name.startswith("regressor.") if block == "shared" else name.startswith(block+".")
    return torch.cat([g for name,g in vectors[loss].items() if include(name)])


def gradient_rows(vectors, numbers, metadata, weight, hypothetical):
    rows = []
    for block in ("shared", "hp_encoder", "mapping", "regressor"):
        r, f, e = (flat(vectors, name, block) for name in ("regression", "neighborhood", "scale"))
        fw, ew = weight*f, .001*e
        nr = float(r.norm())
        row = dict(metadata, block=block, lambda_f=weight, hypothetical_lambda_f=hypothetical,
                   regression_norm=nr, fuzzy_raw_norm=float(f.norm()), fuzzy_weighted_norm=float(fw.norm()),
                   norm_weighted_norm=float(ew.norm()), total_norm=float((r+fw+ew).norm()),
                   cosine_rf=cosine(r,fw), cosine_fe=cosine(fw,ew),
                   R_f=float(fw.norm())/nr if nr>ZERO else None, R_e=float(ew.norm())/nr if nr>ZERO else None,
                   hypothetical_fuzzy_norm=float((hypothetical*f).norm()), hypothetical_cosine_rf=cosine(r,hypothetical*f),
                   hypothetical_R_f=float((hypothetical*f).norm())/nr if nr>ZERO else None,
                   near_zero_threshold=ZERO, cosine_status="defined" if min(nr,float(fw.norm()))>ZERO else "undefined/near_zero",
                   **{k:v for k,v in numbers.items() if k not in ["total","weighted_neighborhood","weighted_pull","weighted_separation"]})
        row.update(total=numbers["regression"]+weight*numbers["neighborhood"]+.001*numbers["scale"],
                   weighted_neighborhood=weight*numbers["neighborhood"],weighted_pull=weight*numbers["pull"],
                   weighted_separation=weight*numbers["separation"])
        rows.append(row)
    return rows


def diagnose_checkpoint(checkpoint, data, weight, hypothetical, folder, metadata):
    folder = Path(folder)
    state = load(checkpoint)
    model = reload_model(state["model"])
    vectors,numbers = gradients(model,data)
    meta = dict(metadata, completed_updates=state["completed_updates"],phase=state["phase"],
                model_hash=state["model_hash"], optimizer_hash=state["optimizer_hash"])
    rows = gradient_rows(vectors,numbers,meta,weight,hypothetical)
    save(folder/"gradients.pt",dict(metadata=meta,vectors=vectors,weights=dict(fuzzy=weight,norm=.001,hypothetical=hypothetical)))
    write(folder/"rows.json",rows)
    return rows


def one_step(checkpoint, data, hypothetical, folder, ledger, metadata):
    folder = Path(folder)
    original_rng = rng_state()
    state = load(checkpoint)
    original_hash = tree_hash(state)
    model = reload_model(state["model"])
    before_cache = embedding_cache(model,data)
    unknown = sorted(set(data["all_ids"])-set(data["ids"]))
    before_predictions = predict(before_cache,data["ids"],unknown)
    train_before = {c:float(before_cache["head"][before_cache["ids"].index(c)]) for c in data["ids"]}
    write(folder/"before_predictions.json",dict(query=before_predictions,train=train_before),immutable=True)
    results = []
    for group,weight in (("C",0.),("F",hypothetical),("W",.1*hypothetical)):
        branch = folder/group
        if (branch/"result.json").exists():
            results.append(__import__("json").loads((branch/"result.json").read_text()))
            continue
        model = reload_model(state["model"])
        optimizer = adam(model.parameters(),.001)
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        restore_rng(state["rng"])
        assert tree_hash(optimizer.state_dict()) == state["optimizer_hash"]
        optimizer.zero_grad(set_to_none=True)
        losses,_,_,_ = components(model,data,weight)
        losses["total"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        task = metadata["task"]+f"/probe/{state['completed_updates']}/{group}"
        gid = ledger.update("mapping",task,1,optimizer,"one_step_probe")
        post = capture(model,optimizer,state["completed_updates"]+1,state["identity"],dict(metadata,branch=group),state["scaler"])
        post.update(source_completed_updates=state["completed_updates"],branch_only_updates=1,source_optimizer_hash=state["optimizer_hash"])
        delta = {n: (p.detach()-state["model"][n]).reshape(-1).double() for n,p in model.named_parameters()}
        post["actual_parameter_delta"] = delta
        save(branch/"post_one_step.pt",post)
        _,numbers,_,_ = components(model,data,weight)
        cache = embedding_cache(model,data)
        predictions = predict(cache,data["ids"],unknown)
        train_after = {c:float(cache["head"][cache["ids"].index(c)]) for c in data["ids"]}
        write(branch/"after_predictions.json",dict(query=predictions,train=train_after),immutable=True)
        actual = dict(metadata,branch=group,lambda_f=weight,completed_updates=state["completed_updates"],
                      source_model_hash=state["model_hash"],source_optimizer_hash=state["optimizer_hash"],
                      source_rng_hash=tree_hash(state["rng"]),post_model_hash=post["model_hash"],ledger_id=gid,post_losses=numbers)
        for block in ("shared","hp_encoder","mapping","regressor"):
            vec = torch.cat([d for n,d in delta.items() if (not n.startswith("regressor.") if block=="shared" else n.startswith(block+"."))])
            actual[block+"_actual_delta_norm"] = float(vec.norm())
        write(branch/"result.json",actual,immutable=True)
        results.append(actual)
    assert original_hash == tree_hash(state)
    restore_rng(original_rng)
    write(folder/"complete.json",dict(branches=3,source_hash=original_hash))
    return results


def loss_shares(model,data):
    with torch.no_grad():
        e,_ = model(data["z"],data["hp"])
        i,j = torch.triu_indices(len(e),len(e),1)
        q=(1/(1+(e[i]-e[j]).square().sum(-1))).clamp(max(torch.finfo(e.dtype).eps,1e-7),1-max(torch.finfo(e.dtype).eps,1e-7))
        p=data["affinity"].float()[i,j]
        pair=data["weights"].float()[i,j]*(-p*q.log()-(1-p)*torch.log1p(-q))
        denom=float(pair.double().sum())
        if denom<=ZERO:
            return {c:None for c in data["ids"]}
        result=torch.zeros(len(e),dtype=torch.float64)
        result.index_add_(0,i,pair.double());result.index_add_(0,j,pair.double())
        result/=2*denom
        assert abs(float(result.sum())-1)<1e-12
        return dict(zip(data["ids"],result.tolist()))


def sensitivity(checkpoint,data,hypothetical,folder,metadata):
    folder=Path(folder)
    state=load(checkpoint)
    model=reload_model(state["model"])
    base,numbers=gradients(model,data)
    vectors={"base":base}
    i,j=torch.triu_indices(len(data["ids"]),len(data["ids"]),1)
    p0=data["affinity"][i,j]
    rows=[];targets=[];shares=[]
    for removed in (None,0,1,2):
        variant=dict(data)
        if removed is None:
            name="three_seed_base";current=base
        else:
            name=f"remove_seed_{removed}"
            y=data["seeds"][:,[k for k in range(3) if k!=removed]].mean(1)
            variant.update(y=y,target64=(y-data["scaler"]["y_mean"])/data["scaler"]["y_scale"],
                           target32=((y-data["scaler"]["y_mean"])/data["scaler"]["y_scale"]).float(),
                           affinity=neighborhood_affinity(y,mode="N2",epsilon=data["epsilon"]))
            current,_=gradients(model,variant)
            vectors[name]=current
            p=variant["affinity"][i,j];change=(p-p0).abs()
            r,f=flat(current,"regression"),flat(current,"neighborhood")
            r0,f0=flat(base,"regression"),flat(base,"neighborhood")
            row=dict(metadata,variant=name,removed_gnn_seed=removed,completed_updates=state["completed_updates"],
                     model_hash=state["model_hash"],hypothetical_lambda_f=hypothetical,
                     target_absolute_change_mean=float((y-data["y"]).abs().mean()),p_absolute_change_mean=float(change.mean()),
                     p_change_quantiles=dict(zip(["q0","q25","q50","q75","q95","q100"],torch.quantile(change,torch.tensor([0,.25,.5,.75,.95,1.],dtype=torch.float64)).tolist())),
                     switched_positive_to_zero=float(((p0>0)&(p==0)).double().mean()),switched_zero_to_positive=float(((p0==0)&(p>0)).double().mean()),
                     regression_norm=float(r.norm()),base_regression_norm=float(r0.norm()),regression_cosine_to_base=cosine(r,r0),
                     fuzzy_raw_norm=float(f.norm()),base_fuzzy_raw_norm=float(f0.norm()),fuzzy_raw_cosine_to_base=cosine(f,f0),
                     fuzzy_hypothetical_norm=float((hypothetical*f).norm()),base_fuzzy_hypothetical_norm=float((hypothetical*f0).norm()),
                     fuzzy_hypothetical_cosine_to_base=cosine(hypothetical*f,hypothetical*f0),
                     regression_fuzzy_raw_cosine=cosine(r,f),weights_hash=tree_hash(data["weights"]),epsilon=data["epsilon"])
            rows.append(row)
            targets.extend(dict(metadata,variant=name,candidate_id=c,base_target=float(data["y"][k]),variant_target=float(y[k]),
                                raw_change=float(y[k]-data["y"][k]),standardized_change=float(variant["target64"][k]-data["target64"][k])) for k,c in enumerate(data["ids"]))
        shares.extend(dict(metadata,variant=name,candidate_id=c,share=s) for c,s in loss_shares(model,variant).items())
    save(folder/"vectors.pt",dict(vectors=vectors,metadata=metadata,model_hash=state["model_hash"],hypothetical_lambda_f=hypothetical))
    write(folder/"rows.json",rows);write(folder/"targets.json",targets);write(folder/"loss_shares.json",shares)
    return rows
