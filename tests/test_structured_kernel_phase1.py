import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from analyse.structured_kernel_phase1.gp import (cpu_setup, tensor, StaticExactGP, settings,
    independent_nll, parameter_values, fit_start, select_start, predict_frozen, make_inputs, positive_median)
from analyse.structured_kernel_phase1.io import sha256, read_json


def test_exact_kernel_formula_and_parameter_counts():
    cpu_setup()
    x=tensor([[0,0,0,0,0,0],[1,2,.1,.2,.3,.4],[1,2,.1,.2,.3,.4],[-1,1,1,0,1,0]])
    y=tensor([0,1,.9,-1]);known=tensor([0,1e-12,.01,.02])
    def matern(d):
        r=np.sqrt(5)*d
        return (1+r+r*r/3)*np.exp(-r)
    a=matern(np.linalg.norm(x.numpy()[:,None,:2]-x.numpy()[None,:,:2],axis=-1))
    h=matern(np.linalg.norm(x.numpy()[:,None,2:]-x.numpy()[None,:,2:],axis=-1))
    for method in ("L-A","L-I","W-A","W-I"):
        m=StaticExactGP(x,y,known,method,2)
        with torch.no_grad(): k=m.covar_module(x).to_dense().numpy()
        expected=.25*(a+h)+.5*a*h if method.endswith("I") else .5*(a+h)
        np.testing.assert_allclose(k,expected,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(np.diag(k),1.,atol=1e-12)
        assert sum(p.numel() for p in m.parameters())==5
        assert torch.equal(m.likelihood.noise_covar.noise,known)
        np.linalg.cholesky(k+np.diag(known.numpy()+.01))


def test_nll_units_normalized_once():
    import gpytorch
    cpu_setup()
    x=tensor([[0,0],[1,1],[2,-1]]);y=tensor([-.5,.1,.6]);known=tensor([.03,.07,.02])
    m=StaticExactGP(x,y,known,"RAW16",2)
    assert sum(p.numel() for p in m.parameters())==4
    with settings(1e-8):
        actual=-gpytorch.mlls.ExactMarginalLogLikelihood(m.likelihood,m)(m(x),y)
    assert float(actual.detach())==pytest.approx(independent_nll(m,x,y,known),abs=1e-10)
    assert parameter_values(m)["extra_noise"]==pytest.approx(.01)


def test_train_only_distance_scaling_and_raw16():
    rng=np.random.default_rng(22)
    x=rng.normal(size=(6,16))
    train={"x_standardized":x.tolist(),"scaler":{"x_mean":[0]*16,"x_scale":[1]*16},"wl":rng.normal(size=(6,30)).tolist()}
    query={"x_raw":rng.normal(size=(3,16)).tolist(),"wl":rng.normal(size=(3,30)).tolist(),
           "y":"DO NOT ACCESS","SE":"DO NOT ACCESS"}
    for method in ("RAW16","L-A","W-I"):
        tx,qx,dim,scales=make_inputs(train,query,method)
        poison={**query,"x_raw":(np.asarray(query["x_raw"])*1e6).tolist(),"y":object(),"SE":object()}
        tx2,_,_,scales2=make_inputs(train,poison,method)
        assert torch.equal(tx,tx2) and scales==scales2
        if method=="RAW16":
            np.testing.assert_allclose(tx.numpy(),x/positive_median(x)["value"])
    assert positive_median(np.zeros((3,4)))=={"value":1.,"degenerate":True,"positive_pairs":0}
    assert "query" not in inspect.signature(fit_start).parameters


def test_fit_budget_resume_and_frozen_prediction(tmp_path):
    cpu_setup()
    x=tensor([[-1,0],[-.3,1],[.1,.5],[.8,-.5],[1,0]])
    y=tensor([-.8,-.1,.2,.9,.7]);known=tensor([.01,.02,.02,.01,.015])
    path=tmp_path/"start0.pt"
    result=fit_start(x,y,known,"RAW16",2,0,path,"synthetic-identity")
    assert result["success"],result
    assert result["adam_updates"]==300
    assert result["final_nll"]==pytest.approx(result["independent_nll"],abs=1e-7)
    checkpoint_hash=sha256(path);log_hash=sha256(path.with_suffix(".jsonl"))
    resumed=fit_start(x,y,known,"RAW16",2,0,path,"synthetic-identity")
    assert resumed==result and sha256(path)==checkpoint_hash and sha256(path.with_suffix(".jsonl"))==log_hash
    # Simulate interruption between terminal checkpoint and JSON summary writes.
    path.with_suffix(".json").unlink()
    recovered=fit_start(x,y,known,"RAW16",2,0,path,"synthetic-identity")
    assert recovered==result and sha256(path)==checkpoint_hash and sha256(path.with_suffix(".jsonl"))==log_hash
    pred=predict_frozen(path,tensor([[0,0],[1,1]]))
    assert np.isfinite(pred["mean_standardized"]).all()
    assert np.min(pred["latent_variance_standardized"])>=-1e-8
    assert sha256(path)==checkpoint_hash
    with pytest.raises(ValueError,match="identity"):
        fit_start(x,y,known,"RAW16",2,0,path,"wrong-identity")


def test_selection_ignores_query_metrics_and_breaks_exact_ties():
    results=[{"success":True,"start_id":2,"final_nll":1.,"Q_MAE":0},
             {"success":True,"start_id":0,"final_nll":1.,"Q_MAE":1000},
             {"success":False,"start_id":1,"final_nll":-100,"Q_MAE":-100}]
    assert select_start(results)==0
