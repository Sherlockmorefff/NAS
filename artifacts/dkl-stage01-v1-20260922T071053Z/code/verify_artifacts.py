"""Recompute selected-data transforms and saved posterior evidence without training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from controlled_dkl import Branch, compare, cpu_setup, independent_nll
from data_adapter import digest, mock_checks
from run_stage1 import all_pass, dump, load_protocol
from source_reference.fuzzy_metric import metric_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args(); root=args.root
    cpu_setup(0)
    protocol,sha=load_protocol(root)
    data=json.loads((root/"selected_data.json").read_text())
    x=np.asarray(data["x_raw"]); y=np.asarray(data["y"]); se=np.asarray(data["measurement_se"])
    xm=x.mean(0); xs=np.maximum(x.std(0),1e-6); ym=y.mean(); ys=max(y.std(),1e-8)
    checks={"selected_data_fingerprint":{"pass":data["identity"]==digest({k:v for k,v in data.items() if k!="identity"})},
            "mock_label_isolation":mock_checks()}
    comparisons={"x":compare(torch.tensor((x-xm)/xs),torch.tensor(data["x_standardized"])),
                 "y":compare(torch.tensor((y-ym)/ys),torch.tensor(data["y_standardized"])),
                 "known_noise":compare(torch.tensor((se/ys)**2),torch.tensor(data["known_variance_standardized"])),
                 "se_from_three_seeds":compare(torch.tensor(np.asarray(data["validation_seeds"]).std(1,ddof=1)/np.sqrt(3)),torch.tensor(se))}
    checks["selected_only_standardization"]={"pass":all_pass(comparisons),**comparisons}
    query=torch.tensor(json.loads((root/"unlabeled_probe_inputs.json").read_text()))
    endpoints={}
    for run in ("seed0","seed1","seed0_replay"):
        expected=json.loads((root/f"{run}_results.json").read_text())
        for name in ("frozen","regression","dkl"):
            state=torch.load(root/"checkpoints"/f"{run}_{name}_endpoint.pt",map_location="cpu",weights_only=False)
            model=Branch.restore(state)
            mean,variance=model.predict(query)
            stored=expected["endpoints"][name]
            item={"state_exact":model.fingerprint()==stored["state"],
                  "kernel":compare(model.kernel(),torch.tensor(stored["kernel"])),
                  "mean":compare(mean,torch.tensor(stored["mean"])),
                  "variance":compare(variance,torch.tensor(stored["variance"])),
                  "independent_nll":independent_nll(model)}
            item["pass"]=item["state_exact"] and all_pass(item)
            endpoints[f"{run}/{name}"]=item
    checks["saved_endpoints"]={"pass":all_pass(endpoints),"items":endpoints}
    # Direct parity with the audited C target, including mapping and head gradients.
    state=torch.load(root/"checkpoints/seed0_shared_init.pt",map_location="cpu",weights_only=False)
    a,b=Branch.restore(state),Branch.restore(state)
    for model in (a,b):
        for parameter in model.mapper.parameters():
            parameter.requires_grad_(True)
        model.mapper.zero_grad(set_to_none=True)
    loss,_=a.objective("regression");loss.backward()
    embedding,pred=b.encode(b.x)
    reference=metric_loss(embedding,pred,b.y,torch.zeros(len(b.y),len(b.y)),lambda_f=0,lambda_e=.001)["total"]
    reference.backward()
    gradients={k:compare(p.grad,dict(b.mapper.named_parameters())[k].grad) for k,p in a.mapper.named_parameters()}
    checks["current_C_objective_and_gradient_parity"]={"pass":compare(loss,reference)["pass"] and all_pass(gradients),
        "loss":compare(loss,reference),"gradients":gradients}
    checks["overall"]={"pass":all_pass(checks),"training_updates":0,"protocol_sha256":sha,
                       "checkpoint_count":9,"B_or_hidden_label_access":False}
    dump(root/"saved_artifact_checks.json",checks)
    print(json.dumps(checks["overall"]))
    if not checks["overall"]["pass"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
