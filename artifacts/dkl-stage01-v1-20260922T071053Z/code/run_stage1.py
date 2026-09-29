"""Run once on synthetic data, then the locked 20-observation engineering protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import gpytorch
import numpy as np
import torch

from controlled_dkl import Branch, Budget, compare, cpu_setup, tensor_hash, train_block
from engineering_checks import check_engineering
from data_adapter import digest, mock_checks


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as stream:
            writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def all_pass(value):
    if isinstance(value, dict):
        if "pass" in value and value["pass"] is not True:
            return False
        return all(all_pass(x) for x in value.values())
    if isinstance(value, list):
        return all(all_pass(x) for x in value)
    return True


def load_protocol(root):
    raw=(root/"stage1_protocol.json").read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    if sha != (root/"stage1_protocol.sha256").read_text().split()[0]:
        raise ValueError("Protocol changed since freeze")
    protocol=json.loads(raw)
    assert protocol["shared"] == {"regression_updates":100,"gp_updates":20,"save_full_math_state":True}
    assert protocol["model_seeds"] == [0,1] and protocol["replay_seeds"] == [0]
    assert protocol["gp"]["extra_noise_lower"] == 1e-6 and protocol["gp"]["lengthscale_lower"] == 1e-4
    assert protocol["next_stage_authorized"] is False
    return protocol,sha


def synthetic(root):
    cpu_setup(73)
    out=root/"synthetic"
    out.mkdir(exist_ok=False)
    x=torch.randn(8,16)
    y=torch.sin(x[:,0])+0.2*x[:,3]
    known=torch.linspace(0.001,0.008,8)
    model=Branch(x,y,known,0)
    traces=[]
    budget=Budget()
    tick=time.perf_counter()
    train_block(model,"regression",5,traces,"synthetic","shared","regression",budget)
    train_block(model,"gp",3,traces,"synthetic","shared","gp",budget)
    checks=check_engineering(model,torch.cat([x,torch.randn(3,16)]),out,budget)
    elapsed=time.perf_counter()-tick
    checks["numerical_probe"]={"pass":all(np.isfinite(r["loss_before_update"]) for r in traces),
                               "seconds":elapsed,"training_seconds":budget.seconds}
    dump(out/"checks.json",checks)
    write_csv(out/"traces.csv",traces)
    dump(out/"estimate.json",{"probe_wall_seconds":elapsed,"probe_updates":len(traces),
                             "estimated_real_seconds_conservative":max(60,elapsed*30),
                             "real_training_cap_seconds":1200,"estimate_uses_hidden_labels":False})
    if not all_pass(checks):
        raise AssertionError("Synthetic engineering checks failed")
    print(json.dumps({"mode":"synthetic","pass":True,"wall_seconds":elapsed,
                      "estimated_real_seconds_conservative":max(60,elapsed*30)}))


def data_tensors(data):
    if len(data["ids"])!=20 or len(set(data["ids"]))!=20:
        raise ValueError("Require exactly 20 unique allowed IDs")
    return tuple(torch.tensor(data[k],dtype=torch.double) for k in
                 ("x_standardized","y_standardized","known_variance_standardized"))


def run_seed(seed, run, x, y, known, query, root, budget, traces, verify, candidate_ids):
    cpu_setup(seed)
    model=Branch(x,y,known,seed)
    initial=model.fingerprint()
    checkpoint_dir=root/"checkpoints"
    tick=time.perf_counter()
    train_block(model,"regression",100,traces,run,"shared","common_regression",budget,checkpoint_dir)
    train_block(model,"gp",20,traces,run,"shared","common_gp",budget,checkpoint_dir)
    shared=model.fingerprint()
    initial_prediction=model.predict(query)
    common_snapshot=model.snapshot()
    torch.save(common_snapshot,checkpoint_dir/f"{run}_shared_init.pt")
    engineering={}
    if verify:
        engineering=check_engineering(model,query,root/"real_verification",budget)
        if not all_pass(engineering):
            raise AssertionError("Real engineering verification failed")
    branches={name:model.clone() for name in ("frozen","regression","dkl")}
    fork_info={name:{"fingerprint":b.fingerprint(),
                     "exact_state":b.fingerprint()==shared,
                     "history_matches":torch.equal(b.gp.train_inputs[0],common_snapshot["history_embedding"]),
                     "mean":compare(b.predict(query)[0],initial_prediction[0]),
                     "variance":compare(b.predict(query)[1],initial_prediction[1])}
               for name,b in branches.items()}
    if not all(v["exact_state"] and v["history_matches"] and all_pass(v) for v in fork_info.values()):
        raise AssertionError("Fork mismatch")
    endpoints={}
    for name,b in branches.items():
        before=tensor_hash(b.mapper.state_dict())
        if name=="frozen":
            train_block(b,"gp",20,traces,run,name,"frozen_gp",budget,checkpoint_dir)
        elif name=="regression":
            train_block(b,"regression",10,traces,run,name,"continued_regression",budget,checkpoint_dir)
            after_mapping=tensor_hash(b.mapper.state_dict())
            train_block(b,"gp",20,traces,run,name,"regression_gp",budget,checkpoint_dir)
            if tensor_hash(b.mapper.state_dict()) != after_mapping:
                raise AssertionError("Regression GP stage changed mapper/buffers")
        else:
            train_block(b,"joint",10,traces,run,name,"joint_dkl",budget,checkpoint_dir)
            after_mapping=tensor_hash(b.mapper.state_dict())
            train_block(b,"gp",10,traces,run,name,"dkl_gp_tail",budget,checkpoint_dir)
            if tensor_hash(b.mapper.state_dict()) != after_mapping:
                raise AssertionError("DKL GP tail changed mapper/buffers")
        mean,var=b.predict(query)
        if not torch.isfinite(mean).all() or not torch.isfinite(var).all() or (var < -1e-8).any():
            raise FloatingPointError("Bad endpoint posterior")
        endpoint=b.snapshot()
        torch.save(endpoint,checkpoint_dir/f"{run}_{name}_endpoint.pt")
        endpoints[name]={"state":b.fingerprint(),"mean":mean.tolist(),"variance":var.tolist(),
                         "kernel":b.kernel().tolist(),"mapper_unchanged":tensor_hash(b.mapper.state_dict())==before,
                         "parameter_counts":{"mapping":sum(p.numel() for p in b.mapping_params()),
                                             "head":sum(p.numel() for p in b.mapper.regressor.parameters()),
                                             "gp":sum(p.numel() for p in b.gp.parameters())}}
        if name=="frozen" and not endpoints[name]["mapper_unchanged"]:
            raise AssertionError("Frozen mapper changed")
    return {"seed":seed,"candidate_ids":list(candidate_ids),
            "selected_tensors_hash":tensor_hash({"x":x,"y":y,"known":known}),
            "initial_state":initial,"shared_state":shared,"fork":fork_info,
            "endpoints":endpoints,"engineering":engineering,"wall_seconds":time.perf_counter()-tick}


def real(root, protocol_sha):
    if (root/"stage1_checks.json").exists() or (root/"stage1_traces.csv").exists():
        raise FileExistsError("Preserve previous attempt; choose a new attempt directory and shared budget ledger")
    if not all_pass(json.loads((root/"synthetic/checks.json").read_text())):
        raise RuntimeError("Synthetic gate not passed")
    data=json.loads((root/"selected_data.json").read_text())
    selection=json.loads((root/"selection_manifest.json").read_text())
    access=json.loads((root/"data_access_audit.json").read_text())
    if data["identity"] != digest({k:v for k,v in data.items() if k!="identity"}):
        raise ValueError("Selected data fingerprint mismatch")
    if data["metadata"]["protocol_sha256"] != protocol_sha or selection["ids"] != data["ids"]:
        raise ValueError("Selected data protocol/order mismatch")
    mock=mock_checks()
    isolation_pass=(mock["pass"] and access["selection_saved_before_label_open"] and
        access["selected_label_rows_decoded"]==20 and access["hidden_label_rows_decoded"]==0 and
        access["unselected_SE_decoded"]==0 and access["B_label_files_opened"]==0 and access["test_metric_files_opened"]==0)
    if not isolation_pass:
        raise ValueError("Data isolation gate failed")
    x,y,known=data_tensors(data)
    cpu_setup(20260922)
    query=torch.cat((x,torch.randn(3,16)),dim=0)
    dump(root/"unlabeled_probe_inputs.json",query.tolist())
    budget=Budget(root/"real_training_budget.json")
    traces=[]; results={}; checks={"label_isolation":{"pass":isolation_pass,"mock":mock,"actual":access}}
    tick=time.perf_counter()
    failure=None
    try:
        for seed,run,verify in [(0,"seed0",True),(1,"seed1",False),(0,"seed0_replay",False)]:
            results[run]=run_seed(seed,run,x,y,known,query,root,budget,traces,verify,data["ids"])
            dump(root/f"{run}_results.json",results[run])
        original,replay=results["seed0"],results["seed0_replay"]
        checks["seed0_full_replay"]={"pass":True,
            "same_frozen_selection_reused":original["candidate_ids"]==replay["candidate_ids"],
            "selected_tensor_values_exact":original["selected_tensors_hash"]==replay["selected_tensors_hash"],
            "initial_exact":original["initial_state"]==replay["initial_state"],
            "shared_exact":original["shared_state"]==replay["shared_state"],"branches":{}}
        for name in original["endpoints"]:
            a,b=original["endpoints"][name],replay["endpoints"][name]
            item={"state_exact":a["state"]==b["state"],
                  "mean":compare(torch.tensor(a["mean"]),torch.tensor(b["mean"])),
                  "variance":compare(torch.tensor(a["variance"]),torch.tensor(b["variance"])),
                  "kernel":compare(torch.tensor(a["kernel"]),torch.tensor(b["kernel"]))}
            item["pass"]=item["state_exact"] and all_pass(item)
            checks["seed0_full_replay"]["branches"][name]=item
        checks["seed0_full_replay"]["pass"]=(checks["seed0_full_replay"]["initial_exact"] and
            checks["seed0_full_replay"]["same_frozen_selection_reused"] and checks["seed0_full_replay"]["selected_tensor_values_exact"] and
            checks["seed0_full_replay"]["shared_exact"] and all_pass(checks["seed0_full_replay"]["branches"]))
        checks["engineering"]=original["engineering"]
        counts={}
        for r in traces:
            key=f'{r["run"]}/{r["branch"]}/{r["block"]}'
            counts[key]=counts.get(key,0)+1
        expected_steps={"common_regression":100,"common_gp":20,"frozen_gp":20,
                        "continued_regression":10,"regression_gp":20,"joint_dkl":10,"dkl_gp_tail":10}
        block_branch={"common_regression":"shared","common_gp":"shared","frozen_gp":"frozen",
            "continued_regression":"regression","regression_gp":"regression","joint_dkl":"dkl","dkl_gp_tail":"dkl"}
        expected_counts={f"{run}/{block_branch[block]}/{block}":n for run in ("seed0","seed1","seed0_replay")
                         for block,n in expected_steps.items()}
        sequences={key:[r["step"] for r in traces if f'{r["run"]}/{r["branch"]}/{r["block"]}'==key]
                   for key in counts}
        counts_pass=counts==expected_counts and all(sequences[key]==list(range(1,n+1)) for key,n in expected_counts.items())
        lr_pass=all(r["lr_mapping"]==(0.001*(0.1 if r["step"]-1>=int(.8*r["planned_steps"]) else 1)
                          if r["mode"] in ("regression","joint") else 0) and
                    r["lr_gp"]==(0.01*(0.1 if r["step"]-1>=int(.8*r["planned_steps"]) else 1)
                          if r["mode"]!="regression" else 0) for r in traces)
        frozen_pass=all(r["mapping_state_delta"]==0 and r["mapping_gradient"]==0 and r["head_gradient"]==0
                        for r in traces if r["mode"]=="gp")
        joint_pass=all(r["mapping_gradient"]>0 and r["mapping_body_delta"]>0 and r["head_gradient"]==0 and r["head_delta"]==0
                       for r in traces if r["mode"]=="joint")
        numeric_columns=["loss_before_update","gradient_before_clip","mapping_gradient","gp_gradient",
                         "head_gradient","mapping_body_delta","extra_noise","lengthscale","outputscale"]
        finite=all(np.isfinite(r[k]) for r in traces for k in numeric_columns)
        checks["fixed_steps_lr_finite_gradients"]={"pass":counts_pass and lr_pass and frozen_pass and joint_pass and finite,
            "counts":counts,"total_formal_updates":len(traces),"schedule_correct":lr_pass,
            "frozen_mapping_and_buffers_unchanged":frozen_pass,"all_joint_steps_body_changes_head_fixed":joint_pass,
            "all_recorded_numerics_finite":finite,
            "min_extra_noise":min(r["extra_noise"] for r in traces),"min_lengthscale":min(r["lengthscale"] for r in traces),
            "min_outputscale":min(r["outputscale"] for r in traces),
            "noise_boundary_distance":min(r["extra_noise"]-1e-6 for r in traces),
            "lengthscale_boundary_distance":min(r["lengthscale"]-1e-4 for r in traces)}
        checks["seed1_ran"]={"pass":"seed1" in results,"statistical_stability_claim":False}
        checks["budget"]={"pass":budget.seconds<1200,"real_training_seconds":budget.seconds,
                           "real_wall_seconds":time.perf_counter()-tick,"limit_seconds":1200}
        if not all_pass(checks):
            raise AssertionError("Required engineering check failed")
    except Exception as exc:
        failure={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        checks["execution"]={"pass":False,**failure}
    finally:
        write_csv(root/"stage1_traces.csv",traces)
        checks["protocol_sha256"]=protocol_sha
        checks["phase1_status"]="PASS" if failure is None and all_pass(checks) else ("INCOMPLETE" if failure and failure["type"]=="TimeoutError" else "FAIL")
        checks["next_stage_authorized"]=False
        dump(root/"stage1_checks.json",checks)
        dump(root/"shared_init_manifest.json",{"candidate_ids":data["ids"],"protocol_sha256":protocol_sha,
             "selected_data_sha256":hashlib.sha256((root/"selected_data.json").read_bytes()).hexdigest(),
             "selected_data_metadata":data.get("metadata",{}),
             "states":{k:{"seed":v["seed"],"initial":v["initial_state"],"shared":v["shared_state"],"fork":v["fork"]}
                       for k,v in results.items()},"rng_in_checkpoints":True,"selection_seed":20260922})
    print(json.dumps({"phase1_status":checks["phase1_status"],"real_training_seconds":budget.seconds,
                      "formal_updates":len(traces),"failure":failure},ensure_ascii=False))
    if failure:
        raise RuntimeError(failure["message"])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["synthetic","real"])
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    torch.set_num_interop_threads(1)
    protocol,sha=load_protocol(args.root)
    dump(args.root/f"environment_{args.mode}.json",{"python":sys.version,"executable":sys.executable,
        "torch":torch.__version__,"gpytorch":gpytorch.__version__,"numpy":np.__version__,
        "platform":platform.platform(),"device":"cpu","threads":1,"protocol_sha256":sha})
    synthetic(args.root) if args.mode=="synthetic" else real(args.root,sha)


if __name__=="__main__":
    main()
