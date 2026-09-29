#!/usr/bin/env python3
"""A-only historical gate / post-seal residual experiment scoring. No model code."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import score_stage2 as base

TOL = 1e-9
BREAK_TOL_RAW = 1e-11
METHODS = ("R_FROZEN", "R_REG", "R_DKL_C", "R_DKL_U", "RAW16")
REPRESENTATIONS = ("RAW_X", "COMMON_REG", "COMMON_GP", "OLD_DKL_1000", "OLD_DKL_4000", "OLD_RAW16")
GATE_METHODS = ("COMMON_REG", "OLD_DKL_1000")
METRICS = base.METRICS + ("IS90_pp", "gain5_pp", "headroom_pp", "break_best_rate", "has_headroom_fraction",
    "random_regret5_pp", "random_gain5_pp", "random_break_best_probability", "E1_pp", "E5_pp",
    "high_correlation_bad_neighbor_fraction")
NN_METRICS = ("E1_pp", "E5_pp", "high_correlation_bad_neighbor_fraction")

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def checked_file(root, relative, expected):
    p = base.safe_child(root, relative)
    require(base.sha256(p) == expected, "Frozen file SHA256 differs: " + relative)
    return p

def read_seal(root, name, expected_count):
    """All identities and prediction rows are verified before full A labels load."""
    manifest = base.read_json(root / name)
    require(manifest.get("sealed") is True, "Uniform prediction/geometry seal required")
    require(manifest.get("execution_status") in ("COMPLETE_VALID", "PARTIAL_BUDGET", "PARTIAL"), "Invalid/blocked diagnostic or experiment cannot be scored")
    for key in ("protocol", "code_manifest", "source_manifest"):
        checked_file(root, manifest.get(key + "_path", key + ".json"), manifest[key + "_sha256"])
    require((root / "frozen_manifest.json").exists(), "Final executable protocol/code must be frozen before any historical scoring")
    frozen = base.read_json(root / "frozen_manifest.json")
    for path, sha in frozen["files"].items(): checked_file(root, path, sha)
    source = base.read_json(root / manifest.get("source_manifest_path", "source_manifest.json"))
    for path, sha in source["files"].items(): checked_file(root, path, sha)
    code = base.read_json(root / manifest.get("code_manifest_path", "code_manifest.json"))
    for path, sha in code["files"].items(): checked_file(root, path, sha)
    require(manifest.get("expected_endpoint_count", expected_count) == expected_count, "Fixed endpoint count changed")
    entries, seen = [], set()
    for item in manifest["files"]:
        p = checked_file(root, item["path"], item["sha256"])
        row = base.read_json(p)
        base.forbid_query_truth(row)
        require(row["unit_id"] == item["unit_id"] and row["unit_id"] not in seen, "Duplicate/mismatched sealed unit ID")
        seen.add(row["unit_id"])
        if "state_path" in item:
            state_hash = item.get("state_file_sha256", item.get("state_sha256"))
            checked_file(root, item["state_path"], state_hash)
        row["prediction_file_sha256"] = item["sha256"]
        entries.append(row)
    complete = manifest["execution_status"] == "COMPLETE_VALID" and len(entries) == expected_count
    require(manifest["execution_status"] != "COMPLETE_VALID" or complete, "Complete status has missing endpoints")
    return manifest, source, entries, complete

def load_labels(root, source, directory, seal_name, labels_from=None):
    path = root / directory / "labels_A.json"
    if not path.exists():
        source_path=Path(labels_from) if labels_from is not None else Path(source["label_source_path"])
        require(base.sha256(source_path)==source["scoring_labels_expected_sha256"],"A label source bytes differ")
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_path,path)
    require(base.sha256(path) == source["scoring_labels_expected_sha256"], "Original legal A scoring-label bytes differ")
    base.write_json(root/directory/"release.json",{"labels_accessed_after_verified_seal_at_utc":datetime.now(timezone.utc).isoformat(),
        "seal_path":seal_name,"seal_sha256":base.sha256(root/seal_name),"labels_A_sha256":base.sha256(path),"only_legal_A":True})
    return base.checked_labels(path, set()), base.sha256(path)

def key(row, method_field="method"):
    return row["fold"], row["sample_seed"], row["n"], row["model_seed"], row[method_field]

def expected(methods):
    return {(f,s,n,m,method) for f in range(5) for s in base.SAMPLE_SEEDS for n in base.NS for method in methods
            for m in ([None] if method in ("RAW16", "RAW_X", "OLD_RAW16") else base.MODEL_SEEDS)}

def nearest_metrics(row, labels):
    train, query = row["training_ids"], row["query_ids"]
    near, five = row["nearest_train_ids"], row["five_nearest_train_ids"]
    require(len(train) in (20,50) and len(set(train)) == len(train), "Neighbor training subset size/identity invalid")
    require(not set(train) & set(query), "Neighbor points leak query IDs into selected training IDs")
    require(len(query) == len(near) == len(five) and len(set(query)) == len(query), "Query/neighbor array alignment invalid")
    for cid, nn, nns in zip(query, near, five):
        require(nn in train and len(nns) == 5 and len(set(nns)) == 5 and all(c in train for c in nns) and nns[0] == nn,
                "Nearest neighbors must come from the actual selected subset in fixed rank order")
    gaps = [abs(labels[q]["validation_mean"] - labels[t]["validation_mean"]) for q,t in zip(query, near)]
    errors5 = [sum(abs(labels[q]["validation_mean"]-labels[t]["validation_mean"]) for t in nns)/5 for q,nns in zip(query,five)]
    rho = row.get("most_correlated_rho",row.get("nearest_correlation"))
    correlated_ids=row.get("most_correlated_train_ids") or near
    require(rho is None or len(rho) == len(query), "Correlation array mismatch")
    require(len(correlated_ids)==len(query) and all(cid in train for cid in correlated_ids),"Most-correlated training IDs must belong to selected subset")
    correlation_gaps=[abs(labels[q]["validation_mean"]-labels[t]["validation_mean"]) for q,t in zip(query,correlated_ids)]
    return {"E1_pp":100*base.mean(gaps), "E5_pp":100*base.mean(errors5),
            "high_correlation_bad_neighbor_fraction":None if rho is None else base.mean([float(r>.99 and 100*g>5) for r,g in zip(rho,correlation_gaps)])}

def random5(y, incumbent):
    ordered = sorted(y); m=len(y); denominator=math.comb(m,5)
    weights = [(value, math.comb(i-1,4)/denominator) for i,value in enumerate(ordered,1) if i>=5]
    best = math.fsum(value*w for value,w in weights)
    return {"random_regret5_pp":100*(max(y)-best),
            "random_gain5_pp":100*math.fsum(max(0,value-incumbent)*w for value,w in weights),
            "random_break_best_probability":math.fsum(w for value,w in weights if value-incumbent>BREAK_TOL_RAW)}

def validate_prediction(p):
    temp = dict(p)
    require(p["method"] in METHODS or p["method"] in ("FROZEN","REG","DKL") or p["method"] in REPRESENTATIONS, "Unknown prediction method")
    temp["method"] = "RAW16" if p["method"] in ("RAW16","OLD_RAW16") else "DKL"
    return base.validate_prediction(temp)

def prediction_metrics(p, nn, labels):
    validate_prediction(p)
    require(p["candidate_ids"] == nn["query_ids"], "Prediction and neighbor query IDs/order differ")
    require(len(nn["training_ids"]) == p["n"] and digest(nn["training_ids"]) == p["training_ids_sha256"], "Prediction and neighbor training subset differ")
    result=base.score_unit(p,labels)
    ids=p["candidate_ids"]; y=[labels[c]["validation_mean"] for c in ids]
    incumbent=max(labels[c]["validation_mean"] for c in nn["training_ids"])
    signed_gain=result["selected_best_validation_mean_raw"]-incumbent
    headroom=max(0,max(y)-incumbent)
    result.update(training_best_raw=incumbent, gain5_pp=100*max(0,signed_gain), headroom_pp=100*headroom,
                  break_best_rate=float(signed_gain>BREAK_TOL_RAW),has_headroom_fraction=float(headroom>BREAK_TOL_RAW))
    result.update(random5(y,incumbent));result.update(nearest_metrics(nn,labels))
    # This protocol explicitly clamps only tolerated roundoff latent negatives in
    # probability scoring; raw prediction variances remain untouched in files.
    variance=[p["y_scale"]**2*(max(latent,0)+p["extra_noise_variance_standardized"])+labels[c]["validation_se"]**2
              for latent,c in zip(p["latent_variance_standardized"],ids)]
    require(all(math.isfinite(v) and v>0 for v in variance),"Observed variance must be positive and finite")
    errors=[mu-target for mu,target in zip(p["mean_raw"],y)]
    result["mean_observed_variance_raw"]=base.mean(variance)
    result["mean_observed_sd_pp"]=100*base.mean([math.sqrt(v) for v in variance])
    result["nlpd_accuracy_units"]=base.mean([.5*(math.log(2*math.pi*v)+e*e/v) for e,v in zip(errors,variance)])
    for level,z in ((90,1.6448536269514722),(95,1.959963984540054)):
        half=[z*math.sqrt(v) for v in variance]
        result[f"coverage{level}_fraction"]=base.mean([float(abs(e)<=h) for e,h in zip(errors,half)])
        result[f"interval{level}_width_pp"]=200*base.mean(half)
        if level==90:
            result["IS90_pp"]=100*base.mean([2*h+20*max(abs(e)-h,0) for e,h in zip(errors,half)])
    return result

def aggregate(rows, fields, metrics):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in fields)].append(r)
    result=[]
    for values, group in sorted(groups.items()):
        r=dict(zip(fields,values));r["child_count"]=len(group)
        for metric in metrics:
            stats=base.spread([g.get(metric) for g in group]);r[metric]=stats["mean"]
            for extra in ("sample_sd","min","max","defined_count"):r[metric+"_"+extra]=stats[extra]
        result.append(r)
    return result

def hierarchy(rows,metrics,method_field="method"):
    sample=aggregate(rows,("n",method_field,"fold","sample_seed"),metrics)
    fold=aggregate(sample,("n",method_field,"fold"),metrics)
    return sample,fold

def direction_counts(values):
    return {"improved":sum(v>TOL for v in values),"tied":sum(abs(v)<=TOL for v in values),"worse":sum(v < -TOL for v in values)}

def gate_decision(fold_deltas, complete):
    comparisons={}; admitted=False
    for n in base.NS:
        for method in GATE_METHODS:
            values=fold_deltas.get((n,method),{})
            full=set(values)==set(range(5))
            support=bool(complete and full and base.mean(list(values.values()))>TOL and sum(x>TOL for x in values.values())>=4)
            comparisons[f"n{n}_{method}"]={"fold_delta_E1_pp":{str(k):v for k,v in values.items()},
                "mean_delta_E1_pp":base.mean(list(values.values())),"positive_folds":sum(x>TOL for x in values.values()),
                "mapping_support":support,"complete_five_folds":full}
            admitted=admitted or support
    route="BLOCKED" if not complete else "RUN_RESIDUAL_ABLATION" if admitted else "STOP_DIAGNOSTIC_ONLY"
    return {"diagnostic_route":route,"route":route,"diagnostic_complete":complete,"comparisons":comparisons,
            "rule":"only COMMON_REG/OLD_DKL_1000 E1 mapped-minus-RAW_X; mean>1e-9pp and >=4/5 positive folds; any n/time admits full B",
            "next_stage_authorized":False,"formal_training_authorized_by_gate":admitted and complete,
            "not_statistical_or_causal_confirmation":True}

def write_gate(root,gate):
    base.write_json(root/"diagnostic_gate.json",gate)
    # This narrow token contains no query outcomes. The B trainer must never
    # receive or open the historical truth-bearing diagnostic gate table.
    admission={"route":gate["route"],"diagnostic_gate_sha256":base.sha256(root/"diagnostic_gate.json"),
        "protocol_sha256":gate.get("protocol_sha256"),"code_manifest_sha256":gate.get("code_manifest_sha256"),
        "diagnostic_complete":gate["diagnostic_complete"]}
    base.write_json(root/"admission.json",admission)

def historical_score(root,labels_from=None):
    seal,source,units,complete=read_seal(root,"historical_manifest.json",280)
    seen={key(r,"representation") for r in units}
    require(len(seen)==len(units) and seen<=expected(REPRESENTATIONS),"Historical identity matrix differs")
    complete=complete and seen==expected(REPRESENTATIONS)
    # Missing any prescribed historical comparison cannot admit B.
    if not complete:
        gate=gate_decision({},False);gate.update(protocol_sha256=seal["protocol_sha256"],code_manifest_sha256=seal["code_manifest_sha256"])
        write_gate(root,gate);return gate
    labels,label_hash=load_labels(root,source,"scoring_A","historical_manifest.json",labels_from)
    rows=[]
    for u in units:
        r={k:u[k] for k in ("unit_id","context_id","subset_id","fold","sample_seed","n","model_seed")}
        r["method"]=u["representation"];r.update(nearest_metrics(u,labels))
        if u.get("prediction") is not None:
            p=prediction_metrics(u["prediction"],u,labels)
            r.update({k:v for k,v in p.items() if k not in r and k not in ("unit_id","context_id","subset_id","fold","sample_seed","n","model_seed","method")})
        else:
            for metric in METRICS:r.setdefault(metric,None)
        rows.append(r)
    sample,fold=hierarchy(rows,METRICS)
    raw={(r["fold"],r["sample_seed"],r["n"]):r for r in rows if r["method"]=="RAW_X"}
    paired=[]
    for r in rows:
        if r["method"] not in GATE_METHODS:continue
        ref=raw[(r["fold"],r["sample_seed"],r["n"])]
        paired.append({**{k:r[k] for k in ("n","method","fold","sample_seed","model_seed","unit_id")},
                       "RAW_X_source_unit_id":ref["unit_id"],"delta_E1_pp":r["E1_pp"]-ref["E1_pp"]})
    gate_sample,gate_folds=hierarchy(paired,("delta_E1_pp",))
    deltas={(n,m):{r["fold"]:r["delta_E1_pp"] for r in gate_folds if r["n"]==n and r["method"]==m}
            for n in base.NS for m in GATE_METHODS}
    gate=gate_decision(deltas,complete)
    gate.update(protocol_sha256=seal["protocol_sha256"],code_manifest_sha256=seal["code_manifest_sha256"],
                historical_manifest_sha256=base.sha256(root/"historical_manifest.json"),labels_A_sha256=label_hash,
                computed_at_utc=datetime.now(timezone.utc).isoformat(),scorer_sha256=base.sha256(__file__))
    out=root/"scoring_A";out.mkdir(exist_ok=True)
    for name,table in (("unit_metrics.csv",rows),("sample_seed_metrics.csv",sample),("fold_metrics.csv",fold),
                       ("gate_unit_deltas.csv",paired),("gate_sample_deltas.csv",gate_sample),("gate_fold_deltas.csv",gate_folds)):
        base.write_csv(out/name,table)
    summary={"diagnostic_complete":True,"diagnostic_route":gate["route"],"per_n":{},
             "RAW_X_and_OLD_RAW16_each_counted_once_per_subset":True,"COMMON_REG_COMMON_GP_not_independent_geometric_evidence":True}
    for n in base.NS:
        summary["per_n"][str(n)]={m:{metric:base.spread([r[metric] for r in fold if r["n"]==n and r["method"]==m])
                                     for metric in METRICS} for m in REPRESENTATIONS}
    base.write_json(out/"summary.json",summary);write_gate(root,gate)
    return gate

def classify(selection_fold_deltas, constraint_fold_deltas, means, seed_reg_deltas, complete):
    if not complete:
        return {"scientific_signal":"不完整，暂不分类","constraint_signal":"不完整，暂不分类",
                "selection_signal":None,"constraint_signal_pass":None,"followup_candidate":None}
    require(len(selection_fold_deltas)==len(constraint_fold_deltas)==5,"Scientific classification needs five matched folds")
    positive=sum(v>TOL for v in selection_fold_deltas)
    c=means["R_DKL_C"]; raw=means["RAW16"];reg=means["R_REG"]
    selection=base.mean(selection_fold_deltas)>TOL and positive>=4 and raw["regret5_pp"]-c["regret5_pp"]>TOL and c["random_regret5_pp"]-c["regret5_pp"]>TOL
    scientific="较一致的开发性离线选优信号" if selection else "局部或混合信号" if positive else "未见相对继续回归的收益"
    gain_ok=all(c["gain5_pp"]>=v-TOL for v in (reg["gain5_pp"],raw["gain5_pp"],c["random_gain5_pp"]))
    break_ok=all(c["break_best_rate"]>=v-1e-12 for v in (reg["break_best_rate"],raw["break_best_rate"],c["random_break_best_probability"]))
    sample=seed_reg_deltas["sample_seeds"];model=seed_reg_deltas["model_seeds"]
    seed_ok=len(sample)==2 and all(v>0 for v in sample.values()) and len(model)==3 and sum(v>0 for v in model.values())>=2
    constraint=base.mean(constraint_fold_deltas)>TOL and sum(v>TOL for v in constraint_fold_deltas)>=4
    if constraint:mechanism="持续约束带来较一致选优改善"
    elif all(abs(v)<=TOL for v in constraint_fold_deltas):mechanism="持续约束的选优影响持平"
    elif any(v>TOL for v in constraint_fold_deltas):mechanism="持续约束的选优影响混合"
    else:mechanism="持续约束选优退步"
    return {"scientific_signal":scientific,"constraint_signal":mechanism,"selection_signal":selection,
            "constraint_signal_pass":constraint,"followup_candidate":bool(selection and gain_ok and break_ok and seed_ok),
            "followup_conditions":{"gain_noninferiority":gain_ok,"break_probability_noninferiority":break_ok,"sample_and_model_seed_consistency":seed_ok}}

PAIR_METRICS=("regret5_improvement_pp","gain5_improvement_pp","break_best_improvement","E1_improvement_pp","E5_improvement_pp","IS90_improvement_pp")

def formal_summary(rows,complete):
    sample,fold=hierarchy(rows,METRICS)
    lookup={key(r):r for r in rows};pairs=[]
    for c in (r for r in rows if r["method"]=="R_DKL_C"):
        for m in ("R_REG","R_DKL_U","R_FROZEN","RAW16"):
            ref=lookup.get((c["fold"],c["sample_seed"],c["n"],None if m=="RAW16" else c["model_seed"],m))
            if ref is None:continue
            r={k:c[k] for k in ("context_id","fold","sample_seed","n","model_seed")}
            r.update(method=m,constrained_unit_id=c["unit_id"],reference_unit_id=ref["unit_id"],RAW16_reference_shared_not_independent=m=="RAW16",
                regret5_improvement_pp=ref["regret5_pp"]-c["regret5_pp"],gain5_improvement_pp=c["gain5_pp"]-ref["gain5_pp"],
                break_best_improvement=c["break_best_rate"]-ref["break_best_rate"],E1_improvement_pp=ref["E1_pp"]-c["E1_pp"],
                E5_improvement_pp=ref["E5_pp"]-c["E5_pp"],IS90_improvement_pp=ref["IS90_pp"]-c["IS90_pp"])
            pairs.append(r)
    ps,pf=hierarchy(pairs,PAIR_METRICS)
    model_fold=aggregate(pairs,("n","method","model_seed","fold"),PAIR_METRICS)
    model_summary=aggregate(model_fold,("n","method","model_seed"),PAIR_METRICS)
    sample_summary=aggregate(ps,("n","method","sample_seed"),PAIR_METRICS)
    # Preserve prediction/error seed decomposition as well as comparison seeds.
    metric_model_fold=aggregate([r for r in rows if r["method"]!="RAW16"],("n","method","model_seed","fold"),METRICS)
    metric_models=aggregate(metric_model_fold,("n","method","model_seed"),METRICS)
    metric_samples=aggregate(sample,("n","method","sample_seed"),METRICS)
    summary={"execution_status":"COMPLETE_VALID" if complete else "PARTIAL_BUDGET","endpoint_count":len(rows),
        "expected_endpoint_count":260,"next_stage_authorized":False,"descriptive_only":True,"B_score_exposed_no_more_training":True,
        "raw_endpoints_counted_once":sum(r["method"]=="RAW16" for r in rows),"per_n":{}}
    for n in base.NS:
        means={m:{metric:base.mean([r[metric] for r in fold if r["n"]==n and r["method"]==m]) for metric in METRICS} for m in METHODS}
        compared={}
        for m in ("R_REG","R_DKL_U","R_FROZEN","RAW16"):
            fr=[r for r in pf if r["n"]==n and r["method"]==m]
            compared[m]={"metrics":{metric:base.spread([r[metric] for r in fr]) for metric in PAIR_METRICS},
                "fold_regret5_improvement_pp":{str(r["fold"]):r["regret5_improvement_pp"] for r in fr},
                "fold_regret5_directions":direction_counts([r["regret5_improvement_pp"] for r in fr]),
                "trajectory_regret5_directions":direction_counts([r["regret5_improvement_pp"] for r in pairs if r["n"]==n and r["method"]==m])}
        seeds={"sample_seeds":{str(r["sample_seed"]):r["regret5_improvement_pp"] for r in sample_summary if r["n"]==n and r["method"]=="R_REG"},
               "model_seeds":{str(r["model_seed"]):r["regret5_improvement_pp"] for r in model_summary if r["n"]==n and r["method"]=="R_REG"}}
        science=classify(list(compared["R_REG"]["fold_regret5_improvement_pp"].values()),
                         list(compared["R_DKL_U"]["fold_regret5_improvement_pp"].values()),means,seeds,complete)
        summary["per_n"][str(n)]={"method_means":means,"paired_comparisons":compared,"REG_comparison_seed_means":seeds,**science}
        summary[f"scientific_signal_n{n}"]=science["scientific_signal"];summary[f"constraint_signal_n{n}"]=science["constraint_signal"]
    with_room=[r for r in rows if r["has_headroom_fraction"]==1]
    room_sample,room_fold=hierarchy(with_room,METRICS)
    summary["headroom_conditioned_descriptive_only"]={str(n):{m:{"endpoint_count":sum(r["n"]==n and r["method"]==m for r in with_room),
        "metrics":{metric:base.spread([r[metric] for r in room_fold if r["n"]==n and r["method"]==m]) for metric in METRICS}}
        for m in METHODS} for n in base.NS}
    tables={"sample_seed_metrics.csv":sample,"fold_metrics.csv":fold,"paired_unit_metrics.csv":pairs,"sample_seed_pairs.csv":ps,
        "fold_pairs.csv":pf,"model_seed_pairs.csv":model_summary,"sample_seed_pair_summary.csv":sample_summary,
        "model_seed_metrics.csv":metric_models,"sample_seed_metric_summary.csv":metric_samples,
        "headroom_conditioned_sample_metrics.csv":room_sample,"headroom_conditioned_fold_metrics.csv":room_fold}
    return summary,tables

def residual_costs(units):
    rows=[];common={}
    fields=("common_training_seconds","branch_training_seconds","common_model_seconds","branch_model_seconds","prediction_seconds","standalone_fit_predict_seconds")
    for p in units:
        r={k:p[k] for k in ("unit_id","context_id","fold","sample_seed","n","model_seed","method")}
        r.update({k:base.finite(p["cost"][k],k) for k in fields})
        require(all(r[k]>=0 for k in fields),"Negative cost")
        require(math.isclose(r["standalone_fit_predict_seconds"],r["common_model_seconds"]+r["branch_model_seconds"],abs_tol=1e-6,rel_tol=1e-6),
                "Standalone model cost must include shared/common and own branch model computations")
        r["standalone_pure_update_seconds"]=r["common_training_seconds"]+r["branch_training_seconds"]
        r["standalone_model_diagnostics_and_other_seconds"]=r["standalone_fit_predict_seconds"]-r["standalone_pure_update_seconds"]
        require(r["standalone_model_diagnostics_and_other_seconds"]>=-1e-6,"Model-inclusive cost below pure update cost")
        r["endpoint_prediction_reused_from_charged_diagnostic"]=True
        if p["method"]!="RAW16":
            context=(p["fold"],p["sample_seed"],p["n"],p["model_seed"])
            if context in common:require(math.isclose(common[context],r["common_model_seconds"],abs_tol=1e-9),"Shared model cost differs between branches")
            common[context]=r["common_model_seconds"]
        rows.append(r)
    summary={"units":"seconds","standalone_includes_own_common_model_computations":True,
        "completed_endpoint_shared_model_seconds":sum(common.values())+sum(r["branch_model_seconds"] for r in rows)+sum(r["common_model_seconds"] for r in rows if r["method"]=="RAW16"),
        "resource_ledger_authoritative_for_A_synthetic_checks_failed_and_incomplete_work":True,
        "per_n_method_standalone_cost":{str(n):{m:base.spread([r["standalone_fit_predict_seconds"] for r in rows if r["n"]==n and r["method"]==m]) for m in METHODS} for n in base.NS}}
    return rows,summary

def formal_score(root,labels_from=None):
    gate=base.read_json(root/"diagnostic_gate.json")
    require(gate["diagnostic_route"]=="RUN_RESIDUAL_ABLATION" and gate["diagnostic_complete"],"Formal B scoring requires the frozen A admission gate")
    seal,source,units,complete=read_seal(root,"predictions_manifest.json",260)
    seen={key(r) for r in units};require(len(seen)==len(units) and seen<=expected(METHODS),"Formal method matrix differs")
    complete=complete and seen==expected(METHODS)
    for p in units:
        validate_prediction(p)
        require("geometry_row" in p,"Every final B endpoint must preserve its own label-free neighbors")
    labels,label_hash=load_labels(root,source,"scoring_B","predictions_manifest.json",labels_from)
    rows=[prediction_metrics(p,p["geometry_row"],labels) for p in units]
    summary,tables=formal_summary(rows,complete)
    summary.update(diagnostic_route=gate["diagnostic_route"],labels_A_sha256=label_hash,
        predictions_manifest_sha256=base.sha256(root/"predictions_manifest.json"),scorer_sha256=base.sha256(__file__),
        scored_at_utc=datetime.now(timezone.utc).isoformat(),missing_endpoint_identities=sorted(expected(METHODS)-seen,key=str))
    cost_rows,cost_summary=residual_costs(units)
    tables.update({"unit_metrics.csv":rows,"cost_table.csv":cost_rows})
    out=root/"scoring_B";out.mkdir(exist_ok=True)
    for name,table in tables.items():base.write_csv(out/name,table)
    refs={r["subset_id"]:{k:r[k] for k in ("subset_id","fold","sample_seed","n","training_best_raw","headroom_pp","random_regret5_pp","random_gain5_pp","random_break_best_probability")} for r in rows}
    base.write_json(out/"random_reference.json",{"by_subset":list(refs.values()),"not_independent_repeats":True})
    calibration=[{k:r[k] for k in ("unit_id","n","method","fold","sample_seed","model_seed","coverage90_fraction","coverage95_fraction","interval90_width_pp","interval95_width_pp","IS90_pp","nlpd_accuracy_units")} for r in rows]
    base.write_csv(out/"calibration.csv",calibration)
    base.write_json(out/"cost_summary.json",cost_summary);base.write_json(out/"summary.json",summary)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("historical","formal"));p.add_argument("--root",type=Path,required=True);p.add_argument("--labels-from",type=Path)
    args=p.parse_args();root=args.root.resolve()
    result=historical_score(root,args.labels_from) if args.mode=="historical" else formal_score(root,args.labels_from)
    print(json.dumps({k:result[k] for k in ("diagnostic_route","execution_status","endpoint_count","next_stage_authorized") if k in result},ensure_ascii=False))

if __name__=="__main__":main()
