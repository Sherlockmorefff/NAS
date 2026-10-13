"""One-thread serial execution with per-start persistence and frozen identities."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gpytorch

from .io import read_json, write_json, append_jsonl, sha256, digest, code_manifest, now
from .protocol import METHODS, DRAWS, frozen_protocol


def expected_settings():
    return [{"setting_id":f"s{d}_f{f}_n{n}_{m}","subset_id":f"s{d}_f{f}_n{n}",
             "fold":f,"draw":d,"n":n,"method":m}
            for d in DRAWS for f in range(5) for n in (20,50) for m in METHODS]


def capture_baseline(repo):
    repo=Path(repo)
    return {"branch":subprocess.check_output(["git","branch","--show-current"],cwd=repo,text=True).strip(),
            "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
            "status":subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True),
            "protected_files":{p:sha256(repo/p) for p in ("bo_phase4.py","eval_utils.py")}}


def preflight(run_dir,repo):
    from .data import prepare_run
    run_dir,repo=Path(run_dir).resolve(),Path(repo).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Use a new empty run directory for preflight")
    run_dir.mkdir(parents=True,exist_ok=True)
    write_json(run_dir/"protocol.json",frozen_protocol())
    write_json(run_dir/"settings.json",expected_settings())
    write_json(run_dir/"baseline.json",capture_baseline(repo))
    source=Path(__file__).parent
    for name in ("PLAN.md","PROMPT.txt"):
        if (source/name).exists():
            shutil.copyfile(source/name,run_dir/name)
    state={"created_at":now(),"updated_at":now(),"settings":{},"run_state":"PENDING"}
    for setting in expected_settings():
        state["settings"][setting["setting_id"]]={**setting,"state":"PENDING","phase":"PENDING"}
    write_json(run_dir/"status.json",state)
    for path in (repo/"analyse/structured_kernel_phase1").glob("*.py"):
        compile(path.read_text(),str(path),"exec")
    diff=subprocess.run(["git","diff","--check"],cwd=repo,capture_output=True,text=True)
    if diff.returncode:
        raise RuntimeError("Existing git diff --check failed: "+diff.stdout+diff.stderr)
    try:
        prepare_run(run_dir,repo)
    except Exception as exc:
        for entry in state["settings"].values():
            entry.update(state="BLOCKED",phase="PREFLIGHT",error=str(exc))
        state.update(run_state="BLOCKED",updated_at=now())
        write_json(run_dir/"status.json",state)
        (run_dir/"preflight_report.md").write_text("# BLOCKED\n\n"+str(exc)+"\n")
        raise
    files=code_manifest(repo)
    inputs={str(p.relative_to(run_dir)):sha256(p) for p in sorted(run_dir.glob("*.json"))
            if p.name not in ("status.json","frozen_manifest.json","commands.jsonl")}
    inputs.update({name:sha256(run_dir/name) for name in
        ("PLAN.md","PROMPT.txt","candidate_manifest.csv","preflight_report.md") if (run_dir/name).exists()})
    inputs.update({str(p.relative_to(run_dir)):sha256(p) for sub in ("training","queries") for p in sorted((run_dir/sub).glob("*.json"))})
    frozen={"created_at":now(),"code_files":files,"code_hash":digest(files),
            "protocol_sha256":sha256(run_dir/"protocol.json"),"input_files":inputs}
    frozen["identity"]=digest(frozen)
    write_json(run_dir/"frozen_manifest.json",frozen)
    write_json(run_dir/"environment.json",{"python":sys.version,"executable":sys.executable,
        "torch":torch.__version__,"gpytorch":gpytorch.__version__,"numpy":np.__version__,
        "device":"cpu","dtype":"float64","workers":1,"threads":1,
        "environment_threads":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}})
    print(json.dumps({"preflight":"PASS","run_dir":str(run_dir),"settings":100,"starts":300}),flush=True)
    return frozen


def verify_frozen(run_dir,repo):
    run_dir=Path(run_dir)
    frozen=read_json(run_dir/"frozen_manifest.json")
    if frozen["identity"]!=digest({k:v for k,v in frozen.items() if k!="identity"}):
        raise ValueError("Frozen identity changed")
    if code_manifest(repo)!=frozen["code_files"]:
        raise ValueError("Source code changed; create a new run version, do not combine predictions")
    for relative,expected in frozen["input_files"].items():
        if sha256(run_dir/relative)!=expected:
            raise ValueError("Frozen input changed: "+relative)
    for relative,expected in read_json(run_dir/"baseline.json")["protected_files"].items():
        if sha256(Path(repo)/relative)!=expected:
            raise ValueError("Protected production file changed: "+relative)
    return frozen


def seal_predictions(run_dir):
    run_dir=Path(run_dir)
    frozen=read_json(run_dir/"frozen_manifest.json")
    sealed={"code_hash":frozen["code_hash"],"protocol_sha256":frozen["protocol_sha256"],"settings":{}}
    for setting in expected_settings():
        folder=run_dir/"settings"/setting["setting_id"]
        if not (folder/"prediction.json").exists() or not (folder/"model.pt").exists():
            continue
        files={"prediction_sha256":sha256(folder/"prediction.json"),"model_sha256":sha256(folder/"model.pt")}
        for start in range(3):
            for suffix,key in (("json",f"start{start}_json_sha256"),("pt",f"start{start}_model_sha256"),
                               ("jsonl",f"start{start}_log_sha256")):
                p=folder/f"start{start}.{suffix}"
                if p.exists(): files[key]=sha256(p)
        sealed["settings"][setting["setting_id"]]=files
    sealed["identity"]=digest(sealed)
    write_json(run_dir/"prediction_seal.json",sealed)
    return sealed


def _prediction_complete(folder,identity):
    if not all((folder/name).exists() for name in ("model.pt","prediction.json","metrics.json")):
        return False
    p=read_json(folder/"prediction.json")
    if p["identity"]!=identity or p["model_hash"]!=sha256(folder/"model.pt"):
        raise ValueError("Completed setting identity/model mismatch")
    for start in range(3):
        if not all((folder/f"start{start}.{suffix}").exists() for suffix in ("pt","json","jsonl")):
            return False
        result=read_json(folder/f"start{start}.json")
        if not result.get("success") or result["identity"]!=identity:
            return False
    return True


def run_setting(run_dir,repo,setting,frozen,status):
    from .data import release_training,query_features
    from .gp import make_inputs,tensor,fit_start,select_start,predict_frozen
    run_dir=Path(run_dir)
    sid=setting["setting_id"]
    entry=status["settings"][sid]
    folder=run_dir/"settings"/sid
    started=time.perf_counter()
    try:
        training=release_training(run_dir,setting["subset_id"])
        query=query_features(run_dir,setting["fold"])
        tx,qx,arch_dim,scales=make_inputs(training,query,setting["method"])
        identity=digest({"frozen_identity":frozen["identity"],"setting":setting,
                         "T_ids":training["ids"],"Q_ids":query["ids"],
                         "scaler":training["scaler"],"distance_scales":scales})
        if entry["state"]=="COMPLETE" and _prediction_complete(folder,identity):
            return
        folder.mkdir(parents=True,exist_ok=True)
        entry.update(state="RUNNING",phase="FITTING",identity=identity,started_at=now())
        write_json(folder/"identity.json",{"identity":identity,"frozen_identity":frozen["identity"],
             **setting,"T_ids":training["ids"],"Q_ids":query["ids"],"scaler":training["scaler"],
             "distance_scales":scales,"optimizer":frozen_protocol()["optimizer"]})
        write_json(run_dir/"status.json",status)
        results=[]
        y=tensor(training["y_standardized"])
        known=tensor(training["known_variance_standardized"])
        for i in range(3):
            result=fit_start(tx,y,known,setting["method"],arch_dim,i,folder/f"start{i}.pt",identity)
            if result.get("success"):
                try:
                    pred=predict_frozen(folder/f"start{i}.pt",qx)
                    result["posterior_finite"]=True
                    result["prediction_jitter_setting"]=pred["jitter_setting"]
                except Exception as exc:
                    result.update(success=False,posterior_finite=False,error=str(exc),error_type=type(exc).__name__)
                write_json(folder/f"start{i}.json",result)
            results.append(result)
            append_jsonl(run_dir/"fit_log.jsonl",{**setting,**result})
            print(json.dumps({"setting":sid,"start":i,"success":result["success"],
                              "nll":result.get("final_nll"),"warning":result.get("convergence_warning")}),flush=True)
        selected=select_start(results)
        shutil.copyfile(folder/f"start{selected}.pt",folder/"model.pt")
        pred=predict_frozen(folder/"model.pt",qx)
        center,scale=training["scaler"]["y_mean"],training["scaler"]["y_scale"]
        prediction={**setting,**pred,"ids":query["ids"],"identity":identity,
                    "T_ids":training["ids"],"selected_start":selected,"y_center":center,"y_scale":scale,
                    "mean_raw":(center+scale*np.asarray(pred["mean_standardized"])).tolist(),
                    "latent_variance_raw":(scale**2*np.asarray(pred["latent_variance_standardized"])).tolist(),
                    "median_known_var_standardized":float(np.median(training["known_variance_standardized"])),
                    "model_hash":sha256(folder/"model.pt"),"distance_scales":scales,
                    "all_starts_success":all(r["success"] for r in results),
                    "start_selection":"training NLL only, exact tie by start id", "sealed_at":now()}
        write_json(folder/"prediction.json",prediction)
        complete=all(r["success"] for r in results)
        entry.update(state="RUNNING" if complete else "FAILED",phase="FITTED" if complete else "FITTED_PARTIAL",
                     selected_start=selected,starts_success=sum(r["success"] for r in results),
                     convergence_warnings=sum(bool(r.get("convergence_warning")) for r in results),
                     elapsed_seconds=time.perf_counter()-started,fit_finished_at=now())
    except Exception as exc:
        entry.update(state="FAILED",phase="FIT_ERROR",error_type=type(exc).__name__,error=str(exc),
                     elapsed_seconds=time.perf_counter()-started)
        write_json(folder/"failure.json",entry)
    status["updated_at"]=now()
    write_json(run_dir/"status.json",status)


def _score_subprocess(run_dir,repo):
    # Scoring lives in a different process after prediction/state seals are written.
    proc=subprocess.run([sys.executable,"-B","-m","analyse.structured_kernel_phase1","score",
                         "--run-dir",str(run_dir),"--repo",str(repo)],cwd=repo)
    if proc.returncode:
        raise RuntimeError("Independent scorer failed; keep frozen predictions for recovery")


def run(run_dir,repo,pilot_then_all=False,resume=False,workers=1):
    from .gp import cpu_setup
    if workers!=1:
        raise ValueError("Frozen protocol requires workers=1")
    cpu_setup()
    run_dir,repo=Path(run_dir).resolve(),Path(repo).resolve()
    frozen=verify_frozen(run_dir,repo)
    status=read_json(run_dir/"status.json")
    if not resume and any(v["state"]!="PENDING" for v in status["settings"].values()):
        raise ValueError("Run already started; use --resume")
    status.update(run_state="RUNNING",pid=os.getpid(),updated_at=now())
    write_json(run_dir/"status.json",status)
    # Finish scoring sealed prior work before deciding whether COMPLETE can be skipped.
    if resume and any((run_dir/"settings").glob("*/prediction.json")):
        seal_predictions(run_dir)
        _score_subprocess(run_dir,repo)
        status=read_json(run_dir/"status.json")
    settings_list=expected_settings()
    pilot=[s for s in settings_list if s["fold"]==0 and s["draw"]==20260922]
    remaining=[s for s in settings_list if s not in pilot]
    for phase,items in (("pilot",pilot),("full",remaining)):
        for setting in items:
            run_setting(run_dir,repo,setting,frozen,status)
            fitted=sum(v["phase"] in ("FITTED","SCORED") or v["state"]=="COMPLETE" for v in status["settings"].values())
            times=[v.get("elapsed_seconds",0) for v in status["settings"].values() if v["phase"] in ("FITTED","SCORED") or v["state"]=="COMPLETE"]
            eta=float(np.mean(times))*(100-fitted) if times else None
            write_json(run_dir/"progress.json",{"phase":phase,"fitted_settings":fitted,
                "failed":sum(v["state"]=="FAILED" for v in status["settings"].values()),
                "remaining_seconds_estimate":eta,"updated_at":now(),"pid":os.getpid()})
        seal_predictions(run_dir)
        _score_subprocess(run_dir,repo)
        status=read_json(run_dir/"status.json")
        if phase=="pilot":
            ok=all(status["settings"][s["setting_id"]]["state"]=="COMPLETE" for s in pilot)
            write_json(run_dir/"pilot_validation.json",{"pass":ok,"settings":10,"starts":30,
                "basis":"artifact completeness, three finite starts, independent NLL, posterior; no Q metric used",
                "unchanged_scientific_protocol":True,"completed_at":now()})
            if not ok:
                status.update(run_state="FAILED",stop_reason="pilot engineering validation failed",updated_at=now())
                write_json(run_dir/"status.json",status)
                raise RuntimeError("Pilot failed engineering validation; no scientific changes made")
        verify_frozen(run_dir,repo)
    counts={k:sum(v["state"]==k for v in status["settings"].values()) for k in ("PENDING","RUNNING","COMPLETE","FAILED","BLOCKED")}
    status.update(run_state="COMPLETE" if counts["COMPLETE"]==100 else "INCOMPLETE",counts=counts,updated_at=now())
    write_json(run_dir/"status.json",status)
    print(json.dumps({"run_finished":status["run_state"],"counts":counts}),flush=True)
    return status
