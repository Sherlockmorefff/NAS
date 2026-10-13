#!/usr/bin/env python3
"""Stage 2: architecture-disjoint development evaluation, with zero GNN calls.

Freeze identity-only splits first. Export only A records at the trusted file
boundary. A learner receives its training records only; query records are read
by the evaluator after predictions, neighbors and model have been persisted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from analyse.fuzzy_metric_replay import digest, file_hash, json_write, validate_candidates


def configuration_grid():
    grid = [{"method": "regression", "epsilon": 0.01, "lambda_f": 0.0,
             "order": 0, "config_id": "regression"}]
    for method in ("N0", "N1", "N2"):
        for ei, epsilon in enumerate((0.005, 0.01, 0.02)):
            for li, weight in enumerate((0.01, 0.1, 1.0)):
                grid.append({"method": method, "epsilon": epsilon, "lambda_f": weight,
                             "order": len(grid), "config_id": f"{method}-e{ei}-w{li}"})
    return grid


def p2_capacity(candidates, config):
    by_id = {c["candidate_id"]: c for c in candidates}
    ids = sorted(by_id)
    size, budget, batch = (config[k] for k in
                           ("future_initial_size", "future_query_budget", "future_batch_size"))
    checks = []
    for seed in config["future_initial_seeds"]:
        initial = np.random.default_rng(seed).permutation(ids)[:size].tolist()
        groups = defaultdict(list)
        for cid in sorted(set(ids) - set(initial)):
            groups[by_id[cid]["architecture_fingerprint"]].append(cid)
        cap = {a: min(2, len(v)) for a, v in groups.items()}
        total = sum(cap.values())
        schedule = []
        for _ in range(budget // batch):
            order = sorted((a for a in cap if cap[a]), key=lambda a: (-cap[a], groups[a][0], a))
            if len(order) < batch:
                break
            picked = []
            for a in order[:batch]:
                picked.append(groups[a].pop(0))
                cap[a] -= 1
            schedule.append(picked)
        ok = len(initial) == size and len(ids) - size >= budget and total >= budget and len(schedule) * batch == budget
        checks.append({"seed": seed, "initial_ids": initial, "initial_ids_hash": digest(initial),
                       "remaining_candidates": len(ids) - len(initial), "remaining_architectures": len(groups),
                       "capacity": total, "schedule": schedule, "feasible": ok,
                       "interpretation": "identity-only existence witness, not a search or arbitrary-ranking guarantee"})
    return checks


def grouped_inner(ids, by_id, seed, fraction):
    arches = sorted({by_id[cid]["architecture_fingerprint"] for cid in ids})
    perm = np.random.default_rng(seed).permutation(arches).tolist()
    n = max(1, min(len(arches) - 1, int(math.floor(len(arches) * fraction))))
    train_arch = set(perm[:n])
    train = sorted(cid for cid in ids if by_id[cid]["architecture_fingerprint"] in train_arch)
    valid = sorted(set(ids) - set(train))
    if len(train) < 5 or not valid:
        raise ValueError("architecture inner split lacks training references or validation queries")
    return {"train_ids": train, "query_ids": valid, "seed": seed}


def build_partition(candidates, mandatory_ids, config):
    validate_candidates(candidates)
    by_id = {c["candidate_id"]: c for c in candidates}
    if len(by_id) != len(candidates) or not set(mandatory_ids) <= set(by_id):
        raise ValueError("duplicate candidate or unknown prior-use ID")
    arches = sorted({c["architecture_fingerprint"] for c in candidates})
    forced = {by_id[cid]["architecture_fingerprint"] for cid in mandatory_ids}
    a_arch = set(forced)
    target = max(len(forced), int(len(arches) * config["a_architecture_fraction"]))
    additions = np.random.default_rng(config["split_seed"]).permutation(sorted(set(arches) - forced)).tolist()
    adjustments = []
    for arch in additions:
        if len(a_arch) >= target:
            break
        proposal = a_arch | {arch}
        b_rows = [c for c in candidates if c["architecture_fingerprint"] not in proposal]
        if not all(x["feasible"] for x in p2_capacity(b_rows, config)):
            adjustments.append({"stopped_before_architecture": arch, "reason": "B identity capacity would fail; no redraw"})
            break
        a_arch = proposal
    a = sorted(cid for cid in by_id if by_id[cid]["architecture_fingerprint"] in a_arch)
    b = sorted(set(by_id) - set(a))
    checks = p2_capacity([by_id[cid] for cid in b], config)
    if len(a_arch) < config["outer_folds"]:
        raise ValueError("A cannot support architecture fivefold")
    outer_arch = np.random.default_rng(config["split_seed"]).permutation(sorted(a_arch))
    folds = []
    for i, group in enumerate(np.array_split(outer_arch, config["outer_folds"])):
        query_arch = set(group.tolist())
        query = sorted(cid for cid in a if by_id[cid]["architecture_fingerprint"] in query_arch)
        train = sorted(set(a) - set(query))
        inner = grouped_inner(train, by_id, config["inner_split_seeds"][i], config["inner_train_fraction"])
        folds.append({"family": f"fold{i}", "train_ids": train, "query_ids": query, "inner": inner})
    final_inner = grouped_inner(a, by_id, config["inner_split_seeds"][-1], config["inner_train_fraction"])
    return {"A_ids": a, "B_ids": b, "A_architectures": sorted(a_arch),
            "B_architectures": sorted(set(arches) - a_arch), "A_hash": digest(a), "B_hash": digest(b),
            "mandatory_ids": sorted(mandatory_ids), "mandatory_architectures": sorted(forced),
            "mandatory_architecture_closure_count": sum(c["architecture_fingerprint"] in forced for c in candidates),
            "identity_adjustments": adjustments, "p2_checks": checks, "folds": folds,
            "final_split": {"family": "final_A", "train_ids": a, "query_ids": [], "inner": final_inner},
            "stage3_original_capacity_feasible": all(c["feasible"] for c in checks),
            "data_use": "both historical development; B excluded from stage2, not independent confirmation"}


def build_task_matrix(partition):
    tasks = []
    for family in partition["folds"] + [partition["final_split"]]:
        name = family["family"]
        for setting in configuration_grid():
            for seed in (0, 1, 2):
                tasks.append({"task_id": f"{name}-inner-{setting['config_id']}-s{seed}",
                              "family": name, "stage": "inner", "seed": seed, "setting": setting,
                              "train_ids": family["inner"]["train_ids"], "query_ids": family["inner"]["query_ids"]})
        for method in ("regression", "N0", "N1", "N2"):
            for seed in (0, 1, 2):
                tasks.append({"task_id": f"{name}-refit-{method}-s{seed}", "family": name,
                              "stage": "refit", "seed": seed, "method": method,
                              "setting_source": f"selection/{name}.json", "train_ids": family["train_ids"],
                              "query_ids": family["query_ids"]})
    if len(tasks) != 576 or len({t["task_id"] for t in tasks}) != 576:
        raise RuntimeError("frozen matrix must have exactly 576 unique tasks")
    return tasks


def read_selected_labels(path, allowed_ids):
    """Parse only requested top-level records from the prepared pretty JSON.

    The trusted exporter scans keys/line boundaries; unrequested value lines are
    discarded without JSON decoding. In particular B values are never formed
    into Python objects or passed to stage-2 training/evaluation. Only A is
    exported; subsequent per-task reads use this A-only file.
    """
    allowed = set(allowed_ids)
    result = {}
    active = None
    record = []
    with Path(path).open() as source:
        for line in source:
            match = re.match(r'^  "([^"\\]+)": \{\s*$', line)
            if match:
                active = match.group(1) if match.group(1) in allowed else None
                record = ["{\n"] if active else []
            elif active is not None:
                if re.match(r"^  }[,]?\s*$", line):
                    record.append("}\n")
                    row = json.loads("".join(record))
                    if active in result:
                        raise ValueError("duplicate selected label")
                    if set(row) != {"validation_mean", "validation_se", "validation_seeds", "training_seeds"}:
                        raise ValueError("unexpected label fields")
                    values = np.asarray(row["validation_seeds"], dtype=float)
                    if (row["training_seeds"] != [0, 1, 2] or values.shape != (3,)
                            or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all()
                            or not np.isclose(values.mean(), row["validation_mean"], rtol=0, atol=1e-12)
                            or not np.isclose(values.std(ddof=1) / math.sqrt(3), row["validation_se"], rtol=0, atol=1e-12)):
                        raise ValueError("invalid selected three-seed validation record")
                    result[active] = row
                    active, record = None, []
                else:
                    record.append(line)
    if set(result) != allowed:
        raise ValueError("requested labels absent from selected-record source")
    return result


def select_settings(inner_results, grid=None):
    grid = grid or configuration_grid()
    grouped = defaultdict(list)
    for row in inner_results:
        grouped[row["setting"]["config_id"]].append(row)
    summaries = {}
    for setting in grid:
        rows = sorted(grouped[setting["config_id"]], key=lambda row: row["seed"])
        if sorted(r["seed"] for r in rows) != [0, 1, 2]:
            raise ValueError("each inner setting must have exactly model seeds 0/1/2")
        summaries[setting["config_id"]] = {
            "setting": setting, "gp_mae": float(np.mean([r["metrics"]["gp_mae"] for r in rows])),
            "primary": float(np.mean([r["metrics"]["main_neighborhood_metric"] for r in rows]))}
    limit = summaries["regression"]["gp_mae"] + 0.001
    n0 = [v for v in summaries.values() if v["setting"]["method"] == "N0"]
    n0_ok = [v for v in n0 if v["gp_mae"] <= limit]
    selected_n0 = min(n0_ok, key=lambda v: (v["primary"], v["setting"]["order"])) if n0_ok else min(
        n0, key=lambda v: (v["gp_mae"], v["primary"], v["setting"]["order"]))
    pairs = []
    for first in (v for v in summaries.values() if v["setting"]["method"] == "N1"):
        second = summaries[first["setting"]["config_id"].replace("N1-", "N2-")]
        pairs.append({"N1": first, "N2": second,
                      "eligible": first["gp_mae"] <= limit and second["gp_mae"] <= limit,
                      "excess": max(0, first["gp_mae"] - limit) + max(0, second["gp_mae"] - limit),
                      "primary": (first["primary"] + second["primary"]) / 2})
    pair_ok = [p for p in pairs if p["eligible"]]
    chosen = min(pair_ok, key=lambda p: (p["primary"], p["N1"]["setting"]["order"])) if pair_ok else min(
        pairs, key=lambda p: (p["excess"], p["primary"], p["N1"]["setting"]["order"]))
    settings = {"regression": grid[0], "N0": selected_n0["setting"],
                "N1": chosen["N1"]["setting"], "N2": chosen["N2"]["setting"]}
    return {"settings": settings, "summaries": summaries, "mae_limit": limit,
            "N0_prediction_gate_satisfied": bool(n0_ok), "N1_N2_joint_prediction_gate_satisfied": bool(pair_ok),
            "N1_N2_diagnostic_excess": chosen["excess"],
            "rule": "mean_of_three_seed_query_medians; paired_N1_N2; fallback_diagnostic_retained"}


def source_identity():
    paths = ["analyse/fuzzy_metric_offline.py", "analyse/fuzzy_metric_offline_core.py",
             "surrogate/fuzzy_metric.py", "analyse/fuzzy_metric_replay.py", "surrogate/accuracy_gp.py",
             "analyse/joint_configuration_metric_gate.py", "architecture_diagnostics.py",
             "configs/fuzzy_metric_stage2_v1.json"]
    return {p: file_hash(ROOT / p) for p in paths}


def finish_label_export(output, prepared, partition, pm):
    oracle_hash = file_hash(prepared / "oracle_validation.json")
    if oracle_hash != pm["oracle_sha256"]:
        raise RuntimeError("historical oracle bytes differ from the verified stage1 source")
    source = output / "A_validation.json"
    a_labels = read_selected_labels(prepared / "oracle_validation.json", partition["A_ids"])
    json_write(source, a_labels, immutable=True)
    json_write(output / "label_boundary.json", {
        "exported_ids": sorted(a_labels), "exported_hash": digest(a_labels), "B_parsed_records": 0,
        "boundary": "trusted selected-record export skips unrequested JSON values; workers only access A_validation.json",
        "protocol_written_before_A_export": True, "test_records": 0,
        "original_oracle_sha256": oracle_hash}, immutable=True)


def freeze(config):
    output, prepared = ROOT / config["output"], ROOT / config["prepared"]
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output / "protocol.json"
    if protocol_file.exists():
        protocol = json.loads(protocol_file.read_text())
        if protocol["config"] != config or protocol["source_identity"] != source_identity():
            raise RuntimeError("existing frozen protocol identity differs; preserve it and investigate")
        partition = json.loads((output / "partition.json").read_text())
        if digest(partition) != protocol["partition_hash"]:
            raise RuntimeError("frozen partition changed")
        finish_label_export(output, prepared, partition, protocol["prepared_manifest"])
        print("Reusing frozen protocol", protocol["protocol_hash"], flush=True)
        return
    candidates = json.loads((prepared / "candidates.json").read_text())
    base = json.loads((prepared / "base_encoding.json").read_text())
    pm = json.loads((prepared / "manifest.json").read_text())
    if digest(candidates) != pm["pool_hash"] or digest(base) != pm["base_encoding_hash"]:
        raise RuntimeError("prepared label-free identity changed")
    root1 = ROOT / config["stage1_root"]
    prior_sets = {}
    for name in ("replay", "replay_resumed", "replay_final", "resume_final"):
        init = json.loads((root1 / name / "initial_selection.json").read_text())["candidate_ids"]
        queries = json.loads((root1 / name / "summary.json").read_text())["query_order"]
        prior_sets[name] = sorted(set(init + queries))
    mandatory = sorted(set().union(*(set(v) for v in prior_sets.values())))
    partition = build_partition(candidates, mandatory, config)
    matrix = build_task_matrix(partition)
    for d in ("tasks", "selection", "source_snapshot", "aggregate"):
        (output / d).mkdir(exist_ok=True)
    identity = source_identity()
    for path in identity:
        target = output / "source_snapshot" / path
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(ROOT / path, target)
    protocol = {"config": config, "source_identity": identity, "partition_hash": digest(partition),
                "task_matrix_hash": digest(matrix), "prepared_manifest": pm,
                "host": socket.gethostname(), "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip(),
                "created_unix": time.time(), "labels_read_before_freeze": [],
                "prior_use": {"sets": prior_sets, "classification": "engineering-only evidence; conservatively assigned all revealed architecture closures to A"},
                "new_identity_only_default": "target half architectures for A; deterministic stop before B capacity failure",
                "defaults_source": "user Codex_Server_Stage2_Prompt.md; no comparative outcomes read",
                "variance_units": "raw accuracy squared: latent + train mean SE squared + learned floor variance"}
    protocol["protocol_hash"] = digest(protocol)
    json_write(output / "partition.json", partition, immutable=True)
    json_write(output / "task_matrix.json", matrix, immutable=True)
    json_write(protocol_file, protocol, immutable=True)
    # Membership and comparison protocol are durable before the first A label parse.
    finish_label_export(output, prepared, partition, pm)
    print(json.dumps({"A": len(partition["A_ids"]), "B": len(partition["B_ids"]),
                      "A_architectures": len(partition["A_architectures"]), "B_architectures": len(partition["B_architectures"]),
                      "stage3_capacity": partition["stage3_original_capacity_feasible"], "planned_tasks": len(matrix)}), flush=True)


def record_fit_start(output, task_id, attempt, kind, limit):
    ledger = Path(output) / "fit_attempts.jsonl"
    with ledger.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        rows = [json.loads(line) for line in stream if line.strip()]
        count = sum(r["kind"] == kind for r in rows)
        if count >= limit:
            raise RuntimeError(f"{kind} hard fit budget reached: {count}/{limit}")
        stream.seek(0, 2)
        stream.write(json.dumps({"kind": kind, "task_id": task_id, "attempt": attempt, "pid": os.getpid(),
                                 "host": socket.gethostname(), "time": time.time(), "threads": torch.get_num_threads()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)


def completed_result(task_dir, expected_identity):
    """Return only a fully persisted, source/input-identical completed task."""
    task_dir = Path(task_dir)
    for done in sorted(task_dir.glob("attempt*/completed.json")):
        row = json.loads(done.read_text())
        if row["identity"] != expected_identity:
            raise RuntimeError("completed task identity differs")
        for name, sha in row["artifact_hashes"].items():
            if file_hash(done.parent / name) != sha:
                raise RuntimeError("completed task artifact changed")
        return row
    return None


def execute_task(task, config, protocol_hash, candidate_rows, base, a_hash):
    from analyse.fuzzy_metric_offline_core import fit_predict, evaluate_predictions
    torch.set_num_threads(config["threads_per_worker"])
    output = ROOT / config["output"]
    task_dir = output / "tasks" / task["task_id"]
    task_dir.mkdir(exist_ok=True)
    identity = digest({"task": task, "protocol": protocol_hash, "A_labels": a_hash})
    completed = completed_result(task_dir, identity)
    if completed is not None:
        return completed
    attempt = len(list(task_dir.glob("attempt*"))) + 1
    path = task_dir / f"attempt{attempt:02d}"
    path.mkdir(exist_ok=False)
    json_write(path / "started.json", {"identity": identity, "task": task, "time": time.time(), "pid": os.getpid()})
    by_id = {c["candidate_id"]: c for c in candidate_rows}
    train = [by_id[cid] for cid in task["train_ids"]]
    query = [by_id[cid] for cid in task["query_ids"]]
    started = time.monotonic()
    try:
        train_labels = read_selected_labels(output / "A_validation.json", task["train_ids"])
        json_write(path / "training_access.json", {"ids": sorted(train_labels), "time": time.time()})
        def callback(kind):
            record_fit_start(output, task["task_id"], attempt, kind, config[f"{kind}_fit_limit"])
        bundle, state = fit_predict(train, query, base, train_labels, task["setting"], task["seed"], config,
                                    on_fit_start=callback)
        torch.save(state, path / "model.pt")
        json_write(path / "predictions_before_reveal.json", bundle, immutable=True)
        # No query label or query SE has been passed to fitting or prediction.
        json_write(path / "evaluation_access.json", {"ids": task["query_ids"], "time": time.time(),
                    "prediction_hash": file_hash(path / "predictions_before_reveal.json"),
                    "model_hash": file_hash(path / "model.pt")})
        query_labels = read_selected_labels(output / "A_validation.json", task["query_ids"])
        evaluation = evaluate_predictions(bundle, train_labels, query_labels)
        json_write(path / "evaluation.json", evaluation)
        result = {"identity": identity, "task_id": task["task_id"], "family": task["family"],
                  "stage": task["stage"], "seed": task["seed"], "setting": task["setting"],
                  "metrics": evaluation["metrics"], "elapsed_seconds": time.monotonic() - started,
                  "attempt": attempt, "directory": str(path),
                  "artifact_hashes": {n: file_hash(path / n) for n in ("model.pt", "predictions_before_reveal.json", "evaluation.json")}}
        json_write(path / "completed.json", result, immutable=True)
        return result
    except Exception:
        json_write(path / "failed.json", {"identity": identity, "task_id": task["task_id"], "attempt": attempt,
                    "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        raise


def run(config):
    output = ROOT / config["output"]
    protocol = json.loads((output / "protocol.json").read_text())
    if config != protocol["config"] or source_identity() != protocol["source_identity"]:
        raise RuntimeError("frozen source/config changed; do not mix task results")
    partition = json.loads((output / "partition.json").read_text())
    matrix = json.loads((output / "task_matrix.json").read_text())
    if digest(partition) != protocol["partition_hash"] or digest(matrix) != protocol["task_matrix_hash"]:
        raise RuntimeError("frozen split/task identity changed")
    boundary = json.loads((output / "label_boundary.json").read_text())
    # Verify the A-only file. No global oracle is opened by this run.
    a_records = read_selected_labels(output / "A_validation.json", partition["A_ids"])
    if digest(a_records) != boundary["exported_hash"]:
        raise RuntimeError("A label source changed")
    del a_records
    prepared = ROOT / config["prepared"]
    candidates = json.loads((prepared / "candidates.json").read_text())
    base = json.loads((prepared / "base_encoding.json").read_text())
    if digest(candidates) != protocol["prepared_manifest"]["pool_hash"] or digest(base) != protocol["prepared_manifest"]["base_encoding_hash"]:
        raise RuntimeError("frozen inputs changed")
    a_set = set(partition["A_ids"])
    candidates = [c for c in candidates if c["candidate_id"] in a_set]
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        for family in partition["folds"] + [partition["final_split"]]:
            name = family["family"]
            inner = [t for t in matrix if t["family"] == name and t["stage"] == "inner"]
            # The first exact matrix task is also a real-input engineering check;
            # reuse it, and do not launch a batch before it has succeeded.
            first = execute_task(inner[0], config, protocol["protocol_hash"], candidates, base, boundary["exported_hash"])
            futures = [pool.submit(execute_task, t, config, protocol["protocol_hash"], candidates, base, boundary["exported_hash"]) for t in inner[1:]]
            inner_results = [first] + [f.result() for f in as_completed(futures)]
            selection = select_settings(inner_results)
            json_write(output / "selection" / f"{name}.json", selection, immutable=True)
            refits = copy.deepcopy([t for t in matrix if t["family"] == name and t["stage"] == "refit"])
            for task in refits:
                task["setting"] = selection["settings"][task["method"]]
            futures = [pool.submit(execute_task, t, config, protocol["protocol_hash"], candidates, base, boundary["exported_hash"]) for t in refits]
            outer_results = [f.result() for f in as_completed(futures)]
            results.extend(inner_results + outer_results)
            print(json.dumps({"completed_family": name, "tasks_so_far": len(results),
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
    json_write(output / "completed_tasks.json", sorted(results, key=lambda r: r["task_id"]), immutable=True)
    summarize(config, results)
    runs = sorted(output.glob("invocation*.json"))
    json_write(output / f"invocation{len(runs)+1:02d}.json", {"elapsed_seconds": time.monotonic() - started,
               "finished_time": time.time(), "completed_tasks": len(results), "gnn_trainings": 0}, immutable=True)


def summarize(config, results):
    from analyse.fuzzy_metric_offline_core import aggregate_evaluations
    output = ROOT / config["output"]
    partition = json.loads((output / "partition.json").read_text())
    aggregate = {}
    for method in ("regression", "N0", "N1", "N2"):
        seeds = []
        for seed in (0, 1, 2):
            rows = [r for r in results if r["stage"] == "refit" and r["family"] != "final_A"
                    and r["seed"] == seed and r["setting"]["method"] == method]
            evaluations = [json.loads((Path(r["directory"]) / "evaluation.json").read_text()) for r in sorted(rows, key=lambda r: r["family"])]
            item = aggregate_evaluations(evaluations)
            if sorted(r["candidate_id"] for r in item["per_query"]) != partition["A_ids"]:
                raise RuntimeError("pooled fivefold evaluation must contain each A candidate once")
            json_write(output / "aggregate" / f"{method}-seed{seed}.json", item, immutable=True)
            seeds.append(item["metrics"])
        numeric = {k: float(np.mean([s[k] for s in seeds])) for k in seeds[0]
                   if all(isinstance(s.get(k), (int, float)) and s[k] is not None for s in seeds)}
        aggregate[method] = {"seed_metrics": seeds, "mean_seed_metrics": numeric}
    base = aggregate["regression"]["mean_seed_metrics"]
    comparisons = {}
    for method in ("N0", "N1", "N2"):
        row = aggregate[method]["mean_seed_metrics"]
        baseline = base["main_neighborhood_metric"]
        improvement = 1 - row["main_neighborhood_metric"] / baseline if baseline > 0 else None
        mae_delta = row["gp_mae"] - base["gp_mae"]
        comparisons[method] = {"relative_neighbor_improvement": improvement,
                               "absolute_neighbor_difference": row["main_neighborhood_metric"] - baseline,
                               "gp_mae_increase": mae_delta, "prediction_gate": mae_delta <= 0.001,
                               "development_gate": improvement is not None and improvement >= 0.1 and mae_delta <= 0.001}
    n1, n2 = [aggregate[m]["mean_seed_metrics"]["main_neighborhood_metric"] for m in ("N1", "N2")]
    difference = 1 - n2 / n1 if n1 > 0 else None
    both_predict = all(comparisons[m]["prediction_gate"] for m in ("N1", "N2"))
    simpler = both_predict and ((difference is not None and difference < 0.02) or (n1 == 0 and n2 == 0))
    eligible = [m for m in ("N0", "N1", "N2") if comparisons[m]["development_gate"]]
    recommended = ("N1" if simpler and "N1" in eligible else min(eligible, key=lambda m: (
        aggregate[m]["mean_seed_metrics"]["main_neighborhood_metric"], ("N0", "N1", "N2").index(m)))) if eligible else None
    ledger = [json.loads(l) for l in (output / "fit_attempts.jsonl").read_text().splitlines()]
    failures = sorted(str(p.relative_to(output)) for p in (output / "tasks").glob("*/attempt*/failed.json"))
    summary = {"status": "completed", "aggregate": aggregate, "comparisons": comparisons,
               "N2_relative_improvement_over_N1": difference, "N1_simplicity_preference": simpler,
               "recommended_method_by_development_rule": recommended,
               "fixed_settings": json.loads((output / "selection/final_A.json").read_text())["settings"],
               "A_candidates": len(partition["A_ids"]), "B_candidates": len(partition["B_ids"]),
               "A_architectures": len(partition["A_architectures"]), "B_architectures": len(partition["B_architectures"]),
               "completed_tasks": len(results), "mapping_fits": sum(r["kind"] == "mapping" for r in ledger),
               "gp_fits": sum(r["kind"] == "gp" for r in ledger), "failed_attempts": failures,
               "sum_task_elapsed_seconds": sum(r["elapsed_seconds"] for r in results),
               "gnn_trainings": 0, "B_labels_parsed": 0, "test_evaluations": 0,
               "interpretation": "development evaluation only; coverage and GP reliability require separate discussion; no stage3 started",
               "stage3_transfer": "parameters/rules/identity-only initial IDs; never full-A trained weights"}
    json_write(output / "summary.json", summary, immutable=True)
    print(json.dumps({"status": "completed", "fits": summary["mapping_fits"], "comparisons": comparisons,
                      "recommendation": recommended}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fuzzy_metric_stage2_v1.json")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if config["gnn_training_budget"] != 0 or config["model_seeds"] != [0, 1, 2] or config["outer_folds"] != 5:
        raise ValueError("stage2 requires zero GNN budget and frozen fivefold/three-seed design")
    torch.set_num_threads(1)
    if args.freeze:
        freeze(config)
    if args.run:
        run(config)
    if not (args.freeze or args.run):
        parser.error("choose --freeze and/or --run")


if __name__ == "__main__":
    main()
