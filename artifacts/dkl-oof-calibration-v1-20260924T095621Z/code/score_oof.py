#!/usr/bin/env python3
"""Statistics-only OOF/outer scoring. No model imports and no partial release.

Inner labels are selected-T snapshots, accessed only after the complete OOF
seal. Outer Q labels are accessed only after all 270 inner sets, 60 tau values,
and 150 outer results pass integrity and accounting checks.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import score_stage2 as base

INNER = ("HEAD", "GP_CONST", "GP_TREND")
OUTER = ("HEAD", "CONST_RAW", "CONST_CAL", "TREND_RAW", "TREND_CAL")
SEEDS = (20260922, 20260923)
Z = {50: .6744897501960817, 80: 1.2815515655446004, 90: 1.6448536269514722, 95: 1.959963984540054}
TOL_PP = 1e-9
META = ("context_id", "subset_id", "fold", "sample_seed", "model_seed")
MEAN_METRICS = ("regret5_pp", "gain5_pp", "headroom_pp", "simple_regret_pp", "break_best_rate", "top1_regret_pp",
                "champion_prediction_rank", "champion_top5_hit_fraction", "mae_pp", "rmse_pp", "bias_pp", "spearman",
                "champion_mae_pp", "champion_rmse_pp", "champion_bias_pp", "top20_mae_pp", "top20_rmse_pp", "top20_bias_pp")
PROB_METRICS = ("nlpd_accuracy_units", "IS90_pp", "champion_IS90_pp", "top20_IS90_pp") + tuple(
    prefix + field + str(level) + suffix for prefix in ("", "champion_", "top20_") for level in Z
    for field, suffix in (("coverage", "_fraction"), ("interval", "_width_pp")))
METRICS = MEAN_METRICS + PROB_METRICS


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def close(a, b, message, atol=1e-12, rtol=1e-10):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol), message)


def checked(root, path, sha):
    p = base.safe_child(root, path)
    require(base.sha256(p) == sha, "Hash mismatch: " + str(path))
    return p


def entries(manifest):
    files = manifest["files"]
    return [{"path": p, "sha256": h} for p, h in files.items()] if isinstance(files, dict) else files


def count_field(record, names, expected):
    available = [record[n] for n in names if n in record]
    require(available and all(v == expected for v in available), "Missing/wrong count " + "/".join(names))


def context_key(row):
    return row["fold"], row["sample_seed"], row["model_seed"]


def sealed_oof(root):
    seal = base.read_json(root / "oof_manifest.json")
    require(seal.get("sealed") is True and seal.get("execution_status") == "COMPLETE_VALID", "Complete OOF seal required before any T holdout scoring")
    count_field(seal, ("completed_pipelines", "pipeline_count"), 90)
    count_field(seal, ("formal_updates",), 387000)
    count_field(seal, ("oof_rows", "oof_row_count"), 4500)
    for name in ("protocol", "worklist"):
        checked(root, name + ".json", seal[name + "_sha256"])
    for name in ("source_manifest", "code_manifest", "frozen_manifest"):
        p = root / (name + ".json")
        require(p.exists(), "Required frozen manifest absent: " + name)
        if name + "_sha256" in seal: checked(root, p.name, seal[name + "_sha256"])
        for path, sha in base.read_json(p)["files"].items(): checked(root, path, sha)
    predictions = []
    for entry in entries(seal):
        path = checked(root, entry["path"], entry["sha256"])
        if not (entry["path"].startswith("predictions_inner/") and entry["path"].endswith(".json")): continue
        p = base.read_json(path)
        base.forbid_query_truth(p)
        require(p["method"] in INNER and p["n_outer"] == 50 and p["n_fit"] in (33, 34) and p["n_holdout"] in (16, 17), "Inner method or sizes differ")
        ids = p["candidate_ids"]
        require(len(ids) == p["n_holdout"] and len(set(ids)) == len(ids), "Inner held-out IDs differ")
        require(len(p["mean_raw"]) == len(p["architecture_fingerprints"]) == len(ids), "OOF vector shape")
        require(all(math.isfinite(v) for v in p["mean_raw"]), "Nonfinite OOF mean")
        if p["method"] == "HEAD":
            require(len(p["train_candidate_ids"]) == len(p["train_mean_raw"]) == p["n_fit"], "Same-model HEAD training prediction absent")
            require(p["actual_updates"] == 1000, "HEAD must use the fixed inner MSE1000 endpoint")
        else:
            require(p["actual_updates"] == 1500, "GP must use the fixed inner readout1500 endpoint")
            require(len(p["model_variance_raw"]) == len(ids), "OOF model variance shape")
            for v, latent in zip(p["model_variance_raw"], p["latent_variance_standardized"]):
                require(latent >= -1e-8 and math.isfinite(latent), "Invalid original latent variance")
                close(v, p["y_scale"] ** 2 * (max(latent, 0) + p["extra_noise_variance_standardized"]), "OOF model variance must include extra exactly once and exclude SE")
                require(v > 0, "Nonpositive model variance")
        if p.get("state_path") and p.get("state_file_sha256"):
            checked(root, p["state_path"], p["state_file_sha256"])
        predictions.append(p)
    require(len(predictions) == 270 and sum(len(p["candidate_ids"]) for p in predictions) == 4500, "Full OOF predictions required")
    work = base.read_json(root / "worklist.json")
    contexts = work["outer_contexts"]
    expected = {(c["context_id"], inner, m) for c in contexts for inner in range(3) for m in INNER}
    seen = {(p["context_id"], p["inner_fold"], p["method"]) for p in predictions}
    require(len(contexts) == 30 and seen == expected and len(seen) == len(predictions), "OOF scientific matrix incomplete or duplicated")
    require({context_key(c) for c in contexts} == {(f, s, m) for f in range(5) for s in SEEDS for m in range(3)}, "Outer context identity differs")
    budget = base.read_json(root / "budget.json"); protocol = base.read_json(root / "protocol.json")
    limit = protocol["budget"].get("model_seconds", protocol["budget"].get("limit"))
    require(budget["limit"] == limit == 1500 and budget["seconds"] <= limit, "Protocol model budget invalid")
    require(budget["formal_updates"] == budget["formal_optimizer_calls"] == 387000 and budget["synthetic_optimizer_calls"] <= 300, "OOF update accounting incomplete")
    return seal, predictions, contexts


def training_snapshots(root, contexts):
    snapshots = {}
    for c in contexts:
        sid = c["subset_id"]
        if sid in snapshots: continue
        t = base.read_json(root / "training" / (sid + ".json"))
        require(len(t["ids"]) == len(t["y"]) == len(t["measurement_se"]) == 50 and len(set(t["ids"])) == 50, "Selected T must contain exactly 50 original observations")
        snapshots[sid] = t
    require(len(snapshots) == 10, "Ten selected training subsets required")
    return snapshots


def tau_moments(y, mu, model_variance, se):
    require(len(y) == len(mu) == len(model_variance) == len(se) and len(y) > 0, "Tau vector lengths differ")
    contribution = [(a - b) ** 2 - v - s * s for a, b, v, s in zip(y, mu, model_variance, se)]
    require(all(math.isfinite(v) for v in contribution), "Nonfinite untruncated tau contribution")
    raw = base.mean(contribution)
    return {"unclipped_tau_squared_raw": raw, "tau_squared_raw": max(0, raw), "tau_sd_pp": 100 * math.sqrt(max(0, raw)), "contributions_raw": contribution}


def evaluate(ids, mu, y, observation_variance=None, incumbent=None):
    require(len(ids) == len(mu) == len(y) and len(ids) >= 5, "Metric vector shape")
    order = sorted(range(len(ids)), key=lambda i: (-mu[i], ids[i])); selected = order[:5]
    ranking = {i: k + 1 for k, i in enumerate(order)}
    errors = [a - b for a, b in zip(mu, y)]; best = max(y)
    champion = [i for i, value in enumerate(y) if best - value <= TOL_PP / 100]
    top20 = sorted(range(len(ids)), key=lambda i: (-y[i], ids[i]))[:math.ceil(.2 * len(ids))]
    r = {"mae_pp": 100 * base.mean([abs(e) for e in errors]), "rmse_pp": 100 * math.sqrt(base.mean([e * e for e in errors])),
         "bias_pp": 100 * base.mean(errors), "mse_pp2": 10000 * base.mean([e * e for e in errors]),
         "spearman": base.spearman(mu, y)[0], "selected_top5_ids": [ids[i] for i in selected],
         "selected_top5_truth_raw": [y[i] for i in selected], "ranking_ids": [ids[i] for i in order],
         "champion_ids": [ids[i] for i in champion], "top20_ids": [ids[i] for i in top20],
         "champion_prediction_rank": min(ranking[i] for i in champion), "champion_top5_hit_fraction": float(bool(set(selected) & set(champion))),
         "regret5_pp": 100 * (best - max(y[i] for i in selected)), "top1_regret_pp": 100 * (best - y[order[0]])}
    if incumbent is not None:
        gain = max(0, max(y[i] for i in selected) - incumbent); headroom = max(0, best - incumbent)
        r.update(gain5_pp=100 * gain, headroom_pp=100 * headroom, simple_regret_pp=100 * (headroom - gain),
                 break_best_rate=float(max(y[i] for i in selected) - incumbent > TOL_PP / 100))
    if observation_variance is not None:
        require(len(observation_variance) == len(ids) and all(math.isfinite(v) and v > 0 for v in observation_variance), "Invalid observation variance")
        sd = [math.sqrt(v) for v in observation_variance]
        nlpd = [.5 * (math.log(2 * math.pi * v) + e * e / v) for e, v in zip(errors, observation_variance)]
    else:
        sd = [None] * len(ids); nlpd = [None] * len(ids)
    r["nlpd_accuracy_units"] = base.mean(nlpd)
    for prefix, ix in (("", list(range(len(ids)))), ("champion_", champion), ("top20_", top20)):
        if prefix:
            r[prefix + "mae_pp"] = 100 * base.mean([abs(errors[i]) for i in ix])
            r[prefix + "rmse_pp"] = 100 * math.sqrt(base.mean([errors[i] ** 2 for i in ix]))
            r[prefix + "bias_pp"] = 100 * base.mean([errors[i] for i in ix])
        for level, z in Z.items():
            r[prefix + f"coverage{level}_fraction"] = None if observation_variance is None else base.mean([float(abs(errors[i]) <= z * sd[i]) for i in ix])
            r[prefix + f"interval{level}_width_pp"] = None if observation_variance is None else 200 * z * base.mean([sd[i] for i in ix])
        r[prefix + "IS90_pp"] = None if observation_variance is None else 100 * base.mean([2 * Z[90] * sd[i] + 20 * max(abs(errors[i]) - Z[90] * sd[i], 0) for i in ix])
    points = []
    for i, cid in enumerate(ids):
        p = {"candidate_id": cid, "mean_raw": mu[i], "truth_raw": y[i], "error_prediction_minus_truth_pp": 100 * errors[i],
             "prediction_rank": ranking[i], "selected_top5": i in selected, "true_champion": i in champion, "true_top20": i in top20,
             "observed_variance_raw": None if observation_variance is None else observation_variance[i], "nlpd_accuracy_units": nlpd[i]}
        for level, z in Z.items():
            p[f"coverage{level}"] = None if sd[i] is None else float(abs(errors[i]) <= z * sd[i])
            p[f"interval{level}_width_pp"] = None if sd[i] is None else 200 * z * sd[i]
        p["IS90_pp"] = None if sd[i] is None else 100 * (2 * Z[90] * sd[i] + 20 * max(abs(errors[i]) - Z[90] * sd[i], 0))
        points.append(p)
    return r, points


def aggregate(rows, keys, metrics):
    groups = defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    return [{**dict(zip(keys, key)), "child_count": len(group), **{m: base.mean([r.get(m) for r in group]) for m in metrics}}
            for key, group in sorted(groups.items(), key=lambda v: str(v[0]))]


def hierarchy(rows, metrics, group="method"):
    subset = aggregate(rows, (group, "fold", "sample_seed"), metrics)
    folds = aggregate(subset, (group, "fold"), metrics)
    seeds = aggregate(subset, (group, "sample_seed"), metrics)
    model_folds = aggregate(rows, (group, "model_seed", "fold"), metrics)
    models = aggregate(model_folds, (group, "model_seed"), metrics)
    return subset, folds, seeds, models


def same_model_gap(parts):
    require(sum(p["n_holdout"] for p in parts) == 50, "OOF held-out weight sum must be 50")
    mse_in = math.fsum(p["n_holdout"] / 50 * base.mean([e * e for e in p["training_errors_raw"]]) for p in parts)
    mse_oof = math.fsum(e * e for p in parts for e in p["holdout_errors_raw"]) / 50
    return {"mse_in_raw": mse_in, "mse_oof_raw": mse_oof, "training_weighted_rmse_pp": 100 * math.sqrt(mse_in),
            "oof_rmse_pp": 100 * math.sqrt(mse_oof), "GAP_RMSE_pp": 100 * (math.sqrt(mse_oof) - math.sqrt(mse_in))}


def inner_statistics(root, predictions, contexts, snapshots):
    lookup = {(p["context_id"], p["inner_fold"], p["method"]): p for p in predictions}
    context_rows, point_rows, gap_rows, fold_rows, correction_rows, tau_rows, stability = [], [], [], [], [], [], []
    for ctx in contexts:
        cid = ctx["context_id"]; t = snapshots[ctx["subset_id"]]
        labels = dict(zip(t["ids"], t["y"])); errors = {}; seen = []; per_method = {m: {} for m in INNER}; gap_parts = []
        se = dict(zip(t["ids"], t["measurement_se"]))
        for j in range(3):
            head = lookup[cid, j, "HEAD"]; v = head["candidate_ids"]; u = head["train_candidate_ids"]
            require(set(u).isdisjoint(v) and set(u) | set(v) == set(t["ids"]), "Same-model U/V partition differs")
            require(len(u) == head["n_fit"] and len(v) == head["n_holdout"], "U/V partition counts differ")
            seen.extend(v)
            eu = [labels[i] - m for i, m in zip(u, head["train_mean_raw"])]; ev = [labels[i] - m for i, m in zip(v, head["mean_raw"])]
            gap_parts.append({"n_holdout": len(v), "training_errors_raw": eu, "holdout_errors_raw": ev})
            for role, ids, mean in (("U", u, head["train_mean_raw"]), ("V", v, head["mean_raw"])):
                met, _ = evaluate(ids, mean, [labels[i] for i in ids])
                fold_rows.append({**{k: ctx[k] for k in META}, "inner_fold": j, "role": role, "count": len(ids),
                                  **{k: met[k] for k in ("mae_pp", "rmse_pp", "bias_pp", "mse_pp2")}})
            for method in INNER:
                p = lookup[cid, j, method]
                require(p["candidate_ids"] == v and p["architecture_fingerprints"] == head["architecture_fingerprints"], "Paired OOF candidate/architecture order differs")
                for k, i in enumerate(v):
                    per_method[method][i] = {"mean": p["mean_raw"][k], "variance": None if method == "HEAD" else p["model_variance_raw"][k],
                                             "architecture": p["architecture_fingerprints"][k], "inner_fold": j}
                if method != "HEAD":
                    s = p["stability"]
                    require(len(s["windows"]) == 3 and [(w["from_step"], w["to_step"]) for w in s["windows"]] == [(1200, 1300), (1300, 1400), (1400, 1500)], "Missing prescribed inner stability windows")
                    stable = all(w["mll_relative_change"] <= 1e-4 and w["query_mean_max_abs_change_raw"] <= .001 and
                                 w["query_latent_sd_max_abs_change_raw"] <= .001 and w["matern_training_kernel_relative_change"] <= .01 and
                                 w["full_training_kernel_relative_change"] <= .01 for w in s["windows"])
                    require(stable == s["numeric_predictive_stable"], "Inner stability mismatch")
                    stability.append({**{k: ctx[k] for k in META}, "inner_fold": j, "method": method,
                                      "numeric_predictive_stable": stable, "top5_stable": s["top5_stable"], "windows": s["windows"]})
        require(len(seen) == len(set(seen)) == 50 and set(seen) == set(t["ids"]), "Each T observation must be OOF exactly once per method")
        gap = {**{k: ctx[k] for k in META}, "method": "HEAD", **same_model_gap(gap_parts)}
        ids = t["ids"]; y = t["y"]; ses = t["measurement_se"]
        correction = []
        for i in ids:
            eh = labels[i] - per_method["HEAD"][i]["mean"]; delta = per_method["GP_TREND"][i]["mean"] - per_method["HEAD"][i]["mean"]
            et = labels[i] - per_method["GP_TREND"][i]["mean"]
            improvement = eh * eh - et * et
            close(improvement, 2 * eh * delta - delta * delta, "Residual-correction identity failed")
            correction.append(improvement * 10000)
            correction_rows.append({**{k: ctx[k] for k in META}, "candidate_id": i, "inner_fold": per_method["HEAD"][i]["inner_fold"],
                "e_HEAD_raw": eh, "delta_TREND_minus_HEAD_raw": delta, "e_TREND_raw": et, "squared_error_improvement_pp2": improvement * 10000,
                "twice_error_times_delta_pp2": 20000 * eh * delta, "delta_squared_pp2": 10000 * delta * delta})
        gap["OOF_squared_error_improvement_pp2"] = base.mean(correction); gap_rows.append(gap)
        for method in INNER:
            values = per_method[method]; mu = [values[i]["mean"] for i in ids]
            vmodel = None if method == "HEAD" else [values[i]["variance"] for i in ids]
            estimated = None if vmodel is None else tau_moments(y, mu, vmodel, ses)
            if estimated is not None:
                squared = [(a - b) ** 2 for a, b in zip(y, mu)]; total = math.fsum(squared)
                arch_sse = defaultdict(float); arch_contributions = defaultdict(list)
                for i, error2, contribution in zip(ids, squared, estimated["contributions_raw"]):
                    arch = values[i]["architecture"]; arch_sse[arch] += error2; arch_contributions[arch].append(contribution)
                sensitivity = [{"architecture": a, "removed_count": len(v), "tau_squared_without_arch_raw": max(0, (sum(estimated["contributions_raw"]) - sum(v)) / (50 - len(v)))} for a, v in arch_contributions.items()]
                tau_rows.append({**{k: ctx[k] for k in META}, "method": method, **estimated,
                    "max_single_candidate_squared_error_fraction": max(squared) / total if total else 0,
                    "max_single_architecture_squared_error_fraction": max(arch_sse.values()) / total if total else 0,
                    "architecture_squared_error_fractions": {a: v / total if total else 0 for a, v in arch_sse.items()},
                    "delete_one_architecture_descriptive_only_no_refit": sensitivity})
            for variant in (["HEAD"] if method == "HEAD" else [method.replace("GP_", "") + "_RAW", method.replace("GP_", "") + "_CAL"]):
                tau = estimated["tau_squared_raw"] if variant.endswith("_CAL") else 0
                obs = None if vmodel is None else [v + tau + s * s for v, s in zip(vmodel, ses)]
                met, pts = evaluate(ids, mu, y, obs)
                row = {**{k: ctx[k] for k in META}, "method": variant, "count": 50, "scope": "apparent calibration on the same OOF records used to estimate tau", **met}
                context_rows.append(row)
                for p in pts:
                    p.update({k: ctx[k] for k in META}); p["method"] = variant; p["inner_fold"] = values[p["candidate_id"]]["inner_fold"]
                    p["model_variance_raw"] = values[p["candidate_id"]]["variance"]; p["tau_squared_raw"] = tau
                    point_rows.append(p)
    return {"context_metrics": context_rows, "candidate_metrics": point_rows, "same_model_inner_fold_errors": fold_rows,
            "gap_contexts": gap_rows, "residual_corrections": correction_rows, "tau_diagnostics": tau_rows, "stability": stability}


def outer_seal(root, oof_seal, contexts):
    seal = base.read_json(root / "outer_predictions_manifest.json")
    require(seal.get("sealed") is True and seal.get("execution_status") == "COMPLETE_VALID", "Complete outer calibration seal required before Q label access")
    count_field(seal, ("expected_endpoint_count", "expected_outer_count"), 150)
    count_field(seal, ("completed_endpoints", "completed_outer_count", "outer_prediction_count"), 150)
    count_field(seal, ("tau_count",), 60)
    for name in ("protocol", "source_manifest", "code_manifest", "worklist"):
        checked(root, name + ".json", seal[name + "_sha256"])
    for path, sha in seal.get("supporting_files", {}).items(): checked(root, path, sha)
    checked(root, "oof_manifest.json", seal["oof_manifest_sha256"])
    tau_path = checked(root, seal.get("tau_path", "calibration/tau.json"), seal["tau_sha256"])
    taus = base.read_json(tau_path)["rows"]
    require(len(taus) == 60 and {(t["context_id"], t["method"]) for t in taus} == {(c["context_id"], m) for c in contexts for m in INNER[1:]}, "Complete unique 60 tau matrix required")
    for t in taus:
        require(math.isfinite(t["tau_squared_raw"]) and t["tau_squared_raw"] >= 0, "Invalid tau variance")
        close(t["tau_squared_raw"], max(0, t["unclipped_tau_squared_raw"]), "Tau nonnegative moment rule differs")
    rows = []
    for e in entries(seal):
        p = base.read_json(checked(root, e["path"], e["sha256"]))
        base.forbid_query_truth(p)
        require(p["method"] in OUTER, "Unprescribed outer method")
        parent = base.read_json(checked(root, p["parent_prediction_path"], p["parent_prediction_sha256"]))
        require(p["candidate_ids"] == parent["candidate_ids"] and p["mean_raw"] == parent["mean_raw"], "Outer mean/query IDs changed from parent")
        require(digest(p["mean_raw"]) == p["mean_array_sha256"], "Mean array hash differs")
        if p["method"] != "HEAD":
            unc = [parent["y_scale"] ** 2 * (max(z, 0) + parent["extra_noise_variance_standardized"]) for z in parent["latent_variance_standardized"]]
            require(len(unc) == len(p["uncalibrated_model_variance_raw"]) == len(p["model_variance_raw"]) == len(p["candidate_ids"]), "Outer variance vectors differ")
            for a, v, w in zip(unc, p["uncalibrated_model_variance_raw"], p["model_variance_raw"]):
                close(a, v, "Outer uncalibrated variance differs from parent latent+extra")
                close(w, v + p["tau_squared_raw"], "Outer CAL variance must add tau exactly once")
            t = next(t for t in taus if t["context_id"] == p["context_id"] and t["method"] == ("GP_CONST" if p["method"].startswith("CONST") else "GP_TREND"))
            close(p["tau_squared_raw"], t["tau_squared_raw"] if p["method"].endswith("_CAL") else 0, "Derived result references incorrect tau")
        rows.append(p)
    require(len(rows) == 150 and {(p["context_id"], p["method"]) for p in rows} == {(c["context_id"], m) for c in contexts for m in OUTER}, "Full unique 150-result outer matrix required")
    for c in contexts:
        pair = {p["method"]: p for p in rows if p["context_id"] == c["context_id"]}
        for m in ("CONST", "TREND"):
            a, b = pair[m + "_RAW"], pair[m + "_CAL"]
            require(a["mean_raw"] == b["mean_raw"] and a["mean_array_sha256"] == b["mean_array_sha256"] and a["candidate_ids"] == b["candidate_ids"], "Calibration changed mean or ID order")
    return seal, rows, taus


def load_q_labels(root, labels_from=None):
    source = base.read_json(root / "source_manifest.json")
    target = root / "scoring" / "labels_A.json"
    expected = source.get("expected_labels_sha256", source.get("scoring_labels_expected_sha256"))
    if not target.exists():
        src = Path(labels_from or source.get("source_label_path") or source["label_source_path"])
        require(base.sha256(src) == expected, "Legal A label source differs")
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, target)
    require(base.sha256(target) == expected, "Released labels differ")
    base.write_json(root / "scoring/release.json", {"released_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_predictions_manifest_sha256": base.sha256(root / "outer_predictions_manifest.json"),
        "oof_manifest_sha256": base.sha256(root / "oof_manifest.json"), "labels_A_sha256": expected,
        "full_90_pipelines_4500_OOF_60_tau_150_outer_verified_first": True, "further_model_training_allowed": False})
    return base.checked_labels(target, set())


def signal_summary(gaps, outer_rows, invariant):
    gsub, gfold, gseed, gmodel = hierarchy(gaps, ("GAP_RMSE_pp", "OOF_squared_error_improvement_pp2"))
    def signed_signal(metric, minimum, tolerance):
        fv = [r[metric] for r in gfold]; sv = [r[metric] for r in gseed]
        mean = base.mean(fv)
        ok = mean >= minimum if minimum is not None else mean > tolerance
        return {"pass": bool(ok and sum(v > tolerance for v in fv) >= 4 and all(v > tolerance for v in sv)),
                "mean": mean, "fold_values": {str(r["fold"]): r[metric] for r in gfold},
                "sample_seed_values": {str(r["sample_seed"]): r[metric] for r in gseed}}
    sub, folds, seeds, models = hierarchy(outer_rows, METRICS)
    means = {m: {k: base.mean([r[k] for r in folds if r["method"] == m]) for k in METRICS} for m in OUTER}
    lookup = {(r["context_id"], r["method"]): r for r in outer_rows}; pairs = []
    for target, reference in (("TREND_CAL", "TREND_RAW"), ("CONST_CAL", "CONST_RAW"), ("TREND_CAL", "CONST_RAW"), ("TREND_CAL", "CONST_CAL")):
        for c in [r for r in outer_rows if r["method"] == target]:
            b = lookup[c["context_id"], reference]
            p = {**{k: c[k] for k in META}, "method": target + "_vs_" + reference}
            p.update(nlpd_improvement_nats=b["nlpd_accuracy_units"] - c["nlpd_accuracy_units"], IS90_improvement_pp=b["IS90_pp"] - c["IS90_pp"],
                     coverage90_change_fraction=c["coverage90_fraction"] - b["coverage90_fraction"],
                     champion_coverage90_change_fraction=c["champion_coverage90_fraction"] - b["champion_coverage90_fraction"],
                     champion_IS90_improvement_pp=b["champion_IS90_pp"] - c["champion_IS90_pp"], top20_IS90_improvement_pp=b["top20_IS90_pp"] - c["top20_IS90_pp"])
            pairs.append(p)
    pair_fields = ("nlpd_improvement_nats", "IS90_improvement_pp", "coverage90_change_fraction", "champion_coverage90_change_fraction", "champion_IS90_improvement_pp", "top20_IS90_improvement_pp")
    ps, pf, pseed, pm = hierarchy(pairs, pair_fields)
    calibration = {}
    for family in ("TREND", "CONST"):
        name = family + "_CAL_vs_" + family + "_RAW"; f = [r for r in pf if r["method"] == name]; s = [r for r in pseed if r["method"] == name]
        raw, cal = means[family + "_RAW"], means[family + "_CAL"]
        distance = abs(raw["coverage90_fraction"] - .9) - abs(cal["coverage90_fraction"] - .9)
        nd = base.mean([r["nlpd_improvement_nats"] for r in f]); iz = base.mean([r["IS90_improvement_pp"] for r in f])
        calibration[family] = {"pass": bool(nd > 1e-10 and iz > 1e-9 and distance > 1e-12 and sum(r["nlpd_improvement_nats"] > 0 for r in f) >= 4 and all(r["nlpd_improvement_nats"] > 0 for r in s)),
            "nlpd_improvement_nats": nd, "IS90_improvement_pp": iz, "coverage_distance_to_0_9_improvement": distance,
            "nlpd_fold_values": {str(r["fold"]): r["nlpd_improvement_nats"] for r in f}, "nlpd_sample_seed_values": {str(r["sample_seed"]): r["nlpd_improvement_nats"] for r in s}}
    c, raw = means["TREND_CAL"], means["CONST_RAW"]
    comp = {"nlpd_increase_nats": c["nlpd_accuracy_units"] - raw["nlpd_accuracy_units"], "IS90_increase_pp": c["IS90_pp"] - raw["IS90_pp"],
            "coverage90_change_percentage_points": 100 * (c["coverage90_fraction"] - raw["coverage90_fraction"]),
            "champion_coverage90_change_percentage_points": 100 * (c["champion_coverage90_fraction"] - raw["champion_coverage90_fraction"])}
    competitive = comp["nlpd_increase_nats"] <= .10 and comp["IS90_increase_pp"] <= 1 and comp["coverage90_change_percentage_points"] >= -5 and comp["champion_coverage90_change_percentage_points"] >= -5
    gap = signed_signal("GAP_RMSE_pp", 1., 1e-9); corr = signed_signal("OOF_squared_error_improvement_pp2", None, 1e-9)
    summary = {"engineering_status": "COMPLETE_VALID", "method_means": means, "gap_signal_detail": gap, "oof_correction_detail": corr,
        "calibration_signal_details": calibration, "competitive_probability_deltas": comp,
        "flags": {"TRAIN_TO_OOF_GAP_SIGNAL": gap["pass"], "OOF_RESIDUAL_CORRECTION_SIGNAL": corr["pass"],
            "CALIBRATION_SIGNAL": calibration["TREND"]["pass"], "CONST_CALIBRATION_SIGNAL": calibration["CONST"]["pass"],
            "PROBABILITY_COMPETITIVE_WITH_CONST_RAW": competitive, "MEAN_AND_SELECTION_INVARIANT": invariant},
        "NLPD_units": "natural log, original accuracy proportion; nat/candidate", "error_width_IS90_units": "percentage points",
        "inner_correction_units": "pp squared", "coverage_units": "fraction", "next_stage_authorized": False,
        "new_mean_selection_capability_tested": False, "A_repeated_development": True}
    tables = {"outer_context_metrics": outer_rows, "outer_subset_metrics": sub, "outer_fold_metrics": folds, "outer_sample_seed_metrics": seeds,
        "outer_model_seed_metrics": models, "paired_context": pairs, "paired_subset": ps, "paired_fold": pf, "paired_sample_seed": pseed, "paired_model_seed": pm,
        "gap_subset_metrics": gsub, "gap_fold_metrics": gfold, "gap_sample_seed_metrics": gseed, "gap_model_seed_metrics": gmodel}
    return summary, tables


def write_inner(root, stats):
    out = root / "scoring_inner"; out.mkdir(exist_ok=True)
    for name, rows in stats.items():
        if name in ("tau_diagnostics", "stability"): base.write_json(out / (name + ".json"), {"rows": rows})
        else: base.write_csv(out / (name + ".csv"), rows)
    metrics = ("mae_pp", "rmse_pp", "bias_pp", "mse_pp2") + PROB_METRICS
    for label, rows in zip(("subset_metrics", "fold_metrics", "sample_seed_metrics", "model_seed_metrics"), hierarchy(stats["context_metrics"], metrics)):
        base.write_csv(out / (label + ".csv"), rows)
    base.write_json(out / "interpretation.json", {"calibrated_OOF_probability_is_apparent_only": True,
        "pooled_RMSE_computed_from_all50_errors_before_sqrt": True, "OOF_small_pool_regret_not_outer_main_metric": True,
        "each_inner_GP_stability_count_out_of90": {m: sum(r["method"] == m and r["numeric_predictive_stable"] for r in stats["stability"]) for m in INNER[1:]}})


def run(root, mode="outer", labels_from=None):
    start = time.perf_counter(); oof, predictions, contexts = sealed_oof(root)
    if mode == "outer": outer, outer_predictions, taus = outer_seal(root, oof, contexts)
    snapshots = training_snapshots(root, contexts)
    stats = inner_statistics(root, predictions, contexts, snapshots)
    write_inner(root, stats)
    if mode == "inner": return {"status": "INNER_STATISTICS_COMPLETE_NO_Q_LABELS", "statistical_seconds": time.perf_counter() - start}
    saved_tau = {(t["context_id"], t["method"]): t for t in taus}
    for t in stats["tau_diagnostics"]:
        for key in ("tau_squared_raw", "unclipped_tau_squared_raw"):
            close(t[key], saved_tau[t["context_id"], t["method"]][key], "Independent per-T tau reconstruction differs")
    labels = load_q_labels(root, labels_from)
    rows, points = [], []
    for p in outer_predictions:
        t = snapshots[p["subset_id"]]; ids = p["candidate_ids"]
        require(not set(t["ids"]) & set(ids), "Outer query leaks into selected T")
        y = [labels[i]["validation_mean"] for i in ids]
        obs = None if p["method"] == "HEAD" else [v + labels[i]["validation_se"] ** 2 for v, i in zip(p["model_variance_raw"], ids)]
        result, candidates = evaluate(ids, p["mean_raw"], y, obs, max(t["y"]))
        meta = {k: p[k] for k in META}; meta["method"] = p["method"]
        rows.append({**meta, **result})
        for c in candidates: c.update(meta)
        points.extend(candidates)
    lookup = {(r["context_id"], r["method"]): r for r in rows}; invariant_rows = []
    for ctx in contexts:
        for family in ("CONST", "TREND"):
            raw, cal = [lookup[ctx["context_id"], family + suffix] for suffix in ("_RAW", "_CAL")]
            require(raw["ranking_ids"] == cal["ranking_ids"] and raw["selected_top5_ids"] == cal["selected_top5_ids"], "RAW/CAL exact rank or Top5 differs")
            deltas = {m: None if raw.get(m) is None else cal[m] - raw[m] for m in MEAN_METRICS}
            require(all(v is None or abs(v) <= 1e-9 for v in deltas.values()), "Calibration mean-based metric changed; experiment invalid")
            invariant_rows.append({**{k: ctx[k] for k in META}, "method": family, "exact_mean_and_rank_identity": True, "mean_metric_differences": deltas})
    require(len(invariant_rows) == 60, "Must verify all 60 RAW/CAL pairs")
    summary, tables = signal_summary(stats["gap_contexts"], rows, True)
    tables.update(outer_candidate_metrics=points, mean_invariance=invariant_rows)
    out = root / "scoring"
    for name, table in tables.items(): base.write_csv(out / (name + ".csv"), table)
    summary.update(oof_row_count=4500, oof_prediction_sets=270, completed_pipelines=90, tau_count=60, outer_result_count=150,
                   RAW_CAL_mean_invariance_pairs=60, new_outer_fits=0, statistical_seconds=time.perf_counter() - start,
                   oof_manifest_sha256=base.sha256(root / "oof_manifest.json"), outer_predictions_manifest_sha256=base.sha256(root / "outer_predictions_manifest.json"),
                   labels_A_sha256=base.sha256(out / "labels_A.json"), scorer_sha256=base.sha256(__file__),
                   stability_counts_out_of90={m: sum(r["method"] == m and r["numeric_predictive_stable"] for r in stats["stability"]) for m in INNER[1:]})
    base.write_json(out / "summary.json", summary)
    return summary


def synthetic_checks():
    import tempfile
    start = time.perf_counter(); checks = []
    r = tau_moments([.7, .9], [.5, .7], [.01, .01], [.1, .1])
    close(r["tau_squared_raw"], .02, "Manual tau example")
    require(tau_moments([.5], [.5], [.01], [.1])["tau_squared_raw"] == 0, "Tau zero boundary")
    checks.append("moment_SE_once_positive_negative_units")
    parts = [{"n_holdout": 17, "training_errors_raw": [.01] * 33, "holdout_errors_raw": [.02] * 17},
             {"n_holdout": 17, "training_errors_raw": [.03] * 33, "holdout_errors_raw": [.04] * 17},
             {"n_holdout": 16, "training_errors_raw": [.05] * 34, "holdout_errors_raw": [.06] * 16}]
    g = same_model_gap(parts)
    close(g["GAP_RMSE_pp"], 100 * (math.sqrt((17 * .02 ** 2 + 17 * .04 ** 2 + 16 * .06 ** 2) / 50) - math.sqrt((17 * .01 ** 2 + 17 * .03 ** 2 + 16 * .05 ** 2) / 50)), "Same-model gap weighted MSE before sqrt")
    checks.append("same_model_gap_pooled_not_mean_RMSE")
    ids = list("abcdefg"); y = [.5, .6, .7, .8, .9, .9, .2]; mu = [.7] * 7
    raw, _ = evaluate(ids, mu, y, [.01] * 7, .8); cal, _ = evaluate(ids, mu, y, [.04] * 7, .8)
    require(raw["selected_top5_ids"] == cal["selected_top5_ids"] == list("abcde"), "Exact ties use IDs")
    for k in MEAN_METRICS: require(raw.get(k) == cal.get(k), "Mean/selection invariance")
    require(raw["champion_prediction_rank"] == 5 and raw["champion_top5_hit_fraction"] == 1, "Champion tie rule")
    zero, _ = evaluate(ids, mu, y, [.01 + 0] * 7, .8); require(raw == zero, "Tau zero must exactly recover raw scores")
    head, _ = evaluate(ids, mu, y, None, .8); require(all(head[k] is None for k in PROB_METRICS), "HEAD uncertainty must be N/A")
    checks.append("all60_pair_logic_exact_ranks_champion_tauzero_HEAD_NA")
    e, d = .13, -.02; close(e * e - (e - d) ** 2, 2 * e * d - d * d, "Correction identity")
    checks.append("correction_identity_pp2")
    all_rows, gaps = [], []
    for f in range(5):
        for s in SEEDS:
            for seed in range(3):
                meta = dict(context_id=f"c{f}_{s}_{seed}", subset_id=f"s{f}_{s}", fold=f, sample_seed=s, model_seed=seed)
                gaps.append({**meta, "method": "HEAD", "GAP_RMSE_pp": 2., "OOF_squared_error_improvement_pp2": .5})
                for m in OUTER:
                    variance = None if m == "HEAD" else [.0001 if m == "TREND_RAW" else .04] * 7
                    met, _ = evaluate(ids, mu, y, variance, .8)
                    all_rows.append({**meta, "method": m, **met})
    summary, tables = signal_summary(gaps, all_rows, True)
    require(summary["flags"]["TRAIN_TO_OOF_GAP_SIGNAL"] and summary["flags"]["OOF_RESIDUAL_CORRECTION_SIGNAL"], "Gap/correction positive direction")
    require(summary["flags"]["CALIBRATION_SIGNAL"] and not summary["flags"]["CONST_CALIBRATION_SIGNAL"], "Calibration strict improvement and tauzero control")
    require(summary["flags"]["PROBABILITY_COMPETITIVE_WITH_CONST_RAW"] and summary["flags"]["MEAN_AND_SELECTION_INVARIANT"], "Separate competitive/invariant flags")
    require(len(tables["paired_subset"]) == 40 and len(tables["paired_fold"]) == 20, "Paired three-level aggregation counts")
    checks.append("six_science_flags_strict_improvement_control_and_hierarchy")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); base.write_json(root / "oof_manifest.json", {"sealed": True, "execution_status": "PARTIAL"})
        try: sealed_oof(root)
        except ValueError: pass
        else: raise AssertionError("Partial OOF must reject without reading labels")
    checks.append("partial_seal_rejection_before_labels")
    return {"pass": True, "checks": checks, "statistical_seconds": time.perf_counter() - start, "model_seconds": 0, "optimizer_updates": 0, "real_labels_accessed": False}


def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("inner", "outer", "synthetic-checks")); p.add_argument("--root", type=Path); p.add_argument("--labels-from")
    a = p.parse_args()
    if a.mode == "synthetic-checks": result = synthetic_checks()
    else:
        require(a.root is not None, "--root is required")
        result = run(a.root, a.mode, a.labels_from)
    print(json.dumps(result if a.mode == "synthetic-checks" else {"status": result.get("status", result.get("engineering_status")), "flags": result.get("flags")}, ensure_ascii=False))


if __name__ == "__main__": main()
