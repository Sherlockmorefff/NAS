#!/usr/bin/env python3
"""Independent OOF-calibration audit using only the Python standard library.

No imports from official scorers, calibration code, model or numeric runtimes.
Every reported effect is reconstructed from the sealed per-candidate arrays.
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

INNER = ("HEAD", "GP_CONST", "GP_TREND")
OUTER = ("HEAD", "CONST_RAW", "CONST_CAL", "TREND_RAW", "TREND_CAL")
SEEDS = (20260922, 20260923)
Z = {50: .6744897501960817, 80: 1.2815515655446004, 90: 1.6448536269514722, 95: 1.959963984540054}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def avg(values):
    x = [v for v in values if v is not None]
    return math.fsum(x) / len(x) if x else None
def check(ok, message):
    if not ok: raise AssertionError(message)
def near(a, b, message, atol=1e-9):
    if a is None:
        check(b is None or b == "", "Unexpected defined metric: " + message); return 0.
    check(b is not None and b != "" and math.isclose(a, float(b), rel_tol=1e-9, abs_tol=atol), "Mismatch " + message + ": " + str((a, b)))
    return abs(a - float(b))
def checked(root, relative, expected):
    p = (root / relative).resolve()
    check(p.is_relative_to(root.resolve()) and sha(p) == expected, "Frozen file path/hash mismatch: " + str(relative))
    return p
def files(manifest):
    v = manifest["files"]
    return [{"path": p, "sha256": h} for p, h in v.items()] if isinstance(v, dict) else v


def rankcorr(a, b):
    def ranks(x): return [1 + sum(t < v for t in x) + .5 * (sum(t == v for t in x) - 1) for v in x]
    x, y = ranks(a), ranks(b); mx, my = avg(x), avg(y)
    xx, yy = math.fsum((v - mx) ** 2 for v in x), math.fsum((v - my) ** 2 for v in y)
    if not xx or not yy: return None
    return math.fsum((v - mx) * (w - my) for v, w in zip(x, y)) / math.sqrt(xx * yy)


def point_metrics(ids, predicted, y, variance, incumbent):
    values = dict(zip(ids, y)); mu = dict(zip(ids, predicted))
    order = sorted(ids, key=lambda cid: (-mu[cid], cid)); selected = order[:5]
    champions = [cid for cid in ids if max(y) - values[cid] <= 1e-11]
    top = sorted(ids, key=lambda cid: (-values[cid], cid))[:math.ceil(.2 * len(ids))]
    error = {cid: mu[cid] - values[cid] for cid in ids}
    gain = max(0, max(values[c] for c in selected) - incumbent); headroom = max(0, max(y) - incumbent)
    result = {"regret5_pp": 100 * (max(y) - max(values[c] for c in selected)), "gain5_pp": 100 * gain,
              "headroom_pp": 100 * headroom, "simple_regret_pp": 100 * (headroom - gain),
              "break_best_rate": float(max(values[c] for c in selected) - incumbent > 1e-11),
              "top1_regret_pp": 100 * (max(y) - values[order[0]]),
              "champion_prediction_rank": min(order.index(c) + 1 for c in champions),
              "champion_top5_hit_fraction": float(bool(set(champions) & set(selected))),
              "spearman": rankcorr(predicted, y), "ranking_ids": order, "selected_top5_ids": selected,
              "champion_ids": champions, "top20_ids": top}
    probability = {}
    for i, cid in enumerate(ids):
        v = variance[i] if variance is not None else None
        if v is None:
            probability[cid] = None
        else:
            check(v > 0 and math.isfinite(v), "Invalid observation variance")
            s = math.sqrt(v)
            d = {"nlpd": math.log(math.sqrt(2 * math.pi * v)) + error[cid] ** 2 / (2 * v)}
            for level, z in Z.items():
                low, high = mu[cid] - z * s, mu[cid] + z * s
                d["coverage" + str(level)] = float(low <= values[cid] <= high)
                d["width" + str(level)] = 100 * (high - low)
                if level == 90: d["IS90"] = 100 * ((high - low) + 20 * max(low - values[cid], 0) + 20 * max(values[cid] - high, 0))
            probability[cid] = d
    result["nlpd_accuracy_units"] = avg([probability[c]["nlpd"] for c in ids]) if variance is not None else None
    for prefix, group in (("", ids), ("champion_", champions), ("top20_", top)):
        result[prefix + "mae_pp"] = 100 * avg([abs(error[c]) for c in group])
        result[prefix + "rmse_pp"] = 100 * math.sqrt(avg([error[c] ** 2 for c in group]))
        result[prefix + "bias_pp"] = 100 * avg([error[c] for c in group])
        for level in Z:
            result[prefix + f"coverage{level}_fraction"] = avg([probability[c]["coverage" + str(level)] for c in group]) if variance is not None else None
            result[prefix + f"interval{level}_width_pp"] = avg([probability[c]["width" + str(level)] for c in group]) if variance is not None else None
        result[prefix + "IS90_pp"] = avg([probability[c]["IS90"] for c in group]) if variance is not None else None
    return result


def nested(rows, field):
    groups = defaultdict(list)
    for r in rows: groups[r["fold"], r["sample_seed"]].append(r[field])
    subsets = {key: avg(v) for key, v in groups.items()}
    folds = {f: avg([v for (ff, s), v in subsets.items() if ff == f]) for f in range(5)}
    seeds = {s: avg([v for (f, ss), v in subsets.items() if ss == s]) for s in SEEDS}
    return avg(folds.values()), folds, seeds


def moment(y, mu, variance, se):
    n = len(y)
    check(n == len(mu) == len(variance) == len(se), "Tau source vector lengths")
    # Independent equivalent computation with separate second-moment sums.
    raw = (math.fsum((a - b) ** 2 for a, b in zip(y, mu)) - math.fsum(variance) - math.fsum(s * s for s in se)) / n
    return raw, max(0, raw)


def run(root):
    inner_seal, outer_seal = read(root / "oof_manifest.json"), read(root / "outer_predictions_manifest.json")
    for name, s in (("OOF", inner_seal), ("outer", outer_seal)):
        check(s.get("sealed") is True and s.get("execution_status") == "COMPLETE_VALID", "Incomplete " + name + " seal; no labels may be read")
    check(inner_seal["completed_pipelines"] == 90 and inner_seal["formal_updates"] == 387000 and inner_seal["oof_rows"] == 4500, "Incomplete OOF accounting")
    protocol = read(checked(root, "protocol.json", outer_seal["protocol_sha256"]))
    source = read(checked(root, "source_manifest.json", outer_seal["source_manifest_sha256"]))
    checked(root, "code_manifest.json", outer_seal["code_manifest_sha256"])
    checked(root, "oof_manifest.json", outer_seal["oof_manifest_sha256"])
    budget = read(root / "budget.json")
    check(budget["limit"] == protocol["budget"].get("model_seconds", protocol["budget"].get("limit")) == 1500 and budget["seconds"] <= 1500, "Budget invalid")
    check(budget["formal_updates"] == budget["formal_optimizer_calls"] == 387000 and budget["synthetic_optimizer_calls"] <= 300, "Incomplete formal calls or synthetic overrun")
    near(budget["seconds"], math.fsum(budget[k] for k in ("training_seconds", "prediction_seconds", "verification_seconds")), "Budget composition")
    predictions = []
    for e in files(inner_seal):
        path = checked(root, e["path"], e["sha256"])
        if e["path"].startswith("predictions_inner/") and e["path"].endswith(".json"):
            predictions.append(read(path))
    for path, digest in outer_seal.get("supporting_files", {}).items(): checked(root, path, digest)
    outer = [read(checked(root, e["path"], e["sha256"])) for e in files(outer_seal)]
    tau = read(checked(root, outer_seal.get("tau_path", "calibration/tau.json"), outer_seal["tau_sha256"]))["rows"]
    contexts = read(root / "worklist.json")["outer_contexts"]
    check(len(contexts) == 30 and {(c["fold"], c["sample_seed"], c["model_seed"]) for c in contexts} == {(f, s, m) for f in range(5) for s in SEEDS for m in range(3)}, "Context identity")
    check(len(predictions) == 270 and sum(len(p["candidate_ids"]) for p in predictions) == 4500, "Incomplete OOF rows")
    check({(p["context_id"], p["inner_fold"], p["method"]) for p in predictions} == {(c["context_id"], j, m) for c in contexts for j in range(3) for m in INNER}, "Incomplete unique inner matrix")
    check(len(outer) == 150 and {(p["context_id"], p["method"]) for p in outer} == {(c["context_id"], m) for c in contexts for m in OUTER}, "Incomplete unique outer matrix")
    check(len(tau) == 60 and {(p["context_id"], p["method"]) for p in tau} == {(c["context_id"], m) for c in contexts for m in INNER[1:]}, "Incomplete unique tau matrix")
    release = read(root / "scoring/release.json")
    check(release["outer_predictions_manifest_sha256"] == sha(root / "outer_predictions_manifest.json") and release["oof_manifest_sha256"] == sha(root / "oof_manifest.json"), "Released labels bound to different seal")
    expected_label_sha = source.get("expected_labels_sha256", source.get("scoring_labels_expected_sha256"))
    labels = read(checked(root, "scoring/labels_A.json", expected_label_sha))["labels"]
    check(len(labels) == 179 and hashlib.sha256(json.dumps(sorted(labels), separators=(",", ":")).encode()).hexdigest() == "d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2", "Legal A labels only")
    train = {c["subset_id"]: read(root / "training" / (c["subset_id"] + ".json")) for c in contexts}
    byinner = {(p["context_id"], p["inner_fold"], p["method"]): p for p in predictions}
    bytau = {(p["context_id"], p["method"]): p for p in tau}
    gaps, tau_report = [], []; max_tau_difference = 0.
    for c in contexts:
        cid = c["context_id"]; t = train[c["subset_id"]]; yy = dict(zip(t["ids"], t["y"])); ss = dict(zip(t["ids"], t["measurement_se"]))
        maps = {m: {} for m in INNER}; train_second = 0.; heldout = []
        for j in range(3):
            h = byinner[cid, j, "HEAD"]; u, v = h["train_candidate_ids"], h["candidate_ids"]
            check(not set(u) & set(v) and set(u) | set(v) == set(t["ids"]), "Same-model split partition")
            train_second += len(v) * math.fsum((yy[i] - m) ** 2 for i, m in zip(u, h["train_mean_raw"])) / len(u) / 50
            heldout.extend(v)
            for method in INNER:
                p = byinner[cid, j, method]
                check(p["candidate_ids"] == v, "Inner methods not paired")
                for k, i in enumerate(v): maps[method][i] = (p["mean_raw"][k], None if method == "HEAD" else p["model_variance_raw"][k])
        check(len(heldout) == len(set(heldout)) == 50 and set(heldout) == set(t["ids"]), "Each T point must occur once OOF")
        mse = {m: avg([(yy[i] - maps[m][i][0]) ** 2 for i in t["ids"]]) for m in INNER}
        gaps.append({**{k: c[k] for k in ("context_id", "fold", "sample_seed", "model_seed")},
                     "gap": 100 * (math.sqrt(mse["HEAD"]) - math.sqrt(train_second)), "correction": 10000 * (mse["HEAD"] - mse["GP_TREND"])})
        for method in INNER[1:]:
            raw, positive = moment(t["y"], [maps[method][i][0] for i in t["ids"]], [maps[method][i][1] for i in t["ids"]], t["measurement_se"])
            saved = bytau[cid, method]
            for key, v in (("unclipped_tau_squared_raw", raw), ("tau_squared_raw", positive)):
                max_tau_difference = max(max_tau_difference, near(v, saved[key], cid + method + key, atol=1e-12))
            tau_report.append({"context_id": cid, "method": method, "unclipped_tau_squared_raw": raw, "tau_squared_raw": positive, "tau_sd_pp": 100 * math.sqrt(positive)})
    csvmetrics = {(r["context_id"], r["method"]): r for r in csv.DictReader((root / "scoring/outer_context_metrics.csv").open())}
    rows = []; max_metric_diff = 0.; means_by_context = {}
    for p in outer:
        cid, method = p["context_id"], p["method"]; t = train[p["subset_id"]]
        parent = read(checked(root, p["parent_prediction_path"], p["parent_prediction_sha256"]))
        check(p["mean_raw"] == parent["mean_raw"] and p["candidate_ids"] == parent["candidate_ids"], "Parent mean changed")
        ids = p["candidate_ids"]; variance = None
        if method != "HEAD":
            origin_method = "GP_CONST" if method.startswith("CONST") else "GP_TREND"
            tau2 = bytau[cid, origin_method]["tau_squared_raw"] if method.endswith("_CAL") else 0
            original = [parent["y_scale"] ** 2 * (max(v, 0) + parent["extra_noise_variance_standardized"]) for v in parent["latent_variance_standardized"]]
            for v, saved, uncal in zip(original, p["model_variance_raw"], p["uncalibrated_model_variance_raw"]):
                near(v, uncal, "Uncalibrated variance", atol=1e-12); near(v + tau2, saved, "Tau variance once", atol=1e-12)
            variance = [v + tau2 + labels[i]["validation_se"] ** 2 for v, i in zip(original, ids)]
        r = point_metrics(ids, p["mean_raw"], [labels[i]["validation_mean"] for i in ids], variance, max(t["y"]))
        official = csvmetrics[cid, method]
        for key, v in r.items():
            if isinstance(v, list): check(v == json.loads(official[key]), "Outer selected/rank/region IDs changed")
            else: max_metric_diff = max(max_metric_diff, near(v, official[key], cid + method + key))
        r.update({k: p[k] for k in ("context_id", "fold", "sample_seed", "model_seed")}); r["method"] = method
        rows.append(r); means_by_context[cid, method] = p["mean_raw"]
    invariance_count = 0
    lookup = {(r["context_id"], r["method"]): r for r in rows}
    invariant_metrics = [k for k in rows[0] if not any(token in k for token in ("coverage", "width", "IS90", "nlpd")) and k not in ("context_id", "fold", "sample_seed", "model_seed", "method")]
    for c in contexts:
        cid = c["context_id"]
        for family in ("CONST", "TREND"):
            check(means_by_context[cid, family + "_RAW"] == means_by_context[cid, family + "_CAL"], "Exact mean invariance failed")
            a, b = lookup[cid, family + "_RAW"], lookup[cid, family + "_CAL"]
            check(all(a[k] == b[k] for k in invariant_metrics), "Exact derived mean/rank invariance failed")
            invariance_count += 1
    summary = read(root / "scoring/summary.json")
    method_means = {}
    for method in OUTER:
        group = [r for r in rows if r["method"] == method]
        method_means[method] = {}
        for field in summary["method_means"][method]:
            v = nested(group, field)[0]; near(v, summary["method_means"][method][field], "Hierarchical mean " + method + field)
            method_means[method][field] = v
    gap, gf, gs = nested(gaps, "gap"); correction, cf, cs = nested(gaps, "correction")
    flags = {"TRAIN_TO_OOF_GAP_SIGNAL": gap >= 1 and sum(v > 1e-9 for v in gf.values()) >= 4 and all(v > 1e-9 for v in gs.values()),
             "OOF_RESIDUAL_CORRECTION_SIGNAL": correction > 1e-9 and sum(v > 1e-9 for v in cf.values()) >= 4 and all(v > 1e-9 for v in cs.values())}
    cal_effects = {}
    for family in ("CONST", "TREND"):
        diffs = []
        for c in contexts:
            raw, cal = [lookup[c["context_id"], family + suffix] for suffix in ("_RAW", "_CAL")]
            diffs.append({"fold": c["fold"], "sample_seed": c["sample_seed"], "nlpd": raw["nlpd_accuracy_units"] - cal["nlpd_accuracy_units"], "IS90": raw["IS90_pp"] - cal["IS90_pp"]})
        nd, nf, ns = nested(diffs, "nlpd"); iz, _, _ = nested(diffs, "IS90")
        distance = abs(method_means[family + "_RAW"]["coverage90_fraction"] - .9) - abs(method_means[family + "_CAL"]["coverage90_fraction"] - .9)
        flag = nd > 1e-10 and iz > 1e-9 and distance > 1e-12 and sum(v > 0 for v in nf.values()) >= 4 and all(v > 0 for v in ns.values())
        flags["CALIBRATION_SIGNAL" if family == "TREND" else "CONST_CALIBRATION_SIGNAL"] = flag
        cal_effects[family] = {"nlpd_improvement_nats": nd, "IS90_improvement_pp": iz, "coverage_distance_improvement": distance, "nlpd_fold_values": nf, "nlpd_sample_seed_values": ns}
    c, b = method_means["TREND_CAL"], method_means["CONST_RAW"]
    flags["PROBABILITY_COMPETITIVE_WITH_CONST_RAW"] = c["nlpd_accuracy_units"] - b["nlpd_accuracy_units"] <= .10 and c["IS90_pp"] - b["IS90_pp"] <= 1 and 100 * (c["coverage90_fraction"] - b["coverage90_fraction"]) >= -5 and 100 * (c["champion_coverage90_fraction"] - b["champion_coverage90_fraction"]) >= -5
    flags["MEAN_AND_SELECTION_INVARIANT"] = invariance_count == 60
    check(flags == summary["flags"], "Independent scientific flags differ")
    near(gap, summary["gap_signal_detail"]["mean"], "Gap mean")
    near(correction, summary["oof_correction_detail"]["mean"], "OOF correction mean")
    result = {"status": "PASS", "audit_script_sha256": sha(__file__), "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_updates": 387000, "inner_predictions": 270, "OOF_rows": 4500, "outer_results": 150, "tau_count": 60,
        "RAW_CAL_exact_invariance_pairs": invariance_count, "max_tau_squared_absolute_difference": max_tau_difference,
        "max_outer_metric_absolute_difference": max_metric_diff, "gap_RMSE_pp": gap, "OOF_squared_error_improvement_pp2": correction,
        "gap_fold_values": gf, "correction_fold_values": cf, "calibration_effects": cal_effects,
        "flags": flags, "method_means": method_means, "recomputed_tau": tau_report,
        "model_seconds": budget["seconds"], "model_calls_in_this_audit": 0, "official_scorer_imported": False,
        "isolated_B_or_test_read": False, "further_execution_authorized": False}
    out = root / "audit/independent_oof_scoring.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({"status": result["status"], "flags": result["flags"]}))
