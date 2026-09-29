#!/usr/bin/env python3
"""Independent block-GP score audit. Standard library only; no scorer imports."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time

A, B, C = "RAW16_ISO", "HP4_ISO", "RAW16_BLOCK2"
NEW = (A, B, C)
OLD = ("HEAD", "H_CONST", "H_TREND")
ALL = NEW + OLD
SEEDS = (20260922, 20260923)
Z = {50: .6744897501960817, 80: 1.2815515655446004, 90: 1.6448536269514722, 95: 1.959963984540054}
EPS = 1e-9
REGIONS = ("all_Q", "HEAD_pred20", "true_top20", "true_bottom20", "champions")
A_IDS_SHA256 = "d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2"


def ensure(ok, text):
    if not ok: raise AssertionError(text)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    v = [x for x in values if x is not None]
    return math.fsum(v) / len(v) if v else None


def checked(root, path, sha):
    child = (root / path).resolve()
    ensure(child.is_relative_to(root.resolve()), "Escaping path")
    ensure(digest(child) == sha, "Frozen bytes changed: " + path)
    return child


def within(actual, wanted, name):
    if wanted is None: ensure(actual is None, "Expected N/A: " + name); return 0.
    ensure(isinstance(actual, (int, float)) and math.isfinite(actual), "Nonfinite/missing " + name)
    delta = abs(actual - wanted)
    ensure(math.isclose(actual, wanted, abs_tol=1e-8, rel_tol=1e-10), "Independent mismatch: " + name + f" ({actual} != {wanted})")
    return delta


def midranks(values):
    order = sorted(range(len(values)), key=values.__getitem__); result = [0.] * len(values); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]: j += 1
        for k in range(i, j): result[order[k]] = (i + j + 1) / 2
        i = j
    return result


def spearman(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2: return None
    a, b = midranks(x), midranks(y); ac, bc = mean(a), mean(b)
    cov = math.fsum((i - ac) * (j - bc) for i, j in zip(a, b))
    denom = math.sqrt(math.fsum((i - ac) ** 2 for i in a) * math.fsum((j - bc) ** 2 for j in b))
    return max(-1., min(1., cov / denom))


def region(pred, truth, variance, ix):
    err = [pred[i] - truth[i] for i in ix]
    out = {"mae_pp": mean([abs(v) for v in err]) * 100, "rmse_pp": math.sqrt(mean([v * v for v in err])) * 100, "bias_pp": mean(err) * 100}
    if variance is None:
        out["nlpd_accuracy_units"] = out["IS90_pp"] = None
        for level in Z:
            out[f"coverage{level}_fraction"] = out[f"interval{level}_width_pp"] = out[f"coverage{level}_target_distance_fraction"] = None
        return out
    sigmas = [math.sqrt(variance[i]) for i in ix]
    out["nlpd_accuracy_units"] = mean([math.log(math.sqrt(2 * math.pi * variance[i])) + (truth[i] - pred[i]) ** 2 / (2 * variance[i]) for i in ix])
    scores = []
    for i, sigma in zip(ix, sigmas):
        lower, upper = pred[i] - Z[90] * sigma, pred[i] + Z[90] * sigma
        scores.append((upper - lower + 20 * max(lower - truth[i], 0) + 20 * max(truth[i] - upper, 0)) * 100)
    out["IS90_pp"] = mean(scores)
    for level, z in Z.items():
        cv = mean([float(abs(e) <= z * sigma) for e, sigma in zip(err, sigmas)])
        out[f"coverage{level}_fraction"] = cv
        out[f"interval{level}_width_pp"] = mean([2 * z * sigma * 100 for sigma in sigmas])
        out[f"coverage{level}_target_distance_fraction"] = abs(cv - level / 100)
    return out


def selection(ids, pred, truth, incumbent):
    order = sorted(range(len(ids)), key=lambda k: (-pred[k], ids[k])); top = order[:5]; best = max(truth)
    champ = [k for k, value in enumerate(truth) if abs(best - value) <= EPS / 100]
    chosen_best = max(truth[k] for k in top); gain = max(0, chosen_best - incumbent) * 100; room = max(0, best - incumbent) * 100
    return {"regret5_pp": (best - chosen_best) * 100, "gain5_pp": gain, "headroom_pp": room, "simple_regret_pp": room - gain,
            "break_best_rate": float(chosen_best - incumbent > EPS / 100), "top1_regret_pp": (best - truth[order[0]]) * 100,
            "champion_prediction_rank": min(order.index(k) + 1 for k in champ), "champion_top5_hit_fraction": float(any(k in top for k in champ)),
            "spearman": spearman(pred, truth), "ranking_ids": [ids[k] for k in order], "selected_top5_ids": [ids[k] for k in top], "champion_ids": [ids[k] for k in champ]}


def check_stability(pred):
    s = pred["stability"]; ws = s["windows"]
    ensure([(w["from_step"], w["to_step"]) for w in ws] == [(1200, 1300), (1300, 1400), (1400, 1500)], "Stability windows incomplete")
    passed = all(w["mll_relative_change"] <= .0001 and w["query_mean_max_abs_change_raw"] <= .001 and w["query_latent_sd_max_abs_change_raw"] <= .001
                 and w.get("training_kernel_relative_change", w.get("full_training_kernel_relative_change", float("inf"))) <= .01 for w in ws)
    ensure(s["numeric_predictive_stable"] == passed, "Stability field incorrect")
    return passed


def average_group(rows, keys, fields):
    groups = defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    return [{**dict(zip(keys, k)), **{f: mean([r[f] for r in values]) for f in fields}} for k, values in groups.items()]


def read_csv(path):
    out = []
    for row in csv.DictReader(Path(path).open()):
        parsed = {}
        for k, v in row.items():
            if v == "": parsed[k] = None
            elif v in ("True", "False"): parsed[k] = v == "True"
            else:
                try: parsed[k] = json.loads(v)
                except (ValueError, TypeError): parsed[k] = v
        out.append(parsed)
    return out


def compare_table(ours, path, keys, fields):
    theirs = read_csv(path); lookup = {tuple(r[k] for k in keys): r for r in theirs}; maximum = 0.
    ensure(len(theirs) == len(lookup) == len(ours), "Table row count mismatch " + str(path))
    for row in ours:
        other = lookup[tuple(row[k] for k in keys)]
        for k in fields: maximum = max(maximum, within(other[k], row[k], str(path.name) + ":" + k))
    return maximum


def gate(root):
    seal = read(root / "predictions_manifest.json")
    ensure(seal.get("sealed") is True and seal.get("execution_status") == "COMPLETE_VALID", "No scoring of partial/invalid matrix")
    for keys, n in ((("new_endpoint_count", "completed_new_endpoints"), 30), (("historical_reference_count",), 90),
                    (("raw_unit_count", "completed_endpoints"), 120), (("mask_count",), 30), (("formal_updates",), 54000)):
        vals = [seal[k] for k in keys if k in seal]; ensure(vals and all(v == n for v in vals), "Missing full-seal count")
    for name in ("protocol", "source_manifest", "code_manifest", "worklist", "masks_manifest", "frozen_manifest", "data_manifest"):
        checked(root, name + ".json", seal[name + "_sha256"])
    for manifest in [seal] + [read(root / (name + ".json")) for name in ("source_manifest", "code_manifest", "frozen_manifest", "masks_manifest")]:
        for path, sha in manifest["files"].items(): checked(root, path, sha)
    budget = read(root / "budget.json"); protocol = read(root / "protocol.json")
    ensure(budget["limit"] == protocol["budget"].get("model_seconds", protocol["budget"].get("limit")) == 600 and budget["seconds"] <= 600, "Budget invalid")
    ensure(budget["formal_updates"] == budget["formal_optimizer_calls"] == 54000 and budget["synthetic_optimizer_calls"] <= 300, "Counts invalid")
    records = seal["records"]; ensure(len(records) == 120 and len({r["unit_id"] for r in records}) == 120, "Raw unit count invalid")
    expected = {(f, s, None, m) for f in range(5) for s in SEEDS for m in NEW} | {(f, s, k, m) for f in range(5) for s in SEEDS for k in range(3) for m in OLD}
    ensure({(r["fold"], r["sample_seed"], r["model_seed"], r["method"]) for r in records} == expected, "Scientific matrix incomplete")
    items = []
    for rec in records:
        ensure(seal["files"].get(rec["path"]) == rec["sha256"], "Unsealed prediction")
        pred = read(checked(root, rec["path"], rec["sha256"]))
        for k in ("subset_id", "fold", "sample_seed", "model_seed"): ensure(pred[k] == rec[k], "Identity mismatch")
        if rec["method"] in NEW:
            checked(root, pred["state_path"], pred["state_file_sha256"])
            ensure(pred.get("actual_updates", pred.get("total_steps")) == 1800 and pred.get("rng_seed", pred.get("execution_rng_seed")) == 0, "Endpoint invalid")
            state = pred["stage_states"]["FIT_GP"]
            ensure(state["path"] == pred["state_path"] and state["sha256"] == pred["state_file_sha256"] and state["fingerprint"] == pred["endpoint_state_hash"], "Final state identity differs")
        items.append((rec, pred))
    refs = read(root / "historical_refs.json")["records"]; lut = {r["unit_id"]: r for r in refs}
    ensure(len(lut) == len(refs) == 90, "Historical identity incomplete")
    for rec, pred in items:
        if rec["method"] in OLD:
            ensure(all(lut[rec["unit_id"]][k] == rec[k] for k in ("path", "sha256", "method", "subset_id", "fold", "sample_seed", "model_seed")), "Historical bytes replaced")
            ensure(pred["method"] == {"HEAD": "HEAD", "H_CONST": "GP_CONST", "H_TREND": "GP_TREND"}[rec["method"]], "Historical method alias")
    masks = read(root / "masks_manifest.json")["records"]
    ensure(len(masks) == 30 and len({(r["subset_id"], r["model_seed"]) for r in masks}) == 30, "Masks incomplete")
    heads = {(r["subset_id"], r["model_seed"]): (r, p) for r, p in items if r["method"] == "HEAD"}
    for mask in masks:
        r, p = heads[mask["subset_id"], mask["model_seed"]]
        ordered = sorted(zip(p["candidate_ids"], p["mean_raw"]), key=lambda q: (-q[1], q[0]))
        wanted = [i for i, _ in ordered[:math.ceil(len(ordered) / 5)]]
        ensure(mask["candidate_ids"] == wanted and mask["query_ids"] == p["candidate_ids"], "Pre-frozen HEAD mask differs")
        ensure(mask["source_prediction_path"] == r["path"] and mask["source_prediction_sha256"] == r["sha256"], "Mask source changed")
        saved = read(checked(root, mask["path"], mask["sha256"]))
        ensure(saved["candidate_ids"] == wanted and saved["query_ids"] == p["candidate_ids"], "Mask file content changed")
    release = read(root / "scoring/release.json")
    ensure(release["predictions_manifest_sha256"] == digest(root / "predictions_manifest.json"), "Release not bound to full seal")
    source = read(root / "source_manifest.json")
    expected_labels = source.get("legal_A_scoring_truth_sha256", source.get("expected_labels_sha256", source.get("scoring_labels_expected_sha256")))
    ensure(digest(root / "scoring/labels_A.json") == expected_labels == release["labels_sha256"], "Released label bytes differ")
    return items, masks, budget


def run(root):
    start = time.perf_counter(); items, masks, budget = gate(root)
    labels = read(root / "scoring/labels_A.json")["labels"]
    ids_hash = hashlib.sha256(json.dumps(sorted(labels), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    ensure(len(labels) == 179 and ids_hash == A_IDS_SHA256, "Legal A identity differs")
    mask_lut = {(m["subset_id"], m["model_seed"]): m["candidate_ids"] for m in masks}
    units, regions, mask_rows, random_rows, stable_rows, ambiguity, snapshots = [], [], [], [], [], [], {}
    for rec, pred in items:
        method, sid = rec["method"], rec["subset_id"]; ids = pred["candidate_ids"]; mu = pred["mean_raw"]
        ensure(len(ids) == len(set(ids)) == (38, 36, 36, 34, 35)[rec["fold"]], "Query identity count differs")
        y = [labels[i]["validation_mean"] for i in ids]; ses = [labels[i]["validation_se"] for i in ids]
        if sid not in snapshots:
            t = read(root / "training" / (sid + ".json")); snapshots[sid] = t
            ensure(len(t["ids"]) == len(t["y"]) == 50 and not set(t["ids"]) & set(ids), "T/Q identity invalid")
            incumbent = max(t["y"]); ys = sorted(y); denominator = math.comb(len(y), 5)
            max_distribution = [(value, math.comb(j, 4) / denominator) for j, value in enumerate(ys) if j >= 4]
            room = 100 * max(0, max(y) - incumbent)
            egain = sum(100 * max(0, value - incumbent) * p for value, p in max_distribution)
            nchamp = sum(max(y) - v <= EPS / 100 for v in y)
            random_rows.append({"subset_id": sid, "expected_regret5_pp": sum((max(y) - value) * p * 100 for value, p in max_distribution),
                                "expected_gain5_pp": egain, "headroom_pp": room, "expected_simple_regret_pp": room - egain,
                                "break_best_probability": sum(p for value, p in max_distribution if value - incumbent > EPS / 100),
                                "champion_hit_probability": 1 - (math.comb(len(y) - nchamp, 5) / denominator if len(y) - nchamp >= 5 else 0)})
        best_t = max(snapshots[sid]["y"])
        if method == "HEAD": obs = None
        else:
            latent = pred["latent_variance_standardized"]; scale2 = pred["y_scale"] ** 2; extra = pred["extra_noise_variance_standardized"]
            ensure(all(v >= -1e-8 and math.isfinite(v) for v in latent), "Invalid latent variance")
            vm = [scale2 * max(v, 0) + scale2 * extra for v in latent]
            if "model_variance_raw" in pred:
                for a, b in zip(pred["model_variance_raw"], vm): within(a, b, "base model variance")
            ensure(pred.get("tau_squared_raw", 0) in (0, None), "Unexpected tau")
            obs = [v + s ** 2 for v, s in zip(vm, ses)]
        meta = {k: rec[k] for k in ("unit_id", "subset_id", "fold", "sample_seed", "model_seed", "method")}
        row = {**meta, **selection(ids, mu, y, best_t)}
        regions_ix = {"all_Q": list(range(len(y))), "true_top20": sorted(range(len(y)), key=lambda i: (-y[i], ids[i]))[:math.ceil(len(y) / 5)],
                      "true_bottom20": sorted(range(len(y)), key=lambda i: (y[i], ids[i]))[:math.ceil(len(y) / 5)],
                      "champions": [i for i, value in enumerate(y) if max(y) - value <= EPS / 100]}
        for name, ix in regions_ix.items():
            stats = region(mu, y, obs, ix); regions.append({**meta, "region": name, **stats})
            row.update({name + "__" + k: value for k, value in stats.items()})
            if name == "all_Q": row.update({k: value for k, value in stats.items() if "target_distance" not in k})
        local_masks = []
        for seed in range(3) if method in NEW else [rec["model_seed"]]:
            ix = [ids.index(i) for i in mask_lut[sid, seed]]; stats = region(mu, y, obs, ix); local_masks.append(stats)
            mask_rows.append({**meta, "head_mask_seed": seed, **stats})
        stats = {k: mean([m[k] for m in local_masks]) for k in local_masks[0]}
        regions.append({**meta, "region": "HEAD_pred20", **stats}); row.update({"HEAD_pred20__" + k: v for k, v in stats.items()})
        units.append(row)
        if method in NEW:
            stable_rows.append({**meta, "stable": check_stability(pred), "nll": pred.get("endpoint_nll_after_update", pred.get("negative_mll"))})
    fields = [k for k in units[0] if k not in ("unit_id", "subset_id", "fold", "sample_seed", "model_seed", "method", "ranking_ids", "selected_top5_ids", "champion_ids")]
    subsets = average_group(units, ("method", "subset_id", "fold", "sample_seed"), fields)
    folds = average_group(subsets, ("method", "fold"), fields)
    samples = average_group(subsets, ("method", "sample_seed"), fields)
    means = {m: {k: mean([r[k] for r in folds if r["method"] == m]) for k in fields} for m in ALL}
    pairs = []; by_subset = {(r["subset_id"], r["method"]): r for r in subsets}
    for target, reference in [(C, A), (C, B), (B, A)] + [(n, o) for n in NEW for o in OLD]:
        for r in [r for r in subsets if r["method"] == target]:
            ref = by_subset[r["subset_id"], reference]; p = {k: r[k] for k in ("subset_id", "fold", "sample_seed")}; p["comparison"] = target + "_vs_" + reference
            for k in fields:
                sign = 1 if ("coverage" in k and "target_distance" not in k) or k in ("gain5_pp", "break_best_rate", "champion_top5_hit_fraction", "spearman") else -1
                p[k] = None if r[k] is None or ref[k] is None else sign * (r[k] - ref[k])
            pairs.append(p)
    pf = average_group(pairs, ("comparison", "fold"), fields); ps = average_group(pairs, ("comparison", "sample_seed"), fields)
    pm = {name: {k: mean([r[k] for r in pf if r["comparison"] == name]) for k in fields} for name in {r["comparison"] for r in pairs}}
    def signal(reference):
        name = C + "_vs_" + reference; p = pm[name]
        return (p["regret5_pp"] >= .2 - EPS and p["gain5_pp"] >= -.02 - EPS
                and sum(r["regret5_pp"] > EPS for r in pf if r["comparison"] == name) >= 4
                and all(r["regret5_pp"] > EPS for r in ps if r["comparison"] == name))
    retention = {m: pm[C + "_vs_" + m]["regret5_pp"] >= -.1 - EPS and pm[C + "_vs_" + m]["gain5_pp"] >= -.02 - EPS for m in ("HEAD", "H_TREND")}
    p = pm[C + "_vs_" + A]
    probability = (p["nlpd_accuracy_units"] >= -.1 - EPS and p["IS90_pp"] >= -1 - EPS and p["coverage90_fraction"] >= -.05 - EPS / 100 and p["champions__coverage90_fraction"] >= -.05 - EPS / 100)
    stable_counts = {m: sum(r["stable"] for r in stable_rows if r["method"] == m) for m in NEW}
    ready = stable_counts[A] >= 8 and stable_counts[C] >= 8
    flags = {"BLOCK_SCALE_SELECTION_SIGNAL": signal(A), "HISTORICAL_SELECTION_RETAINED_VS_HEAD": retention["HEAD"],
             "HISTORICAL_SELECTION_RETAINED_VS_H_TREND": retention["H_TREND"], "HISTORICAL_SELECTION_RETAINED": all(retention.values()),
             "ARCHITECTURE_INCREMENT_SIGNAL": signal(B), "PROBABILITY_NO_LARGE_DEGRADATION_C_VS_A": probability, "OPTIMIZATION_READY": ready,
             "BLOCK_DISTANCE_CANDIDATE_FOR_PROTOCOL_REVIEW": signal(A) and all(retention.values()) and ready}
    nll = {(r["subset_id"], r["method"]): r["nll"] for r in stable_rows}
    for sid in snapshots:
        d = nll[sid, C] - nll[sid, A]
        ambiguity.append({"subset_id": sid, "C_minus_A_negative_mll_nats_per_observation": d, "EXPANDED_MODEL_OPTIMIZATION_AMBIGUITY": d > 1e-6})
    # Only now inspect official score outputs: they did not generate our numbers.
    official = read(root / "scoring/summary.json")
    ensure(official["flags"] == flags and official["stability_counts_out_of10"] == stable_counts, "Independent science flags/stability differ")
    ensure(official["expanded_model_optimization_ambiguity_count"] == sum(r["EXPANDED_MODEL_OPTIMIZATION_AMBIGUITY"] for r in ambiguity), "NLL ambiguity count differs")
    maximum = 0.
    for m in ALL:
        for k in fields: maximum = max(maximum, within(official["method_means"][m][k], means[m][k], m + ":" + k))
    for name, values in pm.items():
        for k, value in values.items(): maximum = max(maximum, within(official["paired_means"][name][k], value, name + ":" + k))
    tables = [(units, "unit_metrics", ("unit_id",), fields), (subsets, "subset_metrics", ("method", "subset_id"), fields),
              (folds, "fold_metrics", ("method", "fold"), fields), (samples, "sample_seed_metrics", ("method", "sample_seed"), fields),
              (pairs, "paired_subset", ("comparison", "subset_id"), fields), (pf, "paired_fold", ("comparison", "fold"), fields),
              (ps, "paired_sample_seed", ("comparison", "sample_seed"), fields),
              (regions, "region_unit_metrics", ("unit_id", "region"), list(stats)), (mask_rows, "head_mask_metrics", ("unit_id", "head_mask_seed"), list(stats)),
              (random_rows, "random5_reference", ("subset_id",), [k for k in random_rows[0] if k != "subset_id"])]
    for own, name, keys, metrics in tables: maximum = max(maximum, compare_table(own, root / "scoring" / (name + ".csv"), keys, metrics))
    saved_units = {r["unit_id"]: r for r in read_csv(root / "scoring/unit_metrics.csv")}
    for r in units:
        for key in ("ranking_ids", "selected_top5_ids", "champion_ids"): ensure(r[key] == saved_units[r["unit_id"]][key], "Exact selected/ranked identities differ")
    output = {"pass": True, "independent_no_official_scorer_imports": True, "no_model_calls": True, "raw_units": len(units), "new_fits": 30,
              "historical_seed_units": 90, "historical_seed_metric_average_verified": True, "mask_metric_rows": len(mask_rows), "region_unit_rows": len(regions),
              "subset_rows": len(subsets), "fold_rows": len(folds), "sample_seed_rows": len(samples), "fixed_pair_subset_rows": len(pairs),
              "random_reference_subsets": len(random_rows), "max_absolute_statistic_difference": maximum, "flags": flags,
              "method_means": means, "paired_means": pm, "stability_counts_out_of10": stable_counts, "expanded_model_optimization_ambiguity": ambiguity,
              "budget": budget, "statistical_seconds": time.perf_counter() - start, "predictions_manifest_sha256": digest(root / "predictions_manifest.json"),
              "official_summary_sha256": digest(root / "scoring/summary.json"), "auditor_sha256": digest(__file__), "scope": "sealed arrays, saved window metrics, state/file identities; no optimizer replay or model restoration"}
    (root / "audit").mkdir(exist_ok=True)
    (root / "audit/independent_block_scoring.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(run(args.root.resolve()), ensure_ascii=False, sort_keys=True, allow_nan=False))
