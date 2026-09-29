#!/usr/bin/env python3
"""Post-seal, training-free scoring of the 1000/2000/4000 DKL length diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import score_stage2 as base

STEPS = (1000, 2000, 4000)
INTERVALS = ((1000, 2000), (1000, 4000), (2000, 4000))
SOURCE_PROTOCOL_SHA256 = "cea701e4aaa64974b8d80053ddfb3cd8af421ff86dda38014284b02d0c67aee0"
SOURCE_PACKAGE_SHA256 = "3898e1189ae17f9d056a2478e94ab4a7f1b6b7539b6aae5fd26153dd5b642717"
METRICS = base.METRICS + ("gain5_pp", "query_gain_upper_bound_pp", "random_expected_gain5_pp", "random_expected_regret5_pp")
COST_FIELDS = ("additional_training_seconds", "diagnostic_prediction_seconds", "verification_seconds",
               "new_fit_predict_seconds", "inherited_stage2_fit_predict_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_file(root, path, sha):
    result = base.safe_child(root, path)
    require(base.sha256(result) == sha, "File hash mismatch: " + str(path))
    return result


def trajectory_identity(unit):
    return (unit["fold"], unit["sample_seed"], unit["n"], unit["model_seed"])


def expected_trajectories():
    return {(f, s, n, m) for f in range(5) for s in base.SAMPLE_SEEDS for n in base.NS for m in base.MODEL_SEEDS}


def load_seal(root, manifest_path):
    """Check the entire new seal and all source identities before opening A labels."""
    seal = base.read_json(manifest_path)
    require(seal.get("sealed") is True, "New predictions are not uniformly sealed")
    status = seal.get("execution_status")
    require(status in ("COMPLETE_VALID", "PARTIAL"), "INVALID/BLOCKED or unknown execution status cannot be scored")
    require(seal.get("expected_endpoint_count") == 180 and seal.get("new_endpoint_count_expected") == 120,
            "Length experiment must retain 180 total / 120 new endpoint expectation")
    for name in ("protocol", "code_manifest", "source_manifest"):
        verify_file(root, seal.get(name + "_path", name + ".json"), seal[name + "_sha256"])
    sources = base.read_json(root / seal.get("source_manifest_path", "source_manifest.json"))
    require(sources["source_protocol_sha256"] == SOURCE_PROTOCOL_SHA256, "Wrong source Stage 2 protocol")
    require(sources["source_package_sha256"] == SOURCE_PACKAGE_SHA256, "Wrong source Stage 2 evidence package identity")
    for relative, sha in sources["files"].items():
        verify_file(root, relative, sha)
    code_manifest = base.read_json(root / seal.get("code_manifest_path", "code_manifest.json"))
    for relative, sha in code_manifest["files"].items():
        verify_file(root, relative, sha)
    contexts = {c["context_id"]: c for c in sources["contexts"]}
    require(len(sources["contexts"]) == len(contexts) == 60 and {trajectory_identity(c) for c in contexts.values()} == expected_trajectories(),
            "Frozen source contexts must contain all 60 original trajectories")
    baseline, training = {}, {}
    for context_id, ctx in contexts.items():
        for prefix in ("checkpoint", "prediction", "training"):
            relative, sha = ctx[prefix + "_path"], ctx[prefix + "_sha256"]
            require(sources["files"].get(relative) == sha, "Source context file absent from frozen source files")
            verify_file(root, relative, sha)
        p = base.validate_prediction(base.read_json(root / ctx["prediction_path"]))
        require(p["method"] == "DKL" and p["context_id"] == context_id and trajectory_identity(p) == trajectory_identity(ctx),
                "Source prediction does not match its DKL context")
        require(p["endpoint_state_hash"] == ctx["model_state_hash"], "Source model mathematical state identity mismatch")
        require(p["training_ids_sha256"] == ctx["training_ids_sha256"], "Source training ID identity mismatch")
        require(p["training_tensor_sha256"] == ctx["training_tensor_sha256"], "Source training tensor identity mismatch")
        t = base.read_json(root / ctx["training_path"])
        require(len(t["ids"]) == ctx["n"] and len(t["y"]) == ctx["n"] and len(set(t["ids"])) == ctx["n"],
                "Invalid source selected training snapshot")
        require(digest(t["ids"]) == ctx["training_ids_sha256"], "Training ID order differs from source")
        require(digest(t["scaler"]) == ctx["scaler_sha256"] and t["scaler"] == ctx["scaler"], "Source scaler changed")
        require(p["y_center"] == t["scaler"]["y_mean"] and p["y_scale"] == t["scaler"]["y_scale"], "Prediction scaler mismatch")
        require(not set(t["ids"]) & set(p["candidate_ids"]), "Training and query candidate overlap")
        scheduler = ctx["source_scheduler"]
        require(scheduler["mode"] == "joint" and scheduler["planned_steps"] == 1000 and scheduler["completed"] == 1000,
                "Source continuation checkpoint scheduler identity mismatch")
        baseline[context_id] = p
        training[context_id] = t
    units, seen, unit_ids = [], set(), set()
    for entry in seal["files"]:
        context_id, step = entry["context_id"], entry["total_joint_step"]
        require(context_id in contexts and step in STEPS, "Unknown trajectory or unregistered scoring checkpoint")
        key = (context_id, step)
        require(key not in seen and entry["unit_id"] not in unit_ids, "Duplicate scientific endpoint")
        seen.add(key); unit_ids.add(entry["unit_id"])
        path = verify_file(root, entry["path"], entry["sha256"])
        p = base.validate_prediction(base.read_json(path))
        ctx, original = contexts[context_id], baseline[context_id]
        require(p["method"] == "DKL" and p["context_id"] == context_id, "Prediction endpoint identity mismatch")
        require(trajectory_identity(p) == trajectory_identity(ctx), "Context ID metadata changed")
        for field in ("candidate_ids", "architecture_fingerprints", "subset_id", "training_ids_sha256", "y_center", "y_scale"):
            require(p[field] == original[field], "Continuation prediction source identity changed: " + field)
        require(p["endpoint_state_hash"] == entry["endpoint_state_hash"], "Manifest mathematical state hash mismatch")
        if step == 1000:
            require(p["unit_id"] == original["unit_id"] and entry["unit_id"] in (original["unit_id"], context_id + "_j1000"),
                    "1000 baseline may only use its original ID or registered j1000 alias")
            require(entry["path"] == ctx["prediction_path"] and entry["sha256"] == ctx["prediction_sha256"], "1000 baseline must reference exact original prediction bytes")
            require(entry["state_path"] == ctx["checkpoint_path"] and entry["state_file_sha256"] == ctx["checkpoint_sha256"],
                    "1000 baseline must reference resumable source checkpoint")
            require(entry["endpoint_state_hash"] == ctx["model_state_hash"], "1000 baseline model identity changed")
            # The original prediction's state_path belongs to the historical
            # artifact, whereas this seal intentionally pins its resumable source
            # checkpoint. Do not reinterpret the old path inside the new bundle.
        else:
            require(p["unit_id"] == entry["unit_id"], "New prediction endpoint ID mismatch")
            require(p.get("total_joint_step") == step, "New endpoint total joint step mismatch")
            require(p["state_path"] == entry["state_path"] and p["state_file_sha256"] == entry["state_file_sha256"],
                    "New endpoint file-state identity mismatch")
        verify_file(root, entry["state_path"], entry["state_file_sha256"])
        p["source_unit_id"] = original["unit_id"]
        p["unit_id"] = entry["unit_id"]
        p["total_joint_step"] = step
        p["prediction_file_sha256"] = entry["sha256"]
        units.append(p)
    expected = {(context_id, step) for context_id in contexts for step in STEPS}
    complete = seen == expected and status == "COMPLETE_VALID"
    require(status != "COMPLETE_VALID" or complete, "COMPLETE_VALID requires all 180 DKL scientific endpoints")
    # All fixed controls are frozen prediction bytes only; no historical scoring
    # table or query truth is available to the training path.
    controls = []
    for relative in sorted(sources["files"]):
        if relative.startswith("source/controls/predictions/") and relative.endswith(".json"):
            p = base.validate_prediction(base.read_json(root / relative))
            require(p["method"] in ("REG", "FROZEN", "RAW16"), "Unexpected fixed control method")
            p["prediction_file_sha256"] = sources["files"][relative]
            controls.append(p)
    expected_controls = {x for x in base.expected_matrix() if x[-1] != "DKL"}
    require(len(controls) == 140 and {base.identity(p) for p in controls} == expected_controls,
            "All original 140 fixed controls must be present once")
    return seal, sources, contexts, training, units, controls, complete, sorted(expected - seen)


def gain_reference(query_y, training_best, k=5):
    transformed = [max(0.0, float(y) - training_best) for y in query_y]
    result = base.exact_random5(transformed, k=k)
    return result["expected_best5_raw"]


def score_endpoint(prediction, training, labels):
    result = base.score_unit(prediction, labels)
    for cid, value in zip(training["ids"], training["y"]):
        require(cid in labels and math.isclose(value, labels[cid]["validation_mean"], abs_tol=1e-12, rel_tol=1e-12),
                "Scoring labels and frozen selected training means disagree")
    best = max(base.finite(value, "selected training y") for value in training["y"])
    query_y = [labels[cid]["validation_mean"] for cid in prediction["candidate_ids"]]
    gain = max(0.0, result["selected_best_validation_mean_raw"] - best)
    upper = max(0.0, max(query_y) - best)
    random_gain = gain_reference(query_y, best)
    random_regret = base.exact_random5(query_y)["expected_regret5_pp"]
    result.update(total_joint_step=prediction.get("total_joint_step"), training_best_raw=best,
                  gain5_raw=gain, gain5_pp=100 * gain, query_gain_upper_bound_raw=upper,
                  query_gain_upper_bound_pp=100 * upper, random_expected_gain5_raw=random_gain,
                  random_expected_gain5_pp=100 * random_gain, random_expected_regret5_pp=random_regret,
                  broke_training_best=(100 * gain > base.TIE_TOLERANCE_PP),
                  original_control_not_continued=prediction["method"] != "DKL")
    result["source_unit_id"] = prediction.get("source_unit_id", prediction["unit_id"])
    return result


def aggregate(rows, group_fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        row = dict(zip(group_fields, key))
        row["child_count"] = len(group)
        for metric in metrics:
            stats = base.spread([r[metric] for r in group])
            row[metric] = stats["mean"]
            for name in ("sample_sd", "min", "max", "defined_count"):
                row[metric + "_" + name] = stats[name]
        output.append(row)
    return output


def hierarchy(rows, metrics=METRICS, checkpoint_field="total_joint_step"):
    sample = aggregate(rows, ("n", checkpoint_field, "fold", "sample_seed"), metrics)
    fold = aggregate(sample, ("n", checkpoint_field, "fold"), metrics)
    final = aggregate(fold, ("n", checkpoint_field), metrics)
    return sample, fold, final


def direction(value):
    return "improved" if value > base.TIE_TOLERANCE_PP else "worse" if value < -base.TIE_TOLERANCE_PP else "tie"


def count_directions(values):
    return {"improved": sum(v > base.TIE_TOLERANCE_PP for v in values),
            "tied": sum(abs(v) <= base.TIE_TOLERANCE_PP for v in values),
            "worse": sum(v < -base.TIE_TOLERANCE_PP for v in values)}


def pair_rows(rows):
    lookup = {(r["context_id"], r["total_joint_step"]): r for r in rows}
    output = []
    for context_id in sorted({r["context_id"] for r in rows}):
        for earlier, later in INTERVALS:
            if (context_id, earlier) not in lookup or (context_id, later) not in lookup:
                continue
            a, b = lookup[(context_id, earlier)], lookup[(context_id, later)]
            row = {k: a[k] for k in ("context_id", "subset_id", "fold", "sample_seed", "n", "model_seed")}
            row.update(comparison=f"{earlier}_to_{later}", earlier_step=earlier, later_step=later,
                       earlier_unit_id=a["unit_id"], later_unit_id=b["unit_id"],
                       regret5_improvement_pp=a["regret5_pp"] - b["regret5_pp"],
                       gain5_improvement_pp=b["gain5_pp"] - a["gain5_pp"],
                       top1_regret_improvement_pp=a["top1_regret_pp"] - b["top1_regret_pp"],
                       mae_improvement_pp=a["mae_pp"] - b["mae_pp"], rmse_improvement_pp=a["rmse_pp"] - b["rmse_pp"],
                       top5_overlap_count=len(set(a["selected_top5_ids"]) & set(b["selected_top5_ids"])),
                       top5_set_changed=set(a["selected_top5_ids"]) != set(b["selected_top5_ids"]),
                       top5_order_changed=a["selected_top5_ids"] != b["selected_top5_ids"],
                       top1_changed=a["selected_top1_id"] != b["selected_top1_id"])
            row["regret5_direction"] = direction(row["regret5_improvement_pp"])
            row["gain5_direction"] = direction(row["gain5_improvement_pp"])
            output.append(row)
    return output


PAIR_METRICS = ("regret5_improvement_pp", "gain5_improvement_pp", "top1_regret_improvement_pp",
                "mae_improvement_pp", "rmse_improvement_pp", "top5_overlap_count")


def summarize(rows, complete):
    samples, folds, overall = hierarchy(rows)
    pairs = pair_rows(rows)
    pair_samples, pair_folds, pair_overall = hierarchy(pairs, PAIR_METRICS, "comparison")
    # Seed-specific views: retain the named seed, average the other seed axis
    # inside each fold, then equally average folds.
    model_fold = aggregate(pairs, ("n", "comparison", "model_seed", "fold"), PAIR_METRICS)
    model_summary = aggregate(model_fold, ("n", "comparison", "model_seed"), PAIR_METRICS)
    sample_summary = aggregate(pair_samples, ("n", "comparison", "sample_seed"), PAIR_METRICS)
    summary = {
        "execution_status": "COMPLETE_VALID" if complete else "PARTIAL",
        "expected_endpoint_count": 180, "scientific_endpoint_count": len(rows),
        "new_endpoint_count_expected": 120, "new_endpoint_count": sum(r["total_joint_step"] != 1000 for r in rows),
        "new_formal_updates_expected": 180000, "next_stage_authorized": False,
        "aggregation": "model seeds within fold/sample first, sample seeds within fold second, five folds equally last",
        "partial_results": "available matched trajectories only; always show denominator; no missing endpoint imputation",
        "tie_tolerance_pp": base.TIE_TOLERANCE_PP,
        "metric_units": {"pp": "percentage points", "coverage": "fraction", "NLPD": "raw accuracy density units"},
        "regret_improvement_direction": "earlier minus later; positive favors longer training",
        "gain_improvement_direction": "later minus earlier; positive favors longer training",
        "source_stage2_classifications_unchanged": True,
        "developmental_followup_after_exposure_to_A": True,
        "no_independent_confirmation_or_significance_claim": True,
        "fixed_controls_label": "原阶段2固定预算参照，未续训；不能声称与4000步DKL等更新预算比较",
        "per_n": {},
    }
    for n in base.NS:
        checkpoints, comparisons = {}, {}
        for step in STEPS:
            selected = [r for r in rows if r["n"] == n and r["total_joint_step"] == step]
            fr = [r for r in folds if r["n"] == n and r["total_joint_step"] == step]
            checkpoints[str(step)] = {
                "trajectory_count": len(selected), "expected_trajectory_count": 30, "fold_count": len(fr),
                "metrics": {metric: base.spread([r[metric] for r in fr]) for metric in METRICS},
                "trajectory_spread": {metric: base.spread([r[metric] for r in selected]) for metric in METRICS},
                "broke_training_best_trajectories": sum(r["broke_training_best"] for r in selected),
                "query_has_improvement_room_trajectories": sum(r["query_gain_upper_bound_pp"] > base.TIE_TOLERANCE_PP for r in selected),
                "practical_stability": "该项历史信息不足" if step == 1000 else "see separately recomputed label-free stability tables",
            }
        for earlier, later in INTERVALS:
            name = f"{earlier}_to_{later}"
            selected = [r for r in pairs if r["n"] == n and r["comparison"] == name]
            fr = [r for r in pair_folds if r["n"] == n and r["comparison"] == name]
            comparisons[name] = {
                "matched_trajectory_count": len(selected), "expected_trajectory_count": 30,
                "metrics": {metric: base.spread([r[metric] for r in fr]) for metric in PAIR_METRICS},
                "trajectory_regret5_directions": count_directions([r["regret5_improvement_pp"] for r in selected]),
                "trajectory_gain5_directions": count_directions([r["gain5_improvement_pp"] for r in selected]),
                "fold_regret5_directions": count_directions([r["regret5_improvement_pp"] for r in fr]),
                "fold_gain5_directions": count_directions([r["gain5_improvement_pp"] for r in fr]),
                "fold_regret5_improvement_pp": {str(r["fold"]): r["regret5_improvement_pp"] for r in fr},
                "fold_gain5_improvement_pp": {str(r["fold"]): r["gain5_improvement_pp"] for r in fr},
                "top5_set_changed_trajectories": sum(r["top5_set_changed"] for r in selected),
                "top5_order_changed_trajectories": sum(r["top5_order_changed"] for r in selected),
                "top1_changed_trajectories": sum(r["top1_changed"] for r in selected),
            }
        summary["per_n"][str(n)] = {"checkpoints": checkpoints, "comparisons": comparisons}
    return summary, {"sample_seed_metrics.csv": samples, "fold_metrics.csv": folds, "trajectory_pairs.csv": pairs,
                     "sample_seed_pairs.csv": pair_samples, "fold_pairs.csv": pair_folds,
                     "model_seed_pairs.csv": model_summary, "sample_seed_pair_summary.csv": sample_summary}


def fixed_control_summary(rows):
    results = {}
    for method in ("FROZEN", "REG", "RAW16"):
        selected = [dict(r, total_joint_step=1000) for r in rows if r["method"] == method]
        _, folds, _ = hierarchy(selected)
        results[method] = {str(n): {"endpoint_count": sum(r["n"] == n for r in selected),
                                     "metrics": {metric: base.spread([r[metric] for r in folds if r["n"] == n]) for metric in METRICS}}
                           for n in base.NS}
    return {"label": "原阶段2固定预算参照，未续训", "continued_in_this_experiment": False,
            "matched_update_budget_comparison": False, "methods": results}


def costs(units):
    rows = []
    for p in units:
        step = p["total_joint_step"]
        row = {k: p[k] for k in ("unit_id", "context_id", "fold", "sample_seed", "n", "model_seed", "total_joint_step")}
        if step == 1000:
            row.update({k: 0.0 for k in COST_FIELDS})
            row["inherited_stage2_fit_predict_seconds"] = p["cost"]["standalone_fit_predict_seconds"]
        else:
            row.update({k: base.finite(p["cost"][k], k) for k in COST_FIELDS})
            require(all(row[k] >= 0 for k in COST_FIELDS), "Negative continuation cost")
            require(math.isclose(row["new_fit_predict_seconds"], row["additional_training_seconds"] + row["diagnostic_prediction_seconds"],
                                 abs_tol=1e-6, rel_tol=1e-6), "New fit/predict cost must exclude separately recorded verification")
        row["inherited_cost_not_charged_to_new_budget"] = True
        row["new_budget_seconds_at_checkpoint"] = row["new_fit_predict_seconds"] + row["verification_seconds"]
        rows.append(row)
    latest = {}
    for row in rows:
        key = row["context_id"]
        if key not in latest or row["total_joint_step"] > latest[key]["total_joint_step"]:
            latest[key] = row
    summary = {"unit": "seconds", "checkpoint_costs_are_cumulative_not_additive": True,
               "inherited_stage2_cost_excluded": True,
               "completed_endpoint_new_budget_seconds_latest_per_context": sum(r["new_budget_seconds_at_checkpoint"] for r in latest.values()),
               "resource_ledger_authoritative_for_global_synthetic_verification_incomplete_steps_failures": True}
    return rows, summary


def run(root, output_dir=None, labels_path=None, manifest_path=None):
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "scoring"
    labels_path = Path(labels_path).resolve() if labels_path else root / "scoring/labels_A.json"
    manifest_path = Path(manifest_path).resolve() if manifest_path else root / "predictions_manifest.json"
    seal, sources, contexts, training, units, controls, complete, missing = load_seal(root, manifest_path)
    # The only full-A label read happens after every new/source prediction and
    # state has passed its seal checks. This module never imports a trainer.
    required_ids = {cid for p in units + controls for cid in p["candidate_ids"]}
    require(base.sha256(labels_path) == sources["scoring_labels_expected_sha256"], "Original A scoring-label byte identity changed")
    release = base.read_json(root / "scoring/release.json")
    require(release["prediction_seal_sha256"] == base.sha256(manifest_path)
            and release["labels_A_sha256"] == base.sha256(labels_path), "New scoring release identity mismatch")
    require(datetime.fromisoformat(release["released_at_utc"]) >= datetime.fromisoformat(seal["sealed_at_utc"]),
            "New scoring labels must be released after the new prediction seal")
    require(release.get("only_A") is True and release.get("new_updates_after_release") is False,
            "Invalid A-only postseal release declaration")
    labels = base.checked_labels(labels_path, required_ids)
    rows = [score_endpoint(p, training[p["context_id"]], labels) for p in units]
    training_by_subset = {c["subset_id"]: training[c["context_id"]] for c in contexts.values()}
    control_rows = [score_endpoint(p, training_by_subset[p["subset_id"]], labels) for p in controls]
    summary, tables = summarize(rows, complete)
    cost_rows, cost_summary = costs(units)
    summary["missing_endpoint_identities"] = missing
    summary["fixed_controls"] = fixed_control_summary(control_rows)
    provenance = {"scored_at_utc": datetime.now(timezone.utc).isoformat(), "scorer_sha256": base.sha256(__file__),
                  "stage2_formula_module_sha256": base.sha256(Path(base.__file__)),
                  "predictions_manifest_sha256": base.sha256(manifest_path), "labels_A_sha256": base.sha256(labels_path),
                  "source_manifest_sha256": seal["source_manifest_sha256"], "sealed_at_utc": seal.get("sealed_at_utc"),
                  "new_scoring_release_sha256": base.sha256(root / "scoring/release.json"),
                  "all_new_and_source_predictions_verified_before_label_read": True,
                  "training_triggered": False, "next_stage_authorized": False}
    summary["provenance"] = provenance
    # Random regret is once per fold; random gain is once per training subset.
    random_folds, random_subsets = {}, {}
    for row in rows:
        random_folds[row["fold"]] = {"fold": row["fold"], "query_size": row["query_size"], "expected_regret5_pp": row["random_expected_regret5_pp"]}
        random_subsets[row["subset_id"]] = {k: row[k] for k in ("subset_id", "fold", "sample_seed", "n", "training_best_raw", "query_gain_upper_bound_pp", "random_expected_gain5_pp")}
    output.mkdir(parents=True, exist_ok=True)
    tables.update({"unit_metrics.csv": rows, "fixed_controls.csv": control_rows, "cost_table.csv": cost_rows})
    calibration_fields = ("unit_id", "context_id", "fold", "sample_seed", "n", "model_seed", "total_joint_step",
                          "coverage90_fraction", "coverage95_fraction", "interval90_width_pp", "interval95_width_pp",
                          "nlpd_accuracy_units", "mean_latent_sd_pp", "mean_observed_sd_pp", "negative_latent_roundoff_count")
    tables["calibration.csv"] = [{k: row[k] for k in calibration_fields} for row in rows]
    for name, content in tables.items():
        base.write_csv(output / name, content)
    base.write_json(output / "summary.json", summary)
    base.write_json(output / "random_reference.json", {"by_fold": list(random_folds.values()), "by_subset": list(random_subsets.values()),
                    "references_are_reused_not_independent_repeats": True,
                    "gain_formula": "exact expected maximum of max(0,query_y-best_training) under uniform five without replacement"})
    base.write_json(output / "fixed_controls_summary.json", summary["fixed_controls"])
    base.write_json(output / "cost_summary.json", cost_summary)
    base.write_json(output / "score_provenance.json", provenance)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--labels")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    summary = run(args.root, args.output_dir, args.labels, args.manifest)
    print(json.dumps({k: summary[k] for k in ("execution_status", "scientific_endpoint_count", "new_endpoint_count", "next_stage_authorized")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
