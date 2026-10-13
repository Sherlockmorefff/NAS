#!/usr/bin/env python3
"""Independent post-seal audit; standard library only, no scorer/trainer imports.

Run only after the main process has sealed, released A labels, and scored:
    python audit/verify_length_scoring.py --root /absolute/artifact
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PP_TOLERANCE = 1e-9
LEGAL_A_HASH = "d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2"
STEPS = (1000, 2000, 4000)
INTERVALS = ((1000, 2000), (1000, 4000), (2000, 4000))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def average(values):
    values = [x for x in values if x is not None]
    return math.fsum(values) / len(values) if values else None


def close(actual, expected, label, absolute=1e-9):
    if actual is None or expected is None:
        require(actual is None and expected is None, label + ": undefined-value mismatch")
    else:
        require(math.isfinite(actual) and math.isfinite(expected), label + ": nonfinite value")
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=absolute),
                label + f": {actual!r} != {expected!r}")


def read(root, relative):
    return json.loads((root / relative).read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_file(root, relative, expected, cache):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Sealed path escapes artifact")
    if path not in cache:
        cache[path] = sha(path)
    require(cache[path] == expected, "Hash mismatch: " + relative)
    return path


def random_expectations(values, incumbent, k=5):
    """Independent CDF gap integration, rather than scorer order-statistic sum.

    E[max] = largest - sum(adjacent_gap * P(max <= lower_score)).
    The same identity applies to the nonnegative gain-transformed scores.
    """
    ordered = sorted(values)
    m = len(ordered)
    require(m >= k, "Too few query candidates for random5")
    denominator = math.comb(m, k)
    regret = math.fsum((ordered[i] - ordered[i - 1]) * math.comb(i, k) / denominator
                       for i in range(k, m))
    gains = sorted(max(0.0, value - incumbent) for value in values)
    gain_gap = math.fsum((gains[i] - gains[i - 1]) * math.comb(i, k) / denominator
                        for i in range(k, m))
    expected_gain = gains[-1] - gain_gap
    return 100 * regret, 100 * expected_gain


def rank_correlation(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    # Pairwise rank construction deliberately differs from the scorer's sort-
    # and-tie-block implementation.
    def ranks(values):
        return [1 + sum(v < a for v in values) + (sum(v == a for v in values) - 1) / 2 for a in values]
    rx, ry = ranks(x), ranks(y)
    mx, my = average(rx), average(ry)
    numerator = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(math.fsum((a - mx) ** 2 for a in rx) * math.fsum((b - my) ** 2 for b in ry))
    return numerator / denominator


def metrics(prediction, selected, labels):
    ids, predictions = prediction["candidate_ids"], prediction["mean_raw"]
    y = [labels[c]["validation_mean"] for c in ids]
    require(not set(ids) & set(selected["ids"]), "Selected training and query overlap")
    require(len(selected["ids"]) == len(selected["y"]) == prediction["n"], "Wrong training snapshot size")
    for cid, value in zip(selected["ids"], selected["y"]):
        close(value, labels[cid]["validation_mean"], "Frozen selected training label", absolute=1e-12)
    incumbent = max(selected["y"])
    order = sorted(range(len(ids)), key=lambda i: (-predictions[i], ids[i]))
    top5 = order[:5]
    chosen_best = min(top5, key=lambda i: (-y[i], ids[i]))
    best = max(y)
    errors = [mu - target for mu, target in zip(predictions, y)]
    variances = prediction["latent_variance_standardized"]
    require(min(variances) >= -1e-8, "Standardized latent variance outside allowed roundoff tolerance")
    extra, scale = prediction["extra_noise_variance_standardized"], prediction["y_scale"]
    observed = [scale ** 2 * (latent + extra) + labels[cid]["validation_se"] ** 2 for latent, cid in zip(variances, ids)]
    require(min(observed) > 0, "Observed variance is nonpositive")
    random_regret, random_gain = random_expectations(y, incumbent)
    gain = max(0.0, y[chosen_best] - incumbent)
    result = {
        "regret5_pp": 100 * (best - y[chosen_best]), "top1_regret_pp": 100 * (best - y[order[0]]),
        "mae_pp": 100 * average([abs(error) for error in errors]),
        "rmse_pp": 100 * math.sqrt(average([error * error for error in errors])),
        "spearman": rank_correlation(predictions, y),
        "gain5_pp": 100 * gain, "query_gain_upper_bound_pp": 100 * max(0.0, best - incumbent),
        "random_expected_regret5_pp": random_regret, "random_expected_gain5_pp": random_gain,
        "mean_latent_sd_pp": 100 * scale * average([math.sqrt(max(0, v)) for v in variances]),
        "mean_observed_sd_pp": 100 * average([math.sqrt(v) for v in observed]),
        "nlpd_accuracy_units": average([math.log(math.sqrt(2 * math.pi * v)) + 0.5 * (error / math.sqrt(v)) ** 2
                                        for error, v in zip(errors, observed)]),
    }
    for level in (90, 95):
        quantile = statistics.NormalDist().inv_cdf((1 + level / 100) / 2)
        halfwidths = [quantile * math.sqrt(v) for v in observed]
        result[f"coverage{level}_fraction"] = sum(abs(e) <= width for e, width in zip(errors, halfwidths)) / len(errors)
        result[f"interval{level}_width_pp"] = 200 * average(halfwidths)
    metadata = {"selected_top5_ids": [ids[i] for i in top5], "selected_best_id": ids[chosen_best],
                "selected_top1_id": ids[order[0]], "training_best_raw": incumbent,
                "broke_training_best": 100 * gain > PP_TOLERANCE,
                "query_has_improvement_room": 100 * max(0.0, best - incumbent) > PP_TOLERANCE,
                "negative_latent_roundoff_count": sum(v < 0 for v in variances)}
    return result, metadata


def directions(values):
    return {"improved": sum(v > PP_TOLERANCE for v in values), "tied": sum(abs(v) <= PP_TOLERANCE for v in values),
            "worse": sum(v < -PP_TOLERANCE for v in values)}


def fold_mean(rows, metric, fold):
    sample_means = []
    for sample in (20260922, 20260923):
        sample_means.append(average([r["metrics"][metric] for r in rows if r["fold"] == fold and r["sample_seed"] == sample]))
    return average(sample_means)


def audit(root):
    root = Path(root).resolve()
    seal = read(root, "predictions_manifest.json")
    require(seal.get("sealed") is True and seal.get("execution_status") in ("COMPLETE_VALID", "PARTIAL"),
            "Valid uniform new prediction seal required")
    require(seal["expected_endpoint_count"] == 180 and seal["new_endpoint_count_expected"] == 120, "Wrong planned matrix")
    require((root / "scoring/summary.json").exists(), "Main scoring must finish before independent review")
    cache = {}
    for name in ("source_manifest", "protocol", "code_manifest"):
        frozen_file(root, name + ".json", seal[name + "_sha256"], cache)
    sources = read(root, "source_manifest.json")
    contexts = {c["context_id"]: c for c in sources["contexts"]}
    require(len(contexts) == 60, "Expected original 60 trajectory contexts")
    predictions = []
    for entry in seal["files"]:
        require(entry["total_joint_step"] in STEPS, "Unregistered checkpoint scoring")
        path = frozen_file(root, entry["path"], entry["sha256"], cache)
        frozen_file(root, entry["state_path"], entry["state_file_sha256"], cache)
        prediction = json.loads(path.read_text())
        require(prediction["method"] == "DKL" and prediction["context_id"] == entry["context_id"], "Wrong method/context")
        require(prediction["endpoint_state_hash"] == entry["endpoint_state_hash"], "Mathematical state identity mismatch")
        ctx = contexts[entry["context_id"]]
        if entry["total_joint_step"] == 1000:
            require(entry["sha256"] == ctx["prediction_sha256"] and entry["state_file_sha256"] == ctx["checkpoint_sha256"],
                    "1000 baseline must retain original prediction and resumable checkpoint identity")
        else:
            require(prediction["total_joint_step"] == entry["total_joint_step"] and prediction["unit_id"] == entry["unit_id"],
                    "New endpoint identity mismatch")
        predictions.append((entry, prediction))
    require(len({(e["context_id"], e["total_joint_step"]) for e, _ in predictions}) == len(predictions), "Duplicate trajectory endpoint")
    if seal["execution_status"] == "COMPLETE_VALID":
        require(len(predictions) == 180 and len({e["context_id"] for e, _ in predictions}) == 60, "Incomplete valid matrix")
    # All label-free identity checks precede the one full scoring-label read.
    release = read(root, "scoring/release.json")
    require(release["prediction_seal_sha256"] == sha(root / "predictions_manifest.json"), "Wrong release seal hash")
    label_hash = sha(root / "scoring/labels_A.json")
    require(label_hash == sources["scoring_labels_expected_sha256"] == release["labels_A_sha256"], "Wrong A-label bytes")
    require(datetime.datetime.fromisoformat(release["released_at_utc"]) >= datetime.datetime.fromisoformat(seal["sealed_at_utc"]),
            "A-label release must follow new seal")
    labels = read(root, "scoring/labels_A.json")["labels"]
    require(len(labels) == 179 and hashlib.sha256(json.dumps(sorted(labels), sort_keys=True, separators=(",", ":")).encode()).hexdigest() == LEGAL_A_HASH,
            "Wrong legal A ID universe")
    for value in labels.values():
        require(set(value) == {"validation_mean", "validation_se", "validation_seeds", "training_seeds"}, "Unexpected A-label fields")
        require(value["training_seeds"] == [0, 1, 2], "Wrong validation seed set")
        close(value["validation_mean"], average(value["validation_seeds"]), "Three-seed mean", absolute=1e-12)
        close(value["validation_se"], statistics.stdev(value["validation_seeds"]) / math.sqrt(3), "Three-seed SE", absolute=1e-12)
    with (root / "scoring/unit_metrics.csv").open() as stream:
        official_units = {r["unit_id"]: r for r in csv.DictReader(stream)}
    with (root / "scoring/trajectory_pairs.csv").open() as stream:
        official_pairs = {(r["context_id"], r["comparison"]): r for r in csv.DictReader(stream)}
    summary = read(root, "scoring/summary.json")
    require(summary["execution_status"] == seal["execution_status"], "Summary status disagrees with seal")
    require(summary["source_stage2_classifications_unchanged"] is True and summary["next_stage_authorized"] is False,
            "Historical conclusion or authorization boundary changed")
    require(summary["fixed_controls"]["continued_in_this_experiment"] is False
            and summary["fixed_controls"]["matched_update_budget_comparison"] is False,
            "Fixed controls must not be presented as continued or budget matched")
    require(len(official_units) == len(predictions), "CSV endpoint count mismatch")
    rows, maximum_errors = [], defaultdict(float)
    for entry, prediction in predictions:
        ctx = contexts[entry["context_id"]]
        path = frozen_file(root, ctx["training_path"], ctx["training_sha256"], cache)
        selected = json.loads(path.read_text())
        computed, extra = metrics(prediction, selected, labels)
        official = official_units[entry["unit_id"]]
        for field in ("selected_best_id", "selected_top1_id"):
            require(extra[field] == official[field], "Selected candidate ID mismatch")
        require(extra["selected_top5_ids"] == json.loads(official["selected_top5_ids"]), "Selected top5 ID/order mismatch")
        require(extra["broke_training_best"] == (official["broke_training_best"] == "True"), "Break-known-best flag mismatch")
        close(extra["training_best_raw"], float(official["training_best_raw"]), "Incumbent training best")
        for metric, value in computed.items():
            target = float(official[metric]) if official[metric] else None
            close(value, target, metric)
            if value is not None:
                maximum_errors[metric] = max(maximum_errors[metric], abs(value - target))
        rows.append({**{k: prediction[k] for k in ("context_id", "fold", "sample_seed", "n", "model_seed")},
                     "step": entry["total_joint_step"], "metrics": computed, **extra})
    lookup = {(r["context_id"], r["step"]): r for r in rows}
    per_n = {}
    for n in (20, 50):
        checkpoints, comparisons = {}, {}
        for step in STEPS:
            part = [r for r in rows if r["n"] == n and r["step"] == step]
            target = summary["per_n"][str(n)]["checkpoints"][str(step)]
            require(len(part) == target["trajectory_count"], "Checkpoint trajectory count mismatch")
            aggregated = {}
            for metric in next(iter(rows))["metrics"] if rows else ():
                fvalues = [fold_mean(part, metric, fold) for fold in range(5)]
                aggregated[metric] = average(fvalues)
                close(aggregated[metric], target["metrics"][metric]["mean"], "Three-level mean: " + metric)
            break_count = sum(r["broke_training_best"] for r in part)
            room_count = sum(r["query_has_improvement_room"] for r in part)
            require(break_count == target["broke_training_best_trajectories"], "Break-best count mismatch")
            require(room_count == target["query_has_improvement_room_trajectories"], "Query opportunity count mismatch")
            checkpoints[str(step)] = {"trajectory_count": len(part), "metrics": aggregated,
                                      "broke_training_best_trajectories": break_count,
                                      "query_has_improvement_room_trajectories": room_count}
        for early, late in INTERVALS:
            name = f"{early}_to_{late}"
            pairs = []
            for cid in contexts:
                if (cid, early) not in lookup or (cid, late) not in lookup:
                    continue
                a, b = lookup[(cid, early)], lookup[(cid, late)]
                if a["n"] != n:
                    continue
                difference = {"regret5_improvement_pp": a["metrics"]["regret5_pp"] - b["metrics"]["regret5_pp"],
                              "gain5_improvement_pp": b["metrics"]["gain5_pp"] - a["metrics"]["gain5_pp"]}
                official = official_pairs[(cid, name)]
                for metric, value in difference.items():
                    close(value, float(official[metric]), "Paired direction: " + metric)
                pairs.append({"fold": a["fold"], "sample_seed": a["sample_seed"], "metrics": difference,
                              "top5_set_changed": set(a["selected_top5_ids"]) != set(b["selected_top5_ids"]),
                              "top5_order_changed": a["selected_top5_ids"] != b["selected_top5_ids"],
                              "top1_changed": a["selected_top1_id"] != b["selected_top1_id"]})
            target = summary["per_n"][str(n)]["comparisons"][name]
            require(len(pairs) == target["matched_trajectory_count"], "Matched trajectory denominator mismatch")
            result = {"matched_trajectory_count": len(pairs)}
            for metric, short in (("regret5_improvement_pp", "regret5"), ("gain5_improvement_pp", "gain5")):
                values = [r["metrics"][metric] for r in pairs]
                counts = directions(values)
                require(counts == target[f"trajectory_{short}_directions"], "Trajectory direction count mismatch")
                folds = [fold_mean(pairs, metric, f) for f in range(5)]
                defined_folds = [x for x in folds if x is not None]
                close(average(folds), target["metrics"][metric]["mean"], "Paired fold mean")
                require(directions(defined_folds) == target[f"fold_{short}_directions"], "Fold direction counts mismatch")
                for f, value in enumerate(folds):
                    if value is not None:
                        close(value, target[f"fold_{short}_improvement_pp"][str(f)], "Fold paired difference")
                result[metric] = {"mean": average(folds), "fold_values": folds, "trajectory_directions": counts}
            for field in ("top5_set_changed", "top5_order_changed", "top1_changed"):
                count = sum(r[field] for r in pairs)
                require(count == target[field + "_trajectories"], "Selection change count mismatch")
                result[field + "_trajectories"] = count
            comparisons[name] = result
        per_n[str(n)] = {"checkpoints": checkpoints, "comparisons": comparisons}
    return {"status": "PASS", "execution_status": seal["execution_status"],
            "reviewed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "review_method": "independent standard-library implementation; no scorer or training module imported",
            "prediction_seal_sha256": sha(root / "predictions_manifest.json"), "labels_A_sha256": label_hash,
            "independent_review_script_sha256": sha(Path(__file__)), "endpoint_count": len(rows),
            "new_endpoint_count": sum(r["step"] != 1000 for r in rows),
            "all_prediction_and_corresponding_state_hashes_verified_before_label_read": True,
            "query_selection_and_incumbent_gain_checked_for_every_endpoint": True,
            "random_reference_check": "exact CDF gap-integration for regret and positive-part gain; no Monte Carlo",
            "maximum_unit_metric_absolute_differences": dict(maximum_errors), "per_n": per_n,
            "source_stage2_classifications_unchanged": True,
            "fixed_controls_not_claimed_to_have_matching_update_budget": True,
            "training_or_prediction_executed": False, "B_or_test_read": False,
            "next_stage_authorized": False, "findings": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    output = args.output or args.root / "audit/final_scoring_review.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "endpoint_count": result["endpoint_count"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
