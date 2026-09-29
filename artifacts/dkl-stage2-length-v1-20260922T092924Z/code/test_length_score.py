"""Synthetic-only length-score tests. No training, network, or real A-label reads."""
import copy
import itertools
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import score_length as score


def prediction(context, method="DKL", step=1000, query_ids=None, architectures=None, scaler=None):
    context_id = context["context_id"]
    ids = query_ids or [f"q{i}" for i in range(7)]
    scaler = scaler or {"y_mean": 0.7, "y_scale": 0.1}
    center, scale = scaler["y_mean"], scaler["y_scale"]
    raw = [0.7 + i / 100 for i in range(len(ids))]
    unit_id = context_id + ("_DKL" if step == 1000 else f"_j{step}")
    if method != "DKL":
        unit_id = context_id + "_" + method
    p = {"unit_id": unit_id, "context_id": context_id, "subset_id": context["subset_id"],
         "fold": context["fold"], "sample_seed": context["sample_seed"], "n": context["n"],
         "model_seed": None if method == "RAW16" else context["model_seed"], "method": method,
         "candidate_ids": ids, "architecture_fingerprints": architectures or ["qa" + str(i // 2) for i in range(len(ids))],
         "mean_raw": raw, "mean_standardized": [(x - center) / scale for x in raw],
         "latent_variance_standardized": [0.2] * len(ids), "latent_variance_raw": [scale ** 2 * 0.2] * len(ids),
         "y_center": center, "y_scale": scale, "extra_noise_variance_standardized": 0.01,
         "endpoint_state_hash": "a" * 64, "training_ids_sha256": context.get("training_ids_sha256", "b" * 64),
         "training_tensor_sha256": context.get("training_tensor_sha256", "c" * 64),
         "state_path": "states/" + unit_id + ".pt", "state_file_sha256": "d" * 64,
         "cost": {"common_training_seconds": 1.0, "branch_training_seconds": 2.0,
                  "prediction_seconds": 0.1, "standalone_fit_predict_seconds": 3.1}}
    if step != 1000:
        k = (step - 1000) / 1000
        p["total_joint_step"] = step
        p["cost"] = {"additional_training_seconds": 2 * k, "diagnostic_prediction_seconds": 0.1 * k,
                     "verification_seconds": 0.01, "new_fit_predict_seconds": 2.1 * k,
                     "inherited_stage2_fit_predict_seconds": 3.1}
    return p


def simple_context(seed=0):
    return {"context_id": f"f0_s20260922_n20_m{seed}", "subset_id": "f0_s20260922_n20",
            "fold": 0, "sample_seed": 20260922, "n": 20, "model_seed": seed}


def synthetic_labels(ids):
    result = {}
    for i, cid in enumerate(ids):
        y = 0.65 + (i % 30) / 100
        seeds = [y - 0.003, y, y + 0.003]
        result[cid] = {"validation_mean": statistics.fmean(seeds),
                       "validation_se": statistics.stdev(seeds) / math.sqrt(3),
                       "validation_seeds": seeds, "training_seeds": [0, 1, 2]}
    return result


def fixture(root, partial=False):
    root = Path(root)
    source_files, contexts, entries = {}, [], []
    ids_by_fold = [[f"f{fold}c{i:03d}" for i in range(count)] for fold, count in enumerate([38, 36, 36, 34, 35])]
    all_ids = [cid for group in ids_by_fold for cid in group]
    labels = synthetic_labels(all_ids)
    def frozen_json(relative, value):
        path = root / relative
        score.base.write_json(path, value)
        source_files[relative] = score.base.sha256(path)
    def frozen_state(relative, text):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        source_files[relative] = score.base.sha256(path)
    for fold, sample, n, seed in sorted(score.expected_trajectories()):
        ctx = {"context_id": f"f{fold}_s{sample}_n{n}_m{seed}", "subset_id": f"f{fold}_s{sample}_n{n}",
               "fold": fold, "sample_seed": sample, "n": n, "model_seed": seed}
        pool = [cid for cid in all_ids if cid not in ids_by_fold[fold]]
        start = 0 if sample == 20260922 else 1
        train_ids = pool[start:start + n]
        y = [labels[c]["validation_mean"] for c in train_ids]
        scaler = {"y_mean": statistics.fmean(y), "y_scale": max(statistics.pstdev(y), 1e-8), "fit_ids": train_ids}
        t = {"ids": train_ids, "y": y, "scaler": scaler}
        train_path = f"source/training/{ctx['subset_id']}.json"
        frozen_json(train_path, t)
        ctx.update(training_path=train_path, training_sha256=source_files[train_path],
                   training_ids_sha256=score.digest(train_ids), training_tensor_sha256="c" * 64,
                   scaler=scaler, scaler_sha256=score.digest(scaler),
                   source_scheduler={"mode": "joint", "planned_steps": 1000, "completed": 1000})
        query_ids = ids_by_fold[fold]
        arch = [f"f{fold}a{i % (20 if fold == 0 else 19)}" for i in range(len(query_ids))]
        baseline = prediction(ctx, query_ids=query_ids, architectures=arch, scaler=scaler)
        path = f"source/predictions/{ctx['context_id']}_DKL.json"
        frozen_json(path, baseline)
        checkpoint = f"source/checkpoints/{ctx['context_id']}_DKL_joint.pt"
        frozen_state(checkpoint, "SYNTHETIC ADAM CHECKPOINT " + ctx["context_id"])
        ctx.update(prediction_path=path, prediction_sha256=source_files[path],
                   checkpoint_path=checkpoint, checkpoint_sha256=source_files[checkpoint], model_state_hash=baseline["endpoint_state_hash"])
        contexts.append(ctx)
        entries.append({"unit_id": ctx["context_id"] + "_j1000", "context_id": ctx["context_id"], "total_joint_step": 1000,
                        "path": path, "sha256": source_files[path], "state_path": checkpoint,
                        "state_file_sha256": source_files[checkpoint], "endpoint_state_hash": baseline["endpoint_state_hash"]})
        for step in (2000, 4000):
            p = prediction(ctx, step=step, query_ids=query_ids, architectures=arch, scaler=scaler)
            state = root / p["state_path"]
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text("SYNTHETIC NEW STATE " + p["unit_id"])
            p["state_file_sha256"] = score.base.sha256(state)
            relative = "predictions/" + p["unit_id"] + ".json"
            score.base.write_json(root / relative, p)
            entries.append({"unit_id": p["unit_id"], "context_id": ctx["context_id"], "total_joint_step": step,
                            "path": relative, "sha256": score.base.sha256(root / relative), "state_path": p["state_path"],
                            "state_file_sha256": p["state_file_sha256"], "endpoint_state_hash": p["endpoint_state_hash"]})
        for method in ("FROZEN", "REG", "RAW16"):
            if method == "RAW16" and seed != 0:
                continue
            p = prediction(ctx, method=method, query_ids=query_ids, architectures=arch, scaler=scaler)
            frozen_json("source/controls/predictions/" + p["unit_id"] + ".json", p)
    score.base.write_json(root / "scoring/labels_A.json", {"labels": labels, "metadata": {"synthetic_only": True}})
    sources = {"source_root": "synthetic://source", "source_protocol_sha256": score.SOURCE_PROTOCOL_SHA256,
               "source_package_sha256": score.SOURCE_PACKAGE_SHA256, "files": source_files, "contexts": contexts,
               "scoring_labels_expected_sha256": score.base.sha256(root / "scoring/labels_A.json")}
    score.base.write_json(root / "source_manifest.json", sources)
    score.base.write_json(root / "protocol.json", {"synthetic_only": True})
    score.base.write_json(root / "code_manifest.json", {"files": {}})
    if partial:
        entries.pop()
    seal = {"sealed": True, "execution_status": "PARTIAL" if partial else "COMPLETE_VALID", "files": entries,
            "expected_endpoint_count": 180, "new_endpoint_count_expected": 120,
            **{name + "_sha256": score.base.sha256(root / (name + ".json")) for name in ("protocol", "code_manifest", "source_manifest")},
            "sealed_at_utc": "2026-09-22T00:00:00+00:00"}
    score.base.write_json(root / "predictions_manifest.json", seal)
    update_release(root)
    return seal, sources, labels


def update_release(root):
    root = Path(root)
    score.base.write_json(root / "scoring/release.json", {
        "released_at_utc": "2026-09-22T00:00:01+00:00",
        "prediction_seal_sha256": score.base.sha256(root / "predictions_manifest.json"),
        "labels_A_sha256": score.base.sha256(root / "scoring/labels_A.json"),
        "only_A": True, "new_updates_after_release": False})


class ArithmeticTests(unittest.TestCase):
    def test_random_gain_matches_enumeration_with_ties_and_zero_gains(self):
        values = [0.1, 0.4, 0.4, 0.5, 0.6, 0.8, 0.9]
        for incumbent in (0.0, 0.4, 0.7, 0.95):
            expected = statistics.fmean(max(0, max(subset) - incumbent) for subset in itertools.combinations(values, 5))
            self.assertAlmostEqual(score.gain_reference(values, incumbent), expected, places=14)

    def test_gain_upper_bound_and_break_best_use_pp(self):
        p = prediction(simple_context())
        labels = synthetic_labels(p["candidate_ids"])
        labels["t0"] = {"validation_mean": 0.67}
        row = score.score_endpoint(p, {"ids": ["t0"], "y": [0.67]}, labels)
        self.assertAlmostEqual(row["gain5_pp"], 4.0)
        self.assertAlmostEqual(row["query_gain_upper_bound_pp"], 4.0)
        self.assertTrue(row["broke_training_best"])
        labels["t0"]["validation_mean"] = 0.9
        row = score.score_endpoint(p, {"ids": ["t0"], "y": [0.9]}, labels)
        self.assertEqual(row["gain5_pp"], 0)
        self.assertEqual(row["query_gain_upper_bound_pp"], 0)
        self.assertEqual(row["random_expected_gain5_pp"], 0)

    def test_paired_directions_and_top5_set_versus_order(self):
        p = prediction(simple_context())
        labels = synthetic_labels(p["candidate_ids"])
        labels["t"] = {"validation_mean": 0.66}
        row = score.score_endpoint(p, {"ids": ["t"], "y": [0.66]}, labels)
        a, b, c = copy.deepcopy(row), copy.deepcopy(row), copy.deepcopy(row)
        a.update(total_joint_step=1000, regret5_pp=3, gain5_pp=1)
        b.update(total_joint_step=2000, regret5_pp=2, gain5_pp=2)
        c.update(total_joint_step=4000, regret5_pp=2, gain5_pp=2)
        b["selected_top5_ids"] = list(reversed(a["selected_top5_ids"]))
        pairs = score.pair_rows([a, b, c])
        self.assertEqual(pairs[0]["regret5_improvement_pp"], 1)
        self.assertEqual(pairs[0]["gain5_improvement_pp"], 1)
        self.assertFalse(pairs[0]["top5_set_changed"])
        self.assertTrue(pairs[0]["top5_order_changed"])
        self.assertEqual(pairs[-1]["regret5_direction"], "tie")

    def test_three_level_aggregation_does_not_weight_partial_counts(self):
        rows = []
        for fold, sample, values in ((0, 20260922, [1, 3, 5]), (0, 20260923, [9]), (1, 20260922, [20])):
            for seed, value in enumerate(values):
                rows.append({"n": 20, "total_joint_step": 4000, "fold": fold, "sample_seed": sample,
                             "model_seed": seed, "regret5_pp": value})
        _, _, overall = score.hierarchy(rows, ("regret5_pp",))
        self.assertEqual(overall[0]["regret5_pp"], 13)

    def test_costs_latest_cumulative_only_and_inherited_excluded(self):
        units = [prediction(simple_context(), step=step) for step in score.STEPS]
        for p, step in zip(units, score.STEPS):
            p["total_joint_step"] = step
        _, summary = score.costs(units)
        self.assertAlmostEqual(summary["completed_endpoint_new_budget_seconds_latest_per_context"], 6.31)
        self.assertTrue(summary["inherited_stage2_cost_excluded"])


class SealTests(unittest.TestCase):
    def setUp(self):
        ids = [f"f{f}c{i:03d}" for f, count in enumerate([38, 36, 36, 34, 35]) for i in range(count)]
        patcher = mock.patch.object(score.base, "A_ID_HASH", score.base.a_id_digest(ids))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_complete_matrix_references_old_checkpoint_not_old_state_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            result = score.run(tmp)
            self.assertEqual(result["execution_status"], "COMPLETE_VALID")
            self.assertEqual(result["scientific_endpoint_count"], 180)
            self.assertEqual(result["new_endpoint_count"], 120)
            self.assertFalse(result["next_stage_authorized"])
            self.assertEqual(result["per_n"]["20"]["comparisons"]["1000_to_4000"]["matched_trajectory_count"], 30)
            self.assertEqual(result["fixed_controls"]["methods"]["RAW16"]["20"]["endpoint_count"], 10)

    def test_partial_matrix_keeps_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp, partial=True)
            result = score.run(tmp)
            self.assertEqual(result["execution_status"], "PARTIAL")
            self.assertEqual(result["scientific_endpoint_count"], 179)
            self.assertEqual(result["new_endpoint_count"], 119)

    def test_partial_budget_stop_may_have_unvisited_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            seal, _, _ = fixture(tmp)
            context_id = seal["files"][0]["context_id"]
            seal["files"] = [entry for entry in seal["files"] if entry["context_id"] == context_id]
            seal["execution_status"] = "PARTIAL"
            score.base.write_json(Path(tmp) / "predictions_manifest.json", seal)
            update_release(tmp)
            result = score.run(tmp)
            self.assertEqual(result["execution_status"], "PARTIAL")
            self.assertEqual(result["scientific_endpoint_count"], 3)
            self.assertEqual(len(result["missing_endpoint_identities"]), 177)

    def test_old_label_metadata_does_not_replace_new_release_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            release_path = Path(tmp) / "scoring/release.json"
            release = score.base.read_json(release_path)
            release["released_at_utc"] = "2026-09-21T23:59:59+00:00"
            score.base.write_json(release_path, release)
            with mock.patch.object(score.base, "checked_labels") as full_labels:
                with self.assertRaisesRegex(ValueError, "after the new prediction seal"):
                    score.run(tmp)
                full_labels.assert_not_called()

    def test_unsealed_and_invalid_status_reject_before_label_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            seal, _, _ = fixture(tmp)
            for sealed, status in ((False, "COMPLETE_VALID"), (True, "INVALID"), (True, "BLOCKED")):
                seal.update(sealed=sealed, execution_status=status)
                score.base.write_json(Path(tmp) / "predictions_manifest.json", seal)
                with mock.patch.object(score.base, "checked_labels") as release:
                    with self.assertRaises(ValueError):
                        score.run(tmp)
                    release.assert_not_called()

    def test_tampered_new_prediction_rejects_before_label_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            seal, _, _ = fixture(tmp)
            path = Path(tmp) / seal["files"][-1]["path"]
            path.write_text(path.read_text() + "\n")
            with mock.patch.object(score.base, "checked_labels") as release:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    score.run(tmp)
                release.assert_not_called()

    def test_unregistered_intermediate_checkpoint_cannot_be_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            seal, _, _ = fixture(tmp)
            seal["files"][-1]["total_joint_step"] = 3000
            score.base.write_json(Path(tmp) / "predictions_manifest.json", seal)
            with mock.patch.object(score.base, "checked_labels") as release:
                with self.assertRaisesRegex(ValueError, "unregistered"):
                    score.run(tmp)
                release.assert_not_called()

    def test_scoring_source_does_not_import_training_libraries(self):
        import ast
        code = ast.parse(Path(score.__file__).read_text())
        roots = {node.module.split(".")[0] for node in ast.walk(code) if isinstance(node, ast.ImportFrom) and node.module}
        roots |= {alias.name.split(".")[0] for node in ast.walk(code) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(roots & {"torch", "gpytorch", "numpy", "stage01_core", "stage2_core", "length_core", "run_length"})
        self.assertIn("score_stage2", roots)


if __name__ == "__main__":
    unittest.main(verbosity=2)
