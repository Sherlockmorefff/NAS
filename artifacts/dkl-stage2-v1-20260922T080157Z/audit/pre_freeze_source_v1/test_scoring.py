"""Synthetic-only checks of Stage 2 scoring; never load experiment labels."""
import itertools
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import score_stage2 as scorer


def unit(fold=0, sample_seed=20260922, n=20, seed=0, method="DKL", count=7, architectures=4):
    model_seed = None if method == "RAW16" else seed
    context = f"f{fold}_s{sample_seed}_n{n}_m{model_seed}"
    return {
        "unit_id": context + "_" + method, "context_id": context,
        "subset_id": f"f{fold}_s{sample_seed}_n{n}", "fold": fold,
        "sample_seed": sample_seed, "n": n, "model_seed": model_seed, "method": method,
        "candidate_ids": [f"f{fold}c{j:03d}" for j in range(count)],
        "architecture_fingerprints": [f"f{fold}a{j % architectures:02d}" for j in range(count)],
        "mean_standardized": [j / 10 for j in range(count)],
        "mean_raw": [0.7 + 0.1 * j / 10 for j in range(count)],
        "latent_variance_standardized": [0.2] * count,
        "latent_variance_raw": [0.002] * count,
        "y_center": 0.7, "y_scale": 0.1, "extra_noise_variance_standardized": 0.01,
        "endpoint_state_hash": "a" * 64, "training_ids_sha256": "b" * 64,
        "prediction_units": {"mean_raw": "accuracy", "latent_variance_raw": "accuracy_squared"},
        "cost": {"common_training_seconds": 1.0 if method != "RAW16" else 0.0,
                 "branch_training_seconds": 2.0, "prediction_seconds": 0.1,
                 "standalone_fit_predict_seconds": 3.1 if method != "RAW16" else 2.1},
    }


def labels_for(prediction, constant=False):
    labels = {}
    for j, cid in enumerate(prediction["candidate_ids"]):
        y = 0.7 if constant else 0.7 + (j % 19) / 100
        values = [y - 0.01, y, y + 0.01]
        labels[cid] = {"validation_mean": statistics.fmean(values),
                       "validation_se": statistics.stdev(values) / math.sqrt(3),
                       "validation_seeds": values, "training_seeds": [0, 1, 2]}
    return labels


def fixture(root, complete=True):
    root = Path(root)
    (root / "predictions").mkdir()
    (root / "scoring").mkdir()
    scorer.write_json(root / "protocol.json", {"synthetic_only": True})
    scorer.write_json(root / "code_manifest.json", {"synthetic_only": True})
    predictions, labels, entries = [], {}, []
    for fold, sample_seed, n, seed, method in sorted(scorer.expected_matrix(), key=str):
        p = unit(fold, sample_seed, n, seed, method,
                 count=[38, 36, 36, 34, 35][fold], architectures=20 if fold == 0 else 19)
        predictions.append(p)
        labels.update(labels_for(p))
        path = root / "predictions" / (p["unit_id"] + ".json")
        scorer.write_json(path, p)
        entries.append({"unit_id": p["unit_id"], "path": str(path.relative_to(root)), "sha256": scorer.sha256(path)})
    if not complete:
        entries.pop()
    manifest = {
        "sealed": True, "phase2_fit_status": "COMPLETE" if complete else "PARTIAL",
        "expected_endpoint_count": 200, "files": entries,
        "protocol_sha256": scorer.sha256(root / "protocol.json"),
        "code_manifest_sha256": scorer.sha256(root / "code_manifest.json"),
        "sealed_at_utc": "2026-09-22T00:00:00+00:00",
    }
    scorer.write_json(root / "predictions_manifest.json", manifest)
    scorer.write_json(root / "scoring" / "labels_A.json", {"labels": labels, "metadata": {"synthetic_only": True}})
    return manifest, predictions, labels


class MetricsTests(unittest.TestCase):
    def test_random5_matches_all_combinations_including_ties(self):
        for y in ([0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.95], [0.2, 0.2, 0.2, 0.5, 0.7, 0.7, 0.8]):
            expected = statistics.fmean(max(group) for group in itertools.combinations(y, 5))
            result = scorer.exact_random5(y)
            self.assertAlmostEqual(result["expected_best5_raw"], expected, places=14)
            self.assertAlmostEqual(result["expected_regret5_pp"], 100 * (max(y) - expected), places=12)

    def test_random5_chooses_entire_five_candidate_query(self):
        self.assertEqual(scorer.exact_random5([0.1, 0.2, 0.3, 0.4, 0.5])["expected_regret5_raw"], 0.0)

    def test_prediction_ranking_and_regret_direction(self):
        p = unit()
        labels = labels_for(p)
        p["candidate_ids"] = list(reversed(p["candidate_ids"]))
        p["mean_raw"] = [0.5] * 7
        result = scorer.score_unit(p, labels)
        self.assertEqual(result["selected_top5_ids"], sorted(p["candidate_ids"])[:5])
        self.assertEqual(result["selected_best_id"], "f0c004")
        self.assertAlmostEqual(result["regret5_pp"], 2.0)
        self.assertAlmostEqual(result["top1_regret_pp"], 6.0)
        self.assertIsNone(result["spearman"])
        self.assertEqual(result["spearman_undefined_reason"], "constant_gp_prediction")

    def test_spearman_tied_ranks_and_constant_truth(self):
        self.assertAlmostEqual(scorer.spearman([1, 2, 2, 4], [4, 3, 3, 1])[0], -1)
        self.assertEqual(scorer.spearman([1, 2, 3], [4, 4, 4]), (None, "constant_query_validation_mean"))

    def test_variance_composition_uses_extra_as_variance_and_se_once(self):
        value = scorer.observed_variance(0.25, 0.04, 0.2, 0.03)
        self.assertAlmostEqual(value, 0.2 ** 2 * (0.25 + 0.04) + 0.03 ** 2, places=15)
        self.assertNotAlmostEqual(value, 0.2 ** 2 * (0.25 + 0.04 ** 2) + 0.03 ** 2 / 3)

    def test_negative_variance_tolerance_is_standardized_only(self):
        p = unit()
        p["y_scale"] = 1e4
        p["mean_raw"] = [p["y_center"] + 1e4 * v for v in p["mean_standardized"]]
        p["latent_variance_standardized"] = [-1e-9] * 7
        p["latent_variance_raw"] = [-0.1] * 7
        scorer.validate_prediction(p)
        result = scorer.score_unit(p, labels_for(p))
        self.assertEqual(result["negative_latent_roundoff_count"], 7)
        self.assertEqual(result["mean_latent_sd_pp"], 0.0)
        self.assertEqual(p["latent_variance_standardized"], [-1e-9] * 7)
        p["latent_variance_standardized"][0] = -1.01e-8
        with self.assertRaisesRegex(ValueError, "Numerical failure"):
            scorer.validate_prediction(p)

    def test_interval_width_and_nlpd_use_same_observed_variance(self):
        p = unit()
        p["mean_raw"] = [0.7] * 7
        labels = labels_for(p, constant=True)
        result = scorer.score_unit(p, labels)
        se = labels[p["candidate_ids"][0]]["validation_se"]
        variance = 0.1 ** 2 * (0.2 + 0.01) + se ** 2
        self.assertEqual(result["coverage90_fraction"], 1.0)
        self.assertEqual(result["coverage95_fraction"], 1.0)
        self.assertAlmostEqual(result["interval90_width_pp"], 100 * 2 * 1.6448536269514722 * math.sqrt(variance))
        self.assertAlmostEqual(result["interval95_width_pp"], 100 * 2 * 1.959963984540054 * math.sqrt(variance))
        self.assertAlmostEqual(result["nlpd_accuracy_units"], 0.5 * math.log(2 * math.pi * variance))

    def test_classification_is_exclusive_and_requires_complete(self):
        self.assertEqual(scorer.classification([1, 1, 1, 1, -0.5], 1, 2, 3, True), "较一致的离线选优信号")
        self.assertEqual(scorer.classification([1, 1, 1, 1, -0.5], 1, 0.5, 3, True), "局部或混合信号")
        self.assertEqual(scorer.classification([0, 0, 0, -1, -1], 1, 2, 3, True), "未见相对继续回归的收益")
        self.assertEqual(scorer.classification([1e-10] * 5, 1, 2, 3, True), "未见相对继续回归的收益")
        self.assertEqual(scorer.classification([], None, None, None, False), "不完整，暂不分类")

    def test_hierarchy_equal_samples_then_equal_folds_even_when_partial(self):
        rows = []
        # Fold 0 sample means are 3 and 9 -> 6; fold 1 mean is 20 -> final 13.
        for fold, sample, values in ((0, 20260922, [1, 3, 5]), (0, 20260923, [9]), (1, 20260922, [20])):
            for seed, value in enumerate(values):
                p = unit(fold=fold, sample_seed=sample, seed=seed)
                r = scorer.score_unit(p, labels_for(p))
                for metric in scorer.METRICS:
                    r[metric] = value
                rows.append(r)
        summary, *_ = scorer.summarize(rows, [], False)
        self.assertEqual(summary["per_n"]["20"]["methods"]["DKL"]["metrics"]["regret5_pp"]["mean"], 13)
        self.assertEqual(summary["scientific_signal_n20"], "不完整，暂不分类")

    def test_pairing_direction_and_raw_reuse_not_independent_repeats(self):
        rows = []
        for method in scorer.METHODS:
            for seed in ([None] if method == "RAW16" else [0, 1, 2]):
                p = unit(method=method, seed=seed)
                row = scorer.score_unit(p, labels_for(p))
                row["regret5_pp"] = {"DKL": 1.0, "REG": 3.0, "FROZEN": 4.0, "RAW16": 5.0}[method]
                rows.append(row)
        summary, _, _, paired, folds = scorer.summarize(rows, [], False)
        self.assertEqual(summary["per_n"]["20"]["methods"]["RAW16"]["endpoint_count"], 1)
        self.assertEqual(summary["per_n"]["20"]["paired_regret5_improvement_pp"]["REG"]["mean"], 2.0)
        self.assertEqual(summary["per_n"]["20"]["paired_regret5_improvement_pp"]["RAW16"]["mean"], 4.0)
        raw_pairs = [r for r in paired if r["comparator"] == "RAW16"]
        self.assertEqual(len(raw_pairs), 3)
        self.assertEqual(len({r["reference_unit_id"] for r in raw_pairs}), 1)
        self.assertTrue(all(r["raw_reference_reused_for_pairing_only"] for r in raw_pairs))


class SealTests(unittest.TestCase):
    def setUp(self):
        # Never relax the production A-pool pin. Only the synthetic fixture's
        # identity is substituted inside each isolated test context.
        synthetic_ids = [f"f{fold}c{j:03d}" for fold, count in enumerate([38, 36, 36, 34, 35])
                         for j in range(count)]
        patcher = mock.patch.object(scorer, "A_ID_HASH", scorer.a_id_digest(synthetic_ids))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_full_synthetic_matrix_scores_200_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            result = scorer.run(tmp)
            self.assertEqual(result["phase2_status"], "COMPLETE_VALID")
            self.assertEqual(result["endpoint_count"], 200)
            self.assertFalse(result["next_stage_authorized"])
            for n in ("20", "50"):
                methods = result["per_n"][n]["methods"]
                self.assertEqual(methods["RAW16"]["endpoint_count"], 10)
                self.assertEqual(methods["DKL"]["endpoint_count"], 30)
            self.assertTrue((Path(tmp) / "scoring" / "fold_pairs.csv").exists())

    def test_partial_matrix_never_receives_full_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp, complete=False)
            result = scorer.run(tmp)
            self.assertEqual(result["phase2_status"], "PARTIAL")
            self.assertEqual(result["endpoint_count"], 199)
            self.assertEqual(result["scientific_signal_n20"], "不完整，暂不分类")
            self.assertEqual(result["scientific_signal_n50"], "不完整，暂不分类")

    def test_unsealed_manifest_never_reads_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _, _ = fixture(tmp)
            manifest["sealed"] = False
            scorer.write_json(Path(tmp) / "predictions_manifest.json", manifest)
            with mock.patch.object(scorer, "checked_labels") as release:
                with self.assertRaisesRegex(ValueError, "uniformly sealed"):
                    scorer.run(tmp)
                release.assert_not_called()

    def test_prediction_tampering_never_reads_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _, _ = fixture(tmp)
            path = Path(tmp) / manifest["files"][-1]["path"]
            path.write_text(path.read_text() + "\n")
            with mock.patch.object(scorer, "checked_labels") as release:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    scorer.run(tmp)
                release.assert_not_called()

    def test_truth_field_is_rejected(self):
        p = unit()
        p["extra"] = {"query_SE": [0.1]}
        with self.assertRaisesRegex(ValueError, "Query truth field"):
            scorer.validate_prediction(p)

    def test_scoring_pool_cannot_add_or_replace_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, labels = fixture(tmp)
            labels["unauthorized_candidate"] = labels.pop(next(iter(labels)))
            scorer.write_json(Path(tmp) / "scoring" / "labels_A.json", {"labels": labels})
            with self.assertRaisesRegex(ValueError, "179 pinned legal A IDs"):
                scorer.run(tmp)

    def test_raw_cannot_have_a_model_seed(self):
        p = unit(method="RAW16")
        p["model_seed"] = 0
        with self.assertRaisesRegex(ValueError, "outside fixed"):
            scorer.validate_prediction(p)

    def test_cost_includes_common_for_each_standalone_method_but_once_in_actual(self):
        units = [unit(method=method) for method in scorer.METHODS]
        rows, summary = scorer.cost_outputs(units)
        self.assertAlmostEqual(summary["completed_endpoint_fit_predict_seconds_with_shared_common_cost_once"], 1 + 4 * 2.1)
        self.assertEqual([r["standalone_fit_predict_seconds"] for r in rows], [3.1, 3.1, 3.1, 2.1])
        units[2]["cost"]["standalone_fit_predict_seconds"] = 2.1
        with self.assertRaisesRegex(ValueError, "include common"):
            scorer.cost_outputs(units)


if __name__ == "__main__":
    unittest.main(verbosity=2)
