#!/usr/bin/env python3
"""Synthetic statistics only. No project observations, tensors, model or fitting."""
import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

import score_readout as score


def fixture(method="GP_CONST", y=None, mu=None, incumbent=.7):
    y = y or [.2, .3, .4, .5, .6, .8, .8]
    mu = mu or [.75, .7, .65, .6, .55, .5, .45]
    ids = ["q" + str(i) for i in range(len(y))]
    trids = ["t" + str(i) for i in range(20)]
    p = {"unit_id": "c_" + method, "context_id": "c", "subset_id": "s", "fold": 0,
         "sample_seed": 20260922, "n": 20, "model_seed": 0, "method": method,
         "candidate_ids": ids, "architecture_fingerprints": ["a" + x for x in ids],
         "mean_raw": mu, "mean_standardized": [(v - .5) / .2 for v in mu],
         "y_center": .5, "y_scale": .2, "latent_variance_standardized": [0.04] * len(y),
         "latent_variance_raw": [.04 * .2 ** 2] * len(y), "extra_noise_variance_standardized": .01,
         "endpoint_state_hash": "a" * 64, "training_ids_sha256": score.digest(trids)}
    p["actual_updates"] = 0 if method == "HEAD" else 1500
    if method == "HEAD":
        for key in ("latent_variance_standardized", "latent_variance_raw", "extra_noise_variance_standardized"):
            p[key] = None
    if method == "GP_SEMI":
        p["semi_decomposition"] = {"trend_mean_standardized": [x - .2 for x in p["mean_standardized"]],
                                    "matern_residual_mean_standardized": [.2] * len(y),
                                    "beta_posterior_mean": [.1, .9], "beta_posterior_covariance": [[.01, -.002], [-.002, .04]]}
    labels = {cid: {"validation_mean": value, "validation_se": .02} for cid, value in zip(ids, y)}
    labels.update({cid: {"validation_mean": incumbent, "validation_se": .01} for cid in trids})
    training = {"ids": trids, "y": [incumbent] * 20}
    return p, training, labels


def synthetic_matrix():
    rows = []
    for n in (20, 50):
        for fold in range(5):
            for seed in score.base.SAMPLE_SEEDS:
                for method in score.METHODS:
                    for model in ([None] if method == "RAW16" else (0, 1, 2)):
                        r = dict(n=n, fold=fold, sample_seed=seed, model_seed=model, method=method,
                                 context_id=f"{n}_{fold}_{seed}_{model}", subset_id=f"{n}_{fold}_{seed}",
                                 unit_id=f"{n}_{fold}_{seed}_{model}_{method}", champion_ids=["champ_" + str(fold)])
                        r.update({metric: 0. for metric in score.METRICS})
                        r.update(regret5_pp={"GP_CONST": 2., "GP_SEMI": 1., "HEAD": 1.05, "GP_TREND": 1.4}.get(method, 3.),
                                 gain5_pp={"GP_CONST": .1, "GP_SEMI": .3, "HEAD": .3}.get(method, .2),
                                 headroom_pp=.5, has_headroom_fraction=1., simple_regret_pp=.2,
                                 nlpd_accuracy_units=-1., IS90_pp=20., coverage90_fraction=.8, champion_coverage90_fraction=.7)
                        rows.append(r)
    stable = [dict(n=n, method=m, model_seed=j, fold=f, sample_seed=s, numeric_predictive_stable=True)
              for n in (20, 50) for m in ("GP_CONST", "GP_TREND", "GP_SEMI")
              for j in (0, 1, 2) for f in range(5) for s in score.base.SAMPLE_SEEDS]
    return rows, stable


class ScoringChecks(unittest.TestCase):
    def test_exact_random_enum_positive_part(self):
        y = [.1, .15, .3, .5, .6, .65, .9]
        inc = .7
        maxima = [max(c) for c in itertools.combinations(y, 5)]
        r = score.random5(y, inc)
        self.assertAlmostEqual(r["random_regret5_pp"], 100 * (max(y) - sum(maxima) / len(maxima)))
        self.assertAlmostEqual(r["random_gain5_pp"], 100 * sum(max(0, x - inc) for x in maxima) / len(maxima))
        self.assertAlmostEqual(r["random_break_best_probability"], sum(x - inc > 1e-11 for x in maxima) / len(maxima))
        self.assertAlmostEqual(r["random_simple_regret_pp"], 20 - r["random_gain5_pp"])
        self.assertNotAlmostEqual(r["random_gain5_pp"], max(0, 100 * (sum(maxima) / len(maxima) - inc)))

    def test_head_no_probability(self):
        p, t, labels = fixture("HEAD")
        score.validate_prediction(p)
        r, candidates = score.score_unit(p, t, labels)
        for metric in ("nlpd_accuracy_units", "coverage90_fraction", "IS90_pp", "champion_coverage90_fraction", "top20_coverage90_fraction"):
            self.assertIsNone(r[metric])
        self.assertTrue(all(c["observed_sd_raw"] is None for c in candidates))
        p["latent_variance_raw"] = [0] * 7
        with self.assertRaises(ValueError): score.validate_prediction(p)

    def test_gain_headroom_simple_regret_distinction(self):
        p, t, labels = fixture(incumbent=.9)
        r, _ = score.score_unit(p, t, labels)
        self.assertAlmostEqual(r["regret5_pp"], 20)
        self.assertEqual(r["headroom_pp"], 0)
        self.assertEqual(r["gain5_pp"], 0)
        self.assertEqual(r["simple_regret_pp"], 0)

    def test_champion_ties_all_errors_and_best_rank(self):
        p, t, labels = fixture(mu=[.5, .5, .5, .5, .5, .7, .9])
        r, _ = score.score_unit(p, t, labels)
        self.assertEqual(r["champion_ids"], ["q5", "q6"])
        self.assertEqual(r["champion_prediction_rank"], 1)
        self.assertEqual(r["champion_top5_hit_fraction"], 1)
        self.assertAlmostEqual(r["champion_mae_pp"], 10)
        self.assertAlmostEqual(r["champion_bias_pp"], 0)
        self.assertEqual(r["top20_count"], 2)
        self.assertEqual(r["top20_ids"], ["q5", "q6"])
        labels["q5"]["validation_mean"] = .8 - .5e-11
        self.assertEqual(score.score_unit(p, t, labels)[0]["champion_count"], 2)
        labels["q5"]["validation_mean"] = .8 - 2e-11
        self.assertEqual(score.score_unit(p, t, labels)[0]["champion_count"], 1)

    def test_exact_prediction_ties_use_id_no_tolerance(self):
        p, t, labels = fixture(mu=[.5] * 7)
        p["candidate_ids"] = list(reversed(p["candidate_ids"]))
        r, _ = score.score_unit(p, t, labels)
        self.assertEqual(r["selected_top5_ids"], ["q0", "q1", "q2", "q3", "q4"])
        self.assertIsNone(r["spearman"])

    def test_variance_units_and_is90(self):
        p, t, labels = fixture()
        score.validate_prediction(p)
        r, cc = score.score_unit(p, t, labels)
        variance = .2 ** 2 * (.04 + .01) + .02 ** 2
        self.assertAlmostEqual(cc[0]["observed_variance_raw"], variance)
        half = score.Z90 * math.sqrt(variance)
        self.assertAlmostEqual(cc[0]["IS90_pp"], 100 * (2 * half + 20 * (.55 - half)))
        self.assertAlmostEqual(cc[0]["nlpd_accuracy_units"], .5 * (math.log(2 * math.pi * variance) + .55 ** 2 / variance))

    def test_semi_decomposition_and_complete_explanation(self):
        candidates = []
        for method in score.NEW_METHODS:
            p, t, labels = fixture(method)
            score.validate_prediction(p)
            candidates += score.score_unit(p, t, labels)[1]
        explanation = score.candidate_explanations(candidates)
        self.assertEqual({r["candidate_id"] for r in explanation}, {"q5", "q6"})
        self.assertTrue(all(r["all_four_new_methods_present"] for r in explanation))
        for r in explanation:
            self.assertAlmostEqual(r["GP_SEMI_mean_raw"], r["semi_trend_raw_including_center"] + r["semi_matern_residual_raw"])
        p["semi_decomposition"]["matern_residual_mean_standardized"][0] += .01
        with self.assertRaises(ValueError): score.validate_prediction(p)

    def test_training_identity_rejects_query_leakage(self):
        p, t, labels = fixture()
        t["ids"][0] = "q0"
        p["training_ids_sha256"] = score.digest(t["ids"])
        with self.assertRaises(ValueError): score.score_unit(p, t, labels)

    def test_endpoint_steps_not_selected_or_additional(self):
        p, _, _ = fixture()
        score.validate_prediction(p)
        for steps in (0, 1000, 1499, 1501):
            p["actual_updates"] = steps
            with self.assertRaises(ValueError): score.validate_prediction(p)

    def test_primary_threshold_and_seed_direction(self):
        self.assertTrue(score.primary_rule([.25] * 4 + [0], [.2, .2]))
        self.assertFalse(score.primary_rule([.249] * 4 + [0], [.2, .2]))
        self.assertFalse(score.primary_rule([1, 1, 1, 0, 0], [1, 1]))
        self.assertFalse(score.primary_rule([1] * 5, [1, 1e-9]))
        self.assertTrue(score.primary_rule([1] * 5, [1, 1.1e-9]))

    def test_full_matrix_all_flags_and_raw_once(self):
        rows, stability = synthetic_matrix()
        summary, tables = score.summarize(rows, stability, True)
        self.assertEqual(summary["new_endpoint_count"], 240)
        self.assertEqual(summary["historical_reference_count"], 200)
        self.assertEqual(summary["RAW16_actual_historical_fits"], 20)
        for n in ("20", "50"):
            flags = summary["per_n"][n]["flags"]
            self.assertEqual(set(flags), set(score.FLAGS))
            self.assertTrue(all(flags[k] for k in flags if k != "ADDITIONAL_OVER_HEAD_SIGNAL"))
            self.assertFalse(flags["ADDITIONAL_OVER_HEAD_SIGNAL"])
            sharing = summary["per_n"][n]["gain_subset_champion_sharing"]
            self.assertTrue(any(x["shared_champion_ids"] for x in sharing))
        raw = [r for r in tables["paired_unit"] if r["reference_method"] == "RAW16"]
        self.assertEqual(len(raw), 60)
        self.assertEqual(len({r["reference_unit_id"] for r in raw}), 20)
        self.assertTrue(all(r["RAW16_shared_reference_not_independent"] for r in raw))

    def test_stability_probability_flags_separate(self):
        rows, stable = synthetic_matrix()
        for r in rows:
            if r["method"] == "GP_SEMI":
                r["nlpd_accuracy_units"] += .11
        for r in stable:
            if r["method"] == "GP_CONST" and r["model_seed"] == 0:
                r["numeric_predictive_stable"] = False
        summary, _ = score.summarize(rows, stable, True)
        flags = summary["per_n"]["50"]["flags"]
        self.assertTrue(flags["PRIMARY_SELECTION_SIGNAL"])
        self.assertFalse(flags["OPTIMIZATION_READY"])
        self.assertFalse(flags["PROBABILITY_NO_LARGE_DEGRADATION"])
        self.assertFalse(flags["READOUT_CANDIDATE_FOR_NEXT_PROTOCOL_REVIEW"])

    def test_champion_coverage_and_gain_are_required(self):
        rows, stable = synthetic_matrix()
        for r in rows:
            if r["method"] == "GP_SEMI":
                r["champion_coverage90_fraction"] -= .06
                r["gain5_pp"] = 0
        summary, _ = score.summarize(rows, stable, True)
        flags = summary["per_n"]["50"]["flags"]
        self.assertFalse(flags["PROBABILITY_NO_LARGE_DEGRADATION"])
        self.assertFalse(flags["INCUMBENT_GAIN_SIGNAL"])
        self.assertFalse(flags["HEAD_SIGNAL_RETAINED"])

    def test_partial_flags_all_unknown(self):
        rows, stable = synthetic_matrix()
        summary, _ = score.summarize(rows[:10], stable[:1], False)
        self.assertTrue(all(v is None for n in summary["per_n"].values() for v in n["flags"].values()))

    def test_stability_requires_three_windows_not_only_last(self):
        p, _, _ = fixture()
        windows = []
        for start in (1200, 1300, 1400):
            windows.append(dict(from_step=start, to_step=start + 100, mll_relative_change=1e-4,
                query_mean_max_abs_change_raw=.001, query_latent_sd_max_abs_change_raw=.001,
                matern_training_kernel_relative_change=.01, full_training_kernel_relative_change=.01,
                criteria=dict(mll_stable=True, mean_stable=True, latent_sd_stable=True, kernels_stable=True),
                numeric_predictive_stable=True, top5_stable=False))
        p["stability"] = dict(windows=windows, numeric_predictive_stable=True, top5_stable=False)
        self.assertTrue(score.stability_rows([p])[0]["numeric_predictive_stable"])
        windows[0]["mll_relative_change"] *= 2
        with self.assertRaises(ValueError): score.stability_rows([p])
        windows[0]["criteria"]["mll_stable"] = False
        windows[0]["numeric_predictive_stable"] = False
        p["stability"]["numeric_predictive_stable"] = False
        self.assertFalse(score.stability_rows([p])[0]["numeric_predictive_stable"])

    def test_scoring_imports_no_model_or_numerics_runtime(self):
        import sys
        self.assertFalse(any(name in sys.modules for name in ("torch", "gpytorch", "numpy", "readout_core", "residual_core")))

    def test_hierarchy_is_not_pooling_candidates_or_raw_reuse(self):
        rows = [dict(n=20, method="HEAD", fold=f, sample_seed=s, model_seed=m, value=f + 100 * (s == 20260923) + m)
                for f in range(5) for s in score.base.SAMPLE_SEEDS for m in (0, 1, 2)]
        subset, folds, seeds, models = score.hierarchy(rows, ("value",))
        self.assertEqual(len(subset), 10)
        self.assertEqual([r["value"] for r in folds], [51, 52, 53, 54, 55])
        self.assertEqual(score.base.mean([r["value"] for r in folds]), 53)
        self.assertEqual([r["value"] for r in seeds], [3, 103])

    def test_unsealed_and_invalid_reject_before_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for seal in ({"sealed": False}, {"sealed": True, "execution_status": "INVALID"}):
                (root / "predictions_manifest.json").write_text(json.dumps(seal))
                with self.assertRaises(ValueError): score.read_sealed(root)
                self.assertFalse((root / "scoring").exists())


if __name__ == "__main__":
    unittest.main()
