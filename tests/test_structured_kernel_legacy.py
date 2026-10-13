import unittest

import numpy as np

from analyse.structured_kernel_phase1.legacy import (
    aggregate_mechanisms, correlation_stats, score_legacy_unit, seed_disagreement,
)


class LegacyTests(unittest.TestCase):
    def test_rank_is_prior_normalized_rank(self):
        self.assertAlmostEqual(correlation_stats(np.eye(4))["effective_rank"], 4)
        self.assertAlmostEqual(correlation_stats(np.ones((4, 4)))["effective_rank"], 1)
        with self.assertRaises(ValueError):
            correlation_stats(2*np.eye(3))

    def test_prospective_score_does_not_use_q_se(self):
        unit = dict(ids=list("abcdef"), mean_raw=[.6,.6,.6,.6,.6,.6],
                    latent_variance_standardized=[.1]*6, y_scale=.1,
                    extra_noise_variance_standardized=.01, median_known_var_standardized=.02)
        labels = {k: {"mean": .5 + i*.01, "se": 0} for i,k in enumerate(unit["ids"])}
        before = score_legacy_unit(unit, labels)
        for value in labels.values():
            value["se"] = 1e6
        self.assertEqual(before, score_legacy_unit(unit, labels))
        self.assertAlmostEqual(before["bias_true_top10_pp"], 5)
        self.assertAlmostEqual(before["bias_pred_top5_pp"], 8)

    def test_model_seed_disagreement_is_not_optimizer_ensemble(self):
        rows = [dict(family="compressed", method="DKL", draw=1, fold=0, n=20, step=1000, model_seed=i) for i in range(3)]
        units = [(i, {"ids": ["a", "b"], "mean_raw": [i*.01, i*.02]}) for i in range(3)]
        seed_disagreement(rows, units)
        self.assertAlmostEqual(rows[0]["seed_prediction_sd_Q_mean_pp"], np.sqrt(2/3)*1.5)
        rows[2]["model_seed"] = None
        for row in rows:
            row.pop("seed_prediction_sd_Q_mean_pp", None)
        seed_disagreement(rows, units)
        self.assertNotIn("seed_prediction_sd_Q_mean_pp", rows[0])

    def test_mechanism_aggregation_equal_weights_folds(self):
        rows=[]
        for fold in range(5):
            for draw in (20260922,20260923):
                for seed in (0,1,2):
                    for step in (0,1000):
                        end = step > 0
                        rows.append(dict(family="compressed", method="DKL", n=50, fold=fold, draw=draw,
                                         model_seed=seed, step=step, train_nll=2-int(end),
                                         q_mae_pp=3+int(end), q_nlpd_prospective=1+.2*int(end),
                                         T_effective_rank=10-2*int(end), TQ_effective_rank=20-3*int(end)))
        result=aggregate_mechanisms(rows)
        self.assertEqual(result["complete_seed_context_pairs"],30)
        self.assertEqual(result["means"][0]["fold_count"],5)
        self.assertEqual(result["means"][0]["supporting_folds_NLL_improves_MAE_worsens_T_rank_drops"],5)
        self.assertEqual(result["means"][0]["delta_q_mae_pp"],1)
        incomplete = aggregate_mechanisms([r for r in rows if not (r["fold"] == 0 and r["draw"] == 20260922 and r["model_seed"] == 2)])
        self.assertEqual(incomplete["means"], [])
        self.assertTrue(incomplete["aggregation_missing"])


if __name__ == "__main__":
    unittest.main()
