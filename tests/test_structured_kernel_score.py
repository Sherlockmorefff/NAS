"""Independent scoring, leakage boundaries, aggregation and frozen-state checks."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from analyse.structured_kernel_phase1.score import (
    DRAWS, METHODS, METRIC_KEYS, Z90, aggregate_metrics, engineering_gate,
    score_frozen, score_prediction,
)
from analyse.structured_kernel_phase1.io import digest, read_json, sha256, write_json


def sample_unit(count=10):
    ids = [f"c{i:02}" for i in range(count)]
    unit = {"ids": ids, "mean_raw": [0.75] * count,
            "latent_variance_standardized": [0.04] * count,
            "extra_noise_variance_standardized": 0.01,
            "median_known_var_standardized": 0.02, "y_scale": 0.1,
            "method": "L-A", "n": 20, "fold": 0, "draw": DRAWS[0],
            "setting_id": "s20260922_f0_n20_L-A", "model_hash": "model-hash"}
    labels = {cid: {"mean": 0.76, "se": 0.01, "seeds": [0.75, 0.76, 0.77]} for cid in ids}
    return unit, labels


def complete_contexts():
    contexts = []
    for method in METHODS:
        for n in (20, 50):
            for fold in range(5):
                for draw in DRAWS:
                    row = {"method": method, "n": n, "fold": fold, "draw": draw}
                    row.update({key: 0.0 for key in METRIC_KEYS})
                    row.update(mae_pp=2.0 if method == "RAW16" else 1.5,
                               nlpd=-2.0, is90_pp=3.0, coverage90=0.9, width90_pp=3.0)
                    contexts.append(row)
    return contexts


class ScoreTests(unittest.TestCase):
    def test_scale_noise_and_nlpd_units(self):
        unit, labels = sample_unit()
        result = score_prediction(unit, labels)
        variance = (0.04 + 0.01 + 0.02) * 0.1 ** 2
        self.assertAlmostEqual(result["mae_pp"], 1.0)
        self.assertAlmostEqual(result["nlpd"], .5 * math.log(2 * math.pi * variance) + .01 ** 2 / (2 * variance))
        self.assertAlmostEqual(result["width90_pp"], 2 * Z90 * math.sqrt(variance) * 100)
        self.assertAlmostEqual(result["is90_pp"], result["width90_pp"])

    def test_q_se_only_changes_retrospective_metrics(self):
        unit, labels = sample_unit()
        changed = copy.deepcopy(labels)
        for row in changed.values():
            row["se"] = 0.2
        before, after = score_prediction(unit, labels), score_prediction(unit, changed)
        for key in METRIC_KEYS:
            if not key.startswith("retrospective_"):
                self.assertEqual(before[key], after[key], key)
        self.assertNotEqual(before["retrospective_nlpd"], after["retrospective_nlpd"])

    def test_input_objects_are_not_modified(self):
        unit, labels = sample_unit()
        saved = copy.deepcopy((unit, labels))
        score_prediction(unit, labels)
        self.assertEqual((unit, labels), saved)

    def test_exact_ties_by_candidate_id_and_ceil_top_ten_percent(self):
        unit, labels = sample_unit(11)
        unit["ids"] = list(reversed(unit["ids"]))
        labels["c00"]["mean"] = labels["c01"]["mean"] = 0.9
        labels["c10"]["mean"] = 1.0
        result = score_prediction(unit, labels)
        # Two true-top candidates: c10 then c00. Tied predictions choose c00..c04.
        self.assertAlmostEqual(result["q_top10_bias_pp"], -20.0)
        self.assertAlmostEqual(result["regret5_pp"], 10.0)
        self.assertAlmostEqual(result["predicted_top5_bias_pp"], ((.75-.9)*2+(.75-.76)*3)/5*100)

    def test_no_predictive_or_variance_clipping(self):
        unit, labels = sample_unit()
        unit["mean_raw"] = [1.05] * len(unit["ids"])
        self.assertAlmostEqual(score_prediction(unit, labels)["mae_pp"], 29.0)
        unit["latent_variance_standardized"][0] = -1e-7
        with self.assertRaises(ValueError):
            score_prediction(unit, labels)

    def test_tiny_negative_latent_is_not_clipped_and_interval_is_missing(self):
        unit, labels = sample_unit()
        unit["latent_variance_standardized"] = [-1e-12] * len(unit["ids"])
        result = score_prediction(unit, labels)
        self.assertIsNone(result["latent_width90_pp"])
        self.assertEqual(result["latent_negative_variance_count"], len(unit["ids"]))
        self.assertEqual(unit["latent_variance_standardized"][0], -1e-12)
        variance = (-1e-12 + .01 + .02) * .1 ** 2
        self.assertAlmostEqual(result["width90_pp"], 2 * Z90 * math.sqrt(variance) * 100)

    def test_zero_y_scale_blocked(self):
        unit, labels = sample_unit()
        unit["y_scale"] = 0.0
        with self.assertRaises(ValueError):
            score_prediction(unit, labels)

    def test_fold_draw_aggregation_is_equal_weight_and_missing_not_dropped(self):
        rows = complete_contexts()
        for row in rows:
            row["mae_pp"] = row["fold"] + (0 if row["draw"] == DRAWS[0] else 2)
        folds, summary, missing = aggregate_metrics(rows)
        self.assertEqual(missing, [])
        self.assertTrue(all(row["mae_pp"] == 3.0 for row in summary))
        partial = [row for row in rows if not(row["method"] == "L-A" and row["n"] == 50 and row["fold"] == 0 and row["draw"] == DRAWS[0])]
        folds, summary, missing = aggregate_metrics(partial)
        self.assertFalse(any(row["method"] == "L-A" and row["n"] == 50 for row in summary))
        self.assertEqual(len(missing), 1)

    def test_gate_cannot_pass_incomplete_run(self):
        folds, summary, _ = aggregate_metrics(complete_contexts())
        gate = engineering_gate(summary, folds, complete=False)
        self.assertEqual(gate["status"], "INCOMPLETE")
        self.assertIsNone(gate["selected_method"])
        self.assertFalse(any(row["passed"] for row in gate["candidates"]))

    def test_gate_ties_additive_then_latent(self):
        folds, summary, _ = aggregate_metrics(complete_contexts())
        gate = engineering_gate(summary, folds, complete=True)
        self.assertEqual(gate["selected_method"], "L-A")
        self.assertEqual(len(gate["comparisons"]), 14)
        for row in summary:
            if row["method"] == "W-I" and row["n"] == 50:
                row["nlpd"] -= .02
        self.assertEqual(engineering_gate(summary, folds, complete=True)["selected_method"], "W-I")

    def test_gate_each_required_threshold_is_effective(self):
        folds, summary, _ = aggregate_metrics(complete_contexts())
        for row in summary:
            if row["method"] == "L-A" and row["n"] == 20:
                row["mae_pp"] = 2.31
        gate = engineering_gate(summary, folds, complete=True)
        candidate = next(row for row in gate["candidates"] if row["method"] == "L-A")
        self.assertFalse(candidate["passed"])
        self.assertFalse(candidate["checks"]["n20_mae_deterioration_at_most_0_3_pp"])


class FrozenScoreTests(unittest.TestCase):
    def make_run(self, root):
        repo, run = root / "repo", root / "run"
        repo.mkdir(); run.mkdir()
        (repo / "source.py").write_text("# frozen source\n")
        write_json(run / "protocol.json", {"frozen": True})
        write_json(run / "public.json", {"features": []})
        codes = {"source.py": sha256(repo / "source.py")}
        unit, labels = sample_unit()
        unit.update(subset_id="s20260922_f0_n20", distance_scales={"architecture": {"value": 1.0}},
                    T_ids=["t0"], y_center=.75)
        setting = {key: unit[key] for key in ("setting_id", "subset_id", "fold", "draw", "n", "method")}
        training = {"ids": unit["T_ids"], "scaler": {"y_mean": .75, "y_scale": .1}, "known_variance_standardized": [.02]}
        write_json(run / "settings.json", [setting])
        training_name = f"training/{setting['subset_id']}.json"
        query_name = "queries/fold0.json"
        write_json(run / training_name, training)
        write_json(run / query_name, {"ids": unit["ids"]})
        freeze = {"code_files": codes, "code_hash": digest(codes), "protocol_sha256": sha256(run / "protocol.json"),
                  "input_files": {name: sha256(run / name) for name in ("public.json", "settings.json", training_name, query_name)}}
        freeze["identity"] = digest(freeze)
        write_json(run / "frozen_manifest.json", freeze)
        sid = unit["setting_id"]
        folder = run / "settings" / sid
        folder.mkdir(parents=True)
        identity = digest({"frozen_identity": freeze["identity"], "setting": setting, "T_ids": training["ids"],
                           "Q_ids": unit["ids"], "scaler": training["scaler"], "distance_scales": unit["distance_scales"]})
        unit.update(identity=identity, selected_start=1)
        hashes = {}
        for start, nll in enumerate((2.0, 1.0, 1.0)):
            (folder / f"start{start}.pt").write_bytes(f"start-{start}".encode())
            (folder / f"start{start}.jsonl").write_text('{"loss": 1.0}\n')
            write_json(folder / f"start{start}.json", {"identity": unit["identity"], "start_id": start, "success": True, "final_nll": nll})
            hashes[f"start{start}_json_sha256"] = sha256(folder / f"start{start}.json")
            hashes[f"start{start}_model_sha256"] = sha256(folder / f"start{start}.pt")
            hashes[f"start{start}_log_sha256"] = sha256(folder / f"start{start}.jsonl")
        (folder / "model.pt").write_bytes((folder / "start1.pt").read_bytes())
        unit["model_hash"] = sha256(folder / "model.pt")
        write_json(folder / "prediction.json", unit)
        hashes.update(model_sha256=unit["model_hash"], prediction_sha256=sha256(folder / "prediction.json"))
        seal = {"code_hash": freeze["code_hash"], "protocol_sha256": freeze["protocol_sha256"], "settings": {sid: hashes}}
        seal["identity"] = digest(seal)
        write_json(run / "prediction_seal.json", seal)
        write_json(run / "status.json", {"settings": {sid: {**{k: unit[k] for k in ("method", "n", "fold", "draw", "identity")}, "state": "RUNNING", "phase": "FITTED"}}})
        return repo, run, unit, labels

    def test_sealed_models_unchanged_and_pilot_still_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, run, unit, labels = self.make_run(Path(temporary))
            before = {str(p): sha256(p) for p in (run / "settings" / unit["setting_id"]).glob("*.pt")}
            data = types.SimpleNamespace(read_A_labels=Mock(return_value=labels))
            with patch.dict(sys.modules, {"analyse.structured_kernel_phase1.data": data}):
                gate = score_frozen(run, repo)
            self.assertEqual(gate["status"], "INCOMPLETE")
            self.assertEqual(gate["complete_settings"], 1)
            self.assertEqual(read_json(run / "status.json")["settings"][unit["setting_id"]]["state"], "COMPLETE")
            self.assertTrue(all(sha256(path) == value for path, value in before.items()))

    def test_code_tamper_rejected_before_label_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, run, _, labels = self.make_run(Path(temporary))
            (repo / "source.py").write_text("# changed\n")
            data = types.SimpleNamespace(read_A_labels=Mock(return_value=labels))
            with patch.dict(sys.modules, {"analyse.structured_kernel_phase1.data": data}):
                with self.assertRaises(ValueError):
                    score_frozen(run, repo)
            data.read_A_labels.assert_not_called()

    def test_query_ids_must_match_frozen_bundle_even_if_prediction_is_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, run, unit, labels = self.make_run(Path(temporary))
            sid = unit["setting_id"]
            unit["ids"] = list(reversed(unit["ids"]))
            prediction_path = run / "settings" / sid / "prediction.json"
            write_json(prediction_path, unit)
            seal = read_json(run / "prediction_seal.json")
            seal["settings"][sid]["prediction_sha256"] = sha256(prediction_path)
            seal["identity"] = digest({key: value for key, value in seal.items() if key != "identity"})
            write_json(run / "prediction_seal.json", seal)
            data = types.SimpleNamespace(read_A_labels=Mock(return_value=labels))
            with patch.dict(sys.modules, {"analyse.structured_kernel_phase1.data": data}):
                gate = score_frozen(run, repo)
            data.read_A_labels.assert_not_called()
            self.assertEqual(gate["complete_settings"], 0)
            self.assertEqual(read_json(run / "status.json")["settings"][sid]["state"], "FAILED")

    def test_start_nll_tie_must_select_lower_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, run, unit, labels = self.make_run(Path(temporary))
            sid = unit["setting_id"]
            unit["selected_start"] = 2
            prediction_path = run / "settings" / sid / "prediction.json"
            write_json(prediction_path, unit)
            seal = read_json(run / "prediction_seal.json")
            seal["settings"][sid]["prediction_sha256"] = sha256(prediction_path)
            seal["identity"] = digest({key: value for key, value in seal.items() if key != "identity"})
            write_json(run / "prediction_seal.json", seal)
            data = types.SimpleNamespace(read_A_labels=Mock(return_value=labels))
            with patch.dict(sys.modules, {"analyse.structured_kernel_phase1.data": data}):
                gate = score_frozen(run, repo)
            data.read_A_labels.assert_not_called()
            self.assertEqual(gate["complete_settings"], 0)
            self.assertIn("training NLL", gate["failures"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
