#!/usr/bin/env python3
"""Stage-1 development replay. This entry point cannot train candidate GNNs.

Preparation alone reads historical shards. The learner gets a label-free
candidate manifest and calls an oracle only after an immutable selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpytorch
import numpy as np
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling.normal import SobolQMCNormalSampler
from surrogate.accuracy_gp import StableQLogExpectedImprovement
from architecture_diagnostics import architecture_fingerprint, encode_architecture_mu, load_joint_vae
from analyse.joint_configuration_metric_gate import hp_feature_vector, load_and_aggregate_source
from surrogate.fuzzy_metric import JointMetricModel, metric_loss, neighborhood_affinity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def json_write(path, value, *, immutable=False):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare(config, output):
    """Trusted preprocessing; no learner or selector receives aggregate rows."""
    output.mkdir(parents=True, exist_ok=False)
    rows, inventory, provenance = load_and_aggregate_source(config)
    candidates, labels = [], {}
    source_by_joint = {}
    for shard in provenance["shards"]:
        for line in Path(shard["path"]).read_text().splitlines():
            raw = json.loads(line)
            source_by_joint[raw["joint_config_fingerprint"]] = raw
    for row in rows:
        candidate = {key: copy.deepcopy(row[key]) for key in (
            "joint_fingerprint", "architecture_fingerprint", "hp_fingerprint",
            "operations", "edges", "raw_hp", "lr", "dropout", "hidden_dim", "l2",
            "training_protocol_identity", "source_file", "source_lines")}
        candidate["candidate_id"] = row["joint_fingerprint"]
        raw = source_by_joint[candidate["candidate_id"]]
        candidate["raw_hp_vector"] = raw["raw_hp_vector"]
        candidate["source_config_id"] = raw["config_id"]
        candidate["effective_layers"] = sum(op != "Identity" for op in candidate["operations"])
        candidate["gcnii_alpha"] = 0.1
        candidate["gcnii_theta"] = 0.5
        for key, expected in {"hp_mode": "global4", "normalization": "none", "gat_heads": 1, "sage_aggr": "mean", "gin_eps": 0.0, "gat_heads_by_layer": None, "sage_aggr_by_layer": None, "gin_eps_by_layer": None}.items():
            if raw.get(key) != expected:
                raise RuntimeError(f"fixed candidate HP differs: {key}")
            candidate[key] = expected
        candidates.append(candidate)
        labels[candidate["candidate_id"]] = {
            "validation_mean": row["validation_mean"],
            "validation_se": row["validation_se"],
            "validation_seeds": [row[f"validation_seed_{seed}"] for seed in (0, 1, 2)],
            "training_seeds": [0, 1, 2],
        }
    if len(candidates) < config["initial_size"] + config["query_budget"]:
        raise RuntimeError("real historical candidate count cannot support requested replay")
    checkpoint = ROOT / config["checkpoint"]
    if file_hash(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("VAE checkpoint differs from the reference manifest")
    vae = load_joint_vae(checkpoint, "global4", device="cpu")
    vae.requires_grad_(False)
    before = tensor_hash(vae.state_dict())
    cache = {}
    for candidate in candidates:
        arch = candidate["architecture_fingerprint"]
        if arch != architecture_fingerprint(candidate):
            raise RuntimeError("architecture identity mismatch")
        if arch not in cache:
            cache[arch] = encode_architecture_mu(vae, candidate)
    # A fresh second pass is a direct deterministic-encoding check.
    deterministic = all(cache[c["architecture_fingerprint"]] == encode_architecture_mu(vae, c)
                        for c in candidates)
    after = tensor_hash(vae.state_dict())
    if before != after or not deterministic:
        raise RuntimeError("VAE changed or encoding is nondeterministic")
    json_write(output / "candidates.json", candidates)
    json_write(output / "oracle_validation.json", labels)
    json_write(output / "base_encoding.json", cache)
    json_write(output / "inventory.json", inventory)
    json_write(output / "provenance.json", provenance)
    json_write(output / "data_use.json", {
        "use": "development_only", "independent_confirmation": [],
        "architecture_groups": sorted(cache),
        "reason": "historical pool previously used for model development; this replay adds no confirmation claim",
    })
    json_write(output / "manifest.json", {
        "pool_hash": digest(candidates), "base_encoding_hash": digest(cache),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "vae_tensor_hash_before": before, "vae_tensor_hash_after": after,
        "vae_requires_grad_count": sum(p.requires_grad for p in vae.parameters()),
        "deterministic_full_pool_reencode": deterministic,
        "vae_performance_supervision": False,
        "pretraining_membership": "exact graph membership unknown; unsupervised pool exposure permitted",
        "oracle_sha256": file_hash(output / "oracle_validation.json"),
        "gnn_trainings": 0,
    })
    return inventory


class HistoryOracle:
    """Evaluation boundary. Only explicitly selected IDs leave this object."""
    def __init__(self, path):
        self.path = Path(path)

    def reveal(self, candidate_ids):
        source = json.loads(self.path.read_text())
        result = {}
        for cid in candidate_ids:
            row = copy.deepcopy(source[cid])
            if set(row) != {"validation_mean", "validation_se", "validation_seeds", "training_seeds"}:
                raise RuntimeError("oracle schema includes unexpected fields")
            if row["training_seeds"] != [0, 1, 2]:
                raise RuntimeError("oracle training seeds changed")
            values = np.asarray(row["validation_seeds"], dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise RuntimeError("invalid revealed validation record")
            if not np.isclose(values.mean(), row["validation_mean"], rtol=0, atol=1e-12):
                raise RuntimeError("revealed seed mean disagrees")
            if not np.isclose(values.std(ddof=1) / math.sqrt(3), row["validation_se"], rtol=0, atol=1e-12):
                raise RuntimeError("revealed seed standard error disagrees")
            result[cid] = row
        return result


def validate_candidates(candidates):
    allowed = {"candidate_id", "joint_fingerprint", "architecture_fingerprint", "hp_fingerprint",
               "operations", "edges", "raw_hp", "lr", "dropout", "hidden_dim", "l2",
               "training_protocol_identity", "source_file", "source_lines", "raw_hp_vector",
               "hp_mode", "normalization", "gat_heads", "sage_aggr", "gin_eps",
               "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
               "gcnii_alpha", "gcnii_theta", "effective_layers", "source_config_id"}
    if any(set(c) - allowed for c in candidates):
        raise ValueError("non-candidate metadata reached P2")
    for candidate in candidates:
        if "raw_hp" not in candidate:
            continue  # P2 also accepts identity-only rows; Replay requires HP features.
        if set(candidate["raw_hp"]) != {"lr", "dropout", "hidden_dim", "l2"}:
            raise ValueError("unexpected nested candidate HP fields")
        if any(key in candidate and candidate[key] != candidate["raw_hp"][key] for key in ("lr", "dropout", "hidden_dim", "l2")):
            raise ValueError("candidate HP representations disagree")


def p2_select(candidates, scores, architecture_counts, batch_size=5):
    """Frozen P2 quota loop from b4_arch_diversity...:1005-1023.

    The adaptation accepts a single batch and a generic score. Exact architecture
    identity, batch uniqueness, total cap 2, and fingerprint tie breaks are fixed.
    Counts apply to new queries, as in the original empty per-replicate Counter.
    """
    validate_candidates(candidates)
    if any(not math.isfinite(float(scores[c["candidate_id"]])) for c in candidates):
        raise ValueError("nonfinite selection score")
    ranked = sorted(candidates, key=lambda c: (-scores[c["candidate_id"]], c["joint_fingerprint"]))
    chosen, used = [], set()
    for candidate in ranked:
        arch = candidate["architecture_fingerprint"]
        if arch in used or architecture_counts.get(arch, 0) >= 2:
            continue
        chosen.append(candidate["candidate_id"])
        used.add(arch)
        if len(chosen) == batch_size:
            return chosen
    raise RuntimeError("P2 cannot fill the requested batch under frozen quotas")


class ReplayGP(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, x, y, se, scale):
        # Same SE² + learned floor², isotropic Matérn-5/2 protocol as joint_metric.
        with gpytorch.settings.min_fixed_noise(float_value=0.0, double_value=0.0, half_value=0.0):
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=(se / scale).square(), learn_additional_noise=True,
                noise_constraint=gpytorch.constraints.GreaterThan((1e-8 / scale) ** 2),
            ).double()
        likelihood.second_noise = (1e-4 / scale) ** 2
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class Replay:
    def __init__(self, config, prepared, output, *, resume=False, oracle=None):
        self.config = copy.deepcopy(config)
        self.prepared, self.output = Path(prepared), Path(output)
        self.candidates = json.loads((self.prepared / "candidates.json").read_text())
        validate_candidates(self.candidates)
        self.identity = digest(self.candidates)
        self.ids = [c["candidate_id"] for c in self.candidates]
        if len(set(self.ids)) != len(self.ids):
            raise RuntimeError("candidate IDs are not unique")
        self.by_id = {c["candidate_id"]: c for c in self.candidates}
        self.index = {cid: i for i, cid in enumerate(self.ids)}
        cache = json.loads((self.prepared / "base_encoding.json").read_text())
        self.manifest = json.loads((self.prepared / "manifest.json").read_text())
        if self.identity != self.manifest["pool_hash"] or digest(cache) != self.manifest["base_encoding_hash"]:
            raise RuntimeError("candidate or base encoding identity changed")
        self.z = torch.tensor([cache[c["architecture_fingerprint"]] for c in self.candidates], dtype=torch.float32)
        self.hp = torch.tensor([hp_feature_vector(c) for c in self.candidates], dtype=torch.float32)
        self.oracle = oracle or HistoryOracle(self.prepared / "oracle_validation.json")
        self.model = JointMetricModel(z_dim=self.z.shape[1], seed=config["model_seed"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.gp = None
        source_paths = ["analyse/fuzzy_metric_replay.py", "surrogate/fuzzy_metric.py",
                        "architecture_diagnostics.py", "models.py", "nas_space.py", "hp_modes.py",
                        "analyse/joint_configuration_metric_gate.py", "surrogate/accuracy_gp.py"]
        source_identity = {str(ROOT / path): file_hash(ROOT / path) for path in source_paths}
        if resume:
            if json.loads((self.output / "source_identity.json").read_text()) != source_identity:
                raise RuntimeError("resume implementation changed")
            self.state = load_torch(self.output / "state.pt")
            if self.state["config_hash"] != digest(config) or self.state["pool_hash"] != self.identity:
                raise RuntimeError("resume configuration or candidate identity differs")
            if self.state["input_manifest_hash"] != digest(self.manifest):
                raise RuntimeError("resume input manifest changed")
            if file_hash(self.prepared / "oracle_validation.json") != self.manifest["oracle_sha256"] and oracle is None:
                raise RuntimeError("historical oracle source changed")
            self.model.load_state_dict(self.state["model"])
            self.optimizer.load_state_dict(self.state["optimizer"])
            torch.set_rng_state(self.state["torch_rng"])
            np.random.set_state(self.state["numpy_rng"])
            random.setstate(self.state["python_rng"])
            if self.state["gp_state"] is not None and not self.state["refresh_needed"]:
                self._make_gp()
                self.gp.load_state_dict(self.state["gp_state"])
                self.gp.eval()
                self.gp.likelihood.eval()
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            json_write(self.output / "source_identity.json", source_identity, immutable=True)
            rng = np.random.default_rng(config["search_seed"])
            initial = [self.ids[int(i)] for i in rng.choice(len(self.ids), config["initial_size"], replace=False)]
            json_write(self.output / "initial_selection.json", {"candidate_ids": initial, "uses_labels": False}, immutable=True)
            self.state = {
                "format_version": 1, "config_hash": digest(config), "pool_hash": self.identity,
                "input_manifest_hash": digest(self.manifest),
                "initial_ids": initial, "known": self.oracle.reveal(initial), "query_order": [],
                "representation_version": -1, "gp_fit_count": 0, "gp_state": None,
                "architecture_counts": {}, "pending": None, "events": [],
                "refresh_needed": True, "gnn_trainings": 0,
            }
            ix = [self.index[cid] for cid in initial]
            self.state["z_mean"], self.state["z_scale"] = self.z[ix].mean(0), self.z[ix].std(0, correction=0).clamp_min(1e-6)
            self.state["hp_mean"], self.state["hp_scale"] = self.hp[ix].mean(0), self.hp[ix].std(0, correction=0).clamp_min(1e-6)
            json_write(self.output / "config.json", config, immutable=True)
            self.save()

    def inputs(self):
        return ((self.z - self.state["z_mean"]) / self.state["z_scale"],
                (self.hp - self.state["hp_mean"]) / self.state["hp_scale"])

    def observed(self):
        ids = self.state["initial_ids"] + self.state["query_order"]
        if len(set(ids)) != len(ids) or set(ids) != set(self.state["known"]):
            raise RuntimeError("query count and revealed identity disagree")
        return ids, torch.tensor([self.index[cid] for cid in ids], dtype=torch.long)

    def save(self):
        self.state["model"] = self.model.state_dict()
        self.state["optimizer"] = self.optimizer.state_dict()
        self.state["torch_rng"] = torch.get_rng_state()
        self.state["numpy_rng"] = np.random.get_state()
        self.state["python_rng"] = random.getstate()
        tmp = self.output / "state.tmp"
        torch.save(self.state, tmp)
        tmp.replace(self.output / "state.pt")

    def update_representation(self):
        ids, ix = self.observed()
        y = torch.tensor([self.state["known"][cid]["validation_mean"] for cid in ids], dtype=torch.float64)
        y_scaled = ((y - y.mean()) / y.std(correction=0).clamp_min(1e-8)).float()
        affinity = neighborhood_affinity(y, mode=self.config["neighborhood"], epsilon=self.config["epsilon"])
        seed_y = torch.tensor([self.state["known"][cid]["validation_seeds"] for cid in ids], dtype=torch.float64)
        differences = seed_y[:, None, :] - seed_y[None, :, :]
        pair_se = differences.std(-1, correction=1) / math.sqrt(3)
        weights = 1 / (1 + (pair_se / self.config["epsilon"]).square())
        z, hp = self.inputs()
        started = time.monotonic()
        self.model.train()
        for _ in range(self.config["representation_steps"]):
            self.optimizer.zero_grad(set_to_none=True)
            embedding, prediction = self.model(z[ix], hp[ix])
            losses = metric_loss(embedding, prediction, y_scaled, affinity,
                                 lambda_f=self.config["lambda_f"], lambda_e=self.config["lambda_e"], pair_weights=weights)
            if not all(torch.isfinite(v).all() for v in losses.values()):
                raise RuntimeError("nonfinite representation loss")
            losses["total"].backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()):
                raise RuntimeError("nonfinite representation gradient")
            self.optimizer.step()
        self.model.eval()
        with torch.no_grad():
            all_e, _ = self.model(z, hp)
        if not torch.isfinite(all_e).all():
            raise RuntimeError("nonfinite candidate representation")
        self.state["embeddings"] = all_e.detach().clone()
        self.state["representation_version"] += 1
        self.state["events"].append({
            "type": "representation_update", "after_queries": len(self.state["query_order"]),
            "representation_version": self.state["representation_version"], "fit_ids": ids,
            "all_candidates_reencoded": len(self.ids), "steps": self.config["representation_steps"],
            "losses": {k: float(v.detach()) for k, v in losses.items()},
            "elapsed_seconds": time.monotonic() - started,
            "noise_weight": "paired_seed_difference_SE_only_observed", "early_stopping": False,
        })

    def _make_gp(self):
        ids, ix = self.observed()
        y = torch.tensor([self.state["known"][cid]["validation_mean"] for cid in ids], dtype=torch.double)
        se = torch.tensor([self.state["known"][cid]["validation_se"] for cid in ids], dtype=torch.double)
        self.y_mean, self.y_scale = y.mean(), y.std(correction=0).clamp_min(1e-8)
        x = self.state["embeddings"][ix].double()
        self.gp = ReplayGP(x, (y - self.y_mean) / self.y_scale, se, float(self.y_scale)).double()
        return ids

    def refresh(self):
        queries = len(self.state["query_order"])
        if self.state["representation_version"] == -1 or queries % self.config["update_every"] == 0:
            self.update_representation()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config["gp_seed"])
            ids = self._make_gp()
        self.gp.train()
        self.gp.likelihood.train()
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.config["gp_learning_rate"])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        started = time.monotonic()
        with gpytorch.settings.cholesky_jitter(double_value=1e-6):
            for _ in range(self.config["gp_steps"]):
                optimizer.zero_grad(set_to_none=True)
                loss = -mll(self.gp(self.gp.train_inputs[0]), self.gp.train_targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite GP loss")
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.gp.parameters()):
                    raise RuntimeError("nonfinite GP gradient")
                optimizer.step()
        self.gp.eval()
        self.gp.likelihood.eval()
        self.state["gp_state"] = self.gp.state_dict()
        self.state["gp_fit_count"] += 1
        self.state["events"].append({
            "type": "gp_refit", "after_queries": queries, "fit_ids": ids,
            "representation_version": self.state["representation_version"],
            "from_scratch": True, "steps": self.config["gp_steps"],
            "elapsed_seconds": time.monotonic() - started,
        })
        self.state["refresh_needed"] = False
        self.save()

    def select(self):
        ids, ix = self.observed()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(double_value=1e-6):
            recomputed, _ = self.model(*self.inputs())
            if not torch.equal(recomputed, self.state["embeddings"]):
                raise RuntimeError("old/new candidate coordinates are inconsistent")
            if not torch.equal(self.gp.train_inputs[0], recomputed[ix].double()):
                raise RuntimeError("GP training points use stale coordinates")
            posterior = self.gp(recomputed.double())
            mean = (posterior.mean * self.y_scale + self.y_mean).numpy()
            latent_sd = posterior.variance.clamp_min(1e-18).sqrt().numpy() * float(self.y_scale)
            # Prospective observation interval uses an observed-only pooled SE;
            # candidate-specific hidden SE is never provided to the predictor.
            observed_noise = np.mean([self.state["known"][cid]["validation_se"] ** 2 for cid in ids])
            floor_var = float(self.gp.likelihood.second_noise.detach()) * float(self.y_scale) ** 2
            observation_sd = np.sqrt(latent_sd ** 2 + observed_noise + floor_var)
        best = max(self.state["known"][cid]["validation_mean"] for cid in ids)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.config["acquisition_samples"]]), seed=self.config["gp_seed"])
        acquisition = StableQLogExpectedImprovement(
            self.gp, best_f=(best - float(self.y_mean)) / float(self.y_scale), sampler=sampler,
        )
        with torch.no_grad():
            score = acquisition(self.state["embeddings"].double().unsqueeze(1)).numpy()
        if not (np.isfinite(mean).all() and np.isfinite(observation_sd).all() and np.isfinite(score).all()):
            raise RuntimeError("nonfinite pre-selection predictions")
        remaining = [c for c in self.candidates if c["candidate_id"] not in self.state["known"]]
        scores = {cid: float(score[i]) for i, cid in enumerate(self.ids)}
        selected = p2_select(remaining, scores, self.state["architecture_counts"], self.config["batch_size"])
        batch = len(self.state["query_order"]) // self.config["batch_size"] + 1
        predictions = [{
            "candidate_id": c["candidate_id"], "mean": float(mean[self.index[c["candidate_id"]]]),
            "latent_sd": float(latent_sd[self.index[c["candidate_id"]]]),
            "observation_sd": float(observation_sd[self.index[c["candidate_id"]]]),
            "interval_lower": float(mean[self.index[c["candidate_id"]]] - 1.96 * observation_sd[self.index[c["candidate_id"]]]),
            "interval_upper": float(mean[self.index[c["candidate_id"]]] + 1.96 * observation_sd[self.index[c["candidate_id"]]]),
            "log_expected_improvement": scores[c["candidate_id"]],
            "representation_version": self.state["representation_version"],
        } for c in remaining]
        record = {"batch": batch, "after_queries": len(self.state["query_order"]),
                  "selected_ids": selected, "predictions": predictions, "known_ids": ids,
                  "candidate_identity_hash": self.identity, "labels_revealed": False}
        json_write(self.output / f"batch_{batch:02d}_before_reveal.json", record, immutable=True)
        self.state["pending"] = record
        self.save()

    def reveal_pending(self):
        pending = self.state["pending"]
        selected = pending["selected_ids"]
        if any(cid in self.state["known"] for cid in selected):
            raise RuntimeError("attempt to reveal an already charged query")
        result = self.oracle.reveal(selected)
        if set(result) != set(selected):
            raise RuntimeError("oracle returned labels for unselected candidates")
        pred = {row["candidate_id"]: row for row in pending["predictions"]}
        for cid in selected:
            self.state["query_order"].append(cid)
            self.state["known"][cid] = result[cid]
            arch = self.by_id[cid]["architecture_fingerprint"]
            counts = self.state["architecture_counts"]
            counts[arch] = counts.get(arch, 0) + 1
            self.state["events"].append({"type": "reveal", "query_number": len(self.state["query_order"]),
                "candidate_id": cid, "label": result[cid], "prediction_before_reveal": pred[cid],
                "representation_version": pred[cid]["representation_version"], "use": "development_only"})
        self.state["pending"] = None
        self.state["refresh_needed"] = True
        self.save()

    def run(self, *, stop_after_queries=None, pause_before_reveal_batch=None):
        limit = self.config["query_budget"] if stop_after_queries is None else stop_after_queries
        if limit < len(self.state["query_order"]) or limit > self.config["query_budget"] or limit % self.config["batch_size"]:
            raise ValueError("stop limit must be a nondecreasing batch boundary within budget")
        while True:
            if self.state["refresh_needed"]:
                self.refresh()
            if len(self.state["query_order"]) >= limit:
                break
            if self.state["pending"] is None:
                self.select()
            if self.state["pending"]["batch"] == pause_before_reveal_batch:
                break
            self.reveal_pending()
        summary = self.summary()
        json_write(self.output / "summary.json", summary)
        json_write(self.output / "events.json", self.state["events"])
        self.save()
        return summary

    def summary(self):
        updates = [e for e in self.state["events"] if e["type"] == "representation_update"]
        if digest(self.candidates) != self.identity:
            raise RuntimeError("candidate configuration mutated during replay")
        if file_hash(self.manifest["checkpoint"]) != self.manifest["checkpoint_sha256"]:
            raise RuntimeError("VAE checkpoint changed during replay")
        return {"initial_records": len(self.state["initial_ids"]), "new_history_queries": len(self.state["query_order"]),
                "logical_records_seen": len(self.state["known"]), "actual_new_gnn_trainings": 0,
                "historical_seed_records_revealed": 3 * len(self.state["known"]),
                "query_budget": self.config["query_budget"], "query_order": self.state["query_order"],
                "representation_updates_after_initial": len(updates) - 1,
                "representation_update_query_counts": [e["after_queries"] for e in updates],
                "gp_refits": self.state["gp_fit_count"],
                "gp_refit_query_counts": [e["after_queries"] for e in self.state["events"] if e["type"] == "gp_refit"],
                "representation_version": self.state["representation_version"],
                "pending_selected_ids": [] if self.state["pending"] is None else self.state["pending"]["selected_ids"],
                "vae_checkpoint_unchanged": True, "candidate_identity_stable": True,
                "use": "engineering_development_replay_only", "test_labels_used": False,
                "complete": len(self.state["query_order"]) == self.config["query_budget"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fuzzy_metric_stage1_v1.json")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-queries", type=int)
    parser.add_argument("--pause-before-reveal-batch", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["gnn_training_budget"] != 0 or config["batch_size"] != 5 or config["update_every"] != 10:
        parser.error("stage 1 requires zero GNN budget, batches of 5 and updates every 10 labels")
    if config["query_budget"] != 20 or config["initial_size"] != 50:
        parser.error("this entry point implements only the requested 50+20 engineering replay")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(config["model_seed"])
    np.random.seed(config["model_seed"])
    torch.manual_seed(config["model_seed"])
    if args.prepare:
        print(json.dumps(prepare(config, args.prepared), indent=2))
    else:
        if args.output is None:
            parser.error("--output is required for replay")
        replay = Replay(config, args.prepared, args.output, resume=args.resume)
        print(json.dumps(replay.run(stop_after_queries=args.stop_after_queries,
                                   pause_before_reveal_batch=args.pause_before_reveal_batch), indent=2))


if __name__ == "__main__":
    main()
