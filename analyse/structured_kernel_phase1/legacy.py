"""Read-only recovery of frozen A-only historical diagnostics. Never fits a model."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


RUNS = {
    "compressed": "dkl-stage2-v1-20260922T080157Z",
    "length": "dkl-stage2-length-v1-20260922T092924Z",
    "residual": "dkl-residual-constraint-v1-20260924T024011Z",
}
REQUESTED = (0, 10, 50, 100, 300, 1000)
DRAWS = (20260922, 20260923)
FAMILY_METHODS = {"compressed": ("RAW16", "FROZEN", "REG", "DKL"),
                  "residual": ("RAW16", "R_FROZEN", "R_REG", "R_DKL_C", "R_DKL_U"),
                  "length": ("DKL",)}
CTX_RE = re.compile(r"s(?P<draw>\d+)_m(?P<seed>\d+)_f(?P<fold>\d+)_n(?P<n>\d+)")


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def correlation_stats(matrix):
    """Noiseless prior correlation only, excluding diagonal from quantiles."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not np.isfinite(r).all():
        raise ValueError("Invalid prior correlation matrix")
    if not np.allclose(r, r.T, atol=1e-8, rtol=1e-6) or not np.allclose(np.diag(r), 1, atol=1e-8, rtol=1e-6):
        raise ValueError("Expected symmetric normalized noiseless prior correlation")
    off = r[np.triu_indices(len(r), 1)]
    return {"correlation_median": float(np.median(off)) if len(off) else None,
            "correlation_p95": float(np.quantile(off, .95)) if len(off) else None,
            "effective_rank": float(np.trace(r) ** 2 / np.sum(r * r.T)),
            "sample_count": len(r)}


def _context(value):
    match = CTX_RE.search(str(value))
    if match is None:
        raise ValueError(f"Unknown historical context identity: {value}")
    c = {k: int(v) for k, v in match.groupdict().items()}
    c["model_seed"] = c.pop("seed")
    c["context_id"] = match.group(0)
    c["subset_id"] = f's{c["draw"]}_f{c["fold"]}_n{c["n"]}'
    return c


def _base(family, method, context, step, path, sources):
    c = _context(context)
    if method == "RAW16":
        c["model_seed"] = None
    return {"family": family, "method": method, **c, "step": int(step),
            "source_path": str(path), "source_sha256": sources.record(path),
            "nll_timing": "after_update", "prior_kernel_definition": "R=K/outputscale; no observation noise"}


class Sources:
    def __init__(self, repo):
        self.repo = Path(repo)
        self.files = {}
        self.checks = []
        self.expected = {}
        self.current_root = None

    def record(self, path):
        path = Path(path)
        key = str(path.relative_to(self.repo))
        if key not in self.files:
            value = _sha(path)
            expected = self.expected.get(str(path))
            if expected is not None and value != expected:
                raise ValueError(f"Historical source hash mismatch: {key}")
            self.files[key] = value
        return self.files[key]

    def register_root(self, root):
        # Reading the hash index does not open or decode any listed labels.
        index = root / "FILES.sha256"
        if not index.exists():
            raise ValueError(f"Missing historical integrity index: {index}")
        self.record(index)
        for line in index.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            self.expected[str(root / relative)] = digest
        for name in ("protocol.json", "code_manifest.json", "frozen_manifest.json"):
            self.record(root / name)
        for relative, value in _read(root / "code_manifest.json")["files"].items():
            if self.record(root / relative) != value:
                raise ValueError(f"Frozen code differs: {relative}")


def audit_checkpoints(repo, sources):
    """Verify all 980 saved checkpoints against their exact, Q-disjoint T."""
    import torch
    old = Path(repo) / "artifacts" / RUNS["compressed"]
    public = _read(old / "public_inputs.json")
    folds = _read(old / "folds.json")["folds"]
    arch = dict(zip(public["ids"], public["architectures"]))
    result = {}
    for family, dirname in RUNS.items():
        root = Path(repo) / "artifacts" / dirname
        sources.register_root(root)
        snapshots = {}
        checked = 0
        for path in sorted((root / "checkpoints").glob("*.pt")):
            c = _context(path.name)
            sid = c["subset_id"]
            if sid not in snapshots:
                folder = "source/training" if family == "length" else "training"
                tpath = root / folder / (sid + ".json")
                sources.record(tpath)
                t = _read(tpath)
                qids = folds[c["fold"]]["query_ids"]
                if set(t["ids"]) & set(qids) or {arch[x] for x in t["ids"]} & {arch[x] for x in qids}:
                    raise ValueError(f"Historical T/Q architecture leakage: {sid}")
                meta = t.get("metadata", {})
                if meta.get("query_labels_released", 0) or meta.get("query_SE_released", 0) or meta.get("access", {}).get("hidden_label_rows_decoded", 0):
                    raise ValueError(f"Historical unapproved label release: {sid}")
                snapshots[sid] = t
            sources.record(path)
            state = torch.load(path, map_location="cpu", weights_only=False)
            for oldkey, newkey in (("x", "x_standardized"), ("y", "y_standardized"), ("known", "known_variance_standardized")):
                if not torch.equal(state[oldkey], torch.tensor(snapshots[sid][newkey], dtype=torch.float64)):
                    raise ValueError(f"Checkpoint/T mismatch: {path.name}:{oldkey}")
            if family == "length":
                expected_step = int(re.search(r"_j(\d+)\.pt$", path.name)[1])
                if state["continuation"]["total_joint_step"] != expected_step:
                    raise ValueError("Continuation step identity differs")
            checked += 1
        expected = {"compressed": 400, "length": 120, "residual": 460}[family]
        if checked != expected or len(snapshots) != 20:
            raise ValueError(f"Incomplete historical checkpoint inventory: {family}: {checked}")
        result[family] = {"checkpoints": checked, "training_snapshots": len(snapshots),
                          "tensor_equality": True, "architecture_disjoint": True}
    sources.checks.append({"checkpoint_audit": result, "total": 980})
    return result


@contextlib.contextmanager
def _old_core(code):
    """Import only frozen math modules; preserve caller RNG and module namespace."""
    import torch
    names = ("stage01_core", "stage2_core", "source_reference", "source_reference.fuzzy_metric")
    saved = {name: sys.modules.get(name) for name in names}
    rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    bytecode = sys.dont_write_bytecode
    path = list(sys.path)
    for name in names:
        sys.modules.pop(name, None)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    try:
        yield importlib.import_module("stage2_core")
    finally:
        sys.path[:] = path
        sys.dont_write_bytecode = bytecode
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]
        torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])


def _measure_model(core, state_path, training, query_x, query_ids, parameter_row=None):
    import gpytorch
    import torch
    model = core.restore(torch.load(state_path, map_location="cpu", weights_only=False))
    if parameter_row is not None:
        model.gp.mean_module.constant = parameter_row["mean"]
        model.gp.covar_module.base_kernel.lengthscale = parameter_row["lengthscale"]
        model.gp.covar_module.outputscale = parameter_row["outputscale"]
        model.gp.likelihood.second_noise = parameter_row["extra_noise_variance_standardized"]
        model.synchronize_history()
    gp = model.gp
    parameters = {"mean": float(gp.mean_module.constant.detach()),
                  "lengthscale": float(gp.covar_module.base_kernel.lengthscale.detach()),
                  "outputscale": float(gp.covar_module.outputscale.detach()),
                  "extra_noise_variance_standardized": float(gp.likelihood.second_noise.detach())}
    with torch.no_grad(), core.numerical_settings():
        hx = model.encode(model.x)[0]
        q = torch.tensor(query_x, dtype=torch.float64)
        hq = model.encode(q)[0]
        gp.train(); gp.likelihood.train()
        gp.set_train_data(inputs=hx, targets=model.y, strict=False)
        nll = float(-gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)(gp(hx), model.y))
        total = torch.cat([hx, hq])
        r = (gp.covar_module(total).to_dense() / parameters["outputscale"]).numpy()
        mu, var = model.predict(q)
        tt = torch.pdist(hx).numpy()
        qt = torch.cdist(hq, hx).numpy().ravel()
    sc = training["scaler"]
    row = {"train_nll": nll, **parameters,
           "mapping_distance_TT_median": float(np.median(tt)),
           "mapping_distance_QT_median": float(np.median(qt)),
           "effective_distance_TT_median": float(np.median(tt) / parameters["lengthscale"]),
           "effective_distance_QT_median": float(np.median(qt) / parameters["lengthscale"])}
    for name, matrix in (("T", r[:len(hx), :len(hx)]), ("TQ", r)):
        row.update({name + "_" + key: value for key, value in correlation_stats(matrix).items()})
    unit = {"ids": query_ids, "mean_raw": (mu.numpy() * sc["y_scale"] + sc["y_mean"]).tolist(),
            "latent_variance_standardized": var.numpy().tolist(), "y_scale": sc["y_scale"],
            "extra_noise_variance_standardized": parameters["extra_noise_variance_standardized"],
            "median_known_var_standardized": float(np.median(training["known_variance_standardized"]))}
    return row, unit


def _compressed(repo, sources, rows, units, missing):
    root = Path(repo) / "artifacts" / RUNS["compressed"]
    public = _read(root / "public_inputs.json")
    folds = _read(root / "folds.json")["folds"]
    positions = {cid: i for i, cid in enumerate(public["ids"])}
    traces_path = root / "training_traces.json"
    sources.record(traces_path)
    traces = {(x["run"], x["branch"], x["block"], x["step"]): x for x in _read(traces_path)}
    with _old_core(root / "code") as core:
        for endpoint in sorted((root / "predictions").glob("*.json")):
            sources.record(endpoint)
            pred = _read(endpoint)
            c = _context(pred["context_id"])
            method = pred["method"]
            training = _read(root / "training" / (c["subset_id"] + ".json"))
            ids = folds[c["fold"]]["query_ids"]
            if pred["candidate_ids"] != ids:
                raise ValueError("Historical prediction Q order differs")
            raw = np.array([public["z"][positions[x]] + public["hp"][positions[x]] for x in ids])
            sc = training["scaler"]
            qx = (raw - np.array(sc["x_mean"])) / np.array(sc["x_scale"])
            cid = c["context_id"]
            block = {"RAW16": "raw_continue", "FROZEN": "gp", "REG": "gp", "DKL": "joint"}[method]
            endpath = root / "checkpoints" / f"{cid}_{method}_{block}.pt"
            start_suffix = {"RAW16": "RAW16_raw_initial", "FROZEN": "COMMON_gp", "DKL": "COMMON_gp", "REG": "REG_regression"}[method]
            startpath = root / "checkpoints" / f"{cid}_{start_suffix}.pt"
            values, unit = _measure_model(core, endpath, training, qx, ids)
            for key in ("mean_raw", "latent_variance_standardized"):
                if not np.allclose(unit[key], pred[key], atol=1e-8, rtol=1e-6):
                    raise ValueError(f"Historical checkpoint posterior reproduction failed: {endpoint}:{key}")
            # Fixed-map reconstruction is admitted only after same logged scalar
            # restoration reproduces the frozen endpoint, including its variance.
            if method != "DKL":
                t = traces[(cid, method, block, 1000)]
                _, reconstructed = _measure_model(core, endpath, training, qx, ids, t)
                for key in ("mean_raw", "latent_variance_standardized"):
                    if not np.allclose(reconstructed[key], pred[key], atol=1e-8, rtol=1e-6):
                        raise ValueError("Fixed-map scalar-log endpoint reconstruction failed")
            for step in (0, 100, 300, 1000):
                path = startpath if step == 0 else endpath
                row = _base("compressed", method, cid, step, path, sources)
                row.update(T_ids_sha256=_digest(training["ids"]), Q_ids_sha256=_digest(ids),
                           stage="final_gp_or_joint_block", source_endpoint=str(endpoint))
                trace = traces.get((cid, method, block, step))
                if trace:
                    row["train_nll_before_update"] = trace.get("data_nll")
                    row["trace_nll_timing"] = "before_update_step_" + str(step)
                if method == "DKL" and step in (100, 300):
                    row.update({k: trace[k] for k in ("mean", "lengthscale", "outputscale", "extra_noise_variance_standardized", "embedding_rms", "embedding_coordinate_std_mean")})
                    row.update(nll_timing="no_post_update_NLL", available="partial_training_trace",
                               source_path=str(traces_path), source_sha256=sources.record(traces_path))
                    missing.append({"family": "compressed", "method": method, "context_id": cid, "step": step,
                                    "missing": "post_update_NLL,Q_posterior,prior_R,exact_distance", "reason": "Intermediate learned mapper overwritten; no retraining"})
                else:
                    values, unit = _measure_model(core, path, training, qx, ids, trace if step in (100, 300) else None)
                    row.update(values, available="complete", recovery="fixed_mapper_scalar_log" if step in (100, 300) else "saved_checkpoint")
                    unit.update({k: row[k] for k in ("method", "n", "fold", "draw")})
                    units.append((len(rows), unit))
                rows.append(row)
            for step in (10, 50):
                missing.append({"family": "compressed", "method": method, "context_id": cid, "step": step,
                                "missing": "all", "reason": "No saved record/checkpoint at requested update; not reconstructed by training"})


def _saved_diagnostics(repo, family, sources, rows, units, missing):
    root = Path(repo) / "artifacts" / RUNS[family]
    for path in sorted((root / "diagnostics").glob("*/j*.json")):
        d = _read(path)
        if family == "residual":
            name = path.parent.name
            if not any(name.endswith("_" + suffix) for suffix in ("RAW16_final", "R_FROZEN_gp", "R_REG_gp", "R_DKL_C_joint", "R_DKL_U_joint")):
                continue
            method = name.split("_n", 1)[1].split("_", 1)[1].rsplit("_", 1)[0]
            step = d["block_step"]
            stage = "final_gp_or_joint_block"
        else:
            method, step, stage = "DKL", d["total_joint_step"], "total_joint_updates"
        c = _context(path.parent.name)
        folder = "source/training" if family == "length" else "training"
        training = _read(root / folder / (c["subset_id"] + ".json"))
        row = _base(family, method, c["context_id"], step, path, sources)
        ids = d["query_ids"]
        row.update(stage=stage, available="complete", recovery="saved_diagnostic", train_nll=d["nll_after_update"],
                   T_ids_sha256=_digest(training["ids"]), Q_ids_sha256=_digest(ids), **d["gp_parameters"])
        matrix_path = root / d.get("kernel_arrays_path", d.get("kernel_path", ""))
        digest = sources.record(matrix_path)
        if digest != d.get("kernel_arrays_sha256", d.get("kernel_sha256")):
            raise ValueError("Historical kernel matrix hash differs")
        with np.load(matrix_path, allow_pickle=False) as arrays:
            for key, prefix in (("training_kernel", "T"), ("combined_kernel", "TQ")):
                r = arrays[key] / d["gp_parameters"]["outputscale"]
                row.update({prefix + "_" + k: v for k, v in correlation_stats(r).items()})
        geometry = d.get("geometry_summary", {})
        for kind, prefix in (("distance_stats", "mapping_distance"), ("effective_distance_stats", "effective_distance")):
            for pair, stats in (geometry.get(kind) or {}).items():
                if isinstance(stats, dict) and "median" in stats:
                    row[prefix + "_" + pair + "_median"] = stats["median"]
        if family == "length":
            row["embedding_rms"] = d.get("representation", {}).get("training_rms")
            row["mapping_distance_status"] = "not_saved_at_intermediate_steps; endpoint_checkpoint_recoverable"
        unit = {"ids": ids, "mean_raw": d["query_mean_raw"],
                "latent_variance_standardized": d["query_latent_variance_standardized"],
                "extra_noise_variance_standardized": d["extra_noise_variance_standardized"],
                "y_scale": d["y_scale"],
                "median_known_var_standardized": float(np.median(training["known_variance_standardized"])),
                **{k: row[k] for k in ("method", "n", "fold", "draw")}}
        units.append((len(rows), unit)); rows.append(row)
    if family == "residual":
        identities = {(row["method"], row["context_id"]) for row in rows if row["family"] == family}
        for method, cid in sorted(identities):
            for step in (10, 50):
                missing.append({"family": family, "method": method, "context_id": cid, "step": step,
                                "missing": "all", "reason": "Diagnostics and sparse update logs only every 100 steps; no retraining"})
    else:
        identities = {row["context_id"] for row in rows if row["family"] == family}
        for cid in sorted(identities):
            for step in (0, 10, 50, 100, 300):
                missing.append({"family": family, "method": "DKL", "context_id": cid, "step": step,
                                "missing": "not_a_continuation_timepoint", "reason": "Long-run source starts at total joint update 1000; never relabel additional updates as total updates"})


def _long_endpoint_distances(repo, sources, rows, units):
    """Recover actual embedding distances only where preserved models exist."""
    root = Path(repo) / "artifacts" / RUNS["length"]
    code = Path(repo) / "artifacts" / RUNS["compressed"] / "code"
    with _old_core(code) as core:
        for ix, unit in units:
            row = rows[ix]
            if row["family"] != "length" or row["step"] not in (1000, 2000, 4000):
                continue
            cid, sid, step = row["context_id"], row["subset_id"], row["step"]
            training = _read(root / "source/training" / (sid + ".json"))
            query_path = root / "source/queries" / f'fold{row["fold"]}.json'
            sources.record(query_path)
            query = _read(query_path)
            if query["ids"] != unit["ids"]:
                raise ValueError("Long-run checkpoint query identity differs")
            sc = training["scaler"]
            qx = (np.asarray(query["x_raw"]) - np.asarray(sc["x_mean"])) / np.asarray(sc["x_scale"])
            checkpoint = root / (f"source/checkpoints/{cid}_DKL_joint.pt" if step == 1000 else f"checkpoints/{cid}_j{step}.pt")
            sources.record(checkpoint)
            values, recovered = _measure_model(core, checkpoint, training, qx, query["ids"])
            for key in ("mean_raw", "latent_variance_standardized"):
                if not np.allclose(recovered[key], unit[key], atol=1e-8, rtol=1e-6):
                    raise ValueError("Long-run checkpoint/diagnostic posterior differs")
            row.update({k: v for k, v in values.items() if k.startswith(("mapping_distance_", "effective_distance_"))})
            row["mapping_distance_status"] = "recovered_from_saved_endpoint_checkpoint"
            row["distance_checkpoint_path"] = str(checkpoint)


def score_legacy_unit(unit, labels):
    ids = unit["ids"]
    y = np.array([labels[x]["mean"] for x in ids], dtype=np.float64)
    mu = np.asarray(unit["mean_raw"], dtype=np.float64)
    latent = np.asarray(unit["latent_variance_standardized"], dtype=np.float64)
    if np.any(latent < -1e-8):
        raise ValueError("Invalid historical posterior variance")
    var = (latent + unit["extra_noise_variance_standardized"] + unit["median_known_var_standardized"]) * unit["y_scale"] ** 2
    if not np.isfinite(var).all() or np.any(var <= 0):
        raise ValueError("Invalid prospective legacy predictive variance")
    true_top = sorted(range(len(ids)), key=lambda i: (-y[i], ids[i]))[:math.ceil(.1 * len(ids))]
    pred_top = sorted(range(len(ids)), key=lambda i: (-mu[i], ids[i]))[:5]
    return {"q_mae_pp": float(np.mean(np.abs(mu-y)) * 100),
            "q_nlpd_prospective": float(np.mean(.5 * (np.log(2*np.pi*var) + (mu-y)**2 / var))),
            "bias_true_top10_pp": float(np.mean((mu-y)[true_top]) * 100),
            "bias_pred_top5_pp": float(np.mean((mu-y)[pred_top]) * 100)}


def seed_disagreement(rows, units):
    groups = defaultdict(list)
    for ix, unit in units:
        row = rows[ix]
        key = tuple(row[k] for k in ("family", "method", "draw", "fold", "n", "step"))
        if row["model_seed"] is not None:
            groups[key].append((ix, unit))
    for group in groups.values():
        if {rows[ix]["model_seed"] for ix, _ in group} != {0, 1, 2} or len(group) != 3:
            continue
        ids = group[0][1]["ids"]
        if any(unit["ids"] != ids for _, unit in group):
            raise ValueError("Seed disagreement Q identities differ")
        sd = np.std([unit["mean_raw"] for _, unit in group], axis=0, ddof=0) * 100
        for ix, _ in group:
            rows[ix]["seed_prediction_sd_Q_mean_pp"] = float(sd.mean())
            rows[ix]["seed_prediction_sd_Q_p95_pp"] = float(np.quantile(sd, .95))
            rows[ix]["seed_prediction_sd_ddof"] = 0


def aggregate_mechanisms(rows):
    """Average seeds, then draws, then equally weighted folds; descriptive only."""
    metrics = ("train_nll", "q_mae_pp", "q_nlpd_prospective", "T_effective_rank", "TQ_effective_rank")
    full = [r for r in rows if all(isinstance(r.get(k), (int, float)) for k in metrics)]
    by_context = defaultdict(dict)
    for row in full:
        key = tuple(row[k] for k in ("family", "method", "n", "fold", "draw", "model_seed"))
        by_context[key][row["step"]] = row
    deltas = []
    for key, points in by_context.items():
        a, b = (1000, 4000) if key[0] == "length" else (0, 1000)
        if a in points and b in points:
            delta = dict(zip(("family", "method", "n", "fold", "draw", "model_seed"), key))
            delta.update(from_step=a, to_step=b)
            delta.update({"delta_" + k: points[b][k] - points[a][k] for k in metrics})
            deltas.append(delta)
    completeness = []
    def reduce(data, keys, member):
        groups = defaultdict(list)
        for row in data:
            groups[tuple(row[k] for k in keys)].append(row)
        out = []
        for key, group in sorted(groups.items()):
            d = dict(zip(keys, key))
            expected = ({None} if d["method"] == "RAW16" else {0, 1, 2}) if member == "model_seed" else (set(DRAWS) if member == "draw" else set(range(5)))
            found = [x[member] for x in group]
            if len(found) != len(expected) or set(found) != expected:
                completeness.append({**d, "stage": member, "expected": sorted(expected, key=str), "found": sorted(found, key=str)})
                continue
            d.update({"delta_" + k: float(np.mean([x["delta_"+k] for x in group])) for k in metrics})
            out.append(d)
        return out
    draw_rows = reduce(deltas, ("family", "method", "n", "fold", "draw"), "model_seed")
    fold_rows = reduce(draw_rows, ("family", "method", "n", "fold"), "draw")
    means = reduce(fold_rows, ("family", "method", "n"), "fold")
    for row in means:
        selected = [r for r in fold_rows if all(r[k] == row[k] for k in ("family", "method", "n"))]
        selected_draws = [r for r in draw_rows if all(r[k] == row[k] for k in ("family", "method", "n"))]
        row["fold_count"] = len(selected)
        row["supporting_folds_NLL_improves_MAE_worsens_T_rank_drops"] = sum(r["delta_train_nll"] < 0 and r["delta_q_mae_pp"] > 0 and r["delta_T_effective_rank"] < 0 for r in selected)
        row["supporting_folds_NLL_improves_NLPD_worsens_T_rank_drops"] = sum(r["delta_train_nll"] < 0 and r["delta_q_nlpd_prospective"] > 0 and r["delta_T_effective_rank"] < 0 for r in selected)
        row["draw_context_count"] = len(selected_draws)
        row["supporting_draw_contexts_NLL_improves_MAE_worsens_T_rank_drops"] = sum(r["delta_train_nll"] < 0 and r["delta_q_mae_pp"] > 0 and r["delta_T_effective_rank"] < 0 for r in selected_draws)
        row["from_step"], row["to_step"] = (1000, 4000) if row["family"] == "length" else (0, 1000)
    return {"aggregation": "model seeds mean -> draws mean -> five folds equal mean; no significance test",
            "means": means, "fold_deltas": fold_rows, "draw_deltas": draw_rows,
            "complete_seed_context_pairs": len(deltas), "aggregation_missing": completeness,
            "interpretation": "Joint directional changes support an explanation; they do not prove causal collapse."}


def aggregate_endpoints(rows, completeness=None):
    completeness = [] if completeness is None else completeness
    keys = ("q_mae_pp", "q_nlpd_prospective", "bias_true_top10_pp", "bias_pred_top5_pp",
            "score_coverage90", "score_width90_pp", "score_is90_pp", "train_nll",
            "T_effective_rank", "TQ_effective_rank", "seed_prediction_sd_Q_mean_pp")
    selected = [r for r in rows if r.get("available") == "complete" and
                r["step"] in ((1000, 2000, 4000) if r["family"] == "length" else (1000,))]
    for grouping, member in ((("family", "method", "n", "step", "fold", "draw"), "model_seed"),
                             (("family", "method", "n", "step", "fold"), "draw"),
                             (("family", "method", "n", "step"), "fold")):
        groups = defaultdict(list)
        for row in selected:
            groups[tuple(row[k] for k in grouping)].append(row)
        result = []
        for identity, group in sorted(groups.items()):
            row = dict(zip(grouping, identity))
            expected = ({None} if row["method"] == "RAW16" else {0, 1, 2}) if member == "model_seed" else (set(DRAWS) if member == "draw" else set(range(5)))
            found = [r[member] for r in group]
            if len(found) != len(expected) or set(found) != expected:
                completeness.append({**row, "stage": member, "expected": sorted(expected, key=str), "found": sorted(found, key=str)})
                continue
            for key in keys:
                values = [r.get(key) for r in group]
                row[key] = None if any(v is None for v in values) else float(np.mean(values))
            result.append(row)
        selected = result
    return selected


def validate_diagnostic_inventory(rows):
    observed = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("family", "method", "n", "fold", "draw", "model_seed"))
        observed[key].append(row["step"])
    failures = []
    for family, methods in FAMILY_METHODS.items():
        expected_steps = set(range(1000, 4001, 100)) if family == "length" else (set(range(0, 1001, 100)) if family == "residual" else {0, 100, 300, 1000})
        for method in methods:
            for n in (20, 50):
                for fold in range(5):
                    for draw in DRAWS:
                        for seed in ((None,) if method == "RAW16" else (0, 1, 2)):
                            key = (family, method, n, fold, draw, seed)
                            found = observed.pop(key, [])
                            if len(found) != len(expected_steps) or set(found) != expected_steps:
                                failures.append(dict(zip(("family", "method", "n", "fold", "draw", "model_seed"), key),
                                    missing_steps=sorted(expected_steps-set(found)), unexpected_steps=sorted(set(found)-expected_steps),
                                    duplicate_steps=len(found)-len(set(found))))
    for key in observed:
        failures.append({"unexpected_context": list(key)})
    return failures


def extract_legacy_diagnostics(run_dir: Path, repo: Path):
    """Audit and extract without changing old sources, running fits, or reading B/test."""
    run_dir, repo = Path(run_dir), Path(repo)
    sources, rows, units, missing = Sources(repo), [], [], []
    audit = audit_checkpoints(repo, sources)
    _compressed(repo, sources, rows, units, missing)
    _saved_diagnostics(repo, "residual", sources, rows, units, missing)
    _saved_diagnostics(repo, "length", sources, rows, units, missing)
    _long_endpoint_distances(repo, sources, rows, units)
    # Everything above uses only T snapshots, frozen states and unlabeled Q.
    from .data import read_A_labels
    from .score import score_prediction
    labels = read_A_labels(repo)
    for ix, unit in units:
        row = rows[ix]
        unit["setting_id"] = f'legacy_{row["family"]}_{row["context_id"]}_{row["method"]}_j{row["step"]}'
        unit["model_hash"] = row["source_sha256"]
        rows[ix].update(score_legacy_unit(unit, labels))
        external = score_prediction(unit, labels)
        rows[ix].update({"score_" + k: v for k, v in external.items() if v is None or isinstance(v, (str, int, float, bool))})
    seed_disagreement(rows, units)
    inventory_missing = validate_diagnostic_inventory(rows)
    endpoint_missing = []
    endpoint_aggregates = aggregate_endpoints(rows, endpoint_missing)
    mechanisms = aggregate_mechanisms(rows)
    unexpected_missing = bool(inventory_missing or endpoint_missing or mechanisms["aggregation_missing"])
    summary = {"status": "INCOMPLETE_DIAGNOSTICS" if unexpected_missing else "COMPLETE_WITH_DOCUMENTED_MISSING_HISTORY", "new_optimizer_updates": 0,
               "checkpoint_audit": audit, "source_files_hashed": len(sources.files),
               "diagnostic_rows": len(rows), "scored_frozen_predictions": len(units),
               "missing_rows": len(missing), "mechanisms": mechanisms,
               "endpoint_aggregates": endpoint_aggregates, "endpoint_aggregation_missing": endpoint_missing,
               "diagnostic_inventory_missing": inventory_missing, "complete_historical_diagnostic_inventory": not unexpected_missing,
               "old_scoring_B_opened": False, "B_labels_opened": False, "test_scores_opened": False,
               "legacy_noise_scoring": "Old reported Q-SE metrics are retrospective; all current primary legacy metrics recomputed with median_T known variance.",
               "model_family": "ConstantMean; isotropic Matern5/2; FixedNoise plus standardized extra variance; distinct from production ARD RBF",
               "budget_warning": "Historical Adam budgets differ from new matched three-start RAW16; historical metrics cannot substitute for E2 RAW16."}
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "mapping_diagnostics.csv", rows)
    _write_csv(run_dir / "legacy_missing.csv", missing)
    _write_json(run_dir / "legacy_source_manifest.json", {"files": sources.files, "checks": sources.checks,
                "provenance": "Frozen initialization/training code chain plus file hashes and exact checkpoint/T tensors verified; not an independent historical process trace."})
    _write_json(run_dir / "legacy_summary.json", summary)
    return summary
