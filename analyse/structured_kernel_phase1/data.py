"""A-only E0 verification and physically separate T/Q input exports.

Only the trusted prepare/scoring process calls read_A_labels. A fitting worker
uses release_training and query_features; neither opens a full-label manifest.
No source artifact is rewritten, imported as executable code, or re-encoded.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from .features import build_operator_graph, directed_wl, fit_distance_scales, graph_hash, isomorphic_operator_graph


SOURCE = Path("results/posthoc/fuzzy-scheduler-regularization-v1-20260921T064826Z")
STAGE2 = Path("artifacts/dkl-stage2-v1-20260922T080157Z")
PREPARED = Path("results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2")
PARTITION = Path("results/posthoc/fuzzy-metric-stage2-20260910-v1/partition.json")
PREREGISTRATION = Path("results/search/high-regime-prior-model-blind-validation-global4-v1/preregistration.json")
VAE = Path("checkpoints/joint_model_pipeline_global4_best.pth")
VAE_LOG = Path("legacy_artifacts/pre_20260813/logs-wgmm1/train_joint/train_joint/train_pipeline_0612_0609")
PUBLIC_HASHES = {
    "features.json": "4aedd1482cdc36c055c4016a25fe4a8158c305321673e16e868b994ab830c797",
    "candidate_identity.json": "c66c2aa70c4cf2d5fe7d6462bd822a35c4fdd64e950b326cfce8fcf013cab46e",
    "recipe_verified.json": "889879271d809e09b4acb5defa2be6d49736ce62f4b2b4efa8b035bd48761d5c",
    "data_checks.json": "b4ba8a976468baba3b2092c25a9b31680a217948e3d190b0f08f6a4d0ecd7429",
}
A_ID_HASH = "d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2"
A_LABEL_HASH = "d6823a6dbe62507f4fdc53de01a69e33cdd3ec8de4fc0aae0cca89563766fa58"
RECIPE_HASH = "40bb99cbc3e33dcc417869a97d969b8e8e1c4ca7c6a5fad66c29ed85edea5978"
VAE_HASH = "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4"
FOLD_IDENTITY = "53f882ca97d2a378e7e56044497c34f3ee98dae8ac5585d0b1634ce67a086a84"
SUBSET_IDENTITY = "3968393a212773cba41da66424965707dfa377c9a7f0328e7c993116c05b383b"
EXTRA_HASHES = {
    PARTITION: "dba3ef021d1d13417b4eedbf83730a8862f4bc7de8a8b831ccb58cd115ab5643",
    PREREGISTRATION: "098240232515aacfb47f8f557765bb3c824faf37d60b8ae680f6e1b2cf2c7260",
    VAE: VAE_HASH,
    PREPARED / "provenance.json": "0d4bb619ef8856383f3a5822ce5c2cb1d28fe773e8cd29e2b9b7cb9a59f19b81",
    Path("legacy/phased_cora_pipeline/train_joint.py"): "da96e8e6524ae94fc06954495d698b12202f90dbe8dbf713040087e9c77a91c2",
    Path("nas_space.py"): "386514dac7bbffabea45bd8a2663bf6da8b740b3fd55eef361aa2ae14350727b",
    VAE_LOG.with_suffix(".json"): "d5bd3f3ebe9b4d49a91579f0aa878e250b81663df4bf0c09a466607c693fc0e9",
    VAE_LOG.with_suffix(".log"): "5be14f7a8142655521300b8d43a95b30bc8fecec59502e3829989d2cc6700805",
    Path("artifacts/reference/reference_manifest.json"): "692401f0165fec979d07788b135655ebe39dd435654ed0ef280385d7999376a5",
}
LABEL_KEYS = {"validation_mean", "validation_se", "validation_seeds", "training_seeds"}
CANDIDATE_KEYS = {
    "candidate_id", "joint_fingerprint", "architecture_fingerprint", "hp_fingerprint",
    "operations", "edges", "raw_hp", "lr", "dropout", "hidden_dim", "l2",
    "training_protocol_identity", "source_file", "source_lines", "raw_hp_vector",
    "hp_mode", "normalization", "gat_heads", "sage_aggr", "gin_eps",
    "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer", "gcnii_alpha",
    "gcnii_theta", "effective_layers", "source_config_id",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError("BLOCKED: " + message)


def _verify(path, expected):
    _require(Path(path).is_file(), "missing frozen source " + str(path))
    actual = _sha(path)
    _require(actual == expected, "source hash changed: " + str(path))
    return actual


def _identity(value):
    _require(value.get("identity") == _digest({k: v for k, v in value.items() if k != "identity"}),
             "semantic manifest identity changed")


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_text() == text, "refusing to replace different E0 artifact: " + str(path))
        return
    # Exclusive creation also avoids overwriting a concurrently created file.
    with path.open("x") as stream:
        stream.write(text)


def _write_json(path, value):
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _safe_existing(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), "manifest path leaves frozen directory")
    return path


def validate_A_labels(ids, rows) -> dict:
    """Reject any non-A row or non-validation field; recompute all mean/SE."""
    _require(len(ids) == len(set(ids)) and set(rows) == set(ids), "A label IDs changed")
    result = {}
    for cid in ids:
        row = rows[cid]
        _require(set(row) == LABEL_KEYS and row["training_seeds"] == [0, 1, 2],
                 "A validation schema or training seeds changed")
        values = np.asarray(row["validation_seeds"], dtype=np.float64)
        mean, se = float(row["validation_mean"]), float(row["validation_se"])
        _require(values.shape == (3,) and np.isfinite(values).all() and
                 bool(((values >= 0) & (values <= 1)).all()) and
                 np.isfinite([mean, se]).all() and 0 <= mean <= 1 and se >= 0,
                 "invalid A validation observation")
        _require(np.isclose(mean, values.mean(), atol=1e-12, rtol=0) and
                 np.isclose(se, values.std(ddof=1) / math.sqrt(3), atol=1e-12, rtol=0),
                 "mean/SE disagree with three-seed validation observations")
        result[cid] = {"mean": mean, "se": se, "seeds": values.tolist()}
    return result


def read_A_labels(repo: Path) -> dict:
    """Trusted scorer/E0 only: decode the byte-pinned A-only validation file."""
    repo = Path(repo).resolve()
    source = repo / SOURCE
    _verify(source / "features.json", PUBLIC_HASHES["features.json"])
    public = _json(source / "features.json")
    _require(set(public) == {"ids", "architectures", "z", "hp"}, "non-public input fields")
    ids = public["ids"]
    _require(len(ids) == 179 and _digest(sorted(ids)) == A_ID_HASH, "A whitelist changed")
    path = source / "private/evaluation_A.json"
    _verify(path, A_LABEL_HASH)
    return validate_A_labels(ids, _json(path))


def training_projection(ids, x_raw, y, known_var, wl, *, subset_id, fold, draw) -> dict:
    """Fit all data transforms exclusively to the supplied T rows, ddof=0."""
    x = np.asarray(x_raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    known_var = np.asarray(known_var, dtype=np.float64)
    wl = np.asarray(wl, dtype=np.float64)
    n = len(ids)
    _require(x.shape == (n, 16) and y.shape == known_var.shape == (n,), "invalid T shapes")
    _require(wl.ndim == 2 and len(wl) == n, "invalid T WL shape")
    _require(all(np.isfinite(a).all() for a in (x, y, known_var, wl)) and
             bool((known_var >= 0).all()), "nonfinite T or negative known variance")
    xm, xs = x.mean(axis=0), x.std(axis=0, ddof=0)
    # Floating mean accumulation can give a tiny nonzero std for exact constants.
    constant_mask = np.ptp(x, axis=0) == 0
    constant = np.flatnonzero(constant_mask).tolist()
    xs[constant_mask] = 1.0
    _require(np.isfinite(xm).all() and np.isfinite(xs).all() and (xs > 0).all(), "invalid T input scale")
    ym, ys = float(y.mean()), float(y.std(ddof=0))
    _require(np.isfinite(ys) and ys > 0 and np.ptp(y) > 0, "zero/nonfinite T y scale: " + subset_id)
    standardized = (x - xm) / xs
    return {
        "ids": list(ids), "x_raw": x.tolist(), "x_standardized": standardized.tolist(),
        "y_standardized": ((y - ym) / ys).tolist(),
        "known_variance_standardized": (known_var / ys ** 2).tolist(),
        "wl": wl.tolist(), "fold": int(fold), "draw": int(draw), "n": n,
        "subset_id": subset_id,
        "scaler": {"x_mean": xm.tolist(), "x_scale": xs.tolist(), "y_mean": ym,
                   "y_scale": ys, "std_correction": 0, "fit_ids": list(ids),
                   "constant_columns": constant},
        "distance_scales": fit_distance_scales(standardized, wl),
        "known_variance_semantics": "raw variance=SE**2; divided by T_y_scale**2 once",
    }


def query_projection(rows) -> dict:
    """Whitelist avoids touching even deliberately poisoned Q label fields."""
    result = {"ids": [row["candidate_id"] for row in rows],
              "x_raw": [row["x_raw"] for row in rows],
              "wl": [row["wl"] for row in rows],
              "architecture_ids": [row["architecture_id"] for row in rows]}
    _require(len(result["ids"]) == len(set(result["ids"])), "duplicate Q IDs")
    x, w = np.asarray(result["x_raw"]), np.asarray(result["wl"])
    _require(x.shape == (len(rows), 16) and w.ndim == 2 and len(w) == len(rows) and
             np.isfinite(x).all() and np.isfinite(w).all(), "invalid unlabelled Q features")
    return result


def _verify_historical_freeze(repo):
    root = repo / STAGE2
    frozen = _json(root / "frozen_manifest.json")
    _require(frozen["labels_released"] is False, "historical freeze was not before label release")
    hashes = {}
    for name, expected in frozen["files"].items():
        path = _safe_existing(root, name)
        hashes[str(path.relative_to(repo))] = _verify(path, expected)
    _verify(root / "protocol.json", frozen["protocol_sha256"])
    _verify(root / "code_manifest.json", frozen["code_manifest_sha256"])
    codes = _json(root / "code_manifest.json")
    for name, expected in codes["files"].items():
        path = _safe_existing(root, name)
        hashes[str(path.relative_to(repo))] = _verify(path, expected)
    hashes[str(STAGE2 / "frozen_manifest.json")] = _sha(root / "frozen_manifest.json")
    return hashes


def _load_sources(repo):
    source = repo / SOURCE
    hashes = {str(SOURCE / name): _verify(source / name, value)
              for name, value in PUBLIC_HASHES.items()}
    hashes.update({str(name): _verify(repo / name, value) for name, value in EXTRA_HASHES.items()})
    public, candidates, recipe, checks = [_json(source / name) for name in PUBLIC_HASHES]
    _require(set(public) == {"ids", "architectures", "z", "hp"}, "public features schema")
    ids = public["ids"]
    _require(len(ids) == len(set(ids)) == 179 and _digest(sorted(ids)) == A_ID_HASH,
             "A candidate universe differs")
    _require(len(set(public["architectures"])) == 96, "A architecture count differs")
    _require([c["candidate_id"] for c in candidates] == ids, "candidate order differs")
    _require(_digest(recipe["training_protocol_identity"]) == RECIPE_HASH, "recipe identity differs")
    recipe_identity = recipe["training_protocol_identity"]
    expected_recipe = {"dataset": "cora", "eval_epochs": 300, "patience": 80,
                       "graph_transform": "NormalizeFeatures", "hp_mode": "global4",
                       "normalization": "none", "training_mode": "full_batch", "metric": "accuracy"}
    _require(all(recipe_identity[k] == v for k, v in expected_recipe.items()), "300/80 recipe differs")
    train = _json(repo / PREREGISTRATION)["training"]
    _require(train == recipe["actual_preregistration_training"] and train["split"] == "public" and
             train["track_test"] is False and train["max_epochs"] == 300 and train["patience"] == 80,
             "actual preregistration differs from cached recipe")
    historical = _json(source / "protocol.json")
    _identity(historical)
    _require(all(historical["inputs"][k] == v for k, v in PUBLIC_HASHES.items()),
             "historical public provenance differs")
    _require(historical["private_inputs"]["private/evaluation_A.json"] == A_LABEL_HASH,
             "historical A label provenance differs")
    hashes[str(SOURCE / "protocol.json")] = _sha(source / "protocol.json")
    graphs, config_identities = {}, set()
    expected_fixed = {"hp_mode": "global4", "normalization": "none", "gat_heads": 1,
                      "sage_aggr": "mean", "gin_eps": 0.0, "gcnii_alpha": 0.1, "gcnii_theta": 0.5,
                      "gat_heads_by_layer": None, "sage_aggr_by_layer": None, "gin_eps_by_layer": None}
    for i, c in enumerate(candidates):
        _require(set(c) == CANDIDATE_KEYS, "candidate schema changed")
        _require(all(c[k] == v for k, v in expected_fixed.items()), "fixed structural attribute changed")
        arch = _digest({"operations": c["operations"], "edges": sorted(c["edges"])})
        hp = _digest([float(v) for v in c["raw_hp_vector"]])
        joint = _digest({"architecture_fingerprint": arch, "hp_fingerprint": hp})
        _require(arch == c["architecture_fingerprint"] == public["architectures"][i] and
                 hp == c["hp_fingerprint"] and joint == c["candidate_id"] == c["joint_fingerprint"],
                 "candidate identity does not match actual configuration")
        _require(c["training_protocol_identity"] == RECIPE_HASH, "mixed candidate recipe")
        raw = c["raw_hp"]
        _require(set(raw) == {"lr", "dropout", "hidden_dim", "l2"} and
                 all(c[k] == raw[k] for k in raw), "inconsistent decoded HP")
        transformed = [math.log10(raw["lr"]), raw["dropout"], math.log2(raw["hidden_dim"]), math.log10(raw["l2"])]
        _require(transformed == public["hp"][i], "HP has been transformed incorrectly")
        g = build_operator_graph(c)
        _require(arch not in graphs or graphs[arch] == g, "one architecture ID maps to different graphs")
        graphs[arch] = g
        config_identities.add(_digest({"architecture": arch, "hp": raw}))
    _require(len(config_identities) == 179, "duplicate actual decoded configuration")
    x = np.c_[np.asarray(public["z"], dtype=np.float64), np.asarray(public["hp"], dtype=np.float64)]
    _require(np.asarray(public["z"]).shape == (179, 12) and
             np.asarray(public["hp"]).shape == (179, 4) and np.isfinite(x).all(), "invalid latent/HP arrays")
    prepared = _json(repo / PREPARED / "manifest.json")
    raw_candidates = _json(repo / PREPARED / "candidates.json")
    encodings = _json(repo / PREPARED / "base_encoding.json")
    _require(prepared == checks["prepared_manifest"] and
             _digest(raw_candidates) == prepared["pool_hash"] and
             _digest(encodings) == prepared["base_encoding_hash"], "prepared_v2 source identity differs")
    _require(all(set(c) == CANDIDATE_KEYS for c in raw_candidates), "non-public prepared metadata")
    by_id = {c["candidate_id"]: c for c in raw_candidates}
    _require(len(by_id) == len(raw_candidates) and all(c == by_id[c["candidate_id"]] for c in candidates),
             "A configurations differ from prepared_v2")
    _require(all(z == encodings[a] for z, a in zip(public["z"], public["architectures"])),
             "A latent cache differs from frozen base encoding")
    _require(prepared["checkpoint_sha256"] == VAE_HASH, "VAE source checkpoint differs")
    provenance = _json(repo / PREPARED / "provenance.json")
    _require(provenance["training_protocol_identity"] == recipe_identity and
             recipe["source_provenance_sha256"] == hashes[str(PREPARED / "provenance.json")],
             "GNN source provenance differs")
    for entry in provenance["shards"] + [provenance["artifact_manifest"]]:
        # Hash opaque source bytes only: never deserialize the mixed-label shard.
        path = Path(entry["path"])
        _require(path.resolve().is_relative_to(repo), "provenance file outside audited repository")
        hashes[str(path.relative_to(repo))] = _verify(path, entry["sha256"])
    for name in ("manifest.json", "candidates.json", "base_encoding.json"):
        hashes[str(PREPARED / name)] = _sha(repo / PREPARED / name)
    reference = _json(repo / "artifacts/reference/reference_manifest.json")
    _require(any(r["path"] == str(VAE) and r["sha256"] == VAE_HASH for r in reference["resources"]),
             "VAE migration provenance differs")
    original_config = _json(repo / VAE_LOG.with_suffix(".json"))
    _require(original_config["data"] == "data/mini_gnn_dataset_global4.pkl" and
             original_config["epochs"] == 100 and original_config["seed"] == 42 and
             original_config["hp_mode"] == "global4", "original VAE training config differs")
    _require("best checkpoint path: results/joint_search/joint_model_pipeline_global4_best.pth" in
             (repo / VAE_LOG.with_suffix(".log")).read_text(), "original VAE output log differs")
    hashes.update(_verify_historical_freeze(repo))
    # These user-edited entrypoints are recorded, not overwritten or executed.
    for name in ("bo_phase4.py", "eval_utils.py", "surrogate/accuracy_gp.py"):
        hashes[name] = _sha(repo / name)
    return public, candidates, graphs, recipe, hashes


def _verify_splits(repo, public):
    old = _json(repo / PARTITION)
    folds = _json(repo / STAGE2 / "folds.json")
    subsets = _json(repo / STAGE2 / "subsets.json")
    _identity(folds)
    _identity(subsets)
    _require(folds["identity"] == FOLD_IDENTITY and subsets["identity"] == SUBSET_IDENTITY,
             "historical fold/subset identity differs")
    ids = set(public["ids"])
    arch = dict(zip(public["ids"], public["architectures"]))
    _require(set(old["A_ids"]) == ids and old["A_hash"] == A_ID_HASH, "original partition A IDs differ")
    by_name = {r["family"]: r for r in old["folds"]}
    _require(len(old["folds"]) == 5 and set(by_name) == {"fold" + str(f) for f in range(5)},
             "original fold identities differ")
    seen = []
    for f, row in enumerate(folds["folds"]):
        original = by_name["fold" + str(f)]
        q, t = row["query_ids"], row["training_pool_ids"]
        _require(row["fold"] == f and q == sorted(original["query_ids"]) and
                 t == sorted(original["train_ids"]), "old fold candidate ordering differs")
        _require(len(q) == len(set(q)) and len(t) == len(set(t)) and set(t) == ids - set(q),
                 "fold train/query complement fails")
        _require({arch[c] for c in t}.isdisjoint({arch[c] for c in q}), "T/Q architecture leakage")
        seen.extend(q)
    _require(len(seen) == 179 and set(seen) == ids and
             [len(r["query_ids"]) for r in folds["folds"]] == [38, 36, 36, 34, 35],
             "fold coverage/counts differ")
    expected_keys = set()
    for draw in (20260922, 20260923):
        for row in folds["folds"]:
            f = row["fold"]
            ordered = np.random.default_rng(np.random.SeedSequence([draw, f])).permutation(
                sorted(row["training_pool_ids"])).tolist()
            for n in (20, 50):
                sid = f"s{draw}_f{f}_n{n}"
                expected_keys.add(sid)
                s = subsets["subsets"][sid]
                _require(s["ids"] == ordered[:n] and s["selected_ids_sha256"] == _digest(ordered[:n]) and
                         s["sample_seed"] == draw and s["fold"] == f and s["n"] == n,
                         "historical training draw differs")
            _require(subsets["subsets"][f"s{draw}_f{f}_n20"]["ids"] ==
                     subsets["subsets"][f"s{draw}_f{f}_n50"]["ids"][:20], "n20 not nested in n50")
    _require(set(subsets["subsets"]) == expected_keys, "training-context matrix differs")
    return folds, subsets


def prepare_run(run_dir: Path, repo: Path) -> dict:
    """Verify immutable real sources then export A manifest, T bundles, Q views."""
    repo, run_dir = Path(repo).resolve(), Path(run_dir).resolve()
    public, candidates, graphs, recipe, source_hashes = _load_sources(repo)
    folds, subsets = _verify_splits(repo, public)
    labels = read_A_labels(repo)
    source_hashes[str(SOURCE / "private/evaluation_A.json")] = A_LABEL_HASH
    wl_features, feature_manifest = directed_wl(graphs)
    # An absent/non-diverse structural representation is never silently replaced.
    _require(not feature_manifest["all_structures_identical"], "all WL graph representations identical")
    pos = {cid: i for i, cid in enumerate(public["ids"])}
    architecture = dict(zip(public["ids"], public["architectures"]))
    x = np.c_[public["z"], public["hp"]]
    fold_by_id = {cid: row["fold"] for row in folds["folds"] for cid in row["query_ids"]}
    architecture_fold = {architecture[cid]: fold for cid, fold in fold_by_id.items()}
    arch_ids = sorted(graphs)
    cross_fold_isomorphic, cross_fold_pairs = [], 0
    for i, left in enumerate(arch_ids):
        for right in arch_ids[i + 1:]:
            if architecture_fold[left] != architecture_fold[right]:
                cross_fold_pairs += 1
                if isomorphic_operator_graph(graphs[left], graphs[right]):
                    cross_fold_isomorphic.append([left, right])
    _require(not cross_fold_isomorphic, "isomorphic operator graphs occur across original folds; no repartition allowed")
    feature_manifest.pop("identity")
    feature_manifest["cross_fold_isomorphism_audit"] = {
        "algorithm": "exact typed directed graph isomorphism by attribute-group permutations",
        "cross_fold_pairs_checked": cross_fold_pairs, "isomorphic_pairs": cross_fold_isomorphic,
        "same_architecture_id_grouping_preserved": True,
    }
    feature_manifest["identity"] = _digest(feature_manifest)
    training_bundles = {}
    for sid, subset in subsets["subsets"].items():
        ids = subset["ids"]
        training_bundles[sid] = training_projection(
            ids, x[[pos[c] for c in ids]], [labels[c]["mean"] for c in ids],
            [labels[c]["se"] ** 2 for c in ids], [wl_features[architecture[c]] for c in ids],
            subset_id=sid, fold=subset["fold"], draw=subset["sample_seed"])
    queries = {}
    for row in folds["folds"]:
        queries[row["fold"]] = query_projection([
            {"candidate_id": c, "x_raw": x[pos[c]].tolist(), "wl": wl_features[architecture[c]],
             "architecture_id": architecture[c]} for c in row["query_ids"]])
    split_manifest = {"folds": folds, "subsets": subsets, "fold_identity": FOLD_IDENTITY,
                      "subset_identity": SUBSET_IDENTITY,
                      "original_partition_sha256": EXTRA_HASHES[PARTITION],
                      "fold_file_sha256": _sha(repo / STAGE2 / "folds.json"),
                      "subset_file_sha256": _sha(repo / STAGE2 / "subsets.json"),
                      "contexts": 20, "nested_n20_in_n50": True, "architecture_disjoint": True}
    split_manifest["identity"] = _digest(split_manifest)
    public_features = {"ids": public["ids"], "architecture_ids": public["architectures"],
                       "x_raw": x.tolist(), "wl": [wl_features[a] for a in public["architectures"]],
                       "graph_hashes": {aid: graph_hash(g) for aid, g in graphs.items()},
                       "feature_identity": feature_manifest["identity"],
                       "A_id_hash": A_ID_HASH, "performance_labels_used": False}
    public_features["identity"] = _digest(public_features)
    source_manifest = {"repo": str(repo), "files": source_hashes, "A_id_hash": A_ID_HASH,
                       "recipe_hash": RECIPE_HASH, "recipe": recipe["training_protocol_identity"],
                       "A_label_sha256": A_LABEL_HASH, "VAE_sha256": VAE_HASH,
                       "A_validation_rows_verified": 179, "training_contexts_verified": 20,
                       "B_label_rows_decoded": 0, "test_scores_decoded": 0,
                       "mixed_shards": "opaque byte hashing only, no JSON deserialization",
                       "search_entrypoint": {
                           "acquisition": "qLogEI; no KG implementation found during read-only audit",
                           "call_chain": ["bo_phase4.py:1609 optimize_acq", "predictor.make_logei",
                                          "surrogate/accuracy_gp.py:778 StableQLogExpectedImprovement"],
                           "search_executed": False},
                       "VAE_provenance": {
                           "supported_training": "graph/HP reconstruction plus KL, no performance objective",
                           "evidence": [str(VAE_LOG.with_suffix(".json")), str(VAE_LOG.with_suffix(".log")),
                                        "legacy/phased_cora_pipeline/train_joint.py",
                                        "artifacts/reference/reference_manifest.json"],
                           "limitation": "historical training pkl/source were not byte-bound by the original log; exact unlabelled graph membership is unknown and permitted"}}
    source_manifest["identity"] = _digest(source_manifest)
    fields = ["candidate_id", "architecture_id", "graph_hash", "recipe_id", "fold",
              "log10_lr", "dropout", "log2_hidden_dim", "log10_l2"]
    fields += [f"latent_{j}" for j in range(12)]
    fields += ["validation_seed_0", "validation_seed_1", "validation_seed_2", "mean", "SE"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for i, c in enumerate(candidates):
        cid, aid = c["candidate_id"], c["architecture_fingerprint"]
        row = {"candidate_id": cid, "architecture_id": aid, "graph_hash": graph_hash(graphs[aid]),
               "recipe_id": RECIPE_HASH, "fold": fold_by_id[cid],
               **dict(zip(fields[5:9], public["hp"][i])),
               **{f"latent_{j}": public["z"][i][j] for j in range(12)},
               **{f"validation_seed_{j}": labels[cid]["seeds"][j] for j in range(3)},
               "mean": labels[cid]["mean"], "SE": labels[cid]["se"]}
        writer.writerow(row)
    _write_text(run_dir / "candidate_manifest.csv", stream.getvalue())
    for name, value in (("split_manifest.json", split_manifest), ("feature_manifest.json", feature_manifest),
                        ("source_manifest.json", source_manifest), ("public_features.json", public_features)):
        _write_json(run_dir / name, value)
    for sid, data in training_bundles.items():
        _write_json(run_dir / "training" / (sid + ".json"), data)
    for f, data in queries.items():
        _write_json(run_dir / "queries" / f"fold{f}.json", data)
    samples = []
    for cid in public["ids"][:5]:
        item = labels[cid]
        samples.append(f"| {cid} | {item['seeds']} | {item['mean']:.15g} | {item['se']:.15g} |")
    y_scales = [data["scaler"]["y_scale"] for data in training_bundles.values()]
    report = ("# E0 preflight\n\nPASS: 179 A candidates, 96 architectures, unique configurations and exact source hashes.\n\n"
              "Cora/public/NormalizeFeatures/full_batch/global4/No-LN; max_epochs=300, patience=80, GNN seeds=0,1,2. "
              "All 179 means and SEs recomputed. SE=sample std(ddof=1)/sqrt(3); known variance=SE², with no extra division by 3.\n\n"
              "Original folds and all 20 training draws exactly reproduced. Q counts=38,36,36,34,35. "
              "T/Q architectures disjoint; n20 is ordered prefix of n50. No label-based selection or repartition.\n\n"
              f"T y scales (ddof=0): {min(y_scales):.15g} to {max(y_scales):.15g}. "
              f"Constant-column contexts: {sum(bool(d['scaler']['constant_columns']) for d in training_bundles.values())}. "
              "Constant x columns use scale=1; zero/nonfinite y scale blocks the setting.\n\n"
              f"Directed operator WL H=2, dictionary sizes={feature_manifest['dictionary_sizes']}, "
              f"phi={feature_manifest['phi_dimension']}, psi={feature_manifest['psi_dimension']}, "
              f"n_max={feature_manifest['n_max']}, e_max={feature_manifest['e_max']}, "
              f"collision fraction={feature_manifest['collision_architecture_fraction']}. "
              "Dictionary uses only frozen A unlabelled graphs, including Q structure. No global HP is duplicated in node labels.\n\n"
              "VAE cache and current checkpoint hashes match. Historical log, unchanged audited training source and migration manifest support "
              "a reconstruction/KL-only VAE. Original pkl byte identity and exact graph membership were not recorded; no claim of full byte-level historical provenance.\n\n"
              "Source mixed shards were hashed as opaque bytes only. B labels/SE and node test scores were not decoded. "
              "The trainer must receive only training/<subset>.json and queries/fold<f>.json. candidate_manifest.csv is scorer/audit-only.\n\n"
              "Search entrypoint (located only; not executed): bo_phase4.py:1609 optimize_acq → predictor.make_logei → "
              "surrogate/accuracy_gp.py:778 StableQLogExpectedImprovement. Current implementation is qLogEI; "
              "no KG implementation was found in the read-only audit. Source hashes for bo_phase4.py, eval_utils.py and "
              "surrogate/accuracy_gp.py are retained in source_manifest.json.\n\n"
              "| candidate_id | seed validation accuracies | mean | SE |\n|---|---|---:|---:|\n" + "\n".join(samples) + "\n")
    _write_text(run_dir / "preflight_report.md", report)
    return {"status": "PASS", "candidate_count": 179, "architecture_count": 96, "contexts": 20,
            "source_identity": source_manifest["identity"], "split_identity": split_manifest["identity"],
            "feature_identity": feature_manifest["identity"], "recipe_hash": RECIPE_HASH,
            "wl_dimension": feature_manifest["psi_dimension"], "subset_ids": sorted(training_bundles)}


def release_training(run_dir: Path, subset_id: str) -> dict:
    """Read exactly one already-exported training bundle, never source labels."""
    _require(isinstance(subset_id, str) and "/" not in subset_id and "\\" not in subset_id,
             "invalid subset identifier")
    allowed = {f"s{s}_f{f}_n{n}" for s in (20260922, 20260923) for f in range(5) for n in (20, 50)}
    _require(subset_id in allowed, "unknown frozen training subset")
    data = _json(Path(run_dir) / "training" / (subset_id + ".json"))
    _require(data["subset_id"] == subset_id and data["ids"] == data["scaler"]["fit_ids"] and
             len(data["ids"]) == data["n"], "training bundle identity differs")
    return data


def query_features(run_dir: Path, fold: int) -> dict:
    """Read and validate exactly the unlabelled Q export for one fold."""
    _require(type(fold) is int and 0 <= fold < 5, "unknown fold")
    data = _json(Path(run_dir) / "queries" / f"fold{fold}.json")
    _require(set(data) == {"ids", "x_raw", "wl", "architecture_ids"}, "Q export contains forbidden fields")
    return query_projection([{"candidate_id": cid, "x_raw": data["x_raw"][i], "wl": data["wl"][i],
                              "architecture_id": data["architecture_ids"][i]}
                             for i, cid in enumerate(data["ids"])])
