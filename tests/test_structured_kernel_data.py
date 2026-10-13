"""Scientific data/WL invariants; real sources are an explicit opt-in check."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from analyse.structured_kernel_phase1 import data
from analyse.structured_kernel_phase1.features import (
    build_operator_graph, directed_wl, fit_distance_scales, graph_hash,
    isomorphic_operator_graph, positive_distance_median,
)


def _config(operations=None, edges=None, **kw):
    operations = operations or ["GCNII", "Identity", "GCNII"]
    if edges is None:
        edges = [[i, i + 1] for i in range(len(operations) + 1)]
    return {"operations": operations, "edges": edges,
            "gcnii_alpha": .1, "gcnii_theta": .5, "gat_heads": 1,
            "sage_aggr": "mean", "gin_eps": 0.0, **kw}


def _relabel(graph, old_to_new):
    nodes = [None] * len(graph["nodes"])
    for old, new in enumerate(old_to_new):
        nodes[new] = copy.deepcopy(graph["nodes"][old])
    return {"nodes": nodes, "edges": [[old_to_new[u], old_to_new[v]] for u, v in graph["edges"]]}


def test_directed_wl_is_invariant_to_vertex_numbering_and_edge_serialization():
    original = build_operator_graph(_config())
    renumbered = _relabel(original, [4, 0, 3, 1, 2])
    renumbered["edges"].reverse()
    features, manifest = directed_wl({"original": original, "permuted": renumbered})
    assert features["original"] == features["permuted"]
    assert isomorphic_operator_graph(original, renumbered)
    assert graph_hash(original) != graph_hash(renumbered)
    assert manifest["collision_architecture_fraction"] == 1.0
    assert "not an isomorphism" in manifest["graph_hash_semantics"]
    # The actual GCNII occurrence is attached to its node, not recomputed from numbering.
    assert renumbered["nodes"][0]["layer"] == 1
    assert renumbered["nodes"][1]["layer"] == 2


def test_same_operators_different_connections_are_distinguished():
    a = build_operator_graph(_config(["GCNConv", "GCNConv"], [[0, 1], [1, 2], [2, 3]]))
    b = build_operator_graph(_config(["GCNConv", "GCNConv"], [[0, 1], [0, 2], [1, 3], [2, 3]]))
    features, manifest = directed_wl({"chain": a, "fork": b})
    assert features["chain"] != features["fork"]
    assert not isomorphic_operator_graph(a, b)
    assert manifest["H"] == 2
    assert manifest["phi_dimension"] == sum(manifest["dictionary_sizes"])
    assert manifest["psi_dimension"] == manifest["phi_dimension"] + 2


def test_operator_features_keep_semantics_and_exclude_global_hp():
    a = _config(["GATConv", "SAGEConv", "GINConv", "Identity", "GCNII"],
                hidden_dim=64, dropout=.1, lr=.01, l2=.001)
    b = {**a, "hidden_dim": 512, "dropout": .7, "lr": .1, "l2": .01}
    ga, gb = build_operator_graph(a), build_operator_graph(b)
    assert ga == gb
    assert ga["nodes"][1]["heads"] == 1 and ga["nodes"][1]["concat"] is False
    assert ga["nodes"][2]["aggr"] == "mean"
    assert ga["nodes"][3]["train_eps"] is False
    assert ga["nodes"][4]["activation"] == "none"
    assert ga["nodes"][5]["shared_weights"] is True
    assert all(not {"hidden_dim", "dropout", "lr", "l2"} & node.keys() for node in ga["nodes"])
    with pytest.raises(ValueError, match="missing actual architecture connections"):
        build_operator_graph({"operations": ["Identity"], "edges": []})


def test_short_edgeless_graph_has_finite_wl_and_zero_edge_feature():
    features, manifest = directed_wl({"tiny": {"nodes": [{"role": "input"}], "edges": []}})
    assert np.isfinite(features["tiny"]).all()
    assert features["tiny"][-2:] == [1.0, 0.0]
    assert manifest["e_max"] == 0
    assert manifest["all_structures_identical"]


def test_three_seed_mean_se_is_known_variance_of_the_mean():
    rows = {"a": {"validation_seeds": [.72, .75, .74], "training_seeds": [0, 1, 2],
                   "validation_mean": np.mean([.72, .75, .74]),
                   "validation_se": np.std([.72, .75, .74], ddof=1) / np.sqrt(3)}}
    observed = data.validate_A_labels(["a"], rows)["a"]
    assert np.isclose(observed["se"] ** 2, np.var(rows["a"]["validation_seeds"], ddof=1) / 3)
    damaged = copy.deepcopy(rows)
    damaged["a"]["validation_se"] /= np.sqrt(3)
    with pytest.raises(ValueError, match="mean/SE disagree"):
        data.validate_A_labels(["a"], damaged)
    damaged = copy.deepcopy(rows)
    damaged["a"]["test_acc"] = .8
    with pytest.raises(ValueError, match="schema"):
        data.validate_A_labels(["a"], damaged)
    with pytest.raises(ValueError, match="IDs"):
        data.validate_A_labels(["a"], {**rows, "B": rows["a"]})


def test_training_scaler_uses_ddof0_constant_scale_one_and_se_squared_once():
    x = np.zeros((3, 16), dtype=np.float64)
    x[:, 0] = [1, 2, 3]
    x[:, 12] = [-3, -2, -1]
    y = np.array([.6, .7, .8])
    se = np.array([.01, .02, .03])
    fit = data.training_projection(["a", "b", "c"], x, y, se ** 2, np.ones((3, 2)),
                                   subset_id="synthetic", fold=0, draw=1)
    assert fit["scaler"]["std_correction"] == 0
    assert fit["scaler"]["fit_ids"] == ["a", "b", "c"]
    assert np.isclose(fit["scaler"]["x_scale"][0], np.std([1, 2, 3], ddof=0))
    assert fit["scaler"]["x_scale"][1] == 1
    assert np.allclose(fit["known_variance_standardized"], se ** 2 / y.std(ddof=0) ** 2)
    reconstructed = fit["scaler"]["y_mean"] + fit["scaler"]["y_scale"] * np.array(fit["y_standardized"])
    assert np.allclose(reconstructed, y)
    assert fit["distance_scales"]["wl"]["degenerate"]
    assert fit["distance_scales"]["wl"]["scale"] == 1
    with pytest.raises(ValueError, match="zero/nonfinite T y scale"):
        data.training_projection(["a", "b", "c"], x, [.7, .7, .7], se ** 2, np.ones((3, 2)),
                                 subset_id="synthetic", fold=0, draw=1)


def test_distance_scaling_excludes_zero_duplicate_pairs_and_query_rows():
    x = np.zeros((3, 16), dtype=np.float64)
    x[:, 0] = [0, 0, 4]
    x[:, 12] = [0, 0, 3]
    scales = fit_distance_scales(x)
    assert scales["latent"]["scale"] == 4
    assert scales["hp"]["scale"] == 3
    assert scales["raw16"]["scale"] == 5
    assert scales["raw16"]["positive_pairs"] == 2
    assert scales["raw16"]["zero_pairs"] == 1
    assert positive_distance_median(np.zeros((1, 3)))["scale"] == 1
    with pytest.raises(ValueError, match="finite"):
        fit_distance_scales(np.full((2, 16), np.nan))


def test_query_projection_does_not_access_poisoned_label_fields():
    class PoisonLabels(dict):
        def __getitem__(self, key):
            if key in {"y", "SE", "validation_mean", "validation_se"}:
                raise AssertionError("Q performance was read")
            return super().__getitem__(key)

    row = PoisonLabels(candidate_id="q", architecture_id="graph", x_raw=[0.0] * 16,
                       wl=[1.0, 1.0, 0.0], y=object(), SE=object())
    view = data.query_projection([row])
    assert set(view) == {"ids", "architecture_ids", "x_raw", "wl"}
    assert view["ids"] == ["q"]


def test_gp_and_exported_distance_scales_agree_without_using_query_distribution():
    from analyse.structured_kernel_phase1.gp import make_inputs
    x = np.arange(4 * 16, dtype=float).reshape(4, 16)
    wl = np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
    training = data.training_projection(["a", "b", "c", "d"], x, [.5, .6, .7, .8],
                                        [.0001] * 4, wl, subset_id="synthetic", fold=0, draw=1)
    q = {"x_raw": np.zeros((2, 16)).tolist(), "wl": np.zeros((2, 2)).tolist()}
    extreme_q = {"x_raw": np.full((2, 16), 1e7).tolist(), "wl": np.full((2, 2), 1e7).tolist()}
    for method in ("L-A", "L-I", "W-A", "W-I", "RAW16"):
        t1, _, _, scales = make_inputs(training, q, method)
        t2, _, _, scales_extreme = make_inputs(training, extreme_q, method)
        assert np.array_equal(t1.numpy(), t2.numpy())
        assert scales == scales_extreme
        if method == "RAW16":
            assert scales["raw16"]["value"] == training["distance_scales"]["raw16"]["scale"]
        else:
            name = "latent" if method.startswith("L-") else "wl"
            assert scales["architecture"]["value"] == training["distance_scales"][name]["scale"]
            assert scales["hp"]["value"] == training["distance_scales"]["hp"]["scale"]


def test_worker_release_reads_only_one_training_file_and_q_file(tmp_path, monkeypatch):
    sid = "s20260922_f0_n20"
    training = {"subset_id": sid, "ids": list(map(str, range(20))), "n": 20,
                "scaler": {"fit_ids": list(map(str, range(20)))}}
    q = {"ids": ["q"], "architecture_ids": ["aq"], "x_raw": [[0.0] * 16], "wl": [[1.0]]}
    allowed = {str(tmp_path / "training" / (sid + ".json")): training,
               str(tmp_path / "queries/fold0.json"): q}
    opened = []

    def narrow_reader(path):
        path = str(path)
        assert path in allowed, "worker attempted to open full labels/manifest"
        opened.append(path)
        return copy.deepcopy(allowed[path])

    monkeypatch.setattr(data, "_json", narrow_reader)
    assert data.release_training(tmp_path, sid)["ids"] == training["ids"]
    assert data.query_features(tmp_path, 0) == q
    assert len(opened) == 2
    allowed[str(tmp_path / "queries/fold0.json")]["validation_se"] = [.1]
    with pytest.raises(ValueError, match="forbidden"):
        data.query_features(tmp_path, 0)
    with pytest.raises(ValueError, match="identifier"):
        data.release_training(tmp_path, "../../candidate_manifest.csv")


@pytest.mark.skipif(not os.environ.get("STRUCTURED_KERNEL_REPO"), reason="requires explicit audited server repo")
def test_frozen_real_data_e0_all_contexts_and_wl(tmp_path):
    repo = Path(os.environ["STRUCTURED_KERNEL_REPO"])
    outcome = data.prepare_run(tmp_path, repo)
    assert outcome["status"] == "PASS"
    assert outcome["contexts"] == 20 and outcome["candidate_count"] == 179
    split = json.loads((tmp_path / "split_manifest.json").read_text())
    assert split["fold_identity"] == data.FOLD_IDENTITY
    assert split["subset_identity"] == data.SUBSET_IDENTITY
    manifest = json.loads((tmp_path / "feature_manifest.json").read_text())
    assert manifest["dictionary_sizes"] == [10, 438, 641]
    assert manifest["psi_dimension"] == 1091
    assert manifest["collision_groups"] == []
    assert manifest["cross_fold_isomorphism_audit"]["isomorphic_pairs"] == []
    for subset in outcome["subset_ids"]:
        t = data.release_training(tmp_path, subset)
        q = data.query_features(tmp_path, t["fold"])
        assert set(t["ids"]).isdisjoint(q["ids"])
        assert t["scaler"]["constant_columns"] == []
        assert t["scaler"]["y_scale"] > 0
    # Exact rerun is idempotent; a changed source/run cannot overwrite the manifest.
    assert data.prepare_run(tmp_path, repo) == outcome
