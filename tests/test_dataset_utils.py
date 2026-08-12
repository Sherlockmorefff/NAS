from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

import dataset_utils


def _data(
    *,
    num_nodes: int = 12,
    num_features: int = 5,
    num_classes: int = 3,
) -> Data:
    labels = torch.arange(num_nodes, dtype=torch.long) % num_classes
    return Data(
        x=torch.arange(
            num_nodes * num_features, dtype=torch.float32
        ).reshape(num_nodes, num_features),
        edge_index=torch.tensor(
            [
                list(range(num_nodes)) + list(range(num_nodes)),
                list(range(1, num_nodes)) + [0] + list(range(num_nodes)),
            ],
            dtype=torch.long,
        ),
        y=labels,
        train_mask=torch.tensor(
            [index < 6 for index in range(num_nodes)], dtype=torch.bool
        ),
        val_mask=torch.tensor(
            [6 <= index < 9 for index in range(num_nodes)], dtype=torch.bool
        ),
        test_mask=torch.tensor(
            [index >= 9 for index in range(num_nodes)], dtype=torch.bool
        ),
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Cora", "cora"),
        ("CiteSeer", "citeseer"),
        ("PubMed", "pubmed"),
        ("CitationFull-DBLP", "dblp"),
        ("FLICKR", "flickr"),
        ("ogbn_arxiv", "ogbn-arxiv"),
    ],
)
def test_dataset_name_canonicalization(raw, expected):
    assert dataset_utils.canonicalize_dataset_name(raw) == expected


def test_dataset_cli_conflicts_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="only be used"):
        dataset_utils.resolve_dataset_request(
            "pubmed", None, cora_root=str(tmp_path)
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataset_utils.resolve_dataset_request(
            "cora", str(tmp_path), cora_root=str(tmp_path / "Cora")
        )
    with pytest.raises(ValueError, match="only applicable"):
        dataset_utils.resolve_dataset_request("flickr", str(tmp_path), split_seed=0)
    with pytest.raises(ValueError, match="only applicable"):
        dataset_utils.resolve_dataset_request(
            "pubmed", str(tmp_path), ogbn_arxiv_edge_mode="undirected"
        )


def test_legacy_cora_root_resolves_to_same_pyg_location(tmp_path):
    legacy = dataset_utils.resolve_dataset_request(
        "cora", None, cora_root=str(tmp_path / "Cora")
    )
    modern = dataset_utils.resolve_dataset_request("cora", str(tmp_path))

    assert dataset_utils._planetoid_roots(legacy) == (
        tmp_path,
        tmp_path / "Cora",
    )
    assert dataset_utils._planetoid_roots(modern) == (
        tmp_path,
        tmp_path / "Cora",
    )


def test_planetoid_public_masks_and_legacy_interface_are_preserved(
    tmp_path, monkeypatch
):
    import torch_geometric.datasets

    original = _data()
    calls = []

    class FakePlanetoid:
        num_classes = 3

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __getitem__(self, index):
            assert index == 0
            return original.clone()

    monkeypatch.setattr(torch_geometric.datasets, "Planetoid", FakePlanetoid)
    legacy = dataset_utils.load_node_classification_dataset(
        "cora",
        str(tmp_path / "Cora"),
        legacy_cora_root=True,
    )
    modern = dataset_utils.load_node_classification_dataset(
        "cora", str(tmp_path)
    )

    assert calls[0]["root"] == calls[1]["root"] == str(tmp_path)
    assert calls[0]["split"] == calls[1]["split"] == "public"
    assert calls[0]["name"] == calls[1]["name"] == "Cora"
    assert torch.equal(legacy.data.train_mask, original.train_mask)
    assert torch.equal(legacy.data.val_mask, original.val_mask)
    assert torch.equal(legacy.data.test_mask, original.test_mask)
    assert legacy.dataset_content_fingerprint == modern.dataset_content_fingerprint
    assert legacy.split_fingerprint == modern.split_fingerprint


def test_dblp_split_is_fixed_disjoint_and_has_exact_counts():
    labels = torch.arange(4, dtype=torch.long).repeat_interleave(500)
    first_masks, first_indices, first_fingerprint = (
        dataset_utils.deterministic_planetoid_like_split(labels, split_seed=0)
    )
    second_masks, second_indices, second_fingerprint = (
        dataset_utils.deterministic_planetoid_like_split(labels, split_seed=0)
    )

    assert first_indices == second_indices
    assert first_fingerprint == second_fingerprint
    assert int(first_masks["train"].sum()) == 80
    assert int(first_masks["valid"].sum()) == 500
    assert int(first_masks["test"].sum()) == 1000
    assert not bool((first_masks["train"] & first_masks["valid"]).any())
    assert not bool((first_masks["train"] & first_masks["test"]).any())
    assert not bool((first_masks["valid"] & first_masks["test"]).any())
    assert {
        int(labels[index]) for index in first_indices["train"]
    } == {0, 1, 2, 3}


def test_dblp_split_is_independent_of_search_and_evaluation_seed():
    labels = torch.arange(3, dtype=torch.long).repeat_interleave(600)
    baseline = dataset_utils.deterministic_planetoid_like_split(
        labels, split_seed=0
    )[1]
    for _search_seed in range(5):
        for _evaluation_seed in (0, 7, 29):
            assert (
                dataset_utils.deterministic_planetoid_like_split(
                    labels, split_seed=0
                )[1]
                == baseline
            )


def test_dblp_split_fails_instead_of_changing_class_quota():
    labels = torch.tensor([0] * 19 + [1] * 2000)
    with pytest.raises(ValueError, match="20 are required"):
        dataset_utils.deterministic_planetoid_like_split(labels)


def test_flickr_official_masks_are_not_replaced(tmp_path, monkeypatch):
    import torch_geometric.datasets

    original = _data()

    class FakeFlickr:
        num_classes = 3

        def __init__(self, root):
            self.root = root

        def __getitem__(self, index):
            assert index == 0
            return original.clone()

    monkeypatch.setattr(torch_geometric.datasets, "Flickr", FakeFlickr)
    bundle = dataset_utils.load_node_classification_dataset(
        "flickr", str(tmp_path)
    )

    assert bundle.split_protocol == "flickr_official"
    assert torch.equal(bundle.data.train_mask, original.train_mask)
    assert torch.equal(bundle.data.val_mask, original.val_mask)
    assert torch.equal(bundle.data.test_mask, original.test_mask)
    assert bundle.graph_transform == "official_features_no_additional_normalization"


def test_ogbn_arxiv_official_split_y_and_undirected_identity(
    tmp_path, monkeypatch
):
    nodeproppred = ModuleType("ogb.nodeproppred")
    ogb = ModuleType("ogb")

    class FakeOGB:
        num_classes = 3

        def __init__(self, name, root):
            assert name == "ogbn-arxiv"
            self.data = Data(
                x=torch.randn(6, 4),
                edge_index=torch.tensor(
                    [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
                ),
                y=torch.tensor([[0], [1], [2], [0], [1], [2]]),
                num_nodes=6,
            )

        def __getitem__(self, index):
            return self.data.clone()

        def get_idx_split(self):
            return {
                "train": torch.tensor([0, 1, 2]),
                "valid": torch.tensor([3]),
                "test": torch.tensor([4, 5]),
            }

    nodeproppred.PygNodePropPredDataset = FakeOGB
    ogb.nodeproppred = nodeproppred
    monkeypatch.setitem(sys.modules, "ogb", ogb)
    monkeypatch.setitem(sys.modules, "ogb.nodeproppred", nodeproppred)

    bundle = dataset_utils.load_node_classification_dataset(
        "ogbn-arxiv",
        str(tmp_path),
        ogbn_arxiv_edge_mode="undirected",
    )

    assert bundle.data.y.shape == (6,)
    assert bundle.data.train_mask.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert bundle.data.val_mask.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert bool(bundle.data.is_undirected())
    assert bundle.manifest["original_graph_directed"] is True
    assert bundle.manifest["experiment_graph_directed"] is False
    assert bundle.graph_transform == "to_undirected"
    assert bundle.context["graph_transform"] == "to_undirected"

    directed = dataset_utils.load_node_classification_dataset(
        "ogbn-arxiv",
        str(tmp_path),
        ogbn_arxiv_edge_mode="directed",
    )
    assert directed.graph_transform == "preserve_directed"
    assert directed.dataset_content_fingerprint != (
        bundle.dataset_content_fingerprint
    )
    assert dataset_utils.evaluation_cache_key(
        "a" * 64,
        directed.context,
        evaluation_fidelity="full",
        evaluation_seed=7,
    ) != dataset_utils.evaluation_cache_key(
        "a" * 64,
        bundle.context,
        evaluation_fidelity="full",
        evaluation_seed=7,
    )


def test_official_split_masks_validate_overlap_and_range():
    with pytest.raises(ValueError, match="overlap"):
        data = _data()
        data.val_mask = data.train_mask.clone()
        dataset_utils.validate_node_classification_data(data)
    with pytest.raises(ValueError, match="out-of-range"):
        dataset_utils.split_indices_to_masks(
            {
                "train": [0, 1],
                "valid": [2],
                "test": [99],
            },
            5,
        )


def test_bundle_fields_cache_identity_and_artifact_mismatch(tmp_path):
    data = _data()
    split_fingerprint, indices = dataset_utils._split_fingerprint_from_masks(
        data, protocol="synthetic", split_seed=None
    )
    cora = dataset_utils._build_bundle(
        data=data.clone(),
        canonical_name="cora",
        source_class="synthetic",
        source_version="1",
        storage_root=tmp_path / "source",
        split_protocol="synthetic",
        split_seed=None,
        split_fingerprint=split_fingerprint,
        split_indices=indices,
        graph_transform="none",
        original_graph_directed=False,
        original_num_edges=int(data.edge_index.shape[1]),
        training_mode="full_batch",
        num_classes=3,
    )
    pubmed = dataset_utils._build_bundle(
        data=data.clone(),
        canonical_name="pubmed",
        source_class="synthetic",
        source_version="1",
        storage_root=tmp_path / "source",
        split_protocol="synthetic",
        split_seed=None,
        split_fingerprint=split_fingerprint,
        split_indices=indices,
        graph_transform="none",
        original_graph_directed=False,
        original_num_edges=int(data.edge_index.shape[1]),
        training_mode="full_batch",
        num_classes=3,
    )

    required = {
        "canonical_name",
        "source_class",
        "num_nodes",
        "num_edges",
        "num_features",
        "num_classes",
        "split_fingerprint",
        "dataset_content_fingerprint",
        "tensor_fingerprints",
        "training_mode",
        "metric",
    }
    assert required.issubset(cora.manifest)
    candidate = "a" * 64
    cora_key = dataset_utils.evaluation_cache_key(
        candidate,
        cora.context,
        evaluation_fidelity="full",
        evaluation_seed=7,
    )
    pubmed_key = dataset_utils.evaluation_cache_key(
        candidate,
        pubmed.context,
        evaluation_fidelity="full",
        evaluation_seed=7,
    )
    assert cora_key != pubmed_key

    output = tmp_path / "run"
    dataset_utils.write_dataset_artifacts(cora, output)
    validated = dataset_utils.validate_expected_dataset_manifest(
        cora, output / "dataset_manifest.json"
    )
    assert validated["dataset_content_fingerprint"] == (
        cora.dataset_content_fingerprint
    )
    with pytest.raises(ValueError, match="dataset context mismatch"):
        dataset_utils.write_dataset_artifacts(pubmed, output)
    with pytest.raises(ValueError, match="dataset context mismatch"):
        dataset_utils.validate_expected_dataset_manifest(
            pubmed, output / "dataset_manifest.json"
        )
    saved = json.loads((output / "dataset_manifest.json").read_text())
    assert saved["canonical_name"] == "cora"


def test_masked_accuracy_matches_ogb_evaluator_semantics():
    from ogb.nodeproppred import Evaluator

    labels = torch.tensor([0, 1, 2, 1])
    predictions = torch.tensor([0, 2, 2, 1])
    mask = torch.tensor([True, True, False, True])
    accuracy = dataset_utils.node_classification_accuracy(
        predictions, labels, mask
    )
    evaluator = Evaluator(name="ogbn-arxiv")
    expected_ogb_accuracy = evaluator.eval(
        {
            "y_true": labels[mask].view(-1, 1).numpy(),
            "y_pred": predictions[mask].view(-1, 1).numpy(),
        }
    )["acc"]
    assert accuracy == pytest.approx(expected_ogb_accuracy)
