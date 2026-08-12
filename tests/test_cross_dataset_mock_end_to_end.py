from __future__ import annotations

import json
import logging
import math

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("botorch")
from torch_geometric.data import Data

import dataset_utils
import final_eval
from initialization_wgmm_ted import candidate_fingerprint
from surrogate.accuracy_gp import AccuracyGPPredictor


def _synthetic_bundle(canonical_name: str, tmp_path) -> dataset_utils.DatasetBundle:
    dataset_index = dataset_utils.CANONICAL_DATASETS.index(canonical_name)
    num_features = 5 + dataset_index
    num_classes = 3 + dataset_index % 3
    num_nodes = 18
    generator = torch.Generator().manual_seed(100 + dataset_index)
    data = Data(
        x=torch.randn(num_nodes, num_features, generator=generator),
        edge_index=torch.tensor(
            [
                list(range(num_nodes)) + list(range(num_nodes)),
                list(range(1, num_nodes)) + [0] + list(range(num_nodes)),
            ],
            dtype=torch.long,
        ),
        y=torch.arange(num_nodes, dtype=torch.long) % num_classes,
        train_mask=torch.tensor(
            [index < 9 for index in range(num_nodes)], dtype=torch.bool
        ),
        val_mask=torch.tensor(
            [9 <= index < 13 for index in range(num_nodes)], dtype=torch.bool
        ),
        test_mask=torch.tensor(
            [index >= 13 for index in range(num_nodes)], dtype=torch.bool
        ),
    )
    split_fingerprint, split_indices = (
        dataset_utils._split_fingerprint_from_masks(
            data,
            protocol="synthetic_mock",
            split_seed=0 if canonical_name == "dblp" else None,
        )
    )
    return dataset_utils._build_bundle(
        data=data,
        canonical_name=canonical_name,
        source_class="tests.SyntheticDataset",
        source_version="1",
        storage_root=tmp_path / "source",
        split_protocol="synthetic_mock",
        split_seed=0 if canonical_name == "dblp" else None,
        split_fingerprint=split_fingerprint,
        split_indices=split_indices,
        graph_transform=(
            "to_undirected" if canonical_name == "ogbn-arxiv" else "synthetic"
        ),
        original_graph_directed=canonical_name == "ogbn-arxiv",
        original_num_edges=int(data.edge_index.shape[1]),
        training_mode="full_batch",
        num_classes=num_classes,
    )


@pytest.mark.parametrize("canonical_name", dataset_utils.CANONICAL_DATASETS)
def test_mock_load_eval_gp_history_and_final_read(
    canonical_name, tmp_path, monkeypatch
):
    bundle = _synthetic_bundle(canonical_name, tmp_path)

    def mock_loader(dataset_name, *_args, **_kwargs):
        assert dataset_utils.canonicalize_dataset_name(dataset_name) == canonical_name
        return bundle

    monkeypatch.setattr(
        dataset_utils, "load_node_classification_dataset", mock_loader
    )
    loaded = dataset_utils.load_node_classification_dataset(
        canonical_name, str(tmp_path / "data")
    )
    assert loaded.num_features == bundle.data.x.shape[1]
    assert loaded.num_classes == int(bundle.data.y.max()) + 1

    generator = torch.Generator().manual_seed(700)
    candidate_vectors = torch.rand(6, 16, generator=generator)
    candidate_vectors[:, :12] = candidate_vectors[:, :12] * 5.0 - 2.5

    def mock_evaluator(z_search: torch.Tensor) -> float:
        dataset_offset = 0.001 * dataset_utils.CANONICAL_DATASETS.index(
            canonical_name
        )
        return float(0.65 + dataset_offset + 0.02 * z_search[0])

    validation_scores = torch.tensor(
        [mock_evaluator(row) for row in candidate_vectors],
        dtype=torch.double,
    )
    predictor = AccuracyGPPredictor.fit_offline(
        candidate_vectors,
        validation_scores,
        arch_nz=12,
        hp_mode="global4",
        z_bound=2.5,
        metadata={"dataset_context": bundle.context},
        device="cpu",
        fit_steps=1,
    )
    predicted = predictor.predict(candidate_vectors[0])
    assert predictor.metadata["dataset_context"] == bundle.context
    assert math.isfinite(predicted["mean"])

    selected = candidate_vectors[int(validation_scores.argmax())]
    fingerprint = candidate_fingerprint(selected.numpy())
    history_record = {
        "step": 0,
        "valid": True,
        "val_acc": float(validation_scores.max()),
        "operations": ["GCNConv"],
        "edges": [[0, 1], [1, 2]],
        "lr": 0.01,
        "dropout": 0.5,
        "hidden_dim": 16,
        "l2": 5e-4,
        "candidate_fingerprint": fingerprint,
        "z_search": selected.tolist(),
        "search_seed": 0,
        "evaluation_fidelity": "full",
        "dataset_context": bundle.context,
    }
    history_dir = tmp_path / "search"
    history_dir.mkdir()
    history_path = history_dir / "history_final.json"
    history_path.write_text(json.dumps([history_record]), encoding="utf-8")
    (history_dir / "history_metadata.json").write_text(
        json.dumps({"dataset_context": bundle.context}), encoding="utf-8"
    )

    args = final_eval.parse_args(
        [
            "--history_path",
            str(history_path),
            "--output",
            str(tmp_path / "final"),
            "--method_label",
            "mock",
            "--search_seed",
            "0",
            "--final_base_seed",
            "123",
            "--top_k",
            "1",
            "--n_replicates",
            "1",
            "--dry_run",
        ]
    )
    summary = final_eval.run_final_evaluation(
        args, logging.getLogger(f"mock-e2e-{canonical_name}")
    )

    assert args.dataset == canonical_name
    assert summary["dataset_context"] == bundle.context
    assert summary["status"] == "dry_run_preflight_passed"
