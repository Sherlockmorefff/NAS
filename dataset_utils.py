"""Unified homogeneous node-classification dataset loading and identity.

The search and final-evaluation paths consume :class:`DatasetBundle` instead
of constructing PyG datasets independently.  Candidate identity intentionally
does not include dataset identity; evaluation/cache identity does.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


CANONICAL_DATASETS = (
    "cora",
    "citeseer",
    "pubmed",
    "dblp",
    "flickr",
    "ogbn-arxiv",
)
DEFAULT_CORA_ROOT = "/tmp/Cora"
DEFAULT_DATA_ROOT = "/tmp/gnn_datasets"
DATASET_MANIFEST_FORMAT_VERSION = 1
DATASET_CONTEXT_FIELDS = (
    "canonical_name",
    "source_class",
    "source_version",
    "torch_version",
    "pyg_version",
    "ogb_version",
    "dataset_content_fingerprint",
    "split_protocol",
    "split_seed",
    "split_fingerprint",
    "graph_transform",
    "training_mode",
    "metric",
)

_ALIASES = {
    "cora": "cora",
    "citeseer": "citeseer",
    "cite-seer": "citeseer",
    "cite_seer": "citeseer",
    "pubmed": "pubmed",
    "pub-med": "pubmed",
    "pub_med": "pubmed",
    "dblp": "dblp",
    "citationfull-dblp": "dblp",
    "citationfull_dblp": "dblp",
    "citationfulldblp": "dblp",
    "flickr": "flickr",
    "ogbn-arxiv": "ogbn-arxiv",
    "ogbn_arxiv": "ogbn-arxiv",
    "ogbnarxiv": "ogbn-arxiv",
}

_PLANETOID_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


@dataclass(frozen=True)
class DatasetRequest:
    canonical_name: str
    data_root: str
    split_seed: int
    ogbn_arxiv_edge_mode: str
    legacy_cora_root: bool


@dataclass
class DatasetBundle:
    data: Any
    canonical_name: str
    source_class: str
    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int
    metric: str
    split_protocol: str
    split_seed: int | None
    split_fingerprint: str
    graph_transform: str
    dataset_content_fingerprint: str
    manifest: dict[str, Any]
    split_indices: dict[str, list[int]] | None = None

    @property
    def context(self) -> dict[str, Any]:
        return dataset_context_from_manifest(self.manifest)

    def to(self, device: torch.device | str) -> "DatasetBundle":
        self.data = self.data.to(device)
        return self


def canonicalize_dataset_name(name: str | None) -> str:
    if name is None:
        return "cora"
    normalized = str(name).strip().lower().replace(" ", "")
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dataset {name!r}; expected one of "
            + ", ".join(CANONICAL_DATASETS)
        ) from exc


def resolve_dataset_request(
    dataset_name: str | None,
    data_root: str | None,
    *,
    cora_root: str | None = None,
    split_seed: int | None = None,
    ogbn_arxiv_edge_mode: str | None = None,
) -> DatasetRequest:
    canonical_name = canonicalize_dataset_name(dataset_name)
    if cora_root is not None and canonical_name != "cora":
        raise ValueError("--cora_root can only be used with --dataset cora")
    if cora_root is not None and data_root is not None:
        raise ValueError("--cora_root and --data_root are mutually exclusive")

    if split_seed is not None and canonical_name != "dblp":
        raise ValueError("--split_seed is only applicable to CitationFull-DBLP")
    resolved_split_seed = 0 if split_seed is None else int(split_seed)
    if resolved_split_seed < 0:
        raise ValueError("--split_seed must be non-negative")

    if ogbn_arxiv_edge_mode is not None and canonical_name != "ogbn-arxiv":
        raise ValueError(
            "--ogbn_arxiv_edge_mode is only applicable to ogbn-arxiv"
        )
    resolved_edge_mode = (
        "undirected"
        if ogbn_arxiv_edge_mode is None
        else str(ogbn_arxiv_edge_mode).strip().lower()
    )
    if resolved_edge_mode not in ("directed", "undirected"):
        raise ValueError(
            "--ogbn_arxiv_edge_mode must be 'directed' or 'undirected'"
        )

    legacy_cora_root = cora_root is not None or (
        canonical_name == "cora" and data_root is None
    )
    if cora_root is not None:
        resolved_root = cora_root
    elif data_root is not None:
        resolved_root = data_root
    elif canonical_name == "cora":
        resolved_root = DEFAULT_CORA_ROOT
    else:
        resolved_root = DEFAULT_DATA_ROOT

    return DatasetRequest(
        canonical_name=canonical_name,
        data_root=os.path.abspath(os.path.expanduser(resolved_root)),
        split_seed=resolved_split_seed,
        ogbn_arxiv_edge_mode=resolved_edge_mode,
        legacy_cora_root=legacy_cora_root,
    )


def _canonical_json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_identifiers(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    identifiers: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        identifiers.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    return identifiers


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def deterministic_planetoid_like_split(
    y: torch.Tensor,
    *,
    split_seed: int = 0,
    train_per_class: int = 20,
    num_val: int = 500,
    num_test: int = 1000,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]], str]:
    labels = torch.as_tensor(y, dtype=torch.long).view(-1).cpu()
    if labels.numel() == 0:
        raise ValueError("cannot split an empty label tensor")
    classes = torch.unique(labels, sorted=True)
    expected_classes = torch.arange(
        int(classes.numel()), dtype=torch.long
    )
    if not torch.equal(classes, expected_classes):
        raise ValueError("DBLP labels must be contiguous and start at zero")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(split_seed))
    train_parts: list[torch.Tensor] = []
    for class_id in classes.tolist():
        indices = torch.nonzero(labels == int(class_id), as_tuple=False).view(-1)
        if indices.numel() < int(train_per_class):
            raise ValueError(
                f"DBLP class {class_id} has {indices.numel()} nodes; "
                f"{train_per_class} are required for training"
            )
        permutation = torch.randperm(indices.numel(), generator=generator)
        train_parts.append(indices[permutation[: int(train_per_class)]])

    train_idx = torch.cat(train_parts).sort().values
    remaining_mask = torch.ones(labels.numel(), dtype=torch.bool)
    remaining_mask[train_idx] = False
    remaining = torch.nonzero(remaining_mask, as_tuple=False).view(-1)
    required_remaining = int(num_val) + int(num_test)
    if remaining.numel() < required_remaining:
        raise ValueError(
            f"DBLP has only {remaining.numel()} nodes after training selection; "
            f"{required_remaining} are required for validation/test"
        )
    remaining = remaining[
        torch.randperm(remaining.numel(), generator=generator)
    ]
    val_idx = remaining[: int(num_val)].sort().values
    test_idx = remaining[int(num_val):required_remaining].sort().values

    masks = {
        "train": torch.zeros(labels.numel(), dtype=torch.bool),
        "valid": torch.zeros(labels.numel(), dtype=torch.bool),
        "test": torch.zeros(labels.numel(), dtype=torch.bool),
    }
    masks["train"][train_idx] = True
    masks["valid"][val_idx] = True
    masks["test"][test_idx] = True
    indices = {
        "train": train_idx.tolist(),
        "valid": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    fingerprint = _canonical_json_fingerprint(
        {
            "protocol": "planetoid_like_20_per_class_500_val_1000_test",
            "split_seed": int(split_seed),
            "indices": indices,
        }
    )
    return masks, indices, fingerprint


def split_indices_to_masks(
    split_indices: Mapping[str, Any],
    num_nodes: int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]], str]:
    aliases = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    normalized: dict[str, torch.Tensor] = {}
    for source_name, target_name in aliases.items():
        if source_name in split_indices and target_name not in normalized:
            normalized[target_name] = torch.as_tensor(
                split_indices[source_name], dtype=torch.long
            ).view(-1).cpu()
    missing = [name for name in ("train", "valid", "test") if name not in normalized]
    if missing:
        raise ValueError(
            "official split is missing required indices: " + ", ".join(missing)
        )

    masks: dict[str, torch.Tensor] = {}
    serialized: dict[str, list[int]] = {}
    for name in ("train", "valid", "test"):
        indices = normalized[name]
        if indices.numel() == 0:
            raise ValueError(f"official {name} split is empty")
        if int(indices.min()) < 0 or int(indices.max()) >= int(num_nodes):
            raise ValueError(f"official {name} split contains out-of-range indices")
        mask = torch.zeros(int(num_nodes), dtype=torch.bool)
        mask[indices] = True
        masks[name] = mask
        serialized[name] = indices.sort().values.tolist()
    fingerprint = _canonical_json_fingerprint(
        {"protocol": "ogb_official", "indices": serialized}
    )
    return masks, serialized, fingerprint


def validate_node_classification_data(
    data: Any,
    *,
    num_classes: int | None = None,
) -> tuple[Any, int]:
    required = (
        "x",
        "edge_index",
        "y",
        "train_mask",
        "val_mask",
        "test_mask",
    )
    missing = [name for name in required if getattr(data, name, None) is None]
    if missing:
        raise ValueError(
            "node-classification data is missing fields: " + ", ".join(missing)
        )

    data.x = torch.as_tensor(data.x).float()
    data.edge_index = torch.as_tensor(data.edge_index, dtype=torch.long)
    data.y = torch.as_tensor(data.y, dtype=torch.long)
    if data.y.ndim == 2 and data.y.shape[1] == 1:
        data.y = data.y.view(-1)
    if data.y.ndim != 1:
        raise ValueError(f"y must have shape [N], got {tuple(data.y.shape)}")
    if data.x.ndim != 2:
        raise ValueError(f"x must have shape [N,D], got {tuple(data.x.shape)}")
    if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2,E], got {tuple(data.edge_index.shape)}"
        )
    num_nodes = int(data.x.shape[0])
    if data.y.numel() != num_nodes:
        raise ValueError("feature and label node counts differ")
    if num_nodes <= 0 or int(data.x.shape[1]) <= 0:
        raise ValueError("dataset must contain nodes and feature columns")
    if data.edge_index.numel() > 0:
        edge_min = int(data.edge_index.min())
        edge_max = int(data.edge_index.max())
        if edge_min < 0 or edge_max >= num_nodes:
            raise ValueError(
                f"edge_index range [{edge_min}, {edge_max}] is invalid for "
                f"{num_nodes} nodes"
            )

    for name in ("train_mask", "val_mask", "test_mask"):
        mask = torch.as_tensor(getattr(data, name), dtype=torch.bool).view(-1)
        if mask.numel() != num_nodes:
            raise ValueError(f"{name} must have length {num_nodes}")
        if not bool(mask.any()):
            raise ValueError(f"{name} is empty")
        setattr(data, name, mask)
    if bool((data.train_mask & data.val_mask).any()):
        raise ValueError("train_mask and val_mask overlap")
    if bool((data.train_mask & data.test_mask).any()):
        raise ValueError("train_mask and test_mask overlap")
    if bool((data.val_mask & data.test_mask).any()):
        raise ValueError("val_mask and test_mask overlap")

    if int(data.y.min()) < 0:
        raise ValueError("labels must be non-negative")
    inferred_classes = int(data.y.max()) + 1
    resolved_classes = inferred_classes if num_classes is None else int(num_classes)
    if resolved_classes <= 1:
        raise ValueError("node classification requires at least two classes")
    if int(data.y.max()) >= resolved_classes:
        raise ValueError("label is outside [0, num_classes-1]")
    observed = torch.unique(data.y, sorted=True)
    expected = torch.arange(resolved_classes, dtype=torch.long)
    if not torch.equal(observed.cpu(), expected):
        raise ValueError("labels must cover every class in [0, num_classes-1]")
    return data, resolved_classes


def _split_fingerprint_from_masks(
    data: Any,
    *,
    protocol: str,
    split_seed: int | None,
) -> tuple[str, dict[str, list[int]]]:
    indices = {
        "train": torch.nonzero(data.train_mask, as_tuple=False).view(-1).tolist(),
        "valid": torch.nonzero(data.val_mask, as_tuple=False).view(-1).tolist(),
        "test": torch.nonzero(data.test_mask, as_tuple=False).view(-1).tolist(),
    }
    return (
        _canonical_json_fingerprint(
            {
                "protocol": protocol,
                "split_seed": split_seed,
                "indices": indices,
            }
        ),
        indices,
    )


def _build_bundle(
    *,
    data: Any,
    canonical_name: str,
    source_class: str,
    source_version: str | None,
    storage_root: Path,
    split_protocol: str,
    split_seed: int | None,
    split_fingerprint: str,
    split_indices: dict[str, list[int]],
    graph_transform: str,
    original_graph_directed: bool,
    original_num_edges: int,
    training_mode: str,
    num_classes: int | None = None,
) -> DatasetBundle:
    data, resolved_classes = validate_node_classification_data(
        data, num_classes=num_classes
    )
    tensor_fingerprints = {
        "x": _tensor_fingerprint(data.x),
        "y": _tensor_fingerprint(data.y),
        "edge_index": _tensor_fingerprint(data.edge_index),
        "train_mask": _tensor_fingerprint(data.train_mask),
        "val_mask": _tensor_fingerprint(data.val_mask),
        "test_mask": _tensor_fingerprint(data.test_mask),
    }
    content_fingerprint = _canonical_json_fingerprint(tensor_fingerprints)
    try:
        experiment_graph_directed = not bool(data.is_undirected())
    except (AttributeError, TypeError):
        experiment_graph_directed = bool(original_graph_directed)
    manifest = {
        "format_version": DATASET_MANIFEST_FORMAT_VERSION,
        "canonical_name": canonical_name,
        "source_class": source_class,
        "source_version": source_version,
        "storage_root": str(storage_root.resolve()),
        "source_files": _source_file_identifiers(storage_root),
        "num_nodes": int(data.x.shape[0]),
        "num_edges": int(data.edge_index.shape[1]),
        "original_num_edges": int(original_num_edges),
        "num_features": int(data.x.shape[1]),
        "num_classes": int(resolved_classes),
        "original_graph_directed": bool(original_graph_directed),
        "experiment_graph_directed": bool(experiment_graph_directed),
        "graph_transform": graph_transform,
        "train_count": int(data.train_mask.sum()),
        "val_count": int(data.val_mask.sum()),
        "test_count": int(data.test_mask.sum()),
        "split_protocol": split_protocol,
        "split_seed": split_seed,
        "split_fingerprint": split_fingerprint,
        "tensor_fingerprints": tensor_fingerprints,
        "dataset_content_fingerprint": content_fingerprint,
        "torch_version": str(torch.__version__),
        "pyg_version": _distribution_version("torch-geometric"),
        "ogb_version": _distribution_version("ogb"),
        "training_mode": training_mode,
        "metric": "accuracy",
    }
    return DatasetBundle(
        data=data,
        canonical_name=canonical_name,
        source_class=source_class,
        num_nodes=int(data.x.shape[0]),
        num_edges=int(data.edge_index.shape[1]),
        num_features=int(data.x.shape[1]),
        num_classes=resolved_classes,
        metric="accuracy",
        split_protocol=split_protocol,
        split_seed=split_seed,
        split_fingerprint=split_fingerprint,
        graph_transform=graph_transform,
        dataset_content_fingerprint=content_fingerprint,
        manifest=manifest,
        split_indices=split_indices if canonical_name == "dblp" else None,
    )


def _planetoid_roots(request: DatasetRequest) -> tuple[Path, Path]:
    dataset_dir_name = _PLANETOID_NAMES[request.canonical_name]
    requested = Path(request.data_root)
    if (
        request.canonical_name == "cora"
        and request.legacy_cora_root
        and requested.name.lower() == "cora"
    ):
        return requested.parent, requested
    return requested, requested / dataset_dir_name


def load_node_classification_dataset(
    dataset_name: str,
    data_root: str,
    *,
    split_seed: int = 0,
    ogbn_arxiv_edge_mode: str = "undirected",
    legacy_cora_root: bool = False,
    training_mode: str = "full_batch",
) -> DatasetBundle:
    request = DatasetRequest(
        canonical_name=canonicalize_dataset_name(dataset_name),
        data_root=os.path.abspath(os.path.expanduser(data_root)),
        split_seed=int(split_seed),
        ogbn_arxiv_edge_mode=str(ogbn_arxiv_edge_mode).lower(),
        legacy_cora_root=bool(legacy_cora_root),
    )
    if training_mode != "full_batch":
        raise ValueError("this experiment currently supports only full_batch training")

    if request.canonical_name in _PLANETOID_NAMES:
        import torch_geometric.transforms as T
        from torch_geometric.datasets import Planetoid

        pyg_root, storage_root = _planetoid_roots(request)
        dataset = Planetoid(
            root=str(pyg_root),
            name=_PLANETOID_NAMES[request.canonical_name],
            split="public",
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        split_protocol = "planetoid_public"
        split_fingerprint, indices = _split_fingerprint_from_masks(
            data, protocol=split_protocol, split_seed=None
        )
        return _build_bundle(
            data=data,
            canonical_name=request.canonical_name,
            source_class="torch_geometric.datasets.Planetoid",
            source_version=_distribution_version("torch-geometric"),
            storage_root=storage_root,
            split_protocol=split_protocol,
            split_seed=None,
            split_fingerprint=split_fingerprint,
            split_indices=indices,
            graph_transform="NormalizeFeatures",
            original_graph_directed=not bool(data.is_undirected()),
            original_num_edges=int(data.edge_index.shape[1]),
            training_mode=training_mode,
            num_classes=int(dataset.num_classes),
        )

    if request.canonical_name == "dblp":
        import torch_geometric.transforms as T
        from torch_geometric.datasets import CitationFull

        storage_root = Path(request.data_root) / "CitationFull-DBLP"
        dataset = CitationFull(
            root=str(storage_root),
            name="DBLP",
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        masks, indices, split_fingerprint = deterministic_planetoid_like_split(
            data.y,
            split_seed=request.split_seed,
            train_per_class=20,
            num_val=500,
            num_test=1000,
        )
        data.train_mask = masks["train"]
        data.val_mask = masks["valid"]
        data.test_mask = masks["test"]
        return _build_bundle(
            data=data,
            canonical_name="dblp",
            source_class="torch_geometric.datasets.CitationFull",
            source_version=_distribution_version("torch-geometric"),
            storage_root=storage_root,
            split_protocol="planetoid_like_20_per_class_500_val_1000_test",
            split_seed=request.split_seed,
            split_fingerprint=split_fingerprint,
            split_indices=indices,
            graph_transform="NormalizeFeatures",
            original_graph_directed=not bool(data.is_undirected()),
            original_num_edges=int(data.edge_index.shape[1]),
            training_mode=training_mode,
            num_classes=int(dataset.num_classes),
        )

    if request.canonical_name == "flickr":
        from torch_geometric.datasets import Flickr

        storage_root = Path(request.data_root) / "Flickr"
        dataset = Flickr(root=str(storage_root))
        data = dataset[0]
        split_protocol = "flickr_official"
        split_fingerprint, indices = _split_fingerprint_from_masks(
            data, protocol=split_protocol, split_seed=None
        )
        return _build_bundle(
            data=data,
            canonical_name="flickr",
            source_class="torch_geometric.datasets.Flickr",
            source_version=_distribution_version("torch-geometric"),
            storage_root=storage_root,
            split_protocol=split_protocol,
            split_seed=None,
            split_fingerprint=split_fingerprint,
            split_indices=indices,
            graph_transform="official_features_no_additional_normalization",
            original_graph_directed=not bool(data.is_undirected()),
            original_num_edges=int(data.edge_index.shape[1]),
            training_mode=training_mode,
            num_classes=int(dataset.num_classes),
        )

    if request.canonical_name == "ogbn-arxiv":
        if request.ogbn_arxiv_edge_mode not in ("directed", "undirected"):
            raise ValueError(
                "ogbn_arxiv_edge_mode must be 'directed' or 'undirected'"
            )
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError as exc:
            raise RuntimeError(
                "ogb is required for ogbn-arxiv; install it without changing "
                "PyTorch/PyG using: python -m pip install ogb"
            ) from exc
        from torch_geometric.utils import is_undirected, to_undirected

        ogb_root = Path(request.data_root) / "OGB"
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(ogb_root))
        data = dataset[0]
        original_num_edges = int(data.edge_index.shape[1])
        original_graph_directed = not bool(
            is_undirected(data.edge_index, num_nodes=int(data.num_nodes))
        )
        if request.ogbn_arxiv_edge_mode == "undirected":
            data.edge_index = to_undirected(
                data.edge_index, num_nodes=int(data.num_nodes)
            )
            graph_transform = "to_undirected"
        else:
            graph_transform = "preserve_directed"
        data.y = torch.as_tensor(data.y, dtype=torch.long).view(-1)
        masks, indices, split_fingerprint = split_indices_to_masks(
            dataset.get_idx_split(), int(data.num_nodes)
        )
        data.train_mask = masks["train"]
        data.val_mask = masks["valid"]
        data.test_mask = masks["test"]
        storage_root = ogb_root / "ogbn_arxiv"
        return _build_bundle(
            data=data,
            canonical_name="ogbn-arxiv",
            source_class="ogb.nodeproppred.PygNodePropPredDataset",
            source_version=_distribution_version("ogb"),
            storage_root=storage_root,
            split_protocol="ogb_official",
            split_seed=None,
            split_fingerprint=split_fingerprint,
            split_indices=indices,
            graph_transform=graph_transform,
            original_graph_directed=original_graph_directed,
            original_num_edges=original_num_edges,
            training_mode=training_mode,
            num_classes=int(dataset.num_classes),
        )

    raise AssertionError(f"unhandled canonical dataset {request.canonical_name!r}")


def load_dataset_from_request(request: DatasetRequest) -> DatasetBundle:
    return load_node_classification_dataset(
        request.canonical_name,
        request.data_root,
        split_seed=request.split_seed,
        ogbn_arxiv_edge_mode=request.ogbn_arxiv_edge_mode,
        legacy_cora_root=request.legacy_cora_root,
    )


def dataset_context_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [key for key in DATASET_CONTEXT_FIELDS if key not in manifest]
    if missing:
        raise ValueError(
            "dataset manifest is missing context fields: " + ", ".join(missing)
        )
    return {key: manifest[key] for key in DATASET_CONTEXT_FIELDS}


def assert_dataset_context_matches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str,
) -> None:
    differing = [
        key
        for key in DATASET_CONTEXT_FIELDS
        if expected.get(key) != actual.get(key)
    ]
    if differing:
        raise ValueError(
            f"{context} dataset context mismatch for fields: "
            + ", ".join(differing)
        )


def evaluation_cache_key(
    candidate_fingerprint: str,
    dataset_context: Mapping[str, Any],
    *,
    evaluation_fidelity: str,
    evaluation_seed: int,
) -> str:
    return _canonical_json_fingerprint(
        {
            "candidate_fingerprint": str(candidate_fingerprint),
            "evaluation_fidelity": str(evaluation_fidelity),
            "evaluation_seed": int(evaluation_seed),
            "dataset_context": {
                key: dataset_context.get(key) for key in DATASET_CONTEXT_FIELDS
            },
        }
    )


def node_classification_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    predicted = torch.as_tensor(predictions).view(-1)
    target = torch.as_tensor(labels, dtype=torch.long).view(-1)
    selected = torch.as_tensor(mask, dtype=torch.bool).view(-1)
    if predicted.numel() != target.numel() or selected.numel() != target.numel():
        raise ValueError("predictions, labels, and mask must have the same length")
    count = int(selected.sum())
    if count <= 0:
        raise ValueError("accuracy mask is empty")
    return float(predicted[selected].eq(target[selected]).sum().item() / count)


def write_dataset_artifacts(
    bundle: DatasetBundle,
    output_dir: str | os.PathLike[str],
) -> tuple[str, str | None]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "dataset_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert_dataset_context_matches(
            dataset_context_from_manifest(saved),
            bundle.context,
            context=str(manifest_path),
        )
    else:
        _atomic_json_dump(bundle.manifest, manifest_path)

    split_path: Path | None = None
    if bundle.canonical_name == "dblp":
        if bundle.split_indices is None:
            raise ValueError("DBLP bundle is missing deterministic split indices")
        split_path = target / "dblp_split_indices.json"
        split_payload = {
            "split_protocol": bundle.split_protocol,
            "split_seed": bundle.split_seed,
            "split_fingerprint": bundle.split_fingerprint,
            "indices": bundle.split_indices,
        }
        if split_path.exists():
            saved_split = json.loads(split_path.read_text(encoding="utf-8"))
            if saved_split != split_payload:
                raise ValueError(
                    f"refusing to overwrite mismatched DBLP split: {split_path}"
                )
        else:
            _atomic_json_dump(split_payload, split_path)
    return str(manifest_path), None if split_path is None else str(split_path)


def validate_expected_dataset_manifest(
    bundle: DatasetBundle,
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"expected dataset manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset manifest must be a JSON object: {path}")
    assert_dataset_context_matches(
        dataset_context_from_manifest(payload),
        bundle.context,
        context=str(path),
    )
    expected_tensors = payload.get("tensor_fingerprints")
    actual_tensors = bundle.manifest.get("tensor_fingerprints")
    if expected_tensors != actual_tensors:
        raise ValueError(
            f"{path} tensor fingerprints do not match the loaded dataset"
        )
    return payload


def _atomic_json_dump(payload: Any, path: Path) -> None:
    import tempfile

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def legacy_cora_context() -> dict[str, Any]:
    return {
        "canonical_name": "cora",
        "source_class": "torch_geometric.datasets.Planetoid",
        "source_version": None,
        "torch_version": None,
        "pyg_version": None,
        "ogb_version": None,
        "dataset_content_fingerprint": "legacy_unrecorded",
        "split_protocol": "planetoid_public",
        "split_seed": None,
        "split_fingerprint": "legacy_unrecorded",
        "graph_transform": "NormalizeFeatures",
        "training_mode": "full_batch",
        "metric": "accuracy",
    }


def history_dataset_context(
    history_path: str | os.PathLike[str],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    history = Path(history_path)
    for filename in ("history_metadata.json", "dataset_manifest.json"):
        candidate = history.parent / filename
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        raw_context = payload.get("dataset_context", payload)
        return dataset_context_from_manifest(raw_context), False

    recorded_contexts = [
        row.get("dataset_context")
        for row in records
        if isinstance(row.get("dataset_context"), Mapping)
    ]
    if recorded_contexts:
        first = dataset_context_from_manifest(recorded_contexts[0])
        for index, context in enumerate(recorded_contexts[1:], start=1):
            assert_dataset_context_matches(
                first,
                dataset_context_from_manifest(context),
                context=f"history record {index}",
            )
        return first, False
    return legacy_cora_context(), True
