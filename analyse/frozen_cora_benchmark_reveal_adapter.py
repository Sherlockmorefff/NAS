"""Frozen Cora adapter for the one-time checkpoint reveal.

This module has no import-time dataset or deep-learning side effects.  Its sole
factory is called by the reveal worker only after the irreversible claim.  The
checkpoint-freeze goal freezes this file's SHA-256 but never calls the factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def build_adapter(bundle: Any, cora_root: Path) -> Mapping[str, Any]:
    # These imports and the first access to the benchmark are intentionally
    # inside the post-claim factory.
    import torch
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from analyse import run_final_validation_checkpoints as runner

    data = Planetoid(root=str(cora_root), name="Cora", transform=NormalizeFeatures())[0]
    view = {
        "model_args": [data.x.cpu(), data.edge_index.cpu()],
        "model_kwargs": {},
        "benchmark_labels": data.y.cpu(),
        "benchmark_mask": data.test_mask.cpu(),
        "num_features": int(data.x.shape[1]),
        "num_classes": 7,
    }

    def load_benchmark_view() -> Mapping[str, Any]:
        return view

    def build_model(payload: Mapping[str, Any], record: Mapping[str, Any]) -> Any:
        shape_view = {
            "num_features": view["num_features"],
            "num_classes": view["num_classes"],
        }
        model = runner.build_model(
            bundle.candidates[str(record["role"])], shape_view, torch.device("cpu")
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        return model

    def score_output(output: Any, values: Mapping[str, Any]) -> float:
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("non-finite benchmark logits")
        mask = values["benchmark_mask"]
        count = int(mask.sum().item())
        if count <= 0:
            raise RuntimeError("benchmark mask is empty")
        correct = (
            output.argmax(dim=1)[mask]
            .eq(values["benchmark_labels"][mask])
            .sum()
            .item()
        )
        return float(correct / count)

    return {
        "load_benchmark_view": load_benchmark_view,
        "build_model": build_model,
        "score_output": score_output,
    }
