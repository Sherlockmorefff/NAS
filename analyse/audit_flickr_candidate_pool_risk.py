"""Decode frozen Flickr pools and report deterministic full-batch risk proxies."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_phase4
from eval_utils import decode_hp_by_mode, stable_seed
from initialization_wgmm_ted import candidate_fingerprint
from surrogate.checkpoint_io import atomic_json_dump


TARGET_FINGERPRINT = (
    "1b09dd32bcdd4427f5de3cc41b9720957945028c9d6f5621c7e987008e997072"
)
TARGET_OPERATIONS = [
    "GATConv",
    "GATConv",
    "GCNConv",
    "SAGEConv",
    "GINConv",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _load_pool(path: Path) -> tuple[torch.Tensor, str]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("z_search"), torch.Tensor):
        raise ValueError(f"invalid candidate pool payload: {path}")
    rows = payload["z_search"].detach().cpu().float()
    if rows.ndim != 2 or rows.shape != (768, 16):
        raise ValueError(f"unexpected candidate pool shape {tuple(rows.shape)}: {path}")
    expected = str(payload.get("fingerprint"))
    actual = bo_phase4.fingerprint_array(rows.numpy())
    if actual != expected:
        raise ValueError(
            f"candidate pool fingerprint mismatch for {path}: {actual} != {expected}"
        )
    return rows, actual


def _decode_row(
    vae,
    z_search: torch.Tensor,
    *,
    seed: int,
    index: int,
    device: torch.device,
) -> dict[str, Any]:
    fingerprint = candidate_fingerprint(z_search.numpy())
    decoder_seed = stable_seed(int(seed), "decoder", z_search[: bo_phase4.ARCH_NZ])
    config = bo_phase4.decode_arch(
        vae,
        z_search[: bo_phase4.ARCH_NZ],
        device,
        n_trials=5,
        decoder_seed=decoder_seed,
    )
    if config is None:
        return {
            "candidate_pool_index": int(index),
            "candidate_fingerprint": fingerprint,
            "decoder_seed": int(decoder_seed),
            "decoded": False,
        }
    hp = decode_hp_by_mode(
        z_search,
        bo_phase4.ARCH_NZ,
        config,
        "global4",
        -4.0,
        -1.5,
        0.1,
        0.6,
    )
    operations = [str(value) for value in config.get("operations", [])]
    effective_layers = len([op for op in operations if op != "Identity"])
    gat_count = operations.count("GATConv")
    hidden = int(hp["hidden_dim"])
    strict_similar_risk = hidden == 512 and effective_layers == 5 and gat_count >= 2
    front_loaded_target_risk = (
        hidden == 512
        and len(operations) >= 2
        and operations[:2] == ["GATConv", "GATConv"]
    )
    exact_architecture_risk = hidden == 512 and operations == TARGET_OPERATIONS
    return {
        "candidate_pool_index": int(index),
        "candidate_fingerprint": fingerprint,
        "decoder_seed": int(decoder_seed),
        "decoded": True,
        "operations": operations,
        "effective_layers": int(effective_layers),
        "hidden_dim": hidden,
        "gat_layer_count": int(gat_count),
        "strict_similar_risk": bool(strict_similar_risk),
        "front_loaded_target_risk": bool(front_loaded_target_risk),
        "exact_architecture_risk": bool(exact_architecture_risk),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    logger = logging.getLogger("audit_flickr_candidate_pool_risk")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    vae = bo_phase4.load_vae(
        SimpleNamespace(hp_mode="global4", checkpoint=args.checkpoint),
        device,
        logger,
    )
    pool_root = Path(args.pool_root)
    all_risk_rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    total = 0
    total_decoded = 0
    for seed in args.seeds:
        path = pool_root / f"search_seed{int(seed)}" / "candidate_pool.pt"
        pool, pool_fingerprint = _load_pool(path)
        rows = [
            _decode_row(vae, z, seed=int(seed), index=index, device=device)
            for index, z in enumerate(pool)
        ]
        decoded = [row for row in rows if row["decoded"]]
        risk_rows = [row for row in decoded if row["strict_similar_risk"]]
        front_rows = [row for row in decoded if row["front_loaded_target_risk"]]
        exact_rows = [row for row in decoded if row["exact_architecture_risk"]]
        target_rows = [
            row for row in rows if row["candidate_fingerprint"] == TARGET_FINGERPRINT
        ]
        all_risk_rows.extend({"search_seed": int(seed), **row} for row in risk_rows)
        total += len(rows)
        total_decoded += len(decoded)
        per_seed[str(int(seed))] = {
            "candidate_pool_path": str(path.resolve()),
            "candidate_pool_fingerprint": pool_fingerprint,
            "candidate_count": len(rows),
            "decoded_count": len(decoded),
            "strict_similar_risk_count": len(risk_rows),
            "strict_similar_risk_fraction_of_pool": len(risk_rows) / len(rows),
            "front_loaded_target_risk_count": len(front_rows),
            "front_loaded_target_risk_fraction_of_pool": len(front_rows) / len(rows),
            "exact_architecture_risk_count": len(exact_rows),
            "target_fingerprint_count": len(target_rows),
            "target_rows": target_rows,
        }
        logger.info(
            "seed=%d decoded=%d strict=%d front_loaded=%d exact=%d target=%d",
            seed,
            len(decoded),
            len(risk_rows),
            len(front_rows),
            len(exact_rows),
            len(target_rows),
        )
    strict_total = len(all_risk_rows)
    front_total = sum(
        row["front_loaded_target_risk_count"] for row in per_seed.values()
    )
    exact_total = sum(row["exact_architecture_risk_count"] for row in per_seed.values())
    payload = {
        "status": "completed",
        "dataset": "flickr",
        "candidate_pool_seeds": [int(seed) for seed in args.seeds],
        "candidate_pool_count": int(total),
        "decoded_candidate_count": int(total_decoded),
        "risk_proxy_definitions": {
            "strict_similar_risk": (
                "hidden_dim == 512 and effective_layers == 5 and "
                "gat_layer_count >= 2"
            ),
            "front_loaded_target_risk": (
                "hidden_dim == 512 and operations[0:2] == "
                "['GATConv', 'GATConv']"
            ),
            "exact_architecture_risk": (
                "hidden_dim == 512 and operations exactly equal the failed candidate"
            ),
            "interpretation": (
                "deterministic structural screening proxies, not measured OOM labels"
            ),
        },
        "strict_similar_risk_count": int(strict_total),
        "strict_similar_risk_fraction_of_all_pools": strict_total / total,
        "front_loaded_target_risk_count": int(front_total),
        "front_loaded_target_risk_fraction_of_all_pools": front_total / total,
        "exact_architecture_risk_count": int(exact_total),
        "exact_architecture_risk_fraction_of_all_pools": exact_total / total,
        "target_fingerprint_count": sum(
            row["target_fingerprint_count"] for row in per_seed.values()
        ),
        "per_seed": per_seed,
        "strict_similar_risk_candidates": all_risk_rows,
        "formal_training_or_evaluation_performed": False,
    }
    atomic_json_dump(payload, args.output)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
