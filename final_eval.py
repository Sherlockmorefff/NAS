"""Final multi-seed evaluation for hp_mode-based NAS search results."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import torch

sys.path.insert(0, "/mnt/project")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from eval_utils import (
    DEFAULT_GAT_HEADS,
    DEFAULT_GIN_EPS,
    DEFAULT_SAGE_AGGR,
    decode_hp_by_mode,
    train_and_eval_arch,
)
from hp_modes import (
    condition_mask_dict_from_ops,
    hp_dim_from_mode,
    hp_names_from_mode,
    validate_hp_mode,
)
from nas_space import JointSpaceVAE


ARCH_NZ_DEFAULT = 12
HP_MODE_CHOICES = ("global4", "hybrid_cond7", "layer_cond19")
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.5
DEFAULT_HIDDEN_DIM = 64
DEFAULT_L2 = 5e-4


def setup_logger(log_dir: str, script_name: str, version: str):
    ts = datetime.now().strftime("%m%d_%H%M")
    log_subdir = os.path.join(log_dir, script_name)
    os.makedirs(log_subdir, exist_ok=True)
    log_path = os.path.join(log_subdir, f"train_{version}_{ts}.log")

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.info(f"log file: {log_path}")
    return logger, log_path


def save_args_json(args: argparse.Namespace, log_path: str) -> str:
    json_path = log_path.replace(".log", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final evaluation with hp_mode decoding")
    parser.add_argument("--cora_root", type=str, default="/tmp/Cora")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--eval_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/final_eval")
    parser.add_argument("--version", type=str, default="final_eval")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch_nz", type=int, default=ARCH_NZ_DEFAULT)
    parser.add_argument("--hp_mode", type=str, default="global4", choices=HP_MODE_CHOICES)
    parser.add_argument("--checkpoint", type=str, default="results/joint_search/joint_model_global4_best.pth")
    parser.add_argument("--auto_best_z", type=str, default="results/bo_phase4/best_z_final.pt")
    parser.add_argument("--disable_auto_best", action="store_true")
    parser.add_argument("--include_official_2gcn_baseline", action="store_true")
    parser.add_argument("--auto_name", type=str, default="NAS_full")
    parser.add_argument("--auto_decode_trials", type=int, default=10)
    parser.add_argument("--log_lr_min", type=float, default=-4.0)
    parser.add_argument("--log_lr_max", type=float, default=-1.5)
    parser.add_argument("--dropout_min", type=float, default=0.1)
    parser.add_argument("--dropout_max", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_L2)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--gcnii_alpha", type=float, default=0.1)
    parser.add_argument("--gcnii_theta", type=float, default=0.5)
    return parser.parse_args()


class ArchArgs:
    def __init__(self, arch_nz: int):
        self.max_n = 7
        self.num_vertex_type = 8
        self.START_TYPE = 0
        self.END_TYPE = 1
        self.hs = 501
        self.nz = int(arch_nz)
        self.bidirectional = True


def _torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_cora(root: str, device: torch.device):
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    dataset = Planetoid(root=pyg_root, name="Cora", transform=T.NormalizeFeatures())
    return dataset[0].to(device), dataset.num_features, dataset.num_classes


def load_vae(args: argparse.Namespace, device: torch.device, logger) -> JointSpaceVAE | None:
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        logger.warning(f"checkpoint not found; skip auto best_z candidates: {args.checkpoint}")
        return None
    hp_dim = hp_dim_from_mode(args.hp_mode)
    model = JointSpaceVAE(
        ArchArgs(args.arch_nz),
        hp_mode=args.hp_mode,
        hp_latent_dim=hp_dim,
        hp_input_dim=hp_dim,
    ).to(device)
    state = _torch_load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(
        f"VAE loaded: {args.checkpoint} "
        f"(arch_nz={args.arch_nz}, hp_mode={args.hp_mode}, hp_dim={hp_dim})"
    )
    return model


def decode_arch_from_z(
    vae: JointSpaceVAE,
    z_arch: torch.Tensor,
    device: torch.device,
    n_trials: int,
) -> dict[str, Any] | None:
    results = []
    z_arch = z_arch.detach().float()
    with torch.no_grad():
        for _ in range(int(n_trials)):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(device))
            graph = graphs[0]
            ops = [
                vae.op_mapping.get(graph.vs[i]["type"], "Identity")
                for i in range(1, graph.vcount() - 1)
            ]
            edges = graph.get_edgelist()
            effective_layers = sum(1 for op in ops if op != "Identity")
            if effective_layers > 0:
                key = (tuple(ops), tuple(edges))
                results.append(
                    (
                        key,
                        {
                            "effective_layers": effective_layers,
                            "operations": ops,
                            "edges": edges,
                        },
                    )
                )
    if not results:
        return None
    best_key = Counter(row[0] for row in results).most_common(1)[0][0]
    return next(config for key, config in results if key == best_key)


def config_from_ops(ops: list[str], edges: list[tuple[int, int]] | list[list[int]]) -> dict[str, Any]:
    return {
        "operations": list(ops),
        "edges": [tuple(edge) for edge in edges],
        "effective_layers": sum(1 for op in ops if op != "Identity"),
    }


def default_hp(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mask = condition_mask_dict_from_ops(list(config.get("operations", [])), args.hp_mode)
    return {
        "lr": DEFAULT_LR,
        "dropout": DEFAULT_DROPOUT,
        "hidden_dim": int(args.hidden_dim),
        "weight_decay": float(args.weight_decay),
        "l2": float(args.weight_decay),
        "gat_heads": DEFAULT_GAT_HEADS,
        "sage_aggr": DEFAULT_SAGE_AGGR,
        "gin_eps": DEFAULT_GIN_EPS,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
        "condition_mask": mask,
        "condition_mask_vector": mask["vector"],
        "hp_norm": None,
        "hp_mode": args.hp_mode,
        "hp_dim": hp_dim_from_mode(args.hp_mode),
    }


def official_2gcn_hp(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    hp = default_hp(config, args)
    hp.update(
        {
            "lr": 0.01,
            "dropout": 0.5,
            "hidden_dim": 16,
            "weight_decay": 5e-4,
            "l2": 5e-4,
        }
    )
    return hp


def candidate_signature(cand: dict[str, Any]) -> tuple:
    return (
        tuple(cand["operations"]),
        tuple(tuple(edge) for edge in cand["edges"]),
        round(float(cand["lr"]), 12),
        round(float(cand["dropout"]), 12),
        int(cand["hidden_dim"]),
        round(float(cand["l2"]), 12),
        int(cand["gat_heads"]),
        str(cand["sage_aggr"]),
        round(float(cand["gin_eps"]), 12),
        tuple(cand["gat_heads_by_layer"] or []),
        tuple(cand["sage_aggr_by_layer"] or []),
        tuple(cand["gin_eps_by_layer"] or []),
    )


def make_candidate(
    name: str,
    group: str,
    description: str,
    config: dict[str, Any],
    hp: dict[str, Any],
    args: argparse.Namespace,
    source_best_z: str | None = None,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "group": group,
        "description": description,
        "hp_mode": args.hp_mode,
        "operations": list(config.get("operations", [])),
        "edges": [list(edge) for edge in config.get("edges", [])],
        "lr": float(hp["lr"]),
        "dropout": float(hp["dropout"]),
        "hidden_dim": int(hp["hidden_dim"]),
        "l2": float(hp.get("weight_decay", hp.get("l2", args.weight_decay))),
        "gat_heads": int(hp.get("gat_heads", DEFAULT_GAT_HEADS)),
        "sage_aggr": str(hp.get("sage_aggr", DEFAULT_SAGE_AGGR)),
        "gin_eps": float(hp.get("gin_eps", DEFAULT_GIN_EPS)),
        "gat_heads_by_layer": hp.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": hp.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": hp.get("gin_eps_by_layer"),
        "gcnii_alpha": float(args.gcnii_alpha),
        "gcnii_theta": float(args.gcnii_theta),
        "condition_mask": hp.get("condition_mask", {}),
        "condition_mask_vector": hp.get("condition_mask_vector"),
        "source_best_z": source_best_z,
        "source_checkpoint": source_checkpoint,
    }


def append_unique(candidates: list[dict[str, Any]], cand: dict[str, Any], logger) -> None:
    sig = candidate_signature(cand)
    if any(candidate_signature(existing) == sig for existing in candidates):
        logger.info(f"skip duplicate final-eval candidate: {cand['name']}")
        return
    candidates.append(cand)


def build_auto_candidates(
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> list[dict[str, Any]]:
    if args.disable_auto_best:
        logger.info("auto best_z disabled by --disable_auto_best")
        return []
    if not args.auto_best_z or not os.path.exists(args.auto_best_z):
        logger.warning(f"auto_best_z not found; skip auto candidates: {args.auto_best_z}")
        return []

    vae = load_vae(args, device, logger)
    if vae is None:
        return []

    hp_dim = hp_dim_from_mode(args.hp_mode)
    search_dim = int(args.arch_nz) + hp_dim
    z_search = torch.as_tensor(_torch_load(args.auto_best_z, map_location="cpu"), dtype=torch.float32).flatten()
    if z_search.numel() != search_dim:
        raise ValueError(
            f"auto_best_z dimension mismatch for hp_mode={args.hp_mode}: "
            f"expected {search_dim}, got {z_search.numel()}"
        )

    nas_config = decode_arch_from_z(vae, z_search[: args.arch_nz], device, args.auto_decode_trials)
    if nas_config is None:
        logger.warning(f"failed to decode auto_best_z architecture: {args.auto_best_z}")
        return []

    full_hp = decode_hp_by_mode(
        z_search=z_search,
        arch_nz=args.arch_nz,
        config=nas_config,
        hp_mode=args.hp_mode,
        log_lr_min=args.log_lr_min,
        log_lr_max=args.log_lr_max,
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
        default_hidden_dim=args.hidden_dim,
        default_weight_decay=args.weight_decay,
    )
    default_nas_hp = default_hp(nas_config, args)

    manual_config = config_from_ops(
        ["GCNConv", "GCNConv", "Identity"],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    manual_hp = decode_hp_by_mode(
        z_search=z_search,
        arch_nz=args.arch_nz,
        config=manual_config,
        hp_mode=args.hp_mode,
        log_lr_min=args.log_lr_min,
        log_lr_max=args.log_lr_max,
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
        default_hidden_dim=args.hidden_dim,
        default_weight_decay=args.weight_decay,
    )

    default_global_full_cond = dict(full_hp)
    default_global_full_cond.update(
        {
            "lr": DEFAULT_LR,
            "dropout": DEFAULT_DROPOUT,
            "hidden_dim": int(args.hidden_dim),
            "weight_decay": float(args.weight_decay),
            "l2": float(args.weight_decay),
        }
    )

    searched_global_default_cond = default_hp(nas_config, args)
    searched_global_default_cond.update(
        {
            "lr": full_hp["lr"],
            "dropout": full_hp["dropout"],
            "hidden_dim": full_hp["hidden_dim"],
            "weight_decay": full_hp["weight_decay"],
            "l2": full_hp["l2"],
        }
    )

    source_best_z = args.auto_best_z
    source_checkpoint = args.checkpoint
    candidates: list[dict[str, Any]] = []
    for cand in [
        make_candidate(
            "Manual_2xGCN_searchedHP",
            "hp_on_manual",
            "Manual 2xGCN architecture with searched global HP decoded from best_z.",
            manual_config,
            manual_hp,
            args,
            source_best_z,
            source_checkpoint,
        ),
        make_candidate(
            "NAS_arch_defaultHP",
            "arch_only",
            "Decoded NAS architecture with default HP.",
            nas_config,
            default_nas_hp,
            args,
            source_best_z,
            source_checkpoint,
        ),
        make_candidate(
            "NAS_arch_globalHPOnly",
            "arch_global_hp",
            "Decoded NAS architecture with searched global HP and default conditional HP.",
            nas_config,
            searched_global_default_cond,
            args,
            source_best_z,
            source_checkpoint,
        ),
        make_candidate(
            "NAS_arch_condHPOnly",
            "cond_only",
            "Decoded NAS architecture with default global HP and searched conditional HP.",
            nas_config,
            default_global_full_cond,
            args,
            source_best_z,
            source_checkpoint,
        ),
        make_candidate(
            args.auto_name,
            "full",
            "Decoded NAS architecture with searched global and conditional HP.",
            nas_config,
            full_hp,
            args,
            source_best_z,
            source_checkpoint,
        ),
    ]:
        append_unique(candidates, cand, logger)

    logger.info(
        f"auto candidates built: {len(candidates)} "
        f"ops={nas_config.get('operations', [])} hp_mode={args.hp_mode}"
    )
    return candidates


def build_candidates(args: argparse.Namespace, device: torch.device, logger) -> list[dict[str, Any]]:
    base_2gcn = config_from_ops(
        ["GCNConv", "GCNConv", "Identity"],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    base_1gcn = config_from_ops(
        ["GCNConv", "Identity", "Identity"],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    candidates: list[dict[str, Any]] = [
        make_candidate(
            "BASE_2xGCN_defaultHP",
            "baseline",
            "Manual 2xGCN baseline with default HP.",
            base_2gcn,
            default_hp(base_2gcn, args),
            args,
        ),
        make_candidate(
            "BASE_1xGCN_defaultHP",
            "baseline",
            "Manual 1xGCN reference with default HP.",
            base_1gcn,
            default_hp(base_1gcn, args),
            args,
        ),
    ]
    if args.include_official_2gcn_baseline:
        append_unique(
            candidates,
            make_candidate(
                "BASE_2xGCN_officialHP",
                "baseline_official_hp",
                "Manual 2xGCN baseline with official GCN paper-style HP.",
                base_2gcn,
                official_2gcn_hp(base_2gcn, args),
                args,
            ),
            logger,
        )
    for cand in build_auto_candidates(args, device, logger):
        append_unique(candidates, cand, logger)
    return candidates


def run_eval(cand: dict[str, Any], data, in_ch: int, out_ch: int, args: argparse.Namespace, device: torch.device, logger):
    config = {
        "operations": list(cand["operations"]),
        "edges": [tuple(edge) for edge in cand["edges"]],
        "effective_layers": sum(1 for op in cand["operations"] if op != "Identity"),
    }
    val_accs: list[float] = []
    test_accs: list[float] = []
    valid_list: list[bool] = []

    for seed in range(int(args.n_seeds)):
        val_acc, is_valid, test_acc = train_and_eval_arch(
            config=config,
            data=data,
            in_ch=in_ch,
            out_ch=out_ch,
            lr=float(cand["lr"]),
            dropout=float(cand["dropout"]),
            hidden_dim=int(cand["hidden_dim"]),
            weight_decay=float(cand["l2"]),
            gcnii_alpha=float(cand.get("gcnii_alpha", args.gcnii_alpha)),
            gcnii_theta=float(cand.get("gcnii_theta", args.gcnii_theta)),
            gat_heads=int(cand.get("gat_heads", DEFAULT_GAT_HEADS)),
            sage_aggr=str(cand.get("sage_aggr", DEFAULT_SAGE_AGGR)),
            gin_eps=float(cand.get("gin_eps", DEFAULT_GIN_EPS)),
            gat_heads_by_layer=cand.get("gat_heads_by_layer"),
            sage_aggr_by_layer=cand.get("sage_aggr_by_layer"),
            gin_eps_by_layer=cand.get("gin_eps_by_layer"),
            device=device,
            max_epochs=int(args.eval_epochs),
            patience=int(args.patience),
            seed=seed,
            track_test=True,
        )
        val_accs.append(float(val_acc))
        test_accs.append(float(test_acc))
        valid_list.append(bool(is_valid))
        logger.info(
            f"  seed {seed:>2d}: val={val_acc:.4f} test={test_acc:.4f} "
            f"{'[valid]' if is_valid else '[invalid]'}"
        )

    valid_val_accs = [value for value, is_valid in zip(val_accs, valid_list) if is_valid]
    valid_test_accs = [value for value, is_valid in zip(test_accs, valid_list) if is_valid]
    n_valid = len(valid_val_accs)
    n_invalid = len(valid_list) - n_valid
    val_mean_all = float(np.mean(val_accs))
    val_std_all = float(np.std(val_accs))
    test_mean_all = float(np.mean(test_accs))
    test_std_all = float(np.std(test_accs))
    if n_valid:
        val_mean_valid_only = float(np.mean(valid_val_accs))
        val_std_valid_only = float(np.std(valid_val_accs))
        test_mean_valid_only = float(np.mean(valid_test_accs))
        test_std_valid_only = float(np.std(valid_test_accs))
    else:
        val_mean_valid_only = None
        val_std_valid_only = None
        test_mean_valid_only = None
        test_std_valid_only = None
        logger.warning(f"all seeds invalid for final-eval candidate: {cand['name']}")

    return {
        "name": cand["name"],
        "group": cand["group"],
        "description": cand["description"],
        "hp_mode": args.hp_mode,
        "val_mean": val_mean_all,
        "val_std": val_std_all,
        "val_list": val_accs,
        "test_mean": test_mean_all,
        "test_std": test_std_all,
        "test_list": test_accs,
        "valid_list": valid_list,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "val_mean_all": val_mean_all,
        "val_std_all": val_std_all,
        "test_mean_all": test_mean_all,
        "test_std_all": test_std_all,
        "val_mean_valid_only": val_mean_valid_only,
        "val_std_valid_only": val_std_valid_only,
        "test_mean_valid_only": test_mean_valid_only,
        "test_std_valid_only": test_std_valid_only,
        "lr": float(cand["lr"]),
        "dropout": float(cand["dropout"]),
        "hidden_dim": int(cand["hidden_dim"]),
        "l2": float(cand["l2"]),
        "gat_heads": int(cand.get("gat_heads", DEFAULT_GAT_HEADS)),
        "sage_aggr": str(cand.get("sage_aggr", DEFAULT_SAGE_AGGR)),
        "gin_eps": float(cand.get("gin_eps", DEFAULT_GIN_EPS)),
        "gat_heads_by_layer": cand.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": cand.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": cand.get("gin_eps_by_layer"),
        "operations": list(cand["operations"]),
        "edges": [list(edge) for edge in cand["edges"]],
        "source_best_z": cand.get("source_best_z"),
        "source_checkpoint": cand.get("source_checkpoint"),
        "condition_mask": cand.get("condition_mask"),
        "condition_mask_vector": cand.get("condition_mask_vector"),
    }


def main() -> None:
    args = parse_args()
    if args.n_seeds <= 0:
        raise ValueError("--n_seeds must be positive")
    args.hp_mode = validate_hp_mode(args.hp_mode)
    hp_dim = hp_dim_from_mode(args.hp_mode)
    search_dim = int(args.arch_nz) + hp_dim
    os.makedirs(args.output, exist_ok=True)

    logger, log_path = setup_logger(args.log_dir, "final_eval", args.version)
    save_args_json(args, log_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    logger.info("=" * 70)
    logger.info(f"final_eval.py -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info(f"Device={device} hp_mode={args.hp_mode} hp_dim={hp_dim} search_dim={search_dim}")
    logger.info(f"hp_names={hp_names_from_mode(args.hp_mode)}")

    data, in_ch, out_ch = load_cora(args.cora_root, device)
    logger.info(f"Cora: {in_ch} features, {out_ch} classes")

    candidates = build_candidates(args, device, logger)
    summary: list[dict[str, Any]] = []

    for cand in candidates:
        logger.info("\n" + "=" * 66)
        logger.info(f"[{cand['group']}] {cand['name']}")
        logger.info(f"  {cand['description']}")
        logger.info(
            f"  lr={cand['lr']:.5f} dropout={cand['dropout']:.3f} "
            f"hidden={cand['hidden_dim']} l2={cand['l2']:.1e} "
            f"gat_heads={cand['gat_heads']} sage_aggr={cand['sage_aggr']} "
            f"gin_eps={cand['gin_eps']:.3f}"
        )
        result = run_eval(cand, data, in_ch, out_ch, args, device, logger)
        summary.append(result)
        logger.info(
            f"  >> val={result['val_mean']:.4f} +/- {result['val_std']:.4f} "
            f"test={result['test_mean']:.4f} +/- {result['test_std']:.4f}"
        )

    out_path = os.path.join(args.output, f"final_results_{args.hp_mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved final results: {out_path}")

    if summary:
        best = max(summary, key=lambda row: row["test_mean"])
        logger.info(f"Best test: {best['name']} {best['test_mean']:.4f} +/- {best['test_std']:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"Run complete. Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
