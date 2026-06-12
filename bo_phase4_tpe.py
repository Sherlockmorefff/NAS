"""Phase 4 Optuna TPE search over joint architecture + HP vectors.

This script searches

    z_search = [z_arch, hp]

where HP semantics are fully controlled by hp_mode. Optuna parameter names are
centralized in the mapping helpers below so normal TPE trials, warm-start
trials, and GMM initialization trials all use the exact same z_search order.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import warnings
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import torch

sys.path.insert(0, "/mnt/project")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Input.*not contained.*unit cube.*",
)

import optuna
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from eval_utils import eval_z_search
from hp_modes import hp_dim_from_mode, hp_names_from_mode, validate_hp_mode
from nas_space import JointSpaceVAE
from weighted_diag_gmm_init import fit_and_sample_gmm_init, load_history_vectors


ARCH_NZ = 12
HP_MODE_CHOICES = ("global4", "hybrid_cond7", "layer_cond19")


class ArchArgs:
    def __init__(self):
        self.max_n = 7
        self.num_vertex_type = 8
        self.START_TYPE = 0
        self.END_TYPE = 1
        self.hs = 501
        self.nz = ARCH_NZ
        self.bidirectional = True


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
    parser = argparse.ArgumentParser(description="Phase4 Optuna TPE joint search with hp_mode + GMM init")
    parser.add_argument("--checkpoint", type=str, default="results/joint_search/joint_model_global4_best.pth")
    parser.add_argument("--warm_start", type=str, default="results/bo_phase3/best_z_final.pt")
    parser.add_argument("--warm_start_repeats", type=int, default=5)
    parser.add_argument("--warm_start_noise", type=float, default=0.15)
    parser.add_argument("--cora_root", type=str, default="/tmp/Cora")
    parser.add_argument("--n_trials", type=int, default=80)
    parser.add_argument("--output", type=str, default="results/bo_phase4_tpe")
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--log_lr_min", type=float, default=-4.0)
    parser.add_argument("--log_lr_max", type=float, default=-1.5)
    parser.add_argument("--dropout_min", type=float, default=0.1)
    parser.add_argument("--dropout_max", type=float, default=0.6)
    parser.add_argument("--version", type=str, default="phase4_tpe")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gcnii_alpha", type=float, default=0.1)
    parser.add_argument("--gcnii_theta", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hp_mode", type=str, default="global4", choices=HP_MODE_CHOICES)
    parser.add_argument("--z_bound", type=float, default=2.5)

    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--load_if_exists", action="store_true")
    parser.add_argument("--tpe_startup_trials", type=int, default=10)
    parser.add_argument("--tpe_multivariate", action="store_true")
    parser.add_argument("--tpe_group", action="store_true")

    parser.add_argument("--gmm_init_history", nargs="*", default=None)
    parser.add_argument("--gmm_init_trials", type=int, default=0)
    parser.add_argument("--gmm_top_frac", type=float, default=0.3)
    parser.add_argument("--gmm_n_components", type=int, default=4)
    parser.add_argument("--gmm_weight_temp", type=float, default=8.0)
    parser.add_argument("--gmm_min_samples", type=int, default=20)
    parser.add_argument("--gmm_var_floor", type=float, default=1e-4)
    parser.add_argument("--gmm_sample_std_scale", type=float, default=1.0)
    parser.add_argument("--gmm_max_iter", type=int, default=100)
    parser.add_argument("--gmm_tol", type=float, default=1e-4)
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-6)
    parser.add_argument("--gmm_save_summary", action="store_true")
    return parser.parse_args()


def _torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def hp_optuna_param_names(hp_mode: str) -> list[str]:
    hp_mode = validate_hp_mode(hp_mode)
    base = [
        "hp_lr_norm",
        "hp_dropout_norm",
        "hp_hidden_norm",
        "hp_l2_norm",
    ]
    if hp_mode == "global4":
        return base
    if hp_mode == "hybrid_cond7":
        return base + [
            "hp_gat_heads_norm",
            "hp_sage_aggr_norm",
            "hp_gin_eps_norm",
        ]
    return (
        base
        + [f"hp_gat_heads_norm_{i}" for i in range(5)]
        + [f"hp_sage_aggr_norm_{i}" for i in range(5)]
        + [f"hp_gin_eps_norm_{i}" for i in range(5)]
    )


def optuna_param_names(hp_mode: str, arch_nz: int) -> list[str]:
    """Return complete Optuna parameter names in exact z_search order."""

    validate_hp_mode(hp_mode)
    arch_names = [f"z_arch_{i:02d}" for i in range(int(arch_nz))]
    return arch_names + hp_optuna_param_names(hp_mode)


def clip_z_search_by_mode(
    z_search,
    hp_mode: str,
    arch_nz: int,
    z_bound: float,
) -> np.ndarray:
    """Clip z_arch to [-z_bound, z_bound] and HP normalized dims to [0, 1]."""

    hp_mode = validate_hp_mode(hp_mode)
    arch_nz = int(arch_nz)
    hp_dim = hp_dim_from_mode(hp_mode)
    expected = arch_nz + hp_dim
    z = np.asarray(z_search, dtype=np.float64).reshape(-1).copy()
    if z.shape[0] != expected:
        raise ValueError(f"Expected z_search length {expected} for hp_mode={hp_mode}, got {z.shape[0]}")
    z_bound = float(z_bound)
    if not np.isfinite(z_bound) or z_bound <= 0.0:
        raise ValueError(f"z_bound must be positive finite, got {z_bound}")
    z = np.nan_to_num(z, nan=0.0, posinf=z_bound, neginf=-z_bound)
    z[:arch_nz] = np.clip(z[:arch_nz], -z_bound, z_bound)
    z[arch_nz:] = np.clip(z[arch_nz:], 0.0, 1.0)
    return z.astype(np.float32, copy=False)


def is_finite_z_search(z_search) -> bool:
    try:
        z = np.asarray(z_search, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(z.size > 0 and np.isfinite(z).all())


def define_optuna_search_space(
    trial,
    hp_mode: str,
    arch_nz: int,
    z_bound: float,
) -> torch.Tensor:
    """Define the full Optuna search space and return z_search Tensor."""

    names = optuna_param_names(hp_mode, arch_nz)
    values: list[float] = []
    for i, name in enumerate(names):
        if i < arch_nz:
            values.append(float(trial.suggest_float(name, -float(z_bound), float(z_bound))))
        else:
            values.append(float(trial.suggest_float(name, 0.0, 1.0)))
    return torch.tensor(values, dtype=torch.float32)


def vector_to_optuna_params(
    z_search,
    hp_mode: str,
    arch_nz: int,
    z_bound: float,
) -> dict[str, float]:
    """Defensively clip z_search, then map it to Optuna params by exact order."""

    z = clip_z_search_by_mode(z_search, hp_mode, arch_nz, z_bound)
    names = optuna_param_names(hp_mode, arch_nz)
    return {name: float(z[i]) for i, name in enumerate(names)}


def params_to_z_search(params: dict, hp_mode: str, arch_nz: int) -> torch.Tensor:
    """Map Optuna params back to z_search using the canonical parameter order."""

    names = optuna_param_names(hp_mode, arch_nz)
    missing = [name for name in names if name not in params]
    if missing:
        raise KeyError(f"Missing Optuna params for z_search mapping: {missing}")
    return torch.tensor([float(params[name]) for name in names], dtype=torch.float32)


_CORA_FILES = [
    "ind.cora.x",
    "ind.cora.tx",
    "ind.cora.allx",
    "ind.cora.y",
    "ind.cora.ty",
    "ind.cora.ally",
    "ind.cora.graph",
    "ind.cora.test.index",
]
_MIRRORS = [
    "https://gitee.com/jiajiewu/planetoid/raw/master/data",
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/kimiyoung/planetoid/master/data",
    "https://cdn.jsdelivr.net/gh/kimiyoung/planetoid@master/data",
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data",
]


def _try_download_file(url: str, dest: str, timeout: int = 30) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 10:
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def load_cora(root: str, device: torch.device, logger):
    raw_dir = os.path.join(root, "raw")
    missing = [
        fname
        for fname in _CORA_FILES
        if not os.path.exists(os.path.join(raw_dir, fname))
        or os.path.getsize(os.path.join(raw_dir, fname)) < 10
    ]
    if missing:
        logger.info(f"Downloading {len(missing)} Cora files")
        os.makedirs(raw_dir, exist_ok=True)
        for fname in missing:
            dest = os.path.join(raw_dir, fname)
            ok = any(_try_download_file(f"{mirror}/{fname}", dest) for mirror in _MIRRORS)
            logger.info(f"  {fname}: {'OK' if ok else 'FAILED'}")
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    dataset = Planetoid(root=pyg_root, name="Cora", transform=T.NormalizeFeatures())
    return dataset[0].to(device), dataset.num_features, dataset.num_classes


def load_vae(args: argparse.Namespace, device: torch.device, logger) -> JointSpaceVAE:
    hp_dim = hp_dim_from_mode(args.hp_mode)
    model = JointSpaceVAE(
        ArchArgs(),
        hp_mode=args.hp_mode,
        hp_latent_dim=hp_dim,
        hp_input_dim=hp_dim,
    ).to(device)
    state = _torch_load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(
        f"VAE loaded: {args.checkpoint} "
        f"(arch_nz={ARCH_NZ}, hp_mode={args.hp_mode}, hp_dim={hp_dim})"
    )
    return model


def decode_arch(vae: JointSpaceVAE, z_arch: torch.Tensor, device: torch.device, n_trials: int = 5):
    results = []
    with torch.no_grad():
        for _ in range(n_trials):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(device))
            graph = graphs[0]
            ops = [
                vae.op_mapping.get(graph.vs[i]["type"], "Identity")
                for i in range(1, graph.vcount() - 1)
            ]
            edges = graph.get_edgelist()
            n_eff = sum(1 for op in ops if op != "Identity")
            if n_eff > 0:
                key = (tuple(ops), tuple(edges))
                results.append(
                    (
                        key,
                        {
                            "effective_layers": n_eff,
                            "operations": ops,
                            "edges": edges,
                        },
                    )
                )
    if not results:
        return None
    best_key = Counter(r[0] for r in results).most_common(1)[0][0]
    return next(config for key, config in results if key == best_key)


def eval_candidate(
    vae: JointSpaceVAE,
    z_search,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    z_np = clip_z_search_by_mode(z_search, args.hp_mode, ARCH_NZ, args.z_bound)
    z_tensor = torch.tensor(z_np, dtype=torch.float32)
    result = eval_z_search(
        vae,
        z_tensor.to(device),
        data,
        in_ch,
        out_ch,
        arch_nz=ARCH_NZ,
        log_lr_min=args.log_lr_min,
        log_lr_max=args.log_lr_max,
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
        device=device,
        use_conditional_params=True,
        gcnii_alpha=args.gcnii_alpha,
        gcnii_theta=args.gcnii_theta,
        max_epochs=args.eval_epochs,
        patience=args.patience,
        hp_mode=args.hp_mode,
        return_hp=True,
    )
    return z_tensor, result


def _edges_json(config: dict | None) -> list[list[int]]:
    if config is None:
        return []
    return [list(edge) for edge in config.get("edges", [])]


def _history_from_result(
    step: int,
    record_type: str,
    z_search: torch.Tensor,
    result: dict[str, Any],
    args: argparse.Namespace,
    best_acc: float | None = None,
    optuna_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    hp = result.get("hp", {})
    config = result.get("config")
    params = {
        "hp_norm": hp.get("hp_norm"),
        "gcnii_alpha": float(args.gcnii_alpha),
        "gcnii_theta": float(args.gcnii_theta),
        "eval_epochs": int(args.eval_epochs),
        "patience": int(args.patience),
    }
    if best_acc is not None:
        params["best"] = float(best_acc)
    if optuna_params is not None:
        params["optuna"] = {k: float(v) for k, v in optuna_params.items()}

    return {
        "step": int(step),
        "type": record_type,
        "hp_mode": args.hp_mode,
        "search_dim": ARCH_NZ + hp_dim_from_mode(args.hp_mode),
        "z_search": z_search.detach().cpu().float().flatten().tolist(),
        "z_arch": z_search[:ARCH_NZ].detach().cpu().float().flatten().tolist(),
        "val_acc": float(result.get("val_acc", 0.0)),
        "value": float(result.get("val_acc", 0.0)),
        "valid": bool(result.get("valid", False)),
        "operations": [] if config is None else list(config.get("operations", [])),
        "edges": _edges_json(config),
        "lr": float(hp.get("lr", result.get("lr", 0.0))),
        "dropout": float(hp.get("dropout", result.get("dropout", 0.0))),
        "hidden_dim": int(hp.get("hidden_dim", result.get("hidden_dim", 64))),
        "l2": float(hp.get("weight_decay", hp.get("l2", result.get("l2", 0.0)))),
        "gat_heads": int(hp.get("gat_heads", result.get("gat_heads", 1))),
        "sage_aggr": str(hp.get("sage_aggr", result.get("sage_aggr", "mean"))),
        "gin_eps": float(hp.get("gin_eps", result.get("gin_eps", 0.0))),
        "gat_heads_by_layer": hp.get("gat_heads_by_layer", result.get("gat_heads_by_layer")),
        "sage_aggr_by_layer": hp.get("sage_aggr_by_layer", result.get("sage_aggr_by_layer")),
        "gin_eps_by_layer": hp.get("gin_eps_by_layer", result.get("gin_eps_by_layer")),
        "condition_mask": hp.get("condition_mask", result.get("condition_mask", {})),
        "condition_mask_vector": hp.get(
            "condition_mask_vector",
            result.get("condition_mask_vector"),
        ),
        "params": params,
    }


def _safe_set_trial_attrs(
    trial,
    record_type: str,
    z_search: torch.Tensor,
    result: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    record = _history_from_result(
        step=int(trial.number),
        record_type=record_type,
        z_search=z_search,
        result=result,
        args=args,
        optuna_params=trial.params,
    )
    for key, value in record.items():
        trial.set_user_attr(key, value)


def make_study(args: argparse.Namespace, logger):
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.tpe_startup_trials,
        multivariate=bool(args.tpe_multivariate),
        group=bool(args.tpe_group),
    )
    study_name = args.study_name or f"phase4_tpe_{args.hp_mode}_{args.version}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
    )
    logger.info(f"Optuna study: {study.study_name}")
    return study


def _enqueue_trial(study, params: dict[str, float], user_attrs: dict[str, Any], logger) -> bool:
    try:
        try:
            study.enqueue_trial(params, user_attrs=user_attrs)
        except TypeError:
            study.enqueue_trial(params)
        return True
    except ValueError as exc:
        logger.warning(f"Skip enqueue_trial: {exc}")
        return False


def enqueue_warm_start(study, args: argparse.Namespace, logger) -> int:
    if not args.warm_start:
        return 0
    if not os.path.exists(args.warm_start):
        logger.warning(f"Warm start file not found: {args.warm_start}")
        return 0

    raw = _torch_load(args.warm_start, map_location="cpu")
    z0 = torch.as_tensor(raw, dtype=torch.float32).flatten().cpu().numpy()
    hp_dim = hp_dim_from_mode(args.hp_mode)
    search_dim = ARCH_NZ + hp_dim
    rng = np.random.default_rng(args.seed)
    candidates: list[np.ndarray] = []

    if z0.shape[0] == ARCH_NZ:
        for _ in range(max(1, int(args.warm_start_repeats))):
            z_arch = z0 + rng.normal(scale=float(args.warm_start_noise), size=ARCH_NZ)
            hp = rng.random(hp_dim)
            candidates.append(np.concatenate([z_arch, hp]))
    elif z0.shape[0] == search_dim:
        candidates.append(z0)
        for _ in range(max(0, int(args.warm_start_repeats) - 1)):
            z = z0.copy()
            z[:ARCH_NZ] += rng.normal(scale=float(args.warm_start_noise), size=ARCH_NZ)
            z[ARCH_NZ:] = np.clip(z[ARCH_NZ:] + rng.normal(scale=0.03, size=hp_dim), 0.0, 1.0)
            candidates.append(z)
    else:
        logger.warning(
            f"Warm start vector length {z0.shape[0]} does not match ARCH_NZ={ARCH_NZ} "
            f"or SEARCH_DIM={search_dim}; skipping."
        )
        return 0

    enqueued = 0
    for idx, z in enumerate(candidates):
        if not is_finite_z_search(z):
            logger.warning(f"Skip non-finite warm start sample #{idx}")
            continue
        params = vector_to_optuna_params(z, args.hp_mode, ARCH_NZ, args.z_bound)
        if _enqueue_trial(study, params, {"queued_type": "warm_start"}, logger):
            enqueued += 1
    logger.info(f"Warm start enqueued: {enqueued}")
    return enqueued


def enqueue_gmm_init(study, args: argparse.Namespace, logger) -> dict[str, Any] | None:
    if int(args.gmm_init_trials) <= 0:
        logger.info("Skipping GMM init enqueue: gmm_init_trials <= 0")
        return None
    if not args.gmm_init_history:
        logger.info("Skipping GMM init enqueue: no --gmm_init_history provided")
        return None

    search_dim = ARCH_NZ + hp_dim_from_mode(args.hp_mode)
    args.search_dim = search_dim
    args.arch_nz = ARCH_NZ

    X, y, meta = load_history_vectors(args.gmm_init_history, args.hp_mode, search_dim, logger)
    sampled_z_list, summary = fit_and_sample_gmm_init(X, y, args.hp_mode, args, logger)
    summary.update({f"loader_{key}": value for key, value in meta.items()})

    enqueued = 0
    for idx, z in enumerate(sampled_z_list):
        if not is_finite_z_search(z):
            logger.warning(f"Skip non-finite GMM sample #{idx}")
            continue
        try:
            z_clipped = clip_z_search_by_mode(z, args.hp_mode, ARCH_NZ, args.z_bound)
            params = vector_to_optuna_params(z_clipped, args.hp_mode, ARCH_NZ, args.z_bound)
        except ValueError as exc:
            logger.warning(f"Skip malformed GMM sample #{idx}: {exc}")
            continue
        if _enqueue_trial(study, params, {"queued_type": "gmm_init"}, logger):
            enqueued += 1

    summary["enqueued_count"] = int(enqueued)
    logger.info(f"GMM init enqueued: {enqueued}/{len(sampled_z_list)}")
    if args.gmm_save_summary:
        os.makedirs(args.output, exist_ok=True)
        summary_path = os.path.join(args.output, f"gmm_init_summary_tpe_{args.hp_mode}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"GMM summary saved: {summary_path}")
    return summary


def objective_factory(
    vae: JointSpaceVAE,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger,
):
    def objective(trial) -> float:
        queued_type = trial.user_attrs.get("queued_type", "tpe")
        z_search = define_optuna_search_space(trial, args.hp_mode, ARCH_NZ, args.z_bound)
        try:
            z_search, result = eval_candidate(vae, z_search, data, in_ch, out_ch, args, device)
            acc = float(result.get("val_acc", 0.0))
        except Exception as exc:
            logger.warning(f"Trial {trial.number} failed: {exc}")
            result = {
                "val_acc": 0.0,
                "valid": False,
                "config": None,
                "hp": {},
            }
            z_search = torch.tensor(
                clip_z_search_by_mode(z_search, args.hp_mode, ARCH_NZ, args.z_bound),
                dtype=torch.float32,
            )
            acc = 0.0

        _safe_set_trial_attrs(trial, queued_type, z_search, result, args)
        hp = result.get("hp", {})
        logger.info(
            f"trial {trial.number:>3d} type={queued_type:<10s} val={acc:.4f} "
            f"valid={bool(result.get('valid', False))} "
            f"lr={float(hp.get('lr', result.get('lr', 0.0))):.5f} "
            f"drop={float(hp.get('dropout', result.get('dropout', 0.0))):.3f} "
            f"hidden={int(hp.get('hidden_dim', result.get('hidden_dim', 64)))} "
            f"l2={float(hp.get('weight_decay', hp.get('l2', result.get('l2', 0.0)))):.1e}"
        )
        return acc

    return objective


def trial_to_history_entry(trial, args: argparse.Namespace) -> dict[str, Any] | None:
    attrs = trial.user_attrs
    if "z_search" not in attrs:
        return None
    record = {
        "step": int(attrs.get("step", trial.number)),
        "type": str(attrs.get("type", attrs.get("queued_type", "tpe"))),
        "hp_mode": str(attrs.get("hp_mode", args.hp_mode)),
        "search_dim": int(attrs.get("search_dim", ARCH_NZ + hp_dim_from_mode(args.hp_mode))),
        "z_search": attrs.get("z_search"),
        "z_arch": attrs.get("z_arch"),
        "val_acc": float(trial.value if trial.value is not None else attrs.get("val_acc", 0.0)),
        "value": float(trial.value if trial.value is not None else attrs.get("value", 0.0)),
        "valid": bool(attrs.get("valid", False)),
        "operations": attrs.get("operations", []),
        "edges": attrs.get("edges", []),
        "lr": float(attrs.get("lr", 0.0)),
        "dropout": float(attrs.get("dropout", 0.0)),
        "hidden_dim": int(attrs.get("hidden_dim", 64)),
        "l2": float(attrs.get("l2", 0.0)),
        "gat_heads": int(attrs.get("gat_heads", 1)),
        "sage_aggr": str(attrs.get("sage_aggr", "mean")),
        "gin_eps": float(attrs.get("gin_eps", 0.0)),
        "gat_heads_by_layer": attrs.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": attrs.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": attrs.get("gin_eps_by_layer"),
        "condition_mask": attrs.get("condition_mask", {}),
        "condition_mask_vector": attrs.get("condition_mask_vector"),
        "params": attrs.get("params", {"optuna": dict(trial.params)}),
    }
    return record


def save_outputs(study, args: argparse.Namespace, logger) -> None:
    os.makedirs(args.output, exist_ok=True)
    records = [
        record
        for record in (trial_to_history_entry(trial, args) for trial in study.trials)
        if record is not None
    ]
    records.sort(key=lambda row: row["step"])

    history_path = os.path.join(args.output, "history_final.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    saved = [history_path]
    if records:
        best_record = max(records, key=lambda row: row["val_acc"])
        best_z = torch.tensor(best_record["z_search"], dtype=torch.float32)
        best_z_search_path = os.path.join(args.output, "best_z_search_final.pt")
        best_z_final_path = os.path.join(args.output, "best_z_final.pt")
        torch.save(best_z, best_z_search_path)
        torch.save(best_z, best_z_final_path)
        saved.extend([best_z_search_path, best_z_final_path])
        logger.info(
            f"Best trial: step={best_record['step']} val={best_record['val_acc']:.4f} "
            f"hp_mode={best_record['hp_mode']}"
        )
    logger.info(f"Saved: {', '.join(saved)}")


def run_tpe(
    vae: JointSpaceVAE,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> None:
    study = make_study(args, logger)
    enqueue_warm_start(study, args, logger)
    enqueue_gmm_init(study, args, logger)
    objective = objective_factory(vae, data, in_ch, out_ch, args, device, logger)
    study.optimize(objective, n_trials=int(args.n_trials))
    save_outputs(study, args, logger)


def main() -> None:
    args = parse_args()
    args.hp_mode = validate_hp_mode(args.hp_mode)
    hp_dim = hp_dim_from_mode(args.hp_mode)
    search_dim = ARCH_NZ + hp_dim
    os.makedirs(args.output, exist_ok=True)

    logger, log_path = setup_logger(args.log_dir, "bo_phase4_tpe", args.version)
    save_args_json(args, log_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    logger.info("=" * 70)
    logger.info(f"bo_phase4_tpe.py -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info(f"Device={device} ARCH_NZ={ARCH_NZ} SEARCH_DIM={search_dim}")
    logger.info(f"hp_mode={args.hp_mode} hp_dim={hp_dim} hp_names={hp_names_from_mode(args.hp_mode)}")
    logger.info(f"z_bound={args.z_bound}")

    vae = load_vae(args, device, logger)
    data, in_ch, out_ch = load_cora(args.cora_root, device, logger)
    logger.info(f"Cora: {in_ch} features, {out_ch} classes")

    run_tpe(vae, data, in_ch, out_ch, args, device, logger)
    elapsed = time.time() - start_time
    logger.info(f"Run complete. Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
