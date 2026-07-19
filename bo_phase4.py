"""Phase 4 BO search using checkpoint or scratch accuracy GP initialization and pure qLogEI.

Search vector:

    z_search = [z_arch, hp]

HP semantics and dimensions are driven only by hp_mode. GMM initialization is
inserted after LHS initialization and before the BO loop. The GMM helper itself
does top-fraction selection, standardization, weighted diagonal GMM fitting,
sampling, inverse standardization, and only then clips to the valid search
domain. This script performs an additional defensive clip before evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/mnt/project")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Input.*not contained.*unit cube.*",
)

from botorch.optim import optimize_acqf

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from eval_utils import (
    SEED_DERIVATION,
    _decode_arch,
    eval_z_search,
    isolated_rng,
    stable_seed,
)
from hp_modes import (
    condition_mask_vector_from_ops,
    hp_dim_from_mode,
    hp_names_from_mode,
    validate_hp_mode,
)
from nas_space import JointSpaceVAE
from surrogate.accuracy_gp import (
    AccuracyGPPredictor,
    denormalize_search_vector,
)
from surrogate.checkpoint_io import atomic_json_dump
from surrogate.history_dataset import architecture_key_from_record
from surrogate.metrics import (
    PREQUENTIAL_METRIC_FIELDS,
    GPConvergenceMonitor,
    prediction_metrics,
    prediction_record_fields,
)
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
    parser = argparse.ArgumentParser(description="Phase4 BO joint search with hp_mode + GMM init")
    parser.add_argument("--checkpoint", type=str, default="results/joint_search/joint_model_global4_best.pth")
    parser.add_argument("--warm_start", type=str, default="results/bo_phase3/best_z_final.pt")
    parser.add_argument("--warm_start_repeats", type=int, default=5)
    parser.add_argument("--warm_start_noise", type=float, default=0.15)
    parser.add_argument("--cora_root", type=str, default="/tmp/Cora")
    parser.add_argument("--n_init", type=int, default=20)
    parser.add_argument("--n_lhs_candidates", type=int, default=160)
    parser.add_argument("--n_iter", type=int, default=60)
    parser.add_argument("--output", type=str, default="results/bo_phase4")
    parser.add_argument("--sigma_arch", type=float, default=0.8)
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--novelty_w", type=float, default=0.0, help="Deprecated and ignored")
    parser.add_argument("--log_lr_min", type=float, default=-4.0)
    parser.add_argument("--log_lr_max", type=float, default=-1.5)
    parser.add_argument("--dropout_min", type=float, default=0.1)
    parser.add_argument("--dropout_max", type=float, default=0.6)
    parser.add_argument("--version", type=str, default="phase4_bo")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gcnii_alpha", type=float, default=0.1)
    parser.add_argument("--gcnii_theta", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hp_mode", type=str, default="global4", choices=HP_MODE_CHOICES)
    parser.add_argument("--z_bound", type=float, default=2.5)

    parser.add_argument("--num_restarts", type=int, default=8)
    parser.add_argument("--raw_samples", type=int, default=256)
    parser.add_argument("--n_extra", type=int, default=128)
    parser.add_argument("--use_conditional_kernel", action="store_true")
    parser.add_argument(
        "--online_candidate_strategy",
        choices=("qlogei", "random"),
        default="qlogei",
    )
    parser.add_argument("--frozen_init_history", type=str, default=None)

    parser.add_argument("--gp_init_mode", choices=("checkpoint", "scratch"), default="checkpoint")
    parser.add_argument("--gp_checkpoint", default=None)
    parser.add_argument("--scratch_gp_min_points", type=int, default=3)
    parser.add_argument("--gp_update_mode", choices=("warm_refit", "append_only"), default="warm_refit")
    parser.add_argument("--gp_refit_every", type=int, default=5)
    parser.add_argument("--gp_refit_steps", type=int, default=20)
    parser.add_argument("--gp_save_every", type=int, default=5)
    parser.add_argument("--adaptive_sampling", action="store_true")
    parser.add_argument("--min_bo_samples", type=int, default=20)
    parser.add_argument("--max_bo_samples", type=int, default=60)
    parser.add_argument("--convergence_check_every", type=int, default=5)
    parser.add_argument("--convergence_patience", type=int, default=3)
    parser.add_argument("--prequential_window", type=int, default=10)
    parser.add_argument("--mae_relative_tol", type=float, default=0.01)
    parser.add_argument("--mae_absolute_tol", type=float, default=0.002)
    parser.add_argument("--std_relative_tol", type=float, default=0.02)
    parser.add_argument("--spearman_tol", type=float, default=0.01)
    parser.add_argument("--degradation_tolerance", type=float, default=0.01)
    parser.add_argument("--best_acc_patience", type=int, default=20)
    parser.add_argument("--best_acc_min_delta", type=float, default=0.001)
    parser.add_argument("--max_wall_time_hours", type=float, default=4.0)
    parser.add_argument("--probe_pool_size", type=int, default=512)
    parser.add_argument("--probe_pool_seed", type=int, default=None)

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


def clip_z_search_by_mode(
    z_search,
    hp_mode: str,
    arch_nz: int,
    z_bound: float,
) -> np.ndarray:
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
    if not np.isfinite(z).all():
        raise ValueError("z_search contains NaN or infinite values")
    z[:arch_nz] = np.clip(z[:arch_nz], -z_bound, z_bound)
    z[arch_nz:] = np.clip(z[arch_nz:], 0.0, 1.0)
    return z.astype(np.float32, copy=False)


def is_finite_z_search(z_search) -> bool:
    try:
        z = np.asarray(z_search, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(z.size > 0 and np.isfinite(z).all())


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


def decode_arch(
    vae: JointSpaceVAE,
    z_arch: torch.Tensor,
    device: torch.device,
    n_trials: int = 5,
    decoder_seed: int | None = None,
):
    return _decode_arch(
        vae,
        z_arch,
        device,
        n_trials=n_trials,
        decoder_seed=decoder_seed,
    )


def eval_candidate(
    vae: JointSpaceVAE,
    z_search,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    step: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    z_np = clip_z_search_by_mode(z_search, args.hp_mode, ARCH_NZ, args.z_bound)
    z_tensor = torch.tensor(z_np, dtype=torch.float32)
    decoder_seed = stable_seed(int(args.seed), "decoder", z_tensor[:ARCH_NZ])
    candidate_eval_seed = stable_seed(int(args.seed), "candidate_eval", int(step))
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
        decoder_seed=decoder_seed,
        candidate_eval_seed=candidate_eval_seed,
    )
    result.update(
        {
            "search_seed": int(args.seed),
            "decoder_seed": decoder_seed,
            "candidate_eval_seed": candidate_eval_seed,
            "seed_derivation": SEED_DERIVATION,
        }
    )
    return z_tensor, result


def _edges_json(config: dict | None) -> list[list[int]]:
    if config is None:
        return []
    return [list(edge) for edge in config.get("edges", [])]


def _architecture_key_from_result(result: dict[str, Any], z_search: torch.Tensor) -> str:
    config = result.get("config")
    row = {
        "operations": [] if config is None else list(config.get("operations", [])),
        "edges": _edges_json(config),
    }
    key, _fallback = architecture_key_from_record(row, z_search.detach().cpu().numpy())
    return key


def _holdout_leakage_reason(
    predictor: AccuracyGPPredictor,
    z_search: torch.Tensor,
    result: dict[str, Any],
    valid: bool,
) -> str:
    if not valid:
        return "invalid_sample"
    if predictor.is_holdout_point(z_search):
        return "fixed_holdout_vector_overlap"
    holdout_keys = set(str(key) for key in predictor.metadata.get("holdout_architecture_keys", []) or [])
    if holdout_keys and _architecture_key_from_result(result, z_search) in holdout_keys:
        return "fixed_holdout_architecture_overlap"
    return ""


def _null_gp_record_fields(train_size_before: int = 0, train_size_after: int = 0) -> dict[str, Any]:
    """Prediction-record fields used before a scratch GP exists."""

    return {
        "gp_pred_mean": None,
        "gp_pred_std": None,
        "gp_pred_95_low": None,
        "gp_pred_95_high": None,
        "gp_residual": None,
        "gp_abs_error": None,
        "gp_squared_error": None,
        "gp_standardized_residual": None,
        "gp_covered_by_95": None,
        "gp_train_size_before": int(train_size_before),
        "gp_train_size_after": int(train_size_after),
    }


def history_record(
    step: int,
    record_type: str,
    z_search: torch.Tensor,
    result: dict[str, Any],
    args: argparse.Namespace,
    best_acc: float | None = None,
    gp_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hp = result.get("hp", {})
    config = result.get("config")
    search_seed = int(result.get("search_seed", args.seed))
    decoder_seed_value = result.get("decoder_seed")
    if decoder_seed_value is None:
        decoder_seed_value = stable_seed(search_seed, "decoder", z_search[:ARCH_NZ])
    decoder_seed = int(decoder_seed_value)
    candidate_eval_seed_value = result.get("candidate_eval_seed")
    if candidate_eval_seed_value is None:
        candidate_eval_seed_value = stable_seed(search_seed, "candidate_eval", int(step))
    candidate_eval_seed = int(candidate_eval_seed_value)
    epochs_ran = int(result.get("epochs_ran") or 0)
    stopped_epoch = int(result.get("stopped_epoch") or epochs_ran)
    params = {
        "hp_norm": hp.get("hp_norm"),
        "gcnii_alpha": float(args.gcnii_alpha),
        "gcnii_theta": float(args.gcnii_theta),
        "eval_epochs": int(args.eval_epochs),
        "patience": int(args.patience),
    }
    if best_acc is not None:
        params["best"] = float(best_acc)

    record = {
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
        "search_seed": search_seed,
        "decoder_seed": decoder_seed,
        "candidate_eval_seed": candidate_eval_seed,
        "seed_derivation": SEED_DERIVATION,
        "best_epoch": result.get("best_epoch"),
        "stopped_epoch": stopped_epoch,
        "epochs_ran": epochs_ran,
        "online_candidate_strategy": result.get("online_candidate_strategy"),
        "candidate_selection_seed": result.get("candidate_selection_seed"),
        "initial_record_replayed": bool(result.get("initial_record_replayed", False)),
        "initial_history_source": result.get("initial_history_source"),
        "params": params,
    }
    if gp_record:
        record.update(gp_record)
    return record


def _sample_unit_lhs(n: int, dim: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, dim), dtype=np.float64)
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        return sampler.random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n, dim))


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    try:
        from scipy.stats import norm

        return norm.ppf(u)
    except Exception:
        rng = np.random.default_rng(0)
        return rng.normal(size=u.shape)


def make_lhs_pool(args: argparse.Namespace) -> list[torch.Tensor]:
    hp_dim = hp_dim_from_mode(args.hp_mode)
    search_dim = ARCH_NZ + hp_dim
    n_pool = max(int(args.n_lhs_candidates), int(args.n_init))
    unit = _sample_unit_lhs(n_pool, search_dim, args.seed)
    arch = _norm_ppf(unit[:, :ARCH_NZ]) * float(args.sigma_arch)
    hp = unit[:, ARCH_NZ:]
    raw = np.concatenate([arch, hp], axis=1)
    clipped = [clip_z_search_by_mode(row, args.hp_mode, ARCH_NZ, args.z_bound) for row in raw]
    return [torch.tensor(row, dtype=torch.float32) for row in clipped]


def load_warm_start_points(args: argparse.Namespace, logger) -> list[torch.Tensor]:
    if not args.warm_start:
        return []
    if not os.path.exists(args.warm_start):
        logger.warning(f"Warm start file not found: {args.warm_start}")
        return []

    hp_dim = hp_dim_from_mode(args.hp_mode)
    search_dim = ARCH_NZ + hp_dim
    raw = _torch_load(args.warm_start, map_location="cpu")
    z0 = torch.as_tensor(raw, dtype=torch.float32).flatten().cpu().numpy()
    rng = np.random.default_rng(args.seed + 17)
    points: list[np.ndarray] = []

    if z0.shape[0] == ARCH_NZ:
        for _ in range(max(1, int(args.warm_start_repeats))):
            z_arch = z0 + rng.normal(scale=float(args.warm_start_noise), size=ARCH_NZ)
            hp = rng.random(hp_dim)
            points.append(np.concatenate([z_arch, hp]))
    elif z0.shape[0] == search_dim:
        points.append(z0)
        for _ in range(max(0, int(args.warm_start_repeats) - 1)):
            z = z0.copy()
            z[:ARCH_NZ] += rng.normal(scale=float(args.warm_start_noise), size=ARCH_NZ)
            z[ARCH_NZ:] = np.clip(z[ARCH_NZ:] + rng.normal(scale=0.03, size=hp_dim), 0.0, 1.0)
            points.append(z)
    else:
        logger.warning(
            f"Warm start vector length {z0.shape[0]} does not match ARCH_NZ={ARCH_NZ} "
            f"or SEARCH_DIM={search_dim}; skipping."
        )
        return []

    clipped = [clip_z_search_by_mode(row, args.hp_mode, ARCH_NZ, args.z_bound) for row in points]
    logger.info(f"Warm start candidates loaded: {len(clipped)}")
    return [torch.tensor(row, dtype=torch.float32) for row in clipped]


def _rbf_kernel_np(X: np.ndarray, lengthscale: float = 1.0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return np.exp(-0.5 * d2 / max(lengthscale * lengthscale, 1e-12))


def schur_greedy_select(candidates: list[torch.Tensor], k: int, jitter: float = 1e-6) -> list[torch.Tensor]:
    """Greedy RBF log-det selection using Schur-complement conditional variance."""

    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)

    X = torch.stack(candidates).float()
    x_min = X.min(0).values
    x_range = (X.max(0).values - x_min).clamp(min=1e-8)
    Xn = ((X - x_min) / x_range).cpu().numpy()
    K = _rbf_kernel_np(Xn, lengthscale=np.sqrt(Xn.shape[1]))
    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while len(selected) < k and remaining:
        if not selected:
            center = Xn.mean(axis=0, keepdims=True)
            d2 = np.sum((Xn[remaining] - center) ** 2, axis=1)
            pick_pos = int(np.argmax(d2))
        else:
            K_ss = K[np.ix_(selected, selected)] + np.eye(len(selected)) * jitter
            K_rs = K[np.ix_(remaining, selected)]
            try:
                solved = np.linalg.solve(K_ss, K_rs.T).T
                cond_var = np.maximum(np.diag(K)[remaining] - np.sum(K_rs * solved, axis=1), 0.0)
            except np.linalg.LinAlgError:
                cond_var = np.ones(len(remaining), dtype=np.float64)
            pick_pos = int(np.argmax(cond_var))
        selected_idx = remaining.pop(pick_pos)
        selected.append(selected_idx)

    return [candidates[i] for i in selected]


def initial_points(args: argparse.Namespace, logger) -> list[torch.Tensor]:
    warm = load_warm_start_points(args, logger)
    lhs_pool = make_lhs_pool(args)
    n_lhs_needed = max(0, int(args.n_init) - len(warm))
    selected_lhs = schur_greedy_select(lhs_pool, n_lhs_needed)
    init = warm[: int(args.n_init)] + selected_lhs
    if len(init) < int(args.n_init):
        init.extend(make_lhs_pool(args)[: int(args.n_init) - len(init)])
    logger.info(
        f"LHS init selected: {len(init)} points "
        f"({len(warm[: int(args.n_init)])} warm + {len(init) - len(warm[: int(args.n_init)])} LHS/Schur)"
    )
    return init[: int(args.n_init)]


def _search_dim(args: argparse.Namespace) -> int:
    return ARCH_NZ + hp_dim_from_mode(args.hp_mode)


def _mask_from_ops(ops: list[str], args: argparse.Namespace) -> list[float]:
    hp_mask = condition_mask_vector_from_ops(ops, args.hp_mode)
    return [1.0] * ARCH_NZ + [float(v) for v in hp_mask]


def _mask_for_candidate(
    vae: JointSpaceVAE,
    z_search: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> list[float]:
    decoder_seed = stable_seed(int(args.seed), "decoder", z_search[:ARCH_NZ])
    try:
        config = decode_arch(
            vae,
            z_search[:ARCH_NZ],
            device,
            n_trials=3,
            decoder_seed=decoder_seed,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to decode candidate condition mask: {exc}") from exc
    ops = [] if config is None else [str(op) for op in config.get("operations", [])]
    return _mask_from_ops(ops, args)


def _candidate_masks(
    vae: JointSpaceVAE,
    z_rows: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> torch.Tensor:
    masks = [
        _mask_for_candidate(vae, z_rows[i].detach().cpu().float(), args, device, logger)
        for i in range(z_rows.shape[0])
    ]
    return torch.tensor(masks, dtype=torch.float32)


def optimize_acq(
    predictor: AccuracyGPPredictor,
    best_f: float,
    args: argparse.Namespace,
    logger,
    vae: JointSpaceVAE,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, list[float]]:
    """Select a candidate strictly by qLogExpectedImprovement."""

    search_dim = _search_dim(args)
    logei = predictor.make_logei(best_f)
    bounds = predictor.normalized_bounds

    candidate_norms: list[torch.Tensor] = []
    if not predictor.use_conditional_kernel:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand, _ = optimize_acqf(
                logei, bounds=bounds, q=1,
                num_restarts=int(args.num_restarts), raw_samples=int(args.raw_samples),
            )
            candidate_norms.append(cand.squeeze(0).detach().cpu().float())

    n_random = int(args.n_extra)
    if predictor.use_conditional_kernel:
        n_random = max(n_random, int(args.raw_samples) + int(args.num_restarts))
    if n_random > 0:
        candidate_norms.extend(torch.rand(n_random, search_dim, generator=generator))
    if not candidate_norms:
        raise RuntimeError("acquisition produced no candidates")

    cand_norm = torch.stack(candidate_norms).float()
    clipped_raw = denormalize_search_vector(
        cand_norm, arch_nz=ARCH_NZ, hp_mode=args.hp_mode, z_bound=args.z_bound
    ).float()

    with torch.no_grad():
        if predictor.use_conditional_kernel:
            cand_masks = _candidate_masks(vae, clipped_raw, args, device, logger)
            eval_input = predictor.acquisition_inputs(clipped_raw, cand_masks)
        else:
            eval_input = predictor.acquisition_inputs(clipped_raw)
        logei_scores = logei(eval_input.double().unsqueeze(1)).float().view(-1)
    if not torch.isfinite(logei_scores).all():
        raise RuntimeError("qLogExpectedImprovement returned non-finite scores")
    best_idx = int(torch.argmax(logei_scores).item())
    best_mask = (
        cand_masks[best_idx].tolist()
        if predictor.use_conditional_kernel
        else _mask_for_candidate(vae, clipped_raw[best_idx], args, device, logger)
    )
    return clipped_raw[best_idx].float(), float(logei_scores[best_idx].item()), best_mask


def sample_random_online_candidate(
    vae: JointSpaceVAE,
    args: argparse.Namespace,
    device: torch.device,
    logger,
    step: int,
) -> tuple[torch.Tensor, list[float], int]:
    """Uniformly sample one online candidate without consuming global RNG state."""

    selection_seed = stable_seed(
        int(args.seed), "random_acquisition", int(step),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(selection_seed)
    normalized = torch.rand(_search_dim(args), generator=generator, dtype=torch.float32)
    raw = denormalize_search_vector(
        normalized,
        arch_nz=ARCH_NZ,
        hp_mode=args.hp_mode,
        z_bound=args.z_bound,
    ).reshape(-1)
    clipped = clip_z_search_by_mode(
        raw, args.hp_mode, ARCH_NZ, args.z_bound,
    )
    z_search = torch.tensor(clipped, dtype=torch.float32)
    condition_mask = _mask_for_candidate(vae, z_search, args, device, logger)
    return z_search, condition_mask, selection_seed


def _append_eval(
    vae: JointSpaceVAE,
    z_search: torch.Tensor,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    history: list[dict[str, Any]],
    X_obs: list[torch.Tensor],
    Y_obs: list[float],
    step: int,
    record_type: str,
    logger,
    predictor: AccuracyGPPredictor | None,
    prediction_records: list[dict[str, Any]],
    gp_stage: str,
    logei_value: float | None,
    update_online: bool,
    online_valid_count: int,
    condition_mask: list[float],
    best_acc: float | None = None,
    convergence_monitor: GPConvergenceMonitor | None = None,
    candidate_selection_seed: int | None = None,
) -> tuple[float, bool, list[float], bool]:
    """Predict, evaluate, record, and optionally update in that strict order."""

    step_started = time.monotonic()
    online_candidate_strategy = (
        getattr(args, "online_candidate_strategy", "qlogei")
        if update_online
        else None
    )
    pre_search_seed = getattr(args, "seed", None)
    pre_decoder_seed = None
    pre_candidate_eval_seed = None
    pre_seed_derivation = None
    if pre_search_seed is not None:
        pre_search_seed = int(pre_search_seed)
        pre_decoder_seed = stable_seed(
            pre_search_seed, "decoder", z_search[:ARCH_NZ],
        )
        pre_candidate_eval_seed = stable_seed(
            pre_search_seed, "candidate_eval", int(step),
        )
        pre_seed_derivation = SEED_DERIVATION
    if update_online and predictor is None:
        raise RuntimeError("online GP update requested before a GP predictor exists")
    train_size_before = 0 if predictor is None else predictor.train_size
    prediction: dict[str, float] | None = None
    if predictor is not None:
        prediction = predictor.predict(
            z_search,
            condition_mask=condition_mask if predictor.use_conditional_kernel else None,
        )
    pre_eval_fields = _null_gp_record_fields(train_size_before, train_size_before)
    if prediction is not None:
        pre_eval_fields.update(
            {
                "gp_pred_mean": float(prediction["mean"]),
                "gp_pred_std": float(prediction["std"]),
                "gp_pred_95_low": float(prediction["lower_95"]),
                "gp_pred_95_high": float(prediction["upper_95"]),
            }
        )
    prediction_row = {
        "step": int(step),
        "record_type": record_type,
        "gp_stage": gp_stage,
        "valid": None,
        "val_acc": None,
        "search_seed": pre_search_seed,
        "decoder_seed": pre_decoder_seed,
        "candidate_eval_seed": pre_candidate_eval_seed,
        "seed_derivation": pre_seed_derivation,
        "best_epoch": None,
        "stopped_epoch": None,
        "epochs_ran": None,
        "online_candidate_strategy": online_candidate_strategy,
        "candidate_selection_seed": candidate_selection_seed,
        "initial_record_replayed": False,
        "initial_history_source": None,
        **pre_eval_fields,
        "logei": None if logei_value is None else float(logei_value),
        "best_val_before": None if best_acc is None else float(best_acc),
        "gp_update_mode": args.gp_update_mode,
        "gp_checkpoint_source": args.gp_checkpoint if args.gp_init_mode == "checkpoint" else None,
        "gp_update_performed": False,
        "gp_update_skipped_reason": "",
        "gp_update_seconds": 0.0,
        "eval_seconds": None,
        "total_step_seconds": float(time.monotonic() - step_started),
    }
    prediction_records.append(prediction_row)
    # Persist the untouched pre-update prediction before the real GNN evaluation.
    _save_prediction_csv(prediction_records, args.output)

    eval_started = time.monotonic()
    z_search, result = eval_candidate(
        vae, z_search, data, in_ch, out_ch, args, device, step=step,
    )
    eval_seconds = time.monotonic() - eval_started
    acc = float(result.get("val_acc", 0.0))
    valid = bool(result.get("valid", False))
    gp_update_skipped_reason = (
        "invalid_sample" if predictor is None and not valid
        else "" if predictor is None
        else _holdout_leakage_reason(predictor, z_search, result, valid)
    )
    safe_for_gp_training = valid and not gp_update_skipped_reason
    gp_fields = (
        _null_gp_record_fields(train_size_before, train_size_before)
        if prediction is None
        else prediction_record_fields(prediction, acc, train_size_before, train_size_before)
    )
    gp_fields.update(
        {
            "logei": None if logei_value is None else float(logei_value),
            "best_val_before": None if best_acc is None else float(best_acc),
            "gp_update_mode": args.gp_update_mode,
            "gp_checkpoint_source": args.gp_checkpoint if args.gp_init_mode == "checkpoint" else None,
            "gp_stage": gp_stage,
            "gp_update_performed": False,
            "gp_update_skipped_reason": gp_update_skipped_reason,
            "gp_update_seconds": 0.0,
            "eval_seconds": float(eval_seconds),
            "total_step_seconds": float(time.monotonic() - step_started),
            "online_candidate_strategy": online_candidate_strategy,
            "candidate_selection_seed": candidate_selection_seed,
        }
    )
    X_obs.append(z_search)
    Y_obs.append(acc)
    history_row = history_record(
        step, record_type, z_search, result, args,
        best_acc=best_acc, gp_record=gp_fields,
    )
    history.append(history_row)
    prediction_row.update(
        {
            "valid": valid,
            "val_acc": acc,
            "search_seed": result.get("search_seed"),
            "decoder_seed": result.get("decoder_seed"),
            "candidate_eval_seed": result.get("candidate_eval_seed"),
            "seed_derivation": result.get("seed_derivation"),
            "best_epoch": result.get("best_epoch"),
            "stopped_epoch": result.get("stopped_epoch"),
            "epochs_ran": result.get("epochs_ran"),
            "online_candidate_strategy": online_candidate_strategy,
            "candidate_selection_seed": candidate_selection_seed,
            **gp_fields,
        }
    )
    if convergence_monitor is not None:
        rolling_metrics = convergence_monitor.observe_bo_result(acc, prediction_row)
        prediction_row.update(rolling_metrics)
        history_row.update(rolling_metrics)
    # Persist the prediction, real result, and metrics before touching the GP.
    _save_prediction_csv(prediction_records, args.output)

    if update_online and safe_for_gp_training:
        update_started = time.monotonic()
        try:
            assert predictor is not None
            predictor.append_observation(
                z_search,
                acc,
                condition_mask=condition_mask if predictor.use_conditional_kernel else None,
            )
            should_optimize = (
                args.gp_update_mode == "warm_refit"
                and (int(online_valid_count) + 1) % int(args.gp_refit_every) == 0
            )
            predictor.refit(
                optimize=should_optimize,
                steps=int(args.gp_refit_steps) if should_optimize else None,
            )
        except Exception as exc:
            prediction_row["gp_update_skipped_reason"] = f"update_failed: {exc}"
            history_row["gp_update_skipped_reason"] = prediction_row["gp_update_skipped_reason"]
            _save_prediction_csv(prediction_records, args.output)
            raise
        update_seconds = time.monotonic() - update_started
        update_fields = {
            "gp_train_size_after": predictor.train_size,
            "gp_update_performed": True,
            "gp_update_seconds": float(update_seconds),
            "total_step_seconds": float(time.monotonic() - step_started),
        }
        prediction_row.update(update_fields)
        history_row.update(update_fields)
        _save_prediction_csv(prediction_records, args.output)
    hp = result.get("hp", {})
    logger.info(
        f"  {record_type:<9s} step={step:>3d}: val={acc:.4f} "
        f"lr={float(hp.get('lr', result.get('lr', 0.0))):.5f} "
        f"drop={float(hp.get('dropout', result.get('dropout', 0.0))):.3f} "
        f"hidden={int(hp.get('hidden_dim', result.get('hidden_dim', 64)))} "
        f"l2={float(hp.get('weight_decay', hp.get('l2', result.get('l2', 0.0)))):.1e} "
        f"valid={valid} "
        + (
            "gp=None"
            if prediction is None
            else f"gp={prediction['mean']:.4f}+/-{prediction['std']:.4f}"
        )
    )
    return acc, valid, condition_mask, safe_for_gp_training


def _save_prediction_csv(records: list[dict[str, Any]], output: str) -> None:
    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, "gp_predictions.csv")
    tmp_path = path + ".tmp"
    fields = [
        "step", "record_type", "gp_stage", "valid", "search_seed",
        "decoder_seed", "candidate_eval_seed", "seed_derivation", "best_epoch",
        "stopped_epoch", "epochs_ran", "online_candidate_strategy",
        "candidate_selection_seed", "initial_record_replayed",
        "initial_history_source", "gp_train_size_before",
        "gp_train_size_after", "gp_pred_mean", "gp_pred_std", "gp_pred_95_low",
        "gp_pred_95_high", "val_acc", "gp_residual", "gp_abs_error",
        "gp_squared_error", "gp_standardized_residual", "gp_covered_by_95",
        "logei", "best_val_before", "gp_update_mode", "gp_checkpoint_source",
        "gp_update_performed", "gp_update_skipped_reason", "eval_seconds",
        "gp_update_seconds", "total_step_seconds",
        *PREQUENTIAL_METRIC_FIELDS,
    ]
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp_path, path)


def validate_frozen_init_configuration(args: argparse.Namespace) -> None:
    """Reject configurations that could mix frozen LHS data with other initialization."""

    if args.frozen_init_history is None:
        return
    if args.gp_init_mode != "scratch":
        raise ValueError("--frozen_init_history requires --gp_init_mode scratch")
    if args.warm_start != "":
        raise ValueError("--frozen_init_history requires --warm_start to be an empty string")
    if args.gmm_init_history:
        raise ValueError("--frozen_init_history cannot be combined with --gmm_init_history")
    if int(args.gmm_init_trials) != 0:
        raise ValueError("--frozen_init_history requires --gmm_init_trials 0")


def _frozen_record_error(source: str, step: Any, field: str, detail: str) -> ValueError:
    return ValueError(
        f"frozen init source={source!r} record step={step!r} field={field!r}: {detail}"
    )


def _finite_record_number(record: dict[str, Any], field: str, source: str, step: Any) -> float:
    if field not in record:
        raise _frozen_record_error(source, step, field, "missing required field")
    value = record[field]
    if isinstance(value, bool):
        raise _frozen_record_error(source, step, field, f"expected finite number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _frozen_record_error(
            source, step, field, f"expected finite number, got {value!r}",
        ) from exc
    if not np.isfinite(number):
        raise _frozen_record_error(source, step, field, f"must be finite, got {value!r}")
    return number


def load_frozen_init_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load and strictly validate only the LHS prefix from a prior final history."""

    validate_frozen_init_configuration(args)
    source = str(args.frozen_init_history)
    try:
        with open(source, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise _frozen_record_error(source, "?", "source", str(exc)) from exc
    if not isinstance(payload, list):
        raise _frozen_record_error(source, "?", "source", "history root must be a JSON list")

    lhs_records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(payload):
        if not isinstance(raw_record, dict):
            raise _frozen_record_error(
                source, f"index:{index}", "record", "history entry must be a dict",
            )
        if raw_record.get("type") == "lhs_init":
            lhs_records.append(copy.deepcopy(raw_record))

    if len(lhs_records) != int(args.n_init):
        raise _frozen_record_error(
            source,
            "?",
            "type",
            f"expected exactly {int(args.n_init)} lhs_init records, found {len(lhs_records)}",
        )

    by_step: dict[int, dict[str, Any]] = {}
    search_dim = _search_dim(args)
    expected_mask_dim = hp_dim_from_mode(args.hp_mode)
    for record in lhs_records:
        raw_step = record.get("step")
        if isinstance(raw_step, bool) or not isinstance(raw_step, int):
            raise _frozen_record_error(source, raw_step, "step", "must be an integer")
        step = int(raw_step)
        if step in by_step:
            raise _frozen_record_error(source, step, "step", "duplicate lhs_init step")
        by_step[step] = record

        if record.get("type") != "lhs_init":
            raise _frozen_record_error(source, step, "type", "must equal 'lhs_init'")
        if record.get("hp_mode") != args.hp_mode:
            raise _frozen_record_error(
                source, step, "hp_mode", f"expected {args.hp_mode!r}, got {record.get('hp_mode')!r}",
            )
        record_search_dim = record.get("search_dim")
        if (
            isinstance(record_search_dim, bool)
            or not isinstance(record_search_dim, int)
            or record_search_dim != search_dim
        ):
            raise _frozen_record_error(
                source,
                step,
                "search_dim",
                f"expected {search_dim}, got {record.get('search_dim')!r}",
            )
        record_search_seed = record.get("search_seed")
        if (
            isinstance(record_search_seed, bool)
            or not isinstance(record_search_seed, int)
            or record_search_seed != int(args.seed)
        ):
            raise _frozen_record_error(
                source,
                step,
                "search_seed",
                f"expected {int(args.seed)}, got {record.get('search_seed')!r}",
            )

        z_search = record.get("z_search")
        if not isinstance(z_search, list):
            raise _frozen_record_error(source, step, "z_search", "must be a JSON list")
        try:
            z_array = np.asarray(z_search, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise _frozen_record_error(source, step, "z_search", str(exc)) from exc
        if z_array.size != search_dim:
            raise _frozen_record_error(
                source, step, "z_search", f"expected length {search_dim}, got {z_array.size}",
            )
        if not np.isfinite(z_array).all():
            raise _frozen_record_error(source, step, "z_search", "contains non-finite values")

        val_acc = _finite_record_number(record, "val_acc", source, step)
        if not 0.0 <= val_acc <= 1.0:
            raise _frozen_record_error(source, step, "val_acc", "must be in [0, 1]")
        if "valid" not in record or not isinstance(record["valid"], bool):
            raise _frozen_record_error(source, step, "valid", "must be present and boolean")
        operations = record.get("operations")
        if not isinstance(operations, list) or not all(
            isinstance(operation, str) for operation in operations
        ):
            raise _frozen_record_error(source, step, "operations", "must be present and a list")
        edges = record.get("edges")
        if not isinstance(edges, list) or not all(
            isinstance(edge, list)
            and len(edge) == 2
            and all(isinstance(vertex, int) and not isinstance(vertex, bool) for vertex in edge)
            for edge in edges
        ):
            raise _frozen_record_error(source, step, "edges", "must be present and a list")
        lr = _finite_record_number(record, "lr", source, step)
        if lr <= 0.0:
            raise _frozen_record_error(source, step, "lr", "must be positive")
        dropout = _finite_record_number(record, "dropout", source, step)
        if not 0.0 <= dropout <= 1.0:
            raise _frozen_record_error(source, step, "dropout", "must be in [0, 1]")
        hidden_dim = record.get("hidden_dim")
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise _frozen_record_error(source, step, "hidden_dim", "must be a positive integer")
        l2_field = "l2" if "l2" in record else "weight_decay"
        l2_value = _finite_record_number(record, l2_field, source, step)
        if l2_value < 0.0:
            raise _frozen_record_error(source, step, l2_field, "must be non-negative")

        mask = record.get("condition_mask_vector")
        if not isinstance(mask, list):
            raise _frozen_record_error(
                source, step, "condition_mask_vector", "must be present and a list",
            )
        try:
            mask_array = np.asarray(mask, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise _frozen_record_error(source, step, "condition_mask_vector", str(exc)) from exc
        if (
            mask_array.size != expected_mask_dim
            or not np.isfinite(mask_array).all()
            or not ((mask_array >= 0.0) & (mask_array <= 1.0)).all()
        ):
            raise _frozen_record_error(
                source,
                step,
                "condition_mask_vector",
                f"expected length {expected_mask_dim} with finite values in [0, 1], "
                f"got length {mask_array.size}",
            )

        derivation = record.get("seed_derivation")
        if derivation is not None and derivation != SEED_DERIVATION:
            raise _frozen_record_error(
                source,
                step,
                "seed_derivation",
                f"expected {SEED_DERIVATION!r}, got {derivation!r}",
            )
        expected_eval_seed = stable_seed(int(args.seed), "candidate_eval", step)
        candidate_eval_seed = record.get("candidate_eval_seed")
        if (
            candidate_eval_seed is not None
            and (
                isinstance(candidate_eval_seed, bool)
                or not isinstance(candidate_eval_seed, int)
                or candidate_eval_seed != expected_eval_seed
            )
        ):
            raise _frozen_record_error(
                source,
                step,
                "candidate_eval_seed",
                f"expected {expected_eval_seed}, got {record.get('candidate_eval_seed')!r}",
            )
        expected_decoder_seed = stable_seed(
            int(args.seed), "decoder", z_array[:ARCH_NZ],
        )
        decoder_seed = record.get("decoder_seed")
        if (
            decoder_seed is not None
            and (
                isinstance(decoder_seed, bool)
                or not isinstance(decoder_seed, int)
                or decoder_seed != expected_decoder_seed
            )
        ):
            raise _frozen_record_error(
                source,
                step,
                "decoder_seed",
                f"expected {expected_decoder_seed}, got {record.get('decoder_seed')!r}",
            )

    expected_steps = list(range(int(args.n_init)))
    actual_steps = sorted(by_step)
    if actual_steps != expected_steps:
        raise _frozen_record_error(
            source,
            "?",
            "step",
            f"expected consecutive steps {expected_steps}, got {actual_steps}",
        )

    records = [by_step[step] for step in expected_steps]
    for record in records:
        record["initial_record_replayed"] = True
        record["initial_history_source"] = source
        record["online_candidate_strategy"] = None
        record["candidate_selection_seed"] = None
    return records


def replay_frozen_initialization(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[torch.Tensor],
    list[float],
    list[tuple[torch.Tensor, float, list[float]]],
    list[dict[str, Any]],
]:
    records = load_frozen_init_records(args)
    X_obs: list[torch.Tensor] = []
    Y_obs: list[float] = []
    init_valid: list[tuple[torch.Tensor, float, list[float]]] = []
    prediction_records: list[dict[str, Any]] = []
    for record in records:
        z_search = torch.tensor(record["z_search"], dtype=torch.float32)
        val_acc = float(record["val_acc"])
        hp_condition_mask = [
            float(value) for value in record["condition_mask_vector"]
        ]
        condition_mask = [1.0] * ARCH_NZ + hp_condition_mask
        X_obs.append(z_search)
        Y_obs.append(val_acc)
        if record["valid"]:
            init_valid.append((z_search, val_acc, condition_mask))
        prediction_records.append(
            {
                "step": int(record["step"]),
                "record_type": "lhs_init",
                "gp_stage": "scratch_init_no_gp",
                "valid": bool(record["valid"]),
                "val_acc": val_acc,
                "search_seed": int(args.seed),
                "decoder_seed": record.get("decoder_seed"),
                "candidate_eval_seed": record.get("candidate_eval_seed"),
                "seed_derivation": record.get("seed_derivation", SEED_DERIVATION),
                "best_epoch": record.get("best_epoch"),
                "stopped_epoch": record.get("stopped_epoch"),
                "epochs_ran": record.get("epochs_ran"),
                "online_candidate_strategy": None,
                "candidate_selection_seed": None,
                "initial_record_replayed": True,
                "initial_history_source": str(args.frozen_init_history),
                "gp_update_performed": False,
                "gp_update_skipped_reason": "replayed_initial_record",
                "eval_seconds": 0.0,
                "gp_update_seconds": 0.0,
                "total_step_seconds": 0.0,
                **_null_gp_record_fields(),
            }
        )
    return records, X_obs, Y_obs, init_valid, prediction_records


def _training_best(predictor: AccuracyGPPredictor) -> float:
    normalized_best = float(predictor.train_Y_norm.max().detach().cpu())
    return normalized_best * predictor.y_std + predictor.y_mean


def _score_logei(
    predictor: AccuracyGPPredictor | None,
    z_search: torch.Tensor,
    condition_mask: list[float],
    best_f: float,
) -> float:
    logei = predictor.make_logei(best_f)
    model_input = predictor.acquisition_inputs(
        z_search,
        condition_masks=condition_mask if predictor.use_conditional_kernel else None,
    )
    with torch.no_grad():
        value = logei(model_input.double().unsqueeze(1)).reshape(-1)[0]
    if not torch.isfinite(value):
        raise RuntimeError("qLogExpectedImprovement returned a non-finite candidate score")
    return float(value.detach().cpu())


def run_gmm_init(
    vae: JointSpaceVAE,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger,
    history: list[dict[str, Any]],
    X_obs: list[torch.Tensor],
    Y_obs: list[float],
    predictor: AccuracyGPPredictor,
    prediction_records: list[dict[str, Any]],
    init_valid: list[tuple[torch.Tensor, float, list[float]]],
    step: int,
) -> int:
    if int(args.gmm_init_trials) <= 0:
        logger.info("Skipping GMM init: gmm_init_trials <= 0")
        return step
    if args.gp_init_mode == "scratch":
        logger.info(
            "Skipping GMM init in scratch GP mode: weighted GMM requires previous history, "
            "and scratch mode is configured to avoid old history."
        )
        return step
    if not args.gmm_init_history:
        logger.info("Skipping GMM init: no --gmm_init_history provided")
        return step
    if predictor is None:
        raise RuntimeError("checkpoint-mode GMM scoring requires a loaded GP predictor")

    search_dim = ARCH_NZ + hp_dim_from_mode(args.hp_mode)
    args.search_dim = search_dim
    args.arch_nz = ARCH_NZ

    X, y, meta = load_history_vectors(args.gmm_init_history, args.hp_mode, search_dim, logger)
    sampled_z_list, summary = fit_and_sample_gmm_init(X, y, args.hp_mode, args, logger)
    summary.update({f"loader_{key}": value for key, value in meta.items()})

    if not sampled_z_list:
        if args.gmm_save_summary:
            os.makedirs(args.output, exist_ok=True)
            summary_path = os.path.join(args.output, f"gmm_init_summary_bo_{args.hp_mode}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"GMM summary saved: {summary_path}")
        return step

    logger.info(f"\n[Step 2] GMM init: evaluating {len(sampled_z_list)} candidates")
    evaluated_count = 0
    for idx, z in enumerate(tqdm(sampled_z_list, desc="GMM init")):
        if not is_finite_z_search(z):
            logger.warning(f"Skip non-finite GMM sample #{idx}")
            continue
        try:
            # Defensive clip only after weighted_diag_gmm_init has inverse-standardized.
            z_np = clip_z_search_by_mode(z, args.hp_mode, ARCH_NZ, args.z_bound)
        except ValueError as exc:
            logger.warning(f"Skip malformed GMM sample #{idx}: {exc}")
            continue
        z_tensor = torch.tensor(z_np, dtype=torch.float32)
        condition_mask = _mask_for_candidate(vae, z_tensor, args, device, logger)
        best_before = max(Y_obs) if Y_obs else None
        logei_value = _score_logei(
            predictor, z_tensor, condition_mask,
            best_before if best_before is not None else _training_best(predictor),
        )
        acc, valid, used_mask, safe_for_gp_training = _append_eval(
            vae,
            z_tensor,
            data,
            in_ch,
            out_ch,
            args,
            device,
            history,
            X_obs,
            Y_obs,
            step,
            "gmm_init",
            logger,
            predictor,
            prediction_records,
            "offline_init",
            logei_value,
            False,
            0,
            condition_mask,
            best_acc=best_before,
        )
        if safe_for_gp_training:
            init_valid.append((z_tensor, acc, used_mask))
        evaluated_count += 1
        step += 1
    summary["evaluated_count"] = int(evaluated_count)
    if args.gmm_save_summary:
        os.makedirs(args.output, exist_ok=True)
        summary_path = os.path.join(args.output, f"gmm_init_summary_bo_{args.hp_mode}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"GMM summary saved: {summary_path}")
    return step


def _scratch_random_candidate(args: argparse.Namespace, seed: int) -> torch.Tensor:
    unit = torch.tensor(
        _sample_unit_lhs(1, _search_dim(args), int(seed)),
        dtype=torch.float32,
    )
    raw = denormalize_search_vector(
        unit,
        arch_nz=ARCH_NZ,
        hp_mode=args.hp_mode,
        z_bound=args.z_bound,
    ).reshape(-1)
    clipped = clip_z_search_by_mode(raw, args.hp_mode, ARCH_NZ, args.z_bound)
    return torch.tensor(clipped, dtype=torch.float32)


def _ensure_scratch_min_points(
    vae: JointSpaceVAE,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger,
    history: list[dict[str, Any]],
    X_obs: list[torch.Tensor],
    Y_obs: list[float],
    prediction_records: list[dict[str, Any]],
    init_valid: list[tuple[torch.Tensor, float, list[float]]],
    step: int,
) -> int:
    required = int(args.scratch_gp_min_points)
    if len(init_valid) >= required:
        return step
    max_extra = max(int(args.n_lhs_candidates), required * 10, 10)
    logger.warning(
        "Scratch GP has %d valid init samples; sampling up to %d extra LHS/random points to reach %d.",
        len(init_valid), max_extra, required,
    )
    attempts = 0
    while len(init_valid) < required and attempts < max_extra:
        z = _scratch_random_candidate(args, int(args.seed) + 170003 + attempts)
        condition_mask = _mask_for_candidate(vae, z, args, device, logger)
        best_before = max(Y_obs) if Y_obs else None
        acc, valid, used_mask, safe_for_gp_training = _append_eval(
            vae,
            z,
            data,
            in_ch,
            out_ch,
            args,
            device,
            history,
            X_obs,
            Y_obs,
            step,
            "scratch_extra_init",
            logger,
            None,
            prediction_records,
            "scratch_init_no_gp",
            None,
            False,
            0,
            condition_mask,
            best_acc=best_before,
        )
        if safe_for_gp_training:
            init_valid.append((z, acc, used_mask))
        step += 1
        attempts += 1
    if len(init_valid) < required:
        raise RuntimeError(
            f"cold-start GP training requires at least {required} valid initialization "
            f"samples, but only {len(init_valid)} were collected after {attempts} extra attempts"
        )
    return step


def _fit_scratch_predictor(
    init_valid: list[tuple[torch.Tensor, float, list[float]]],
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> AccuracyGPPredictor:
    if len(init_valid) < int(args.scratch_gp_min_points):
        raise RuntimeError(
            f"scratch GP requires at least {args.scratch_gp_min_points} valid samples, got {len(init_valid)}"
        )
    init_X = torch.stack([item[0] for item in init_valid])
    init_y = [item[1] for item in init_valid]
    init_masks = [item[2] for item in init_valid]
    metadata = {
        "gp_init_mode": "scratch",
        "offline_train_size": 0,
        "used_previous_history": False,
        "used_offline_checkpoint": False,
        "scratch_initial_train_size": int(len(init_valid)),
        "dataset": "Cora",
        "metric": "val_acc",
        "eval_epochs": int(args.eval_epochs),
        "patience": int(args.patience),
        "vae_checkpoint": args.checkpoint,
        "vae_version": args.version,
        "seed": int(args.seed),
    }
    started = time.monotonic()
    predictor = AccuracyGPPredictor.fit_offline(
        init_X,
        init_y,
        arch_nz=ARCH_NZ,
        hp_mode=args.hp_mode,
        z_bound=args.z_bound,
        use_conditional_kernel=bool(args.use_conditional_kernel),
        condition_masks=init_masks if args.use_conditional_kernel else None,
        metadata=metadata,
        device=device,
        fit_steps=int(args.gp_refit_steps),
    )
    predictor.offline_train_size = 0
    predictor.metadata.update(metadata)
    path = os.path.join(args.output, "accuracy_gp_scratch_initial.pt")
    predictor.save(path)
    logger.info(
        "Scratch GP initial fit: train_size=%d seconds=%.3f saved=%s",
        predictor.train_size, time.monotonic() - started, path,
    )
    return predictor


def run_bo(
    vae: JointSpaceVAE,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger,
    predictor: AccuracyGPPredictor | None,
) -> None:
    history: list[dict[str, Any]] = []
    X_obs: list[torch.Tensor] = []
    Y_obs: list[float] = []
    prediction_records: list[dict[str, Any]] = []
    init_valid: list[tuple[torch.Tensor, float, list[float]]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 7919)
    run_started = time.monotonic()
    initialization_source = (
        "frozen_history" if args.frozen_init_history is not None else "evaluated"
    )
    replayed_initial_samples = 0

    logger.info(f"\n[Step 1] LHS init: n_init={args.n_init}")
    logger.info(f"  gp_init_mode={args.gp_init_mode}")
    logger.info(f"  online candidate strategy={args.online_candidate_strategy}")
    logger.info(f"  initialization source={initialization_source}")
    logger.info(f"  frozen history path={args.frozen_init_history}")
    logger.info(
        "  online budget mode=%s",
        "adaptive" if args.adaptive_sampling else "fixed",
    )
    logger.info("  adaptive sampling budgets count online BO samples after initial GP fit")
    logger.info(f"  hp_mode={args.hp_mode}")
    logger.info(f"  hp_dim={hp_dim_from_mode(args.hp_mode)}")
    logger.info(f"  hp_names={hp_names_from_mode(args.hp_mode)}")
    logger.info(f"  ARCH_NZ={ARCH_NZ} SEARCH_DIM={ARCH_NZ + hp_dim_from_mode(args.hp_mode)}")
    logger.info(f"  z_bound={args.z_bound}")
    logger.info(f"  LR: 10^[{args.log_lr_min},{args.log_lr_max}] Dropout: [{args.dropout_min},{args.dropout_max}]")

    if args.frozen_init_history is not None:
        (
            history,
            X_obs,
            Y_obs,
            init_valid,
            prediction_records,
        ) = replay_frozen_initialization(args)
        replayed_initial_samples = len(history)
        step = int(args.n_init)
        _save_prediction_csv(prediction_records, args.output)
        logger.info("  replayed initial sample count=%d", replayed_initial_samples)
    else:
        step = 0
        for z in tqdm(initial_points(args, logger), desc="LHS init"):
            condition_mask = _mask_for_candidate(vae, z, args, device, logger)
            best_before = max(Y_obs) if Y_obs else None
            if predictor is None:
                logei_value = None
                gp_stage = "scratch_init_no_gp"
            else:
                logei_value = _score_logei(
                    predictor, z, condition_mask,
                    best_before if best_before is not None else _training_best(predictor),
                )
                gp_stage = "offline_init"
            acc, valid, used_mask, safe_for_gp_training = _append_eval(
                vae,
                z,
                data,
                in_ch,
                out_ch,
                args,
                device,
                history,
                X_obs,
                Y_obs,
                step,
                "lhs_init",
                logger,
                predictor,
                prediction_records,
                gp_stage,
                logei_value,
                False,
                0,
                condition_mask,
                best_acc=best_before,
            )
            if safe_for_gp_training:
                init_valid.append((z, acc, used_mask))
            step += 1
        logger.info("  replayed initial sample count=0")

    lhs_best = max(Y_obs) if Y_obs else 0.0
    logger.info(f"\nLHS done. Best={lhs_best:.4f}")

    if args.frozen_init_history is None:
        step = run_gmm_init(
            vae,
            data,
            in_ch,
            out_ch,
            args,
            device,
            logger,
            history,
            X_obs,
            Y_obs,
            predictor,
            prediction_records,
            init_valid,
            step,
        )

    if args.gp_init_mode == "scratch":
        if args.frozen_init_history is None:
            step = _ensure_scratch_min_points(
                vae,
                data,
                in_ch,
                out_ch,
                args,
                device,
                logger,
                history,
                X_obs,
                Y_obs,
                prediction_records,
                init_valid,
                step,
            )
        elif len(init_valid) < int(args.scratch_gp_min_points):
            raise _frozen_record_error(
                str(args.frozen_init_history),
                "?",
                "valid",
                f"requires at least {int(args.scratch_gp_min_points)} valid lhs_init records, "
                f"found {len(init_valid)}",
            )
        predictor = _fit_scratch_predictor(init_valid, args, device, logger)
        logger.info(
            "Scratch init summary: scratch_init_samples=%d scratch_init_valid_samples=%d "
            "total_evaluated_samples=%d",
            len([row for row in prediction_records if row["gp_stage"] == "scratch_init_no_gp"]),
            len(init_valid),
            len(history),
        )
    elif not init_valid:
        logger.error("All LHS/warm-start/GMM initialization points were invalid.")
        assert predictor is not None
        predictor.save(os.path.join(args.output, "accuracy_gp_online_final.pt"))
        summary = {
            "gp_init_mode": args.gp_init_mode,
            "used_previous_history": True,
            "used_offline_checkpoint": True,
            "converged": False,
            "stop_reason": "no_valid_initialization_samples",
            "offline_train_size": predictor.offline_train_size,
            "online_added": 0,
            "scratch_init_samples": 0,
            "scratch_init_valid_samples": 0,
            "online_bo_samples": 0,
            "total_evaluated_samples": len(history),
            "gp_train_size_final": predictor.train_size,
            "best_actual_val_acc": float(max(Y_obs)) if Y_obs else 0.0,
            "elapsed_seconds": float(time.monotonic() - run_started),
            "search_seed": int(args.seed),
            "seed_derivation": SEED_DERIVATION,
            "rng_isolation_enabled": True,
            "online_candidate_strategy": args.online_candidate_strategy,
            "initialization_source": initialization_source,
            "frozen_init_history": args.frozen_init_history,
            "replayed_initial_samples": int(replayed_initial_samples),
            "newly_evaluated_online_samples": 0,
        }
        atomic_json_dump(summary, os.path.join(args.output, "gp_metrics.json"))
        _save(history, X_obs, Y_obs, args, logger, "final", predictor=predictor, summary=summary)
        return
    else:
        assert predictor is not None
        init_X = torch.stack([item[0] for item in init_valid])
        init_y = [item[1] for item in init_valid]
        init_masks = [item[2] for item in init_valid]
        update_started = time.monotonic()
        predictor.append_observations(
            init_X, init_y,
            condition_masks=init_masks if predictor.use_conditional_kernel else None,
        )
        predictor.refit(optimize=True, steps=int(args.gp_refit_steps))
        logger.info(
            "Offline init batch update: added=%d train_size=%d seconds=%.3f",
            len(init_valid), predictor.train_size, time.monotonic() - update_started,
        )

    assert predictor is not None
    best_acc = max(Y_obs)
    logger.info(
        f"\n[Step 3] BO: n_iter={args.n_iter} "
        f"SEARCH_DIM={ARCH_NZ + hp_dim_from_mode(args.hp_mode)}"
    )

    monitor = GPConvergenceMonitor(
        min_bo_samples=args.min_bo_samples,
        max_bo_samples=args.max_bo_samples,
        convergence_check_every=args.convergence_check_every,
        convergence_patience=args.convergence_patience,
        prequential_window=args.prequential_window,
        mae_relative_tol=args.mae_relative_tol,
        mae_absolute_tol=args.mae_absolute_tol,
        std_relative_tol=args.std_relative_tol,
        spearman_tol=args.spearman_tol,
        degradation_tolerance=args.degradation_tolerance,
        best_acc_patience=args.best_acc_patience,
        best_acc_min_delta=args.best_acc_min_delta,
        max_wall_time_hours=args.max_wall_time_hours,
    )
    probe_seed = args.probe_pool_seed if args.probe_pool_seed is not None else int(args.seed) + 104729
    probe_norm = torch.tensor(
        _sample_unit_lhs(int(args.probe_pool_size), _search_dim(args), int(probe_seed)),
        dtype=torch.float32,
    )
    probe_raw = denormalize_search_vector(
        probe_norm, arch_nz=ARCH_NZ, hp_mode=args.hp_mode, z_bound=args.z_bound
    ).float()
    probe_masks = (
        _candidate_masks(vae, probe_raw, args, device, logger)
        if predictor.use_conditional_kernel else None
    )
    online_valid_count = 0
    eval_times: list[float] = []
    update_times: list[float] = []
    online_limit = int(args.max_bo_samples) if args.adaptive_sampling else int(args.n_iter)
    stop_reason = "fixed_iteration_complete"

    for it in range(online_limit):
        if args.adaptive_sampling:
            last_estimate = 0.0
            if eval_times or update_times:
                last_estimate = float(np.median(eval_times[-5:])) + float(np.median(update_times[-5:]))
            decision = monitor.stop_decision(it, time.monotonic() - run_started, last_estimate)
            if decision.deferred_reason is not None:
                logger.info("Adaptive stop deferred: %s", decision.deferred_reason)
            if decision.stop_reason is not None:
                stop_reason = decision.stop_reason
                break
        if args.online_candidate_strategy == "qlogei":
            acquisition_seed = stable_seed(
                int(args.seed), "acquisition", int(step),
            )
            candidate_selection_seed = acquisition_seed
            with isolated_rng(acquisition_seed, device):
                z_next, logei_value, condition_mask = optimize_acq(
                    predictor, best_acc, args, logger, vae, device, generator,
                )
        elif args.online_candidate_strategy == "random":
            (
                z_next,
                condition_mask,
                candidate_selection_seed,
            ) = sample_random_online_candidate(vae, args, device, logger, step)
            logei_value = None
        else:
            raise ValueError(
                f"unsupported online candidate strategy: {args.online_candidate_strategy!r}"
            )

        z_np = clip_z_search_by_mode(z_next, args.hp_mode, ARCH_NZ, args.z_bound)
        z_next = torch.tensor(z_np, dtype=torch.float32)
        acc, valid, _, _safe_for_gp_training = _append_eval(
            vae,
            z_next,
            data,
            in_ch,
            out_ch,
            args,
            device,
            history,
            X_obs,
            Y_obs,
            step,
            "bo",
            logger,
            predictor,
            prediction_records,
            "online_bo",
            logei_value,
            True,
            online_valid_count,
            condition_mask,
            best_acc=best_acc,
            convergence_monitor=monitor,
            candidate_selection_seed=candidate_selection_seed,
        )
        current_record = prediction_records[-1]
        if current_record["gp_update_performed"]:
            online_valid_count += 1
            if online_valid_count % int(args.gp_save_every) == 0:
                predictor.save(os.path.join(args.output, "accuracy_gp_online_latest.pt"))
        eval_times.append(float(current_record["eval_seconds"]))
        update_times.append(float(current_record["gp_update_seconds"]))
        if acc > best_acc:
            best_acc = acc
            logger.info(f"  iter {it:>3d}: <-- NEW BEST {best_acc:.4f}")
        else:
            logger.info(f"  iter {it:>3d}: best={best_acc:.4f}")

        step += 1
        online_samples = it + 1
        if monitor.should_check(online_samples):
            holdout = _evaluate_fixed_holdout(predictor, logger)
            probe_predictions = predictor.predict_batch(
                probe_raw,
                condition_masks=probe_masks if predictor.use_conditional_kernel else None,
            )
            check = monitor.add_check(
                online_samples=online_samples,
                gp_train_size=predictor.train_size,
                holdout_metrics=holdout,
                probe_stds=[row["std"] for row in probe_predictions],
                elapsed_seconds=time.monotonic() - run_started,
                eval_seconds=eval_times,
                gp_update_seconds=update_times,
            )
            _write_convergence_files(monitor.history, args.output)
            logger.info("GP convergence check: %s", json.dumps(check, ensure_ascii=False))
        if (it + 1) % 10 == 0:
            _save(history, X_obs, Y_obs, args, logger, f"step{it}", predictor=predictor)
    else:
        if args.adaptive_sampling:
            stop_reason = "sample_budget_reached"

    online_bo_samples = len([row for row in prediction_records if row["gp_stage"] == "online_bo"])
    final_decision = monitor.stop_decision(
        online_bo_samples,
        time.monotonic() - run_started,
        monitor.history[-1]["estimated_next_step_seconds"] if monitor.history else 0.0,
    )
    if args.adaptive_sampling and final_decision.stop_reason is not None:
        stop_reason = final_decision.stop_reason

    best_idx = int(np.argmax(np.asarray(Y_obs, dtype=np.float64)))
    best_z = X_obs[best_idx]
    best_decoder_seed = stable_seed(int(args.seed), "decoder", best_z[:ARCH_NZ])
    best_cfg = decode_arch(
        vae,
        best_z[:ARCH_NZ],
        device,
        n_trials=10,
        decoder_seed=best_decoder_seed,
    )

    logger.info("\n" + "=" * 66)
    logger.info(f"          Phase 4 BO hp_mode={args.hp_mode} Final")
    logger.info("=" * 66)
    logger.info(f"  Best val_acc  : {max(Y_obs):.4f}")
    logger.info(f"  GNN config    : {best_cfg}")
    logger.info(f"  SEARCH_DIM    : {ARCH_NZ + hp_dim_from_mode(args.hp_mode)}")
    logger.info("=" * 66)

    predictor.save(os.path.join(args.output, "accuracy_gp_online_final.pt"))
    summary = {
        "gp_init_mode": args.gp_init_mode,
        "used_previous_history": bool(args.gp_init_mode == "checkpoint"),
        "used_offline_checkpoint": bool(args.gp_init_mode == "checkpoint"),
        "converged": bool(final_decision.converged),
        "stop_reason": stop_reason,
        "offline_train_size": int(predictor.offline_train_size),
        "online_added": int(predictor.train_size - predictor.offline_train_size),
        "scratch_init_samples": int(
            len([row for row in prediction_records if row["gp_stage"] == "scratch_init_no_gp"])
        ),
        "scratch_init_valid_samples": int(len(init_valid)) if args.gp_init_mode == "scratch" else 0,
        "online_bo_samples": int(online_bo_samples),
        "total_evaluated_samples": int(len(history)),
        "gp_train_size_final": predictor.train_size,
        "best_actual_val_acc": float(max(Y_obs)),
        "elapsed_seconds": float(time.monotonic() - run_started),
        "search_seed": int(args.seed),
        "seed_derivation": SEED_DERIVATION,
        "rng_isolation_enabled": True,
        "online_candidate_strategy": args.online_candidate_strategy,
        "initialization_source": initialization_source,
        "frozen_init_history": args.frozen_init_history,
        "replayed_initial_samples": int(replayed_initial_samples),
        "newly_evaluated_online_samples": int(online_bo_samples),
        "convergence_checks": int(len(monitor.history)),
        "valid_convergence_checks": int(monitor.valid_convergence_checks),
        "stagnation_deferred_events": list(monitor.stagnation_deferred_events),
        **monitor.current_prequential_metrics(),
    }
    atomic_json_dump(summary, os.path.join(args.output, "gp_metrics.json"))
    _write_convergence_files(monitor.history, args.output)
    _save(history, X_obs, Y_obs, args, logger, "final", predictor=predictor, summary=summary)
    _plot(history, args, logger)
    try:
        from analyse.plot_gp_prediction_records import plot_gp_records

        plot_gp_records(args.output)
    except Exception as exc:
        logger.warning("GP prediction plotting failed after results were saved: %s", exc)


def _evaluate_fixed_holdout(
    predictor: AccuracyGPPredictor,
    logger,
) -> dict[str, Any] | None:
    if predictor.holdout_X_raw is None or predictor.holdout_Y is None:
        logger.warning("GP checkpoint has no fixed holdout; holdout convergence metrics unavailable")
        return None
    masks = predictor.holdout_condition_masks if predictor.use_conditional_kernel else None
    rows = predictor.predict_batch(predictor.holdout_X_raw, condition_masks=masks)
    return prediction_metrics(
        predictor.holdout_Y.tolist(),
        [row["mean"] for row in rows],
        [row["std"] for row in rows],
    )


def _write_convergence_files(rows: list[dict[str, Any]], output: str) -> None:
    os.makedirs(output, exist_ok=True)
    atomic_json_dump(rows, os.path.join(output, "gp_convergence_history.json"))
    path = os.path.join(output, "gp_convergence_history.csv")
    tmp_path = path + ".tmp"
    fields = list(rows[0]) if rows else [
        "check_index", "online_samples", "gp_train_size", "holdout_mae",
        "prequential_window_mae", "probe_mean_std", "converged",
    ]
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _save(
    history: list[dict[str, Any]],
    X_obs: list[torch.Tensor],
    Y_obs: list[float],
    args: argparse.Namespace,
    logger,
    suffix: str,
    predictor: AccuracyGPPredictor | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    os.makedirs(args.output, exist_ok=True)
    history_path = os.path.join(args.output, f"history_{suffix}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    saved = [history_path]
    if X_obs and Y_obs:
        best_z = X_obs[int(np.argmax(np.asarray(Y_obs, dtype=np.float64)))]
        best_z_search_path = os.path.join(args.output, f"best_z_search_{suffix}.pt")
        torch.save(best_z, best_z_search_path)
        saved.append(best_z_search_path)
        if suffix == "final":
            best_z_final_path = os.path.join(args.output, "best_z_final.pt")
            torch.save(best_z, best_z_final_path)
            saved.append(best_z_final_path)
    if predictor is not None:
        checkpoint_path = os.path.join(args.output, "accuracy_gp_online_latest.pt")
        predictor.save(checkpoint_path)
        saved.append(checkpoint_path)
    if summary is not None:
        summary_path = os.path.join(args.output, "run_summary.json")
        atomic_json_dump(summary, summary_path)
        saved.append(summary_path)
    logger.info(f"Saved: {', '.join(saved)}")


def _plot(history: list[dict[str, Any]], args: argparse.Namespace, logger) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [h["step"] for h in history]
        accs = [h["val_acc"] for h in history]
        valids = [h.get("valid", True) for h in history]
        bests = []
        cur = 0.0
        for h in history:
            cur = max(cur, h["val_acc"])
            bests.append(cur)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        valid_steps = [s for s, v in zip(steps, valids) if v]
        valid_accs = [a for a, v in zip(accs, valids) if v]
        invalid_steps = [s for s, v in zip(steps, valids) if not v]
        invalid_accs = [a for a, v in zip(accs, valids) if not v]
        ax.scatter(valid_steps, valid_accs, alpha=0.6, color="steelblue", label="Valid")
        ax.scatter(invalid_steps, invalid_accs, alpha=0.4, color="red", marker="x", label="Invalid")
        ax.plot(steps, bests, "r-", lw=2, label="Best so far")
        ax.set(xlabel="Step", ylabel="Val Accuracy", title=f"Phase4 BO {args.hp_mode}")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(args.output, f"convergence_phase4_bo_{args.hp_mode}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved: {path}")
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot")


def main() -> None:
    args = parse_args()
    args.hp_mode = validate_hp_mode(args.hp_mode)
    hp_dim = hp_dim_from_mode(args.hp_mode)
    search_dim = ARCH_NZ + hp_dim
    os.makedirs(args.output, exist_ok=True)

    for name in (
        "n_init", "n_lhs_candidates", "n_iter", "num_restarts", "raw_samples",
        "gp_refit_every", "gp_refit_steps", "gp_save_every", "min_bo_samples",
        "max_bo_samples", "convergence_check_every", "convergence_patience",
        "prequential_window", "best_acc_patience", "probe_pool_size",
        "scratch_gp_min_points",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if int(args.scratch_gp_min_points) < 2:
        raise ValueError("--scratch_gp_min_points must be at least 2 for Exact GP training")
    if int(args.n_extra) < 0:
        raise ValueError("--n_extra must be non-negative")
    if int(args.min_bo_samples) > int(args.max_bo_samples):
        raise ValueError("--min_bo_samples cannot exceed --max_bo_samples")
    if float(args.max_wall_time_hours) <= 0.0:
        raise ValueError("--max_wall_time_hours must be positive")
    validate_frozen_init_configuration(args)
    if float(args.novelty_w) != 0.0:
        warnings.warn(
            "--novelty_w is deprecated and ignored by all online candidate strategies",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.gp_init_mode == "checkpoint" and not args.gp_checkpoint:
        raise ValueError("--gp_checkpoint is required when --gp_init_mode checkpoint")
    if args.gp_init_mode == "scratch" and args.gp_checkpoint:
        warnings.warn(
            "--gp_checkpoint is ignored when --gp_init_mode scratch",
            RuntimeWarning,
            stacklevel=2,
        )

    logger, log_path = setup_logger(args.log_dir, "bo_phase4", args.version)
    save_args_json(args, log_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    logger.info("=" * 70)
    logger.info(f"bo_phase4.py -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info(f"Device={device} ARCH_NZ={ARCH_NZ} SEARCH_DIM={search_dim}")
    logger.info(f"hp_mode={args.hp_mode} hp_dim={hp_dim} hp_names={hp_names_from_mode(args.hp_mode)}")
    logger.info(f"z_bound={args.z_bound}")
    logger.info(f"gp_init_mode={args.gp_init_mode}")
    logger.info(f"online_candidate_strategy={args.online_candidate_strategy}")
    logger.info(
        "initialization_source=%s frozen_init_history=%s",
        "frozen_history" if args.frozen_init_history is not None else "evaluated",
        args.frozen_init_history,
    )
    logger.info(
        "online_budget=%s",
        "adaptive" if args.adaptive_sampling else "fixed",
    )

    vae = load_vae(args, device, logger)
    data, in_ch, out_ch = load_cora(args.cora_root, device, logger)
    logger.info(f"Cora: {in_ch} features, {out_ch} classes")

    predictor: AccuracyGPPredictor | None = None
    if args.gp_init_mode == "checkpoint":
        predictor = AccuracyGPPredictor.load(
            args.gp_checkpoint,
            device=device,
            expected={
                "arch_nz": ARCH_NZ,
                "hp_mode": args.hp_mode,
                "search_dim": search_dim,
                "z_bound": float(args.z_bound),
                "dataset": "Cora",
                "eval_epochs": int(args.eval_epochs),
                "patience": int(args.patience),
                "vae_checkpoint": args.checkpoint,
                "vae_version": os.path.basename(args.checkpoint),
                "use_conditional_kernel": bool(args.use_conditional_kernel),
            },
        )
        holdout_size = 0 if predictor.holdout_Y is None else int(predictor.holdout_Y.numel())
        logger.info(
            "Accuracy GP loaded: source=%s model=SingleTaskGP kernel=%s offline_train=%d holdout=%d",
            args.gp_checkpoint, predictor.kernel_type, predictor.offline_train_size, holdout_size,
        )
    else:
        logger.info(
            "Scratch GP mode: no previous history and no offline GP checkpoint will be used; "
            "initial GP will be trained after at least %d valid initialization samples.",
            int(args.scratch_gp_min_points),
        )
        if args.gmm_init_history:
            logger.warning("Ignoring --gmm_init_history in scratch GP mode to avoid previous-history leakage.")

    run_bo(vae, data, in_ch, out_ch, args, device, logger, predictor)
    elapsed = time.time() - start_time
    logger.info(f"Run complete. Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
