"""Phase 4 BO search with LHS init, geometric novelty hooks, and GMM init.

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
from tqdm import tqdm

sys.path.insert(0, "/mnt/project")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Input.*not contained.*unit cube.*",
)

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from eval_utils import eval_z_search
from hp_modes import (
    condition_mask_vector_from_ops,
    hp_dim_from_mode,
    hp_names_from_mode,
    validate_hp_mode,
)
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


class GPNDNASNovelty:
    """Lightweight novelty scorer used by the BO acquisition reranking step.

    The original Phase4 path used geometric novelty terms together with LogEI.
    This local implementation keeps the same role in the pipeline: it updates
    from observed joint vectors and rewards candidates that are far from the
    current observation set in normalized joint space.
    """

    def __init__(self):
        self._x_obs: torch.Tensor | None = None

    def update(self, X_obs: torch.Tensor) -> None:
        self._x_obs = X_obs.detach().cpu().float().clone()

    def batch_score(self, z_list: list[torch.Tensor], normalize: bool = True) -> np.ndarray:
        if self._x_obs is None or len(z_list) == 0:
            return np.zeros(len(z_list), dtype=np.float64)
        X = self._x_obs
        x_min = X.min(0).values
        x_range = (X.max(0).values - x_min).clamp(min=1e-8)
        Xn = (X - x_min) / x_range
        Z = torch.stack([z.detach().cpu().float() for z in z_list])
        Zn = (Z - x_min) / x_range
        dists = torch.cdist(Zn, Xn).min(dim=1).values.cpu().numpy()
        if normalize and dists.size > 0:
            d_min = float(np.min(dists))
            d_max = float(np.max(dists))
            if d_max - d_min > 1e-12:
                dists = (dists - d_min) / (d_max - d_min)
        return np.nan_to_num(dists, nan=0.0, posinf=0.0, neginf=0.0)


class ConditionalMaskedKernel(Kernel):
    """RBF kernel over [features, masks] with pairwise active HP dimensions."""

    has_lengthscale = True

    def __init__(self, feature_dim: int, arch_nz: int, **kwargs):
        super().__init__(ard_num_dims=int(feature_dim), **kwargs)
        self.feature_dim = int(feature_dim)
        self.arch_nz = int(arch_nz)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("ConditionalMaskedKernel does not support last_dim_is_batch")
        if x2 is None:
            x2 = x1

        d = self.feature_dim
        feat1 = x1[..., :d]
        feat2 = x2[..., :d]
        mask1 = x1[..., d:2 * d].clamp(0.0, 1.0)
        mask2 = x2[..., d:2 * d].clamp(0.0, 1.0)

        if feat1.shape[-1] != d or feat2.shape[-1] != d:
            raise ValueError("ConditionalMaskedKernel received malformed augmented inputs")
        if mask1.shape[-1] != d or mask2.shape[-1] != d:
            raise ValueError("ConditionalMaskedKernel requires mask dimensions appended after features")

        diff = feat1.unsqueeze(-2) - feat2.unsqueeze(-3)
        pair_mask = torch.sqrt((mask1.unsqueeze(-2) * mask2.unsqueeze(-3)).clamp(min=0.0))
        diff = diff * pair_mask

        lengthscale = self.lengthscale.to(device=diff.device, dtype=diff.dtype).reshape(-1)
        if lengthscale.numel() != d:
            lengthscale = lengthscale[-d:]
        lengthscale = lengthscale.clamp(min=1e-8).view(*([1] * (diff.dim() - 1)), d)

        scaled = diff / lengthscale
        r2 = (scaled * scaled).sum(dim=-1)
        K = torch.exp(-0.5 * r2.clamp(min=0.0))
        if diag:
            return K.diagonal(dim1=-2, dim2=-1)
        return K


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
    parser.add_argument("--novelty_w", type=float, default=0.15)
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


def history_record(
    step: int,
    record_type: str,
    z_search: torch.Tensor,
    result: dict[str, Any],
    args: argparse.Namespace,
    best_acc: float | None = None,
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


def _minmax_normalize(X: torch.Tensor):
    xmin = X.min(0).values
    xrng = (X.max(0).values - xmin).clamp(min=1e-8)
    return (X - xmin) / xrng, xmin, xrng


def _search_dim(args: argparse.Namespace) -> int:
    return ARCH_NZ + hp_dim_from_mode(args.hp_mode)


def _mask_from_ops(ops: list[str], args: argparse.Namespace) -> list[float]:
    hp_mask = condition_mask_vector_from_ops(ops, args.hp_mode)
    return [1.0] * ARCH_NZ + [float(v) for v in hp_mask]


def _mask_from_history_row(row: dict[str, Any], args: argparse.Namespace) -> list[float]:
    search_dim = _search_dim(args)
    hp_dim = hp_dim_from_mode(args.hp_mode)
    vec = row.get("condition_mask_vector")
    if vec is None and isinstance(row.get("condition_mask"), dict):
        cond = row["condition_mask"]
        vec = cond.get("vector", cond.get("mask"))

    if vec is not None:
        vals = [float(v) for v in list(vec)]
        if len(vals) == search_dim:
            return vals
        if len(vals) == hp_dim:
            return [1.0] * ARCH_NZ + vals

    return _mask_from_ops([str(op) for op in row.get("operations", [])], args)


def _mask_for_candidate(
    vae: JointSpaceVAE,
    z_search: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> list[float]:
    try:
        config = decode_arch(vae, z_search[:ARCH_NZ], device, n_trials=3)
        ops = [] if config is None else [str(op) for op in config.get("operations", [])]
        return _mask_from_ops(ops, args)
    except Exception as exc:
        logger.warning(f"Failed to decode candidate mask; using all-active mask: {exc}")
        return [1.0] * _search_dim(args)


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


def _fit_standard_gp(X_train: torch.Tensor, Y_train: torch.Tensor):
    gp = SingleTaskGP(X_train.double(), Y_train.double())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    return gp


def build_gp_model(
    X_norm: torch.Tensor,
    Y_norm: torch.Tensor,
    M_obs: torch.Tensor | None,
    args: argparse.Namespace,
    logger,
):
    """Build conditional GP when requested; otherwise return standard GP.

    The conditional path consumes augmented inputs concat([X_norm, M_obs]).
    If fitting fails for any numerical reason, the caller receives a standard
    SingleTaskGP and can continue the BO iteration without interruption.
    """

    search_dim = _search_dim(args)
    hp_dim = hp_dim_from_mode(args.hp_mode)

    if args.use_conditional_kernel and args.hp_mode != "global4" and M_obs is not None:
        try:
            if M_obs.shape != X_norm.shape:
                raise ValueError(f"M_obs shape {tuple(M_obs.shape)} != X_norm shape {tuple(X_norm.shape)}")
            X_aug = torch.cat([X_norm.float(), M_obs.float().clamp(0.0, 1.0)], dim=-1)
            gp = SingleTaskGP(X_aug.double(), Y_norm.double())
            gp.covar_module = ScaleKernel(
                ConditionalMaskedKernel(feature_dim=search_dim, arch_nz=ARCH_NZ)
            ).to(device=X_aug.device, dtype=torch.double)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            logger.info(
                f"ConditionalMaskedKernel enabled "
                f"(hp_mode={args.hp_mode}, hp_dim={hp_dim}, train_n={X_norm.shape[0]})"
            )
            return gp, True
        except Exception as exc:
            logger.warning(
                f"ConditionalMaskedKernel failed; falling back to standard SingleTaskGP: {exc}"
            )
    elif args.use_conditional_kernel and args.hp_mode == "global4":
        logger.info("Conditional kernel requested for global4; using standard SingleTaskGP.")

    gp = _fit_standard_gp(X_norm, Y_norm)
    return gp, False


def _random_candidate_raw(args: argparse.Namespace) -> torch.Tensor:
    hp_dim = hp_dim_from_mode(args.hp_mode)
    z_arch = torch.randn(ARCH_NZ) * float(args.sigma_arch)
    hp = torch.rand(hp_dim)
    z = torch.cat([z_arch, hp])
    clipped = clip_z_search_by_mode(z, args.hp_mode, ARCH_NZ, args.z_bound)
    return torch.tensor(clipped, dtype=torch.float32)


def optimize_acq(
    X_obs: list[torch.Tensor],
    Y_obs: list[float],
    novelty: GPNDNASNovelty,
    args: argparse.Namespace,
    logger,
    vae: JointSpaceVAE,
    device: torch.device,
    history: list[dict[str, Any]] | None = None,
    valid_flags: list[bool] | None = None,
) -> torch.Tensor:
    search_dim = _search_dim(args)
    if valid_flags is None:
        X_fit = X_obs
        Y_fit = Y_obs
        history_fit = history
    else:
        keep_idx = [idx for idx, valid in enumerate(valid_flags) if valid]
        X_fit = [X_obs[idx] for idx in keep_idx]
        Y_fit = [Y_obs[idx] for idx in keep_idx]
        history_fit = [history[idx] for idx in keep_idx] if history is not None else None

    if len(X_fit) < 3:
        return _random_candidate_raw(args)

    X_t = torch.stack(X_fit).float()
    Y_t = torch.tensor(Y_fit, dtype=torch.float32).unsqueeze(-1)
    X_n, Xmin, Xrng = _minmax_normalize(X_t)
    Y_std = Y_t.std().clamp(min=1e-6)
    Y_n = (Y_t - Y_t.mean()) / Y_std

    M_obs = None
    if history_fit is not None:
        try:
            M_obs = torch.tensor(
                [_mask_from_history_row(row, args) for row in history_fit],
                dtype=torch.float32,
            )
        except Exception as exc:
            logger.warning(f"Failed to build observed condition masks; using standard GP: {exc}")
            M_obs = None

    gp, using_conditional = build_gp_model(X_n, Y_n, M_obs, args, logger)

    logei = qLogExpectedImprovement(gp, best_f=Y_n.double().max())
    bounds = torch.zeros(2, search_dim, dtype=torch.double)
    bounds[1] = 1.0

    candidate_norms: list[torch.Tensor] = []
    if not using_conditional:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cand, _ = optimize_acqf(
                    logei,
                    bounds=bounds,
                    q=1,
                    num_restarts=int(args.num_restarts),
                    raw_samples=int(args.raw_samples),
                )
                candidate_norms.append(cand.squeeze(0).float())
            except Exception as exc:
                logger.warning(f"optimize_acqf failed: {exc}; using random candidate pool")

    n_random = int(args.n_extra)
    if using_conditional:
        n_random = max(n_random, int(args.raw_samples) + int(args.num_restarts))
    if n_random > 0:
        candidate_norms.extend(torch.rand(n_random, search_dim))
    if not candidate_norms:
        return _random_candidate_raw(args)

    cand_norm = torch.stack(candidate_norms).float()
    raw = cand_norm * Xrng + Xmin
    clipped_raw = torch.stack(
        [
            torch.tensor(
                clip_z_search_by_mode(row, args.hp_mode, ARCH_NZ, args.z_bound),
                dtype=torch.float32,
            )
            for row in raw
        ]
    )

    with torch.no_grad():
        eval_norm = ((clipped_raw - Xmin) / Xrng).clamp(0.0, 1.0)
        if using_conditional:
            cand_masks = _candidate_masks(vae, clipped_raw, args, device, logger)
            eval_input = torch.cat([eval_norm.float(), cand_masks.float().clamp(0.0, 1.0)], dim=-1)
        else:
            eval_input = eval_norm
        try:
            ei_scores = logei(eval_input.double().unsqueeze(1)).float().view(-1)
        except Exception:
            if using_conditional:
                logger.warning("Conditional LogEI scoring failed; using zero EI scores for candidate rerank.")
            ei_scores = torch.zeros(clipped_raw.shape[0], dtype=torch.float32)

    if float(args.novelty_w) > 0.0:
        novelty.update(torch.stack(X_fit).float())
        nov = torch.tensor(
            novelty.batch_score([clipped_raw[i] for i in range(clipped_raw.shape[0])], normalize=True),
            dtype=torch.float32,
        )
        scores = ei_scores + float(args.novelty_w) * nov
    else:
        scores = ei_scores

    best_idx = int(torch.argmax(scores).item())
    return clipped_raw[best_idx].float()


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
    best_acc: float | None = None,
) -> float:
    z_search, result = eval_candidate(vae, z_search, data, in_ch, out_ch, args, device)
    acc = float(result.get("val_acc", 0.0))
    X_obs.append(z_search)
    Y_obs.append(acc)
    history.append(history_record(step, record_type, z_search, result, args, best_acc=best_acc))
    hp = result.get("hp", {})
    logger.info(
        f"  {record_type:<9s} step={step:>3d}: val={acc:.4f} "
        f"lr={float(hp.get('lr', result.get('lr', 0.0))):.5f} "
        f"drop={float(hp.get('dropout', result.get('dropout', 0.0))):.3f} "
        f"hidden={int(hp.get('hidden_dim', result.get('hidden_dim', 64)))} "
        f"l2={float(hp.get('weight_decay', hp.get('l2', result.get('l2', 0.0)))):.1e} "
        f"valid={bool(result.get('valid', False))}"
    )
    return acc


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
    step: int,
) -> int:
    if int(args.gmm_init_trials) <= 0:
        logger.info("Skipping GMM init: gmm_init_trials <= 0")
        return step
    if not args.gmm_init_history:
        logger.info("Skipping GMM init: no --gmm_init_history provided")
        return step

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
        _append_eval(
            vae,
            torch.tensor(z_np, dtype=torch.float32),
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
            best_acc=max(Y_obs) if Y_obs else None,
        )
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


def run_bo(
    vae: JointSpaceVAE,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> None:
    history: list[dict[str, Any]] = []
    X_obs: list[torch.Tensor] = []
    Y_obs: list[float] = []
    novelty = GPNDNASNovelty()

    logger.info(f"\n[Step 1] LHS init: n_init={args.n_init}")
    logger.info(f"  hp_mode={args.hp_mode}")
    logger.info(f"  hp_dim={hp_dim_from_mode(args.hp_mode)}")
    logger.info(f"  hp_names={hp_names_from_mode(args.hp_mode)}")
    logger.info(f"  ARCH_NZ={ARCH_NZ} SEARCH_DIM={ARCH_NZ + hp_dim_from_mode(args.hp_mode)}")
    logger.info(f"  z_bound={args.z_bound}")
    logger.info(f"  LR: 10^[{args.log_lr_min},{args.log_lr_max}] Dropout: [{args.dropout_min},{args.dropout_max}]")

    step = 0
    for z in tqdm(initial_points(args, logger), desc="LHS init"):
        _append_eval(
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
        )
        step += 1

    if not any(h["valid"] for h in history):
        logger.error("All LHS init points invalid.")
        _save(history, X_obs, Y_obs, args, logger, "final")
        return

    logger.info(f"\nLHS done. Best={max(Y_obs):.4f}")

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
        step,
    )

    best_acc = max(Y_obs)
    logger.info(
        f"\n[Step 3] BO: n_iter={args.n_iter} "
        f"SEARCH_DIM={ARCH_NZ + hp_dim_from_mode(args.hp_mode)}"
    )

    for it in range(int(args.n_iter)):
        try:
            valid_flags = [bool(row.get("valid", False)) for row in history]
            z_next = optimize_acq(
                X_obs,
                Y_obs,
                novelty,
                args,
                logger,
                vae,
                device,
                history=history,
                valid_flags=valid_flags,
            )
        except Exception as exc:
            logger.warning(f"  [iter {it}] acqf error: {exc}; random fallback")
            z_next = _random_candidate_raw(args)

        z_np = clip_z_search_by_mode(z_next, args.hp_mode, ARCH_NZ, args.z_bound)
        z_next = torch.tensor(z_np, dtype=torch.float32)
        acc = _append_eval(
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
            best_acc=best_acc,
        )

        if acc > best_acc:
            best_acc = acc
            logger.info(f"  iter {it:>3d}: <-- NEW BEST {best_acc:.4f}")
        else:
            logger.info(f"  iter {it:>3d}: best={best_acc:.4f}")

        step += 1
        if (it + 1) % 10 == 0:
            _save(history, X_obs, Y_obs, args, logger, f"step{it}")

    best_idx = int(np.argmax(np.asarray(Y_obs, dtype=np.float64)))
    best_z = X_obs[best_idx]
    best_cfg = decode_arch(vae, best_z[:ARCH_NZ], device, n_trials=10)

    logger.info("\n" + "=" * 66)
    logger.info(f"          Phase 4 BO hp_mode={args.hp_mode} Final")
    logger.info("=" * 66)
    logger.info(f"  Best val_acc  : {max(Y_obs):.4f}")
    logger.info(f"  GNN config    : {best_cfg}")
    logger.info(f"  SEARCH_DIM    : {ARCH_NZ + hp_dim_from_mode(args.hp_mode)}")
    logger.info("=" * 66)

    _save(history, X_obs, Y_obs, args, logger, "final")
    _plot(history, args, logger)


def _save(
    history: list[dict[str, Any]],
    X_obs: list[torch.Tensor],
    Y_obs: list[float],
    args: argparse.Namespace,
    logger,
    suffix: str,
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

    vae = load_vae(args, device, logger)
    data, in_ch, out_ch = load_cora(args.cora_root, device, logger)
    logger.info(f"Cora: {in_ch} features, {out_ch} classes")

    run_bo(vae, data, in_ch, out_ch, args, device, logger)
    elapsed = time.time() - start_time
    logger.info(f"Run complete. Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
