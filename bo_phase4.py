"""Phase 4 BO search using checkpoint or scratch accuracy GP initialization and pure qLogEI.

Search vector:

    z_search = [z_arch, hp]

HP semantics and dimensions are driven only by hp_mode. The default Schur and
legacy weighted-GMM path is preserved. Optional WGMM-clustered TED strategies
use one unlabeled LHS pool, an Exact-GP-normalized RBF kernel, and a separate
two-stage full/low-fidelity initialization before the same online BO policy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from typing import Any, Callable

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
from surrogate.dkl_accuracy_gp import DKLAccuracyGPPredictor, validate_dkl_config
from surrogate.checkpoint_io import atomic_json_dump, atomic_torch_save
from surrogate.history_dataset import architecture_key_from_record
from surrogate.metrics import (
    PREQUENTIAL_METRIC_FIELDS,
    GPConvergenceMonitor,
    prediction_metrics,
    prediction_record_fields,
)
from weighted_diag_gmm_init import fit_and_sample_gmm_init, load_history_vectors
from initialization_wgmm_ted import (
    QUOTA_MODES,
    WGMM_ASSIGNMENT_SOURCES,
    allocate_cluster_quotas,
    assign_clusters,
    atomic_json_dump as atomic_initialization_json_dump,
    candidate_fingerprint as make_candidate_fingerprint,
    canonical_json_fingerprint,
    combined_expansion_scores,
    fit_pool_gmm,
    fingerprint_array,
    greedy_ted_select,
    load_checkpoint_wgmm,
    rbf_kernel,
    select_by_cluster_scores,
    sha256_file,
    ted_features,
)


ARCH_NZ = 12
HP_MODE_CHOICES = ("global4", "hybrid_cond7", "layer_cond19")
CANDIDATE_EVALUATION_SEED_SCHEME = (
    "sha256_v1_candidate_evaluation_search_seed_fingerprint_fidelity"
)
FULL_FIDELITY_SEED_CONTEXT = ["search_seed", "candidate_fingerprint", "full"]
LOW_FIDELITY_SEED_CONTEXT = ["search_seed", "candidate_fingerprint", "low"]


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
    payload = {
        **vars(args),
        **candidate_evaluation_seed_provenance(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
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
    parser.add_argument(
        "--initial_selection_strategy",
        choices=("schur", "wgmm_ted", "wgmm_ted_lowfid"),
        default="schur",
        help="initial point selector; schur preserves the legacy path",
    )
    parser.add_argument(
        "--initial_seed_evals", type=int, default=None,
        help="full-fidelity seed evaluations (default: n_init)",
    )
    parser.add_argument(
        "--initial_expand_evals", type=int, default=0,
        help="second-stage full-fidelity evaluations; excluded from n_iter",
    )
    parser.add_argument(
        "--max_total_full_evals", type=int, default=None,
        help="optional strict check: seed + expansion + online",
    )
    parser.add_argument(
        "--wgmm_source", choices=WGMM_ASSIGNMENT_SOURCES, default="checkpoint",
        help=(
            "recover a serialized checkpoint diagonal mixture, or fit an ordinary "
            "diagonal GMM with uniform weights on the unlabeled LHS pool"
        ),
    )
    parser.add_argument(
        "--wgmm_checkpoint", type=str, default=None,
        help="mixture checkpoint/sidecar (default: VAE checkpoint)",
    )
    parser.add_argument(
        "--wgmm_n_components", type=int, default=None,
        help="required for gmm_fit_pool; otherwise verified against checkpoint",
    )
    parser.add_argument("--wgmm_covariance_regularization", type=float, default=1e-6)
    parser.add_argument("--wgmm_assignment", choices=("hard",), default="hard")
    parser.add_argument("--wgmm_quota_mode", choices=QUOTA_MODES, default="hybrid")
    parser.add_argument("--wgmm_equal_weight", type=float, default=0.5)
    parser.add_argument("--ted_kernel_lengthscale", type=float, default=None)
    parser.add_argument("--ted_regularization", type=float, default=0.1)
    parser.add_argument("--ted_jitter", type=float, default=1e-8)
    parser.add_argument(
        "--ted_shortlist_per_cluster", type=int, default=100,
        help="maximum conditioned-TED shortlist size in each nonempty cluster",
    )
    parser.add_argument(
        "--low_fidelity_epochs", type=int, default=20,
        help="fixed low-fidelity epoch budget",
    )
    parser.add_argument(
        "--low_fidelity_patience", type=int, default=0,
        help="0 disables low-fidelity early stopping",
    )
    parser.add_argument("--low_fidelity_score_weight", type=float, default=0.50)
    parser.add_argument("--gp_mean_score_weight", type=float, default=0.25)
    parser.add_argument("--gp_std_score_weight", type=float, default=0.25)
    parser.add_argument(
        "--resume_initialization", action="store_true",
        help="strictly resume matching atomic WGMM-TED artifacts",
    )
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
    parser.add_argument("--surrogate_type", choices=("exact_gp", "dkl_gp"), default="exact_gp")
    parser.add_argument("--dkl_hidden_dim", type=int, default=32)
    parser.add_argument("--dkl_feature_dim", type=int, choices=(4, 8), default=8)
    parser.add_argument("--dkl_activation", choices=("silu", "relu"), default="silu")
    parser.add_argument("--dkl_lr", type=float, default=0.01)
    parser.add_argument("--dkl_weight_decay", type=float, default=1e-4)
    parser.add_argument("--dkl_grad_clip", type=float, default=5.0)
    parser.add_argument("--dkl_init_steps", type=int, default=200)
    parser.add_argument("--dkl_refit_steps", type=int, default=50)
    parser.add_argument("--dkl_early_stopping_patience", type=int, default=25)
    parser.add_argument("--dkl_min_delta", type=float, default=1e-5)
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


def candidate_evaluation_seed(
    search_seed: int,
    candidate_fingerprint: str,
    evaluation_fidelity: str,
) -> int:
    """Derive a method-, stage-, and order-independent candidate training seed."""

    if not isinstance(candidate_fingerprint, str) or not candidate_fingerprint.strip():
        raise ValueError("candidate_fingerprint must be a non-empty string")
    if evaluation_fidelity not in ("full", "low"):
        raise ValueError(
            "evaluation_fidelity must be exactly 'full' or 'low', "
            f"got {evaluation_fidelity!r}"
        )
    return stable_seed(
        int(search_seed),
        "candidate_evaluation",
        candidate_fingerprint,
        evaluation_fidelity,
    )


def candidate_evaluation_seed_provenance() -> dict[str, Any]:
    """Return JSON-safe provenance for the unified candidate seed scheme."""

    return {
        "candidate_evaluation_seed_scheme": CANDIDATE_EVALUATION_SEED_SCHEME,
        "full_fidelity_seed_context": list(FULL_FIDELITY_SEED_CONTEXT),
        "low_fidelity_seed_context": list(LOW_FIDELITY_SEED_CONTEXT),
    }


def _canonical_candidate_identity(
    z_search: Any,
    *,
    hp_mode: str,
    z_bound: float,
    candidate_fingerprint: str | None = None,
) -> tuple[np.ndarray, str]:
    """Canonicalize a candidate and validate any caller-provided fingerprint."""

    clipped = clip_z_search_by_mode(z_search, hp_mode, ARCH_NZ, z_bound)
    expected_fingerprint = make_candidate_fingerprint(clipped)
    if candidate_fingerprint is not None:
        if not isinstance(candidate_fingerprint, str) or not candidate_fingerprint.strip():
            raise ValueError("candidate_fingerprint must be a non-empty string")
        if candidate_fingerprint != expected_fingerprint:
            raise ValueError(
                "candidate_fingerprint does not match canonical clipped float32 z_search: "
                f"expected {expected_fingerprint}, got {candidate_fingerprint}"
            )
    return clipped, expected_fingerprint


def _validated_explicit_seed(
    value: Any,
    *,
    expected: int,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    actual = int(value)
    if actual != int(expected):
        raise ValueError(
            f"{field} does not match {CANDIDATE_EVALUATION_SEED_SCHEME}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _validate_candidate_evaluation_record(
    record: dict[str, Any],
    z_search: Any,
    *,
    search_seed: int,
    hp_mode: str,
    z_bound: float,
    evaluation_fidelity: str,
    context: str,
) -> tuple[np.ndarray, str, int]:
    """Strictly validate persisted candidate identity and training-seed provenance."""

    record_fingerprint = record.get("candidate_fingerprint")
    if not isinstance(record_fingerprint, str) or not record_fingerprint.strip():
        raise ValueError(f"{context} candidate_fingerprint is missing or empty")
    canonical_z, fingerprint = _canonical_candidate_identity(
        z_search,
        hp_mode=hp_mode,
        z_bound=z_bound,
        candidate_fingerprint=record_fingerprint,
    )
    record_search_seed = record.get("search_seed")
    if (
        isinstance(record_search_seed, bool)
        or not isinstance(record_search_seed, (int, np.integer))
        or int(record_search_seed) != int(search_seed)
    ):
        raise ValueError(
            f"{context} search_seed mismatch: expected {int(search_seed)}, "
            f"got {record_search_seed!r}"
        )
    if record.get("seed_derivation") != SEED_DERIVATION:
        raise ValueError(
            f"{context} seed_derivation mismatch: expected {SEED_DERIVATION!r}, "
            f"got {record.get('seed_derivation')!r}"
        )
    if record.get("candidate_evaluation_seed_scheme") != CANDIDATE_EVALUATION_SEED_SCHEME:
        raise ValueError(
            f"{context} candidate_evaluation_seed_scheme mismatch: expected "
            f"{CANDIDATE_EVALUATION_SEED_SCHEME!r}, got "
            f"{record.get('candidate_evaluation_seed_scheme')!r}; legacy stage/step seed "
            "artifacts cannot be resumed under the current scheme"
        )
    if record.get("evaluation_fidelity") != evaluation_fidelity:
        raise ValueError(
            f"{context} evaluation_fidelity mismatch: expected {evaluation_fidelity!r}, "
            f"got {record.get('evaluation_fidelity')!r}"
        )
    expected_seed = candidate_evaluation_seed(
        int(search_seed), fingerprint, evaluation_fidelity,
    )
    if "evaluation_seed" not in record:
        raise ValueError(f"{context} evaluation_seed is missing")
    if "candidate_eval_seed" not in record:
        raise ValueError(f"{context} candidate_eval_seed is missing")
    _validated_explicit_seed(
        record["evaluation_seed"], expected=expected_seed, field=f"{context} evaluation_seed",
    )
    _validated_explicit_seed(
        record["candidate_eval_seed"],
        expected=expected_seed,
        field=f"{context} candidate_eval_seed",
    )
    expected_decoder_seed = stable_seed(
        int(search_seed), "decoder", canonical_z[:ARCH_NZ],
    )
    _validated_explicit_seed(
        record.get("decoder_seed"),
        expected=expected_decoder_seed,
        field=f"{context} decoder_seed",
    )
    initialization_strategy = record.get("initialization_strategy")
    if not isinstance(initialization_strategy, str) or not initialization_strategy:
        raise ValueError(f"{context} initialization_strategy is missing or empty")
    return canonical_z, fingerprint, expected_seed


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
    evaluation_stage: str | None = None,
    evaluation_fidelity: str = "full",
    candidate_fingerprint: str | None = None,
    evaluation_seed: int | None = None,
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    z_np, expected_fingerprint = _canonical_candidate_identity(
        z_search,
        hp_mode=args.hp_mode,
        z_bound=args.z_bound,
        candidate_fingerprint=candidate_fingerprint,
    )
    z_tensor = torch.tensor(z_np, dtype=torch.float32)
    decoder_seed = stable_seed(int(args.seed), "decoder", z_tensor[:ARCH_NZ])
    expected_evaluation_seed = candidate_evaluation_seed(
        int(args.seed), expected_fingerprint, evaluation_fidelity,
    )
    candidate_eval_seed = (
        expected_evaluation_seed
        if evaluation_seed is None
        else _validated_explicit_seed(
            evaluation_seed,
            expected=expected_evaluation_seed,
            field="evaluation_seed",
        )
    )
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
        max_epochs=(int(args.eval_epochs) if max_epochs_override is None else int(max_epochs_override)),
        patience=(int(args.patience) if patience_override is None else int(patience_override)),
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
            "evaluation_seed": candidate_eval_seed,
            "evaluation_stage": evaluation_stage,
            "evaluation_fidelity": evaluation_fidelity,
            "candidate_fingerprint": expected_fingerprint,
            "candidate_evaluation_seed_scheme": CANDIDATE_EVALUATION_SEED_SCHEME,
            "architecture_fingerprint": _decoded_architecture_fingerprint(
                result.get("config")
            ),
            "hp_fingerprint": _hp_configuration_fingerprint(result),
            "seed_derivation": SEED_DERIVATION,
        }
    )
    return z_tensor, result


def _edges_json(config: dict | None) -> list[list[int]]:
    if config is None:
        return []
    return [list(edge) for edge in config.get("edges", [])]


def _decoded_architecture_fingerprint(config: dict | None) -> str | None:
    if config is None:
        return None
    operations = [str(value) for value in config.get("operations", [])]
    edges = sorted([list(map(int, edge)) for edge in config.get("edges", [])])
    return canonical_json_fingerprint({"operations": operations, "edges": edges})


def _hp_configuration_fingerprint(result: dict[str, Any]) -> str:
    hp = result.get("hp") if isinstance(result.get("hp"), dict) else {}
    fields = (
        "lr", "dropout", "hidden_dim", "weight_decay", "gat_heads",
        "sage_aggr", "gin_eps", "gat_heads_by_layer", "sage_aggr_by_layer",
        "gin_eps_by_layer", "condition_mask_vector",
    )
    payload = {
        key: hp.get(key, result.get(key))
        for key in fields
    }
    return canonical_json_fingerprint(payload)


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
    evaluation_fidelity = result.get("evaluation_fidelity", "full")
    canonical_z, expected_fingerprint = _canonical_candidate_identity(
        z_search,
        hp_mode=args.hp_mode,
        z_bound=args.z_bound,
        candidate_fingerprint=result.get("candidate_fingerprint"),
    )
    expected_eval_seed = candidate_evaluation_seed(
        search_seed, expected_fingerprint, evaluation_fidelity,
    )
    decoder_seed_value = result.get("decoder_seed")
    if decoder_seed_value is None:
        decoder_seed_value = stable_seed(search_seed, "decoder", canonical_z[:ARCH_NZ])
    decoder_seed = int(decoder_seed_value)
    candidate_eval_seed_value = result.get("candidate_eval_seed")
    if candidate_eval_seed_value is None:
        candidate_eval_seed_value = expected_eval_seed
    candidate_eval_seed = _validated_explicit_seed(
        candidate_eval_seed_value,
        expected=expected_eval_seed,
        field="candidate_eval_seed",
    )
    evaluation_seed_value = result.get("evaluation_seed", candidate_eval_seed)
    evaluation_seed = _validated_explicit_seed(
        evaluation_seed_value,
        expected=expected_eval_seed,
        field="evaluation_seed",
    )
    scheme = result.get("candidate_evaluation_seed_scheme")
    if scheme is not None and scheme != CANDIDATE_EVALUATION_SEED_SCHEME:
        raise ValueError(
            "candidate_evaluation_seed_scheme mismatch: "
            f"expected {CANDIDATE_EVALUATION_SEED_SCHEME!r}, got {scheme!r}"
        )
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
        "z_search": canonical_z.tolist(),
        "z_arch": canonical_z[:ARCH_NZ].tolist(),
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
        "evaluation_seed": evaluation_seed,
        "candidate_fingerprint": expected_fingerprint,
        "evaluation_stage": result.get("evaluation_stage"),
        "evaluation_fidelity": evaluation_fidelity,
        "candidate_evaluation_seed_scheme": CANDIDATE_EVALUATION_SEED_SCHEME,
        "initialization_strategy": result.get(
            "initialization_strategy", getattr(args, "initial_selection_strategy", "schur")
        ),
        "full_evaluation_index": result.get("full_evaluation_index"),
        "architecture_fingerprint": result.get("architecture_fingerprint"),
        "hp_fingerprint": result.get("hp_fingerprint"),
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
    record_metadata: dict[str, Any] | None = None,
    pre_update_persist: Callable[[], None] | None = None,
) -> tuple[float, bool, list[float], bool]:
    """Predict, evaluate, record, and optionally update in that strict order."""

    step_started = time.monotonic()
    canonical_z, expected_fingerprint = _canonical_candidate_identity(
        z_search,
        hp_mode=args.hp_mode,
        z_bound=args.z_bound,
        candidate_fingerprint=(
            None if record_metadata is None else record_metadata.get("candidate_fingerprint")
        ),
    )
    z_search = torch.tensor(canonical_z, dtype=torch.float32)
    online_candidate_strategy = (
        getattr(args, "online_candidate_strategy", "qlogei")
        if update_online
        else None
    )
    pre_search_seed = getattr(args, "seed", None)
    pre_decoder_seed = None
    pre_candidate_eval_seed = None
    pre_seed_derivation = None
    metadata = {} if record_metadata is None else dict(record_metadata)
    default_stage = (
        "online_bo"
        if update_online or record_type == "bo"
        else "initial_seed"
        if record_type == "lhs_init"
        else "initial_expand"
    )
    evaluation_stage = str(metadata.get("evaluation_stage", default_stage))
    evaluation_fidelity = str(metadata.get("evaluation_fidelity", "full"))
    expected_evaluation_seed = candidate_evaluation_seed(
        int(args.seed), expected_fingerprint, evaluation_fidelity,
    )
    if "evaluation_seed" in metadata:
        _validated_explicit_seed(
            metadata["evaluation_seed"],
            expected=expected_evaluation_seed,
            field="evaluation_seed",
        )
    if "candidate_eval_seed" in metadata:
        _validated_explicit_seed(
            metadata["candidate_eval_seed"],
            expected=expected_evaluation_seed,
            field="candidate_eval_seed",
        )
    supplied_scheme = metadata.get("candidate_evaluation_seed_scheme")
    if supplied_scheme is not None and supplied_scheme != CANDIDATE_EVALUATION_SEED_SCHEME:
        raise ValueError(
            "candidate_evaluation_seed_scheme mismatch: "
            f"expected {CANDIDATE_EVALUATION_SEED_SCHEME!r}, got {supplied_scheme!r}"
        )
    metadata.update(
        {
            "search_seed": int(args.seed),
            "candidate_fingerprint": expected_fingerprint,
            "evaluation_stage": evaluation_stage,
            "evaluation_fidelity": evaluation_fidelity,
            "evaluation_seed": expected_evaluation_seed,
            "candidate_eval_seed": expected_evaluation_seed,
            "candidate_evaluation_seed_scheme": CANDIDATE_EVALUATION_SEED_SCHEME,
            "initialization_strategy": metadata.get(
                "initialization_strategy",
                getattr(args, "initial_selection_strategy", "schur"),
            ),
            "full_evaluation_index": metadata.get(
                "full_evaluation_index", len(history) if evaluation_fidelity == "full" else None,
            ),
            "seed_derivation": SEED_DERIVATION,
        }
    )
    fingerprint = expected_fingerprint
    if pre_search_seed is not None:
        pre_search_seed = int(pre_search_seed)
        pre_decoder_seed = stable_seed(
            pre_search_seed, "decoder", z_search[:ARCH_NZ],
        )
        pre_candidate_eval_seed = candidate_evaluation_seed(
            pre_search_seed,
            fingerprint,
            evaluation_fidelity,
        )
        pre_seed_derivation = SEED_DERIVATION
    if update_online and predictor is None:
        raise RuntimeError("online GP update requested before a GP predictor exists")
    train_size_before = 0 if predictor is None else predictor.train_size
    surrogate_type = getattr(args, "surrogate_type", "exact_gp")
    predictor_metadata = getattr(predictor, "metadata", {}) if predictor is not None else {}
    surrogate_provenance = {
        "surrogate_type": surrogate_type,
        "dkl_hidden_dim": getattr(args, "dkl_hidden_dim", None) if surrogate_type == "dkl_gp" else None,
        "dkl_feature_dim": getattr(args, "dkl_feature_dim", None) if surrogate_type == "dkl_gp" else None,
        "dkl_activation": getattr(args, "dkl_activation", None) if surrogate_type == "dkl_gp" else None,
        "dkl_kernel_type": "small_feature_rbf" if surrogate_type == "dkl_gp" else None,
        "dkl_mask_mode": (
            "masked_features_and_mask"
            if surrogate_type == "dkl_gp" and bool(getattr(args, "use_conditional_kernel", False))
            else "none" if surrogate_type == "dkl_gp" else None
        ),
        "dkl_training_seed": predictor_metadata.get("dkl_training_seed"),
        "dkl_seed_derivation": (
            "stable_seed(search_seed, 'dkl_initial_fit'|'dkl_refit', train_size, online_valid_count)"
            if surrogate_type == "dkl_gp" else None
        ),
    }
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
        "evaluation_seed": pre_candidate_eval_seed,
        "candidate_evaluation_seed_scheme": CANDIDATE_EVALUATION_SEED_SCHEME,
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
        **surrogate_provenance,
        "gp_update_performed": False,
        "gp_update_skipped_reason": "",
        "gp_update_seconds": 0.0,
        "eval_seconds": None,
        "total_step_seconds": float(time.monotonic() - step_started),
        **metadata,
    }
    prediction_records.append(prediction_row)
    # Persist the untouched pre-update prediction before the real GNN evaluation.
    _save_prediction_csv(prediction_records, args.output)

    eval_started = time.monotonic()
    z_search, result = eval_candidate(
        vae,
        z_search,
        data,
        in_ch,
        out_ch,
        args,
        device,
        step=step,
        evaluation_stage=evaluation_stage,
        evaluation_fidelity=evaluation_fidelity,
        candidate_fingerprint=fingerprint,
        evaluation_seed=pre_candidate_eval_seed,
    )
    result.update(metadata)
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
            **surrogate_provenance,
            "gp_stage": gp_stage,
            "gp_update_performed": False,
            "gp_update_skipped_reason": gp_update_skipped_reason,
            "gp_update_seconds": 0.0,
            "eval_seconds": float(eval_seconds),
            "total_step_seconds": float(time.monotonic() - step_started),
            "online_candidate_strategy": online_candidate_strategy,
            "candidate_selection_seed": candidate_selection_seed,
            **metadata,
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
            "evaluation_seed": result.get("evaluation_seed"),
            "candidate_evaluation_seed_scheme": result.get(
                "candidate_evaluation_seed_scheme"
            ),
            "architecture_fingerprint": result.get("architecture_fingerprint"),
            "hp_fingerprint": result.get("hp_fingerprint"),
            "seed_derivation": result.get("seed_derivation"),
            "best_epoch": result.get("best_epoch"),
            "stopped_epoch": result.get("stopped_epoch"),
            "epochs_ran": result.get("epochs_ran"),
            "online_candidate_strategy": online_candidate_strategy,
            "candidate_selection_seed": candidate_selection_seed,
            **gp_fields,
            **metadata,
        }
    )
    if convergence_monitor is not None:
        rolling_metrics = convergence_monitor.observe_bo_result(acc, prediction_row)
        prediction_row.update(rolling_metrics)
        history_row.update(rolling_metrics)
    # Persist the prediction, real result, and metrics before touching the GP.
    _save_prediction_csv(prediction_records, args.output)
    if pre_update_persist is not None:
        pre_update_persist()

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
            if getattr(args, "surrogate_type", "exact_gp") == "dkl_gp":
                dkl_refit_seed = stable_seed(
                    int(args.seed),
                    "dkl_refit",
                    int(predictor.train_size),
                    int(online_valid_count) + 1,
                )
                predictor.refit(
                    optimize=should_optimize,
                    steps=int(args.dkl_refit_steps) if should_optimize else None,
                    training_seed=dkl_refit_seed,
                )
                prediction_row["dkl_training_seed"] = int(dkl_refit_seed)
                history_row["dkl_training_seed"] = int(dkl_refit_seed)
            else:
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
        "decoder_seed", "candidate_eval_seed", "evaluation_seed",
        "candidate_evaluation_seed_scheme", "seed_derivation", "best_epoch",
        "architecture_fingerprint", "hp_fingerprint",
        "stopped_epoch", "epochs_ran", "online_candidate_strategy",
        "candidate_selection_seed", "initial_record_replayed",
        "initial_history_source", "gp_train_size_before",
        "gp_train_size_after", "gp_pred_mean", "gp_pred_std", "gp_pred_95_low",
        "gp_pred_95_high", "val_acc", "gp_residual", "gp_abs_error",
        "gp_squared_error", "gp_standardized_residual", "gp_covered_by_95",
        "logei", "best_val_before", "gp_update_mode", "gp_checkpoint_source",
        "gp_update_performed", "gp_update_skipped_reason", "eval_seconds",
        "gp_update_seconds", "total_step_seconds",
        "surrogate_type", "dkl_hidden_dim", "dkl_feature_dim",
        "dkl_activation", "dkl_kernel_type", "dkl_mask_mode",
        "dkl_training_seed", "dkl_seed_derivation",
        *PREQUENTIAL_METRIC_FIELDS,
    ]
    initialization_fields = [
        "initialization_strategy", "evaluation_stage", "evaluation_fidelity",
        "cluster_id", "cluster_responsibility", "candidate_pool_index",
        "candidate_fingerprint", "selection_rank", "ted_score", "quota_mode",
        "low_fidelity_used", "low_fidelity_record_id",
        "full_evaluation_index", "online_iteration",
    ]
    if any(any(field in row for field in initialization_fields) for row in records):
        fields.extend(initialization_fields)
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp_path, path)


def _surrogate_provenance(
    args: argparse.Namespace,
    predictor: AccuracyGPPredictor | None = None,
) -> dict[str, Any]:
    surrogate_type = getattr(args, "surrogate_type", "exact_gp")
    if surrogate_type == "exact_gp":
        return {
            "surrogate_type": "exact_gp",
            "dkl_hidden_dim": None,
            "dkl_feature_dim": None,
            "dkl_activation": None,
            "dkl_kernel_type": None,
            "dkl_mask_mode": None,
            "dkl_training_seed": None,
            "dkl_seed_derivation": None,
        }
    metadata = {} if predictor is None else predictor.metadata
    return {
        "surrogate_type": "dkl_gp",
        "dkl_hidden_dim": int(args.dkl_hidden_dim),
        "dkl_feature_dim": int(args.dkl_feature_dim),
        "dkl_activation": str(args.dkl_activation),
        "dkl_kernel_type": "small_feature_rbf",
        "dkl_mask_mode": (
            "masked_features_and_mask" if bool(args.use_conditional_kernel) else "none"
        ),
        "dkl_training_seed": metadata.get("dkl_training_seed"),
        "dkl_seed_derivation": (
            "stable_seed(search_seed, 'dkl_initial_fit'|'dkl_refit', train_size, online_valid_count)"
        ),
    }


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
        clipped_z = clip_z_search_by_mode(
            z_array, args.hp_mode, ARCH_NZ, args.z_bound,
        )
        expected_fingerprint = make_candidate_fingerprint(clipped_z)
        if record.get("candidate_fingerprint") != expected_fingerprint:
            raise _frozen_record_error(
                source,
                step,
                "candidate_fingerprint",
                f"expected {expected_fingerprint}, got {record.get('candidate_fingerprint')!r}",
            )
        if record.get("candidate_evaluation_seed_scheme") != CANDIDATE_EVALUATION_SEED_SCHEME:
            raise _frozen_record_error(
                source,
                step,
                "candidate_evaluation_seed_scheme",
                f"expected {CANDIDATE_EVALUATION_SEED_SCHEME!r}, got "
                f"{record.get('candidate_evaluation_seed_scheme')!r}; legacy stage/step "
                "seed artifacts cannot be replayed",
            )
        if record.get("evaluation_stage") != "initial_seed":
            raise _frozen_record_error(
                source, step, "evaluation_stage", "must equal 'initial_seed'",
            )
        if record.get("evaluation_fidelity") != "full":
            raise _frozen_record_error(
                source, step, "evaluation_fidelity", "must equal 'full'",
            )
        if record.get("initialization_strategy") != "schur":
            raise _frozen_record_error(
                source, step, "initialization_strategy", "must equal 'schur'",
            )
        if record.get("full_evaluation_index") != step:
            raise _frozen_record_error(
                source,
                step,
                "full_evaluation_index",
                f"expected {step}, got {record.get('full_evaluation_index')!r}",
            )

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
        if derivation != SEED_DERIVATION:
            raise _frozen_record_error(
                source,
                step,
                "seed_derivation",
                f"expected {SEED_DERIVATION!r}, got {derivation!r}",
            )
        expected_eval_seed = candidate_evaluation_seed(
            int(args.seed), expected_fingerprint, "full",
        )
        candidate_eval_seed = record.get("candidate_eval_seed")
        if (
            candidate_eval_seed is None
            or (
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
        evaluation_seed = record.get("evaluation_seed")
        if (
            isinstance(evaluation_seed, bool)
            or not isinstance(evaluation_seed, int)
            or evaluation_seed != expected_eval_seed
        ):
            raise _frozen_record_error(
                source,
                step,
                "evaluation_seed",
                f"expected {expected_eval_seed}, got {record.get('evaluation_seed')!r}",
            )
        expected_decoder_seed = stable_seed(
            int(args.seed), "decoder", clipped_z[:ARCH_NZ],
        )
        decoder_seed = record.get("decoder_seed")
        if (
            decoder_seed is None
            or (
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
                "evaluation_seed": record.get("evaluation_seed"),
                "candidate_evaluation_seed_scheme": record.get(
                    "candidate_evaluation_seed_scheme"
                ),
                "candidate_fingerprint": record.get("candidate_fingerprint"),
                "evaluation_stage": record.get("evaluation_stage"),
                "evaluation_fidelity": record.get("evaluation_fidelity"),
                "initialization_strategy": record.get("initialization_strategy"),
                "full_evaluation_index": record.get("full_evaluation_index"),
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
        "surrogate_type": getattr(args, "surrogate_type", "exact_gp"),
        "seed_derivation": SEED_DERIVATION,
        **candidate_evaluation_seed_provenance(),
    }
    if getattr(args, "initial_selection_strategy", "schur") != "schur":
        metadata.update(
            {
                "initialization_strategy": args.initial_selection_strategy,
                "initialization_config_fingerprint": args.initialization_config_fingerprint,
            }
        )
    started = time.monotonic()
    if getattr(args, "surrogate_type", "exact_gp") == "dkl_gp":
        initial_fit_seed = stable_seed(int(args.seed), "dkl_initial_fit")
        metadata["dkl_training_seed"] = int(initial_fit_seed)
        predictor = DKLAccuracyGPPredictor.fit_offline(
            init_X,
            init_y,
            arch_nz=ARCH_NZ,
            hp_mode=args.hp_mode,
            z_bound=args.z_bound,
            use_conditional_kernel=bool(args.use_conditional_kernel),
            condition_masks=init_masks if args.use_conditional_kernel else None,
            metadata=metadata,
            device=device,
            fit_steps=int(args.dkl_init_steps),
            training_seed=initial_fit_seed,
            hidden_dim=int(args.dkl_hidden_dim),
            feature_dim=int(args.dkl_feature_dim),
            activation=args.dkl_activation,
            lr=float(args.dkl_lr),
            weight_decay=float(args.dkl_weight_decay),
            grad_clip=float(args.dkl_grad_clip),
            init_steps=int(args.dkl_init_steps),
            refit_steps=int(args.dkl_refit_steps),
            early_stopping_patience=int(args.dkl_early_stopping_patience),
            min_delta=float(args.dkl_min_delta),
        )
    else:
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
        "Scratch GP initial fit: surrogate_type=%s train_size=%d seconds=%.3f saved=%s",
        getattr(args, "surrogate_type", "exact_gp"), predictor.train_size,
        time.monotonic() - started, path,
    )
    return predictor


def _resolved_initial_seed_evals(args: argparse.Namespace) -> int:
    return int(args.n_init) if args.initial_seed_evals is None else int(args.initial_seed_evals)


def validate_two_stage_initialization_config(args: argparse.Namespace) -> None:
    """Validate new budgets without changing any legacy Schur semantics."""

    seed_evals = _resolved_initial_seed_evals(args)
    expand_evals = int(args.initial_expand_evals)
    if seed_evals <= 0:
        raise ValueError("--initial_seed_evals must be positive")
    if expand_evals < 0:
        raise ValueError("--initial_expand_evals must be non-negative")
    if args.max_total_full_evals is not None:
        expected = seed_evals + expand_evals + int(args.n_iter)
        if expected != int(args.max_total_full_evals):
            raise ValueError(
                "full-evaluation budget mismatch: initial_seed_evals "
                f"{seed_evals} + initial_expand_evals {expand_evals} + n_iter "
                f"{int(args.n_iter)} = {expected}, not --max_total_full_evals "
                f"{int(args.max_total_full_evals)}"
            )
    if args.initial_selection_strategy == "schur":
        if args.initial_seed_evals is not None and seed_evals != int(args.n_init):
            raise ValueError("legacy Schur requires --initial_seed_evals to equal --n_init")
        if expand_evals != 0:
            raise ValueError("legacy Schur does not support --initial_expand_evals")
        return
    if args.surrogate_type != "exact_gp":
        raise ValueError("WGMM-clustered TED requires --surrogate_type exact_gp")
    if args.gp_init_mode != "scratch":
        raise ValueError("WGMM-clustered TED requires --gp_init_mode scratch")
    if args.gp_checkpoint:
        raise ValueError("WGMM-clustered TED does not accept --gp_checkpoint")
    if args.warm_start != "":
        raise ValueError("WGMM-clustered TED requires --warm_start to be an empty string")
    if args.frozen_init_history is not None:
        raise ValueError("WGMM-clustered TED uses strict initialization resume, not --frozen_init_history")
    if args.gmm_init_history or int(args.gmm_init_trials) != 0:
        raise ValueError("WGMM-clustered TED cannot be combined with legacy GMM initialization")
    if seed_evals != int(args.n_init):
        raise ValueError("--initial_seed_evals must equal --n_init for replay compatibility")
    total_initial = seed_evals + expand_evals
    if int(args.n_lhs_candidates) < total_initial:
        raise ValueError(
            f"--n_lhs_candidates {args.n_lhs_candidates} is smaller than the "
            f"{total_initial} unique full-fidelity initialization points"
        )
    if args.wgmm_source == "gmm_fit_pool" and args.wgmm_n_components is None:
        raise ValueError("--wgmm_n_components is required with --wgmm_source gmm_fit_pool")
    if args.wgmm_n_components is not None and int(args.wgmm_n_components) <= 0:
        raise ValueError("--wgmm_n_components must be positive")
    if float(args.wgmm_covariance_regularization) < 0.0:
        raise ValueError("--wgmm_covariance_regularization must be non-negative")
    if not 0.0 <= float(args.wgmm_equal_weight) <= 1.0:
        raise ValueError("--wgmm_equal_weight must be in [0, 1]")
    if float(args.ted_regularization) <= 0.0:
        raise ValueError("--ted_regularization must be positive")
    if float(args.ted_jitter) < 0.0:
        raise ValueError("--ted_jitter must be non-negative")
    if int(args.ted_shortlist_per_cluster) <= 0:
        raise ValueError("--ted_shortlist_per_cluster must be positive")
    if int(args.low_fidelity_epochs) <= 0:
        raise ValueError("--low_fidelity_epochs must be positive")
    if int(args.low_fidelity_patience) < 0:
        raise ValueError("--low_fidelity_patience must be non-negative")
    score_weights = np.asarray(
        [
            args.low_fidelity_score_weight,
            args.gp_mean_score_weight,
            args.gp_std_score_weight,
        ],
        dtype=np.float64,
    )
    if np.any(score_weights < 0.0) or not np.isfinite(score_weights).all():
        raise ValueError("low-fidelity/GP score weights must be finite and non-negative")
    if not np.isclose(float(score_weights.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("low-fidelity/GP score weights must sum to 1")


def _atomic_csv_dump(rows: list[dict[str, Any]], path: str, fields: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _initialization_paths(output: str) -> dict[str, str]:
    names = (
        "initialization_config.json",
        "candidate_pool.pt",
        "candidate_pool_metadata.json",
        "wgmm_assignments.csv",
        "cluster_quota.json",
        "ted_seed_trace.csv",
        "ted_shortlist_trace.csv",
        "expansion_scores.csv",
        "selected_seed_indices.json",
        "selected_expand_indices.json",
        "low_fidelity_history.json",
        "initialization_full_history.json",
        "rng_provenance.json",
        "budget_summary.json",
    )
    return {name: os.path.join(output, name) for name in names}


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_initialization_artifacts(
    *,
    args: argparse.Namespace,
    pool: torch.Tensor,
    masks: torch.Tensor,
    assignments: dict[str, np.ndarray],
    wgmm_parameters,
    config: dict[str, Any],
    paths: dict[str, str],
) -> list[dict[str, Any]]:
    pool_fingerprint = str(config["candidate_pool_fingerprint"])
    atomic_torch_save(
        {
            "format_version": 1,
            "z_search": pool.detach().cpu().float(),
            "condition_masks": masks.detach().cpu().float(),
            "fingerprint": pool_fingerprint,
        },
        paths["candidate_pool.pt"],
    )
    assignment_rows: list[dict[str, Any]] = []
    responsibilities = assignments["responsibilities"]
    cluster_ids = assignments["cluster_ids"]
    entropies = assignments["entropy"]
    for index in range(pool.shape[0]):
        row = pool[index].detach().cpu().float().tolist()
        cluster_id = int(cluster_ids[index])
        assignment_rows.append(
            {
                "candidate_index": index,
                "candidate_fingerprint": make_candidate_fingerprint(row),
                "z_arch_json": json.dumps(row[:ARCH_NZ], separators=(",", ":")),
                "hp_json": json.dumps(row[ARCH_NZ:], separators=(",", ":")),
                "z_search_json": json.dumps(row, separators=(",", ":")),
                "cluster_id": cluster_id,
                "responsibilities_json": json.dumps(
                    responsibilities[index].tolist(), separators=(",", ":")
                ),
                "cluster_responsibility": float(responsibilities[index, cluster_id]),
                "responsibility_entropy": float(entropies[index]),
                "component_weight": float(wgmm_parameters.weights[cluster_id]),
                "assignment_source": wgmm_parameters.source,
            }
        )
    _atomic_csv_dump(assignment_rows, paths["wgmm_assignments.csv"])
    atomic_initialization_json_dump(
        {
            "format_version": 1,
            "candidate_count": int(pool.shape[0]),
            "search_dim": int(pool.shape[1]),
            "arch_dim": ARCH_NZ,
            "hp_mode": args.hp_mode,
            "candidate_pool_fingerprint": pool_fingerprint,
            "candidate_pool_seed": int(args.seed),
            "vae_checkpoint": os.path.abspath(args.checkpoint),
            "vae_checkpoint_sha256": sha256_file(args.checkpoint),
            "wgmm": wgmm_parameters.to_json(),
            "ted_feature_transform": "exact_gp_normalize_then_inactive_mask",
        },
        paths["candidate_pool_metadata.json"],
    )
    atomic_initialization_json_dump(config, paths["initialization_config.json"])
    return assignment_rows


def _load_resume_full_history(
    path: str,
    *,
    config_fingerprint: str,
    search_seed: int,
    hp_mode: str,
    z_bound: float,
) -> tuple[
    list[dict[str, Any]],
    list[torch.Tensor],
    list[float],
    list[tuple[torch.Tensor, float, list[float]]],
    list[dict[str, Any]],
]:
    if not os.path.exists(path):
        return [], [], [], [], []
    records = _read_json(path)
    if not isinstance(records, list):
        raise ValueError("initialization_full_history.json must contain a JSON list")
    history: list[dict[str, Any]] = []
    X_obs: list[torch.Tensor] = []
    Y_obs: list[float] = []
    valid_rows: list[tuple[torch.Tensor, float, list[float]]] = []
    predictions: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("initialization history contains a non-object record")
        if raw.get("initialization_config_fingerprint") != config_fingerprint:
            raise ValueError("resume initialization config fingerprint mismatch")
        stage = str(raw.get("evaluation_stage"))
        if stage not in ("initial_seed", "initial_expand"):
            raise ValueError(f"resume initialization has unsupported stage {stage!r}")
        if raw.get("evaluation_fidelity") != "full":
            raise ValueError("resume initialization history must contain full-fidelity records only")
        pool_index = int(raw.get("candidate_pool_index"))
        key = (stage, pool_index)
        if key in seen:
            raise ValueError(f"resume history contains duplicate candidate {key}")
        seen.add(key)
        try:
            canonical_z, _fingerprint, _evaluation_seed = _validate_candidate_evaluation_record(
                raw,
                raw["z_search"],
                search_seed=search_seed,
                hp_mode=hp_mode,
                z_bound=z_bound,
                evaluation_fidelity="full",
                context="resume initialization",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"resume initialization candidate seed validation failed: {exc}") from exc
        z = torch.tensor(canonical_z, dtype=torch.float32)
        if int(raw.get("full_evaluation_index", -1)) != len(history):
            raise ValueError(
                "resume initialization full_evaluation_index values are not contiguous"
            )
        acc = float(raw["val_acc"])
        mask_hp = [float(value) for value in raw["condition_mask_vector"]]
        full_mask = [1.0] * ARCH_NZ + mask_hp
        history.append(copy.deepcopy(raw))
        X_obs.append(z)
        Y_obs.append(acc)
        if bool(raw.get("valid")):
            valid_rows.append((z, acc, full_mask))
        prediction = copy.deepcopy(raw)
        prediction["record_type"] = raw.get("type")
        predictions.append(prediction)
    return history, X_obs, Y_obs, valid_rows, predictions


def _cluster_component_mass(assignments: dict[str, np.ndarray]) -> dict[int, float]:
    responsibilities = np.asarray(assignments["responsibilities"], dtype=np.float64)
    mass = responsibilities.sum(axis=0)
    if float(mass.sum()) <= 0.0:
        raise RuntimeError("WGMM responsibilities have zero total mass")
    mass /= mass.sum()
    return {index: float(value) for index, value in enumerate(mass)}


def _select_clustered_ted(
    *,
    kernel: np.ndarray,
    cluster_ids: np.ndarray,
    quotas: dict[int, int],
    conditioned: Sequence[int] = (),
    shortlist_limit: int | None = None,
    regularization: float,
    jitter: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    conditioned_set = set(int(value) for value in conditioned)
    selected: list[int] = []
    traces: list[dict[str, Any]] = []
    for cluster_id in sorted(quotas):
        cluster_indices = np.flatnonzero(cluster_ids == int(cluster_id)).astype(int).tolist()
        cluster_conditioned = [index for index in cluster_indices if index in conditioned_set]
        available = len(cluster_indices) - len(cluster_conditioned)
        requested = int(quotas[cluster_id])
        if shortlist_limit is not None:
            requested = min(requested, int(shortlist_limit))
        requested = min(requested, available)
        if requested <= 0:
            continue
        cluster_kernel = kernel[np.ix_(cluster_indices, cluster_indices)]
        chosen, cluster_trace = greedy_ted_select(
            cluster_kernel,
            cluster_indices,
            requested,
            conditioned_indices=cluster_conditioned,
            regularization=float(regularization),
            jitter=float(jitter),
            cluster_id=int(cluster_id),
        )
        selected.extend(chosen)
        traces.extend(cluster_trace)
    return selected, traces


def _low_fidelity_eval(
    *,
    vae: JointSpaceVAE,
    z: torch.Tensor,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    candidate_index: int,
    cluster_id: int,
    fingerprint: str,
) -> dict[str, Any]:
    canonical_z, fingerprint = _canonical_candidate_identity(
        z,
        hp_mode=args.hp_mode,
        z_bound=args.z_bound,
        candidate_fingerprint=fingerprint,
    )
    z = torch.tensor(canonical_z, dtype=torch.float32)
    evaluation_seed = candidate_evaluation_seed(int(args.seed), fingerprint, "low")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    z_result, result = eval_candidate(
        vae,
        z,
        data,
        in_ch,
        out_ch,
        args,
        device,
        step=int(candidate_index),
        evaluation_stage="initial_shortlist",
        evaluation_fidelity="low",
        candidate_fingerprint=fingerprint,
        evaluation_seed=evaluation_seed,
        max_epochs_override=int(args.low_fidelity_epochs),
        patience_override=(
            int(args.low_fidelity_epochs) + 1
            if int(args.low_fidelity_patience) == 0
            else int(args.low_fidelity_patience)
        ),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.monotonic() - started
    hp = result.get("hp", {})
    return {
        "record_id": f"lowfid-{candidate_index}-{fingerprint[:12]}",
        "candidate_pool_index": int(candidate_index),
        "candidate_fingerprint": fingerprint,
        "z_search": z_result.detach().cpu().float().tolist(),
        "cluster_id": int(cluster_id),
        "evaluation_stage": "initial_shortlist",
        "evaluation_fidelity": "low",
        "search_seed": int(args.seed),
        "evaluation_seed": int(evaluation_seed),
        "candidate_eval_seed": int(evaluation_seed),
        "candidate_evaluation_seed_scheme": CANDIDATE_EVALUATION_SEED_SCHEME,
        "initialization_strategy": args.initial_selection_strategy,
        "full_evaluation_index": None,
        "decoder_seed": result.get("decoder_seed"),
        "architecture_fingerprint": result.get("architecture_fingerprint"),
        "hp_fingerprint": result.get("hp_fingerprint"),
        "seed_derivation": SEED_DERIVATION,
        "requested_epochs": int(args.low_fidelity_epochs),
        "requested_patience": int(args.low_fidelity_patience),
        "actual_epochs": int(result.get("epochs_ran") or 0),
        "val_acc": float(result.get("val_acc", 0.0)),
        "valid": bool(result.get("valid", False)),
        "runtime_seconds": float(runtime),
        "gpu_seconds": float(runtime) if device.type == "cuda" else None,
        "equivalent_full_evaluations": float(result.get("epochs_ran") or 0)
        / max(1, int(args.eval_epochs)),
        "failure_reason": "" if bool(result.get("valid", False)) else "invalid_evaluation",
        "operations": [] if result.get("config") is None else list(result["config"].get("operations", [])),
        "edges": [] if result.get("config") is None else [list(edge) for edge in result["config"].get("edges", [])],
        "hp": hp,
        "gp_mean": None,
        "gp_std": None,
        "low_fidelity_rank": None,
        "gp_mean_rank": None,
        "gp_std_rank": None,
        "combined_score": None,
        "selected_for_full_expansion": False,
    }


def _validate_low_full_candidate_consistency(
    low_row: dict[str, Any],
    full_row: dict[str, Any],
) -> None:
    """Reject fidelity comparisons that changed candidate, decode, or HP semantics."""

    for field in (
        "candidate_fingerprint",
        "decoder_seed",
        "architecture_fingerprint",
        "hp_fingerprint",
    ):
        if low_row.get(field) != full_row.get(field):
            raise RuntimeError(f"low/full {field} mismatch for the same expansion candidate")
    if low_row.get("operations") != full_row.get("operations"):
        raise RuntimeError("low/full decoded operations mismatch for the same expansion candidate")
    if low_row.get("edges") != full_row.get("edges"):
        raise RuntimeError("low/full decoded edges mismatch for the same expansion candidate")
    low_z = np.asarray(low_row.get("z_search"), dtype=np.float32)
    full_z = np.asarray(full_row.get("z_search"), dtype=np.float32)
    if low_z.shape != full_z.shape or not np.array_equal(low_z, full_z):
        raise RuntimeError("low/full z_search mismatch for the same expansion candidate")
    if int(low_row.get("evaluation_seed", -1)) == int(full_row.get("evaluation_seed", -1)):
        raise RuntimeError("low/full training seeds must be fidelity-isolated")


def _validate_online_resume_boundary(
    *,
    raw_history_count: int,
    completed_count: int,
    adaptive_sampling: bool,
    n_iter: int,
    max_bo_samples: int,
) -> None:
    """Validate the transaction boundary represented by online resume artifacts."""

    if raw_history_count not in (completed_count, completed_count + 1):
        raise ValueError("WGMM online resume state/history count mismatch")
    resume_limit = int(max_bo_samples) if adaptive_sampling else int(n_iter)
    if raw_history_count > resume_limit:
        raise ValueError("WGMM online resume contains more evaluations than the online budget")
    if adaptive_sampling and raw_history_count != completed_count:
        raise ValueError(
            "adaptive WGMM online resume requires a clean committed iteration boundary; "
            "a pending pre-GP-update record cannot preserve exact timing/check state"
        )


def _initialization_method_id(strategy: str, wgmm_source: str) -> str:
    """Return the unambiguous method label persisted with initialization artifacts."""

    return f"{strategy}__{wgmm_source}"


def run_wgmm_two_stage_initialization(
    vae: JointSpaceVAE,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> tuple[
    list[dict[str, Any]],
    list[torch.Tensor],
    list[float],
    list[dict[str, Any]],
    list[tuple[torch.Tensor, float, list[float]]],
    AccuracyGPPredictor,
    int,
    dict[str, Any],
]:
    """Run label-free design, seed fit, optional LF ranking, and expansion fit."""

    validate_two_stage_initialization_config(args)
    paths = _initialization_paths(args.output)
    config_path = paths["initialization_config.json"]
    if os.path.exists(config_path) and not args.resume_initialization:
        raise FileExistsError(
            f"initialization artifacts already exist in {args.output!r}; use a new output "
            "directory or --resume_initialization"
        )
    pool_list = make_lhs_pool(args)
    pool = torch.stack(pool_list).cpu().float()
    pool_fingerprint = fingerprint_array(pool.numpy())
    if args.hp_mode == "global4":
        masks = torch.ones_like(pool)
    else:
        masks = _candidate_masks(vae, pool, args, device, logger).cpu().float()
    features = ted_features(
        pool.numpy(),
        arch_nz=ARCH_NZ,
        hp_mode=args.hp_mode,
        z_bound=args.z_bound,
        condition_masks=masks.numpy(),
    )
    lengthscale = (
        math.sqrt(features.shape[1])
        if args.ted_kernel_lengthscale is None
        else float(args.ted_kernel_lengthscale)
    )
    kernel = rbf_kernel(features, lengthscale=lengthscale)
    if args.wgmm_source == "checkpoint":
        wgmm_path = args.checkpoint if args.wgmm_checkpoint is None else args.wgmm_checkpoint
        wgmm_parameters = load_checkpoint_wgmm(
            wgmm_path,
            arch_dim=ARCH_NZ,
            covariance_regularization=float(args.wgmm_covariance_regularization),
        )
        if args.wgmm_n_components is not None and int(args.wgmm_n_components) != wgmm_parameters.n_components:
            raise ValueError(
                "--wgmm_n_components does not match the component count stored in the checkpoint"
            )
    else:
        fit_seed = stable_seed(int(args.seed), "wgmm_fit_pool", pool_fingerprint)
        wgmm_parameters = fit_pool_gmm(
            pool[:, :ARCH_NZ].numpy(),
            n_components=int(args.wgmm_n_components),
            random_state=fit_seed,
            covariance_regularization=float(args.wgmm_covariance_regularization),
            var_floor=float(args.gmm_var_floor),
            max_iter=int(args.gmm_max_iter),
            tol=float(args.gmm_tol),
        )
    assignments = assign_clusters(pool[:, :ARCH_NZ].numpy(), wgmm_parameters)
    cluster_ids = assignments["cluster_ids"]
    capacities = {
        component: int(np.sum(cluster_ids == component))
        for component in range(wgmm_parameters.n_components)
    }
    component_mass = _cluster_component_mass(assignments)
    seed_evals = _resolved_initial_seed_evals(args)
    seed_quotas, seed_quota_diagnostics = allocate_cluster_quotas(
        capacities,
        seed_evals,
        mode=args.wgmm_quota_mode,
        component_weights=component_mass,
        equal_weight=float(args.wgmm_equal_weight),
    )
    seed_indices, seed_trace = _select_clustered_ted(
        kernel=kernel,
        cluster_ids=cluster_ids,
        quotas=seed_quotas,
        regularization=float(args.ted_regularization),
        jitter=float(args.ted_jitter),
    )
    if len(seed_indices) != seed_evals:
        raise RuntimeError("TED seed selection did not fill initial_seed_evals")
    config_without_fingerprint = {
        "format_version": 1,
        "initialization_strategy": args.initial_selection_strategy,
        "initialization_method_id": _initialization_method_id(
            args.initial_selection_strategy, wgmm_parameters.source,
        ),
        "search_seed": int(args.seed),
        **candidate_evaluation_seed_provenance(),
        "hp_mode": args.hp_mode,
        "search_dim": int(pool.shape[1]),
        "initial_seed_evals": seed_evals,
        "initial_expand_evals": int(args.initial_expand_evals),
        "n_iter": int(args.n_iter),
        "max_total_full_evals": args.max_total_full_evals,
        "candidate_pool_fingerprint": pool_fingerprint,
        "condition_masks_fingerprint": fingerprint_array(masks.numpy()),
        "candidate_pool_size": int(pool.shape[0]),
        "lhs_sigma_arch": float(args.sigma_arch),
        "z_bound": float(args.z_bound),
        "vae_checkpoint_sha256": sha256_file(args.checkpoint),
        "wgmm_source": wgmm_parameters.source,
        "wgmm_estimator_semantics": wgmm_parameters.estimator_semantics,
        "wgmm_source_fingerprint": wgmm_parameters.source_fingerprint,
        "wgmm_parameter_fingerprint": wgmm_parameters.parameter_fingerprint,
        "wgmm_n_components": wgmm_parameters.n_components,
        "wgmm_assignment": args.wgmm_assignment,
        "wgmm_covariance_regularization": float(args.wgmm_covariance_regularization),
        "wgmm_quota_mode": args.wgmm_quota_mode,
        "wgmm_equal_weight": float(args.wgmm_equal_weight),
        "ted_kernel": "rbf",
        "ted_transform": "exact_gp_normalize_then_inactive_mask",
        "ted_kernel_lengthscale": float(lengthscale),
        "ted_regularization": float(args.ted_regularization),
        "ted_jitter": float(args.ted_jitter),
        "ted_shortlist_per_cluster": int(args.ted_shortlist_per_cluster),
        "low_fidelity_epochs": int(args.low_fidelity_epochs),
        "low_fidelity_patience": int(args.low_fidelity_patience),
        "full_fidelity_epochs": int(args.eval_epochs),
        "full_fidelity_patience": int(args.patience),
        "use_conditional_kernel": bool(args.use_conditional_kernel),
        "gp_refit_steps": int(args.gp_refit_steps),
        "gp_update_mode": args.gp_update_mode,
        "gp_refit_every": int(args.gp_refit_every),
        "online_candidate_strategy": args.online_candidate_strategy,
        "adaptive_sampling": bool(args.adaptive_sampling),
        "min_bo_samples": int(args.min_bo_samples),
        "max_bo_samples": int(args.max_bo_samples),
        "convergence_check_every": int(args.convergence_check_every),
        "convergence_patience": int(args.convergence_patience),
        "prequential_window": int(args.prequential_window),
        "mae_relative_tol": float(args.mae_relative_tol),
        "mae_absolute_tol": float(args.mae_absolute_tol),
        "std_relative_tol": float(args.std_relative_tol),
        "spearman_tol": float(args.spearman_tol),
        "degradation_tolerance": float(args.degradation_tolerance),
        "best_acc_patience": int(args.best_acc_patience),
        "best_acc_min_delta": float(args.best_acc_min_delta),
        "max_wall_time_hours": float(args.max_wall_time_hours),
        "probe_pool_size": int(args.probe_pool_size),
        "probe_pool_seed": args.probe_pool_seed,
        "num_restarts": int(args.num_restarts),
        "raw_samples": int(args.raw_samples),
        "n_extra": int(args.n_extra),
        "score_weights": [
            float(args.low_fidelity_score_weight),
            float(args.gp_mean_score_weight),
            float(args.gp_std_score_weight),
        ],
    }
    config_fingerprint = canonical_json_fingerprint(config_without_fingerprint)
    initialization_config = {
        **config_without_fingerprint,
        "initialization_config_fingerprint": config_fingerprint,
    }
    args.initialization_config_fingerprint = config_fingerprint
    if args.resume_initialization:
        if not os.path.exists(config_path):
            raise FileNotFoundError("--resume_initialization requires initialization_config.json")
        prior_config = _read_json(config_path)
        if prior_config.get("candidate_evaluation_seed_scheme") != CANDIDATE_EVALUATION_SEED_SCHEME:
            raise ValueError(
                "resume initialization uses a legacy or missing "
                "candidate_evaluation_seed_scheme; old stage/step seed artifacts cannot be mixed"
            )
        if prior_config.get("initialization_config_fingerprint") != config_fingerprint:
            raise ValueError("resume initialization metadata/fingerprint mismatch")
        if prior_config != initialization_config:
            raise ValueError("resume initialization configuration differs from the saved configuration")
        saved_seed_indices = _read_json(paths["selected_seed_indices.json"])
        if saved_seed_indices != seed_indices:
            raise ValueError("resume seed selection differs from saved indices")
    else:
        _write_initialization_artifacts(
            args=args,
            pool=pool,
            masks=masks,
            assignments=assignments,
            wgmm_parameters=wgmm_parameters,
            config=initialization_config,
            paths=paths,
        )
        atomic_initialization_json_dump(seed_indices, paths["selected_seed_indices.json"])
        _atomic_csv_dump(seed_trace, paths["ted_seed_trace.csv"])
        atomic_initialization_json_dump(
            {"seed": seed_quota_diagnostics}, paths["cluster_quota.json"]
        )
    history, X_obs, Y_obs, init_valid, prediction_records = _load_resume_full_history(
        paths["initialization_full_history.json"],
        config_fingerprint=config_fingerprint,
        search_seed=int(args.seed),
        hp_mode=args.hp_mode,
        z_bound=float(args.z_bound),
    )
    for row in history:
        index = int(row["candidate_pool_index"])
        if index < 0 or index >= pool.shape[0]:
            raise ValueError("resume initialization candidate index is outside the saved pool")
        if not np.array_equal(
            np.asarray(row["z_search"], dtype=np.float32), pool[index].numpy(),
        ):
            raise ValueError("resume initialization candidate does not match its pool index")
    completed = {
        (str(row.get("evaluation_stage")), int(row.get("candidate_pool_index")))
        for row in history
    }
    trace_by_index = {int(row["candidate_index"]): row for row in seed_trace}
    for selection_rank, index in enumerate(seed_indices):
        if ("initial_seed", int(index)) in completed:
            continue
        z = pool[int(index)].clone()
        mask = masks[int(index)].tolist()
        fingerprint = make_candidate_fingerprint(z.numpy())
        trace_row = trace_by_index[int(index)]
        metadata = {
            "initialization_strategy": args.initial_selection_strategy,
            "evaluation_stage": "initial_seed",
            "evaluation_fidelity": "full",
            "cluster_id": int(cluster_ids[index]),
            "cluster_responsibility": float(
                assignments["responsibilities"][index, cluster_ids[index]]
            ),
            "candidate_pool_index": int(index),
            "candidate_fingerprint": fingerprint,
            "selection_rank": int(selection_rank),
            "ted_score": float(trace_row["ted_score"]),
            "quota_mode": args.wgmm_quota_mode,
            "low_fidelity_used": False,
            "low_fidelity_record_id": None,
            "full_evaluation_index": len(history),
            "online_iteration": None,
            "initialization_config_fingerprint": config_fingerprint,
        }
        best_before = max(Y_obs) if Y_obs else None
        acc, valid, used_mask, safe = _append_eval(
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
            len(history),
            "wgmm_ted_seed",
            logger,
            None,
            prediction_records,
            "scratch_init_no_gp",
            None,
            False,
            0,
            mask,
            best_acc=best_before,
            record_metadata=metadata,
        )
        if safe:
            init_valid.append((z, acc, used_mask))
        atomic_initialization_json_dump(history, paths["initialization_full_history.json"])
    seed_valid = []
    for row in history:
        if row.get("evaluation_stage") == "initial_seed" and bool(row.get("valid")):
            z = torch.tensor(row["z_search"], dtype=torch.float32)
            mask = [1.0] * ARCH_NZ + [float(value) for value in row["condition_mask_vector"]]
            seed_valid.append((z, float(row["val_acc"]), mask))
    if len(seed_valid) < int(args.scratch_gp_min_points):
        raise RuntimeError(
            "WGMM-TED seed phase produced too few valid full-fidelity points; "
            "the strict full budget does not permit scratch_extra_init"
        )
    predictor = _fit_scratch_predictor(seed_valid, args, device, logger)
    expand_evals = int(args.initial_expand_evals)
    shortlist_indices: list[int] = []
    shortlist_trace: list[dict[str, Any]] = []
    expand_indices: list[int] = []
    low_history: list[dict[str, Any]] = []
    expansion_score_rows: list[dict[str, Any]] = []
    expand_quota_diagnostics: dict[str, Any] | None = None
    if expand_evals:
        shortlist_capacities = {
            component: min(
                int(args.ted_shortlist_per_cluster),
                max(0, capacities.get(component, 0) - seed_quotas.get(component, 0)),
            )
            for component in range(wgmm_parameters.n_components)
        }
        shortlist_quotas = dict(shortlist_capacities)
        shortlist_indices, shortlist_trace = _select_clustered_ted(
            kernel=kernel,
            cluster_ids=cluster_ids,
            quotas=shortlist_quotas,
            conditioned=seed_indices,
            shortlist_limit=int(args.ted_shortlist_per_cluster),
            regularization=float(args.ted_regularization),
            jitter=float(args.ted_jitter),
        )
        if len(shortlist_indices) < expand_evals:
            raise ValueError(
                f"TED shortlist contains {len(shortlist_indices)} candidates, fewer than "
                f"--initial_expand_evals {expand_evals}; increase --n_lhs_candidates or "
                "--ted_shortlist_per_cluster"
            )
        if not args.resume_initialization:
            _atomic_csv_dump(shortlist_trace, paths["ted_shortlist_trace.csv"])
        shortlist_cluster_ids = np.asarray([cluster_ids[index] for index in shortlist_indices])
        shortlist_capacity = {
            component: int(np.sum(shortlist_cluster_ids == component))
            for component in range(wgmm_parameters.n_components)
        }
        expand_quotas, expand_quota_diagnostics = allocate_cluster_quotas(
            shortlist_capacity,
            expand_evals,
            mode=args.wgmm_quota_mode,
            component_weights=component_mass,
            equal_weight=float(args.wgmm_equal_weight),
        )
        trace_rank = {int(row["candidate_index"]): int(row["selection_rank"]) for row in shortlist_trace}
        if args.initial_selection_strategy == "wgmm_ted_lowfid":
            if os.path.exists(paths["low_fidelity_history.json"]):
                if not args.resume_initialization:
                    raise FileExistsError("low_fidelity_history.json already exists")
                low_history = _read_json(paths["low_fidelity_history.json"])
                if not isinstance(low_history, list):
                    raise ValueError("low_fidelity_history.json must contain a list")
                seen_low: set[int] = set()
                shortlist_set = set(int(value) for value in shortlist_indices)
                for row in low_history:
                    if not isinstance(row, dict):
                        raise ValueError("low-fidelity resume history contains a non-object")
                    index = int(row.get("candidate_pool_index", -1))
                    if index not in shortlist_set or index in seen_low:
                        raise ValueError("low-fidelity resume candidate set is invalid or duplicated")
                    seen_low.add(index)
                    fingerprint = make_candidate_fingerprint(pool[index].numpy())
                    if row.get("candidate_fingerprint") != fingerprint:
                        raise ValueError("low-fidelity resume candidate fingerprint mismatch")
                    if row.get("initialization_config_fingerprint") != config_fingerprint:
                        raise ValueError("low-fidelity resume configuration fingerprint mismatch")
                    _validate_candidate_evaluation_record(
                        row,
                        pool[index].numpy(),
                        search_seed=int(args.seed),
                        hp_mode=args.hp_mode,
                        z_bound=float(args.z_bound),
                        evaluation_fidelity="low",
                        context="low-fidelity resume",
                    )
            low_by_index = {int(row["candidate_pool_index"]): row for row in low_history}
            for index in shortlist_indices:
                if int(index) in low_by_index:
                    continue
                fingerprint = make_candidate_fingerprint(pool[index].numpy())
                row = _low_fidelity_eval(
                    vae=vae,
                    z=pool[index].clone(),
                    data=data,
                    in_ch=in_ch,
                    out_ch=out_ch,
                    args=args,
                    device=device,
                    candidate_index=int(index),
                    cluster_id=int(cluster_ids[index]),
                    fingerprint=fingerprint,
                )
                row["initialization_config_fingerprint"] = config_fingerprint
                low_history.append(row)
                low_by_index[int(index)] = row
                atomic_initialization_json_dump(low_history, paths["low_fidelity_history.json"])
            shortlist_tensor = pool[shortlist_indices]
            shortlist_masks = masks[shortlist_indices]
            gp_predictions = predictor.predict_batch(
                shortlist_tensor,
                condition_masks=shortlist_masks if predictor.use_conditional_kernel else None,
            )
            combined_values = np.full(len(shortlist_indices), -1.0, dtype=np.float64)
            for component in sorted(expand_quotas):
                positions = [
                    pos for pos, index in enumerate(shortlist_indices)
                    if int(cluster_ids[index]) == int(component)
                ]
                indices = [shortlist_indices[pos] for pos in positions]
                lf = [float(low_by_index[index]["val_acc"]) for index in indices]
                valid = [bool(low_by_index[index]["valid"]) for index in indices]
                means = [float(gp_predictions[pos]["mean"]) for pos in positions]
                stds = [float(gp_predictions[pos]["std"]) for pos in positions]
                score_parts = combined_expansion_scores(
                    indices,
                    lf,
                    means,
                    stds,
                    low_fidelity_valid=valid,
                    weights=(
                        float(args.low_fidelity_score_weight),
                        float(args.gp_mean_score_weight),
                        float(args.gp_std_score_weight),
                    ),
                )
                for local_pos, global_pos in enumerate(positions):
                    index = shortlist_indices[global_pos]
                    low_row = low_by_index[index]
                    low_row.update(
                        {
                            "gp_mean": means[local_pos],
                            "gp_std": stds[local_pos],
                            "low_fidelity_rank": float(score_parts["low_fidelity_rank"][local_pos]),
                            "gp_mean_rank": float(score_parts["gp_mean_rank"][local_pos]),
                            "gp_std_rank": float(score_parts["gp_std_rank"][local_pos]),
                            "combined_score": float(score_parts["combined_score"][local_pos]),
                        }
                    )
                    combined_values[global_pos] = score_parts["combined_score"][local_pos]
            expand_indices = select_by_cluster_scores(
                shortlist_indices,
                shortlist_cluster_ids,
                combined_values,
                expand_quotas,
            )
            selected_set = set(expand_indices)
            for row in low_history:
                row["selected_for_full_expansion"] = int(row["candidate_pool_index"]) in selected_set
            atomic_initialization_json_dump(low_history, paths["low_fidelity_history.json"])
            expansion_score_rows = [
                {
                    **row,
                    "hp": json.dumps(row.get("hp", {}), sort_keys=True),
                    "operations": json.dumps(row.get("operations", [])),
                    "edges": json.dumps(row.get("edges", [])),
                    "z_search": json.dumps(row.get("z_search", [])),
                }
                for row in low_history
            ]
        else:
            shortlist_order = {int(row["candidate_index"]): int(row["selection_rank"]) for row in shortlist_trace}
            deterministic_scores = [-float(shortlist_order[index]) for index in shortlist_indices]
            expand_indices = select_by_cluster_scores(
                shortlist_indices,
                shortlist_cluster_ids,
                deterministic_scores,
                expand_quotas,
            )
            expansion_score_rows = [
                {
                    "candidate_pool_index": int(index),
                    "cluster_id": int(cluster_ids[index]),
                    "ted_selection_rank": int(shortlist_order[index]),
                    "selected_for_full_expansion": int(index) in set(expand_indices),
                }
                for index in shortlist_indices
            ]
        if len(expand_indices) != expand_evals:
            raise RuntimeError("expansion selection did not fill initial_expand_evals")
        if os.path.exists(paths["selected_expand_indices.json"]):
            saved_expand = _read_json(paths["selected_expand_indices.json"])
            if saved_expand != expand_indices:
                raise ValueError("resume expansion selection differs from saved indices")
        else:
            atomic_initialization_json_dump(expand_indices, paths["selected_expand_indices.json"])
        _atomic_csv_dump(expansion_score_rows, paths["expansion_scores.csv"])
        quota_payload = {"seed": seed_quota_diagnostics, "expand": expand_quota_diagnostics}
        atomic_initialization_json_dump(quota_payload, paths["cluster_quota.json"])
        shortlist_by_index = {int(row["candidate_index"]): row for row in shortlist_trace}
        low_by_index = {int(row["candidate_pool_index"]): row for row in low_history}
        for selection_rank, index in enumerate(expand_indices):
            if ("initial_expand", int(index)) in completed:
                continue
            z = pool[int(index)].clone()
            mask = masks[int(index)].tolist()
            fingerprint = make_candidate_fingerprint(z.numpy())
            trace_row = shortlist_by_index[int(index)]
            low_row = low_by_index.get(int(index))
            metadata = {
                "initialization_strategy": args.initial_selection_strategy,
                "evaluation_stage": "initial_expand",
                "evaluation_fidelity": "full",
                "cluster_id": int(cluster_ids[index]),
                "cluster_responsibility": float(
                    assignments["responsibilities"][index, cluster_ids[index]]
                ),
                "candidate_pool_index": int(index),
                "candidate_fingerprint": fingerprint,
                "selection_rank": int(selection_rank),
                "ted_score": float(trace_row["ted_score"]),
                "quota_mode": args.wgmm_quota_mode,
                "low_fidelity_used": low_row is not None,
                "low_fidelity_record_id": None if low_row is None else low_row["record_id"],
                "full_evaluation_index": len(history),
                "online_iteration": None,
                "initialization_config_fingerprint": config_fingerprint,
            }
            best_before = max(Y_obs) if Y_obs else None
            acc, valid, used_mask, safe = _append_eval(
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
                len(history),
                "wgmm_ted_expand",
                logger,
                predictor,
                prediction_records,
                "initial_expand_gp50",
                _score_logei(predictor, z, mask, best_before),
                False,
                0,
                mask,
                best_acc=best_before,
                record_metadata=metadata,
            )
            if low_row is not None:
                _validate_low_full_candidate_consistency(low_row, history[-1])
            if safe:
                init_valid.append((z, acc, used_mask))
            atomic_initialization_json_dump(history, paths["initialization_full_history.json"])
        all_valid: list[tuple[torch.Tensor, float, list[float]]] = []
        for row in history:
            if row.get("evaluation_stage") in ("initial_seed", "initial_expand") and bool(row.get("valid")):
                z = torch.tensor(row["z_search"], dtype=torch.float32)
                mask = [1.0] * ARCH_NZ + [float(value) for value in row["condition_mask_vector"]]
                all_valid.append((z, float(row["val_acc"]), mask))
        predictor = _fit_scratch_predictor(all_valid, args, device, logger)
        init_valid = all_valid
    low_actual_epochs = sum(int(row.get("actual_epochs") or 0) for row in low_history)
    low_wall = sum(float(row.get("runtime_seconds") or 0.0) for row in low_history)
    low_equivalent = sum(float(row.get("equivalent_full_evaluations") or 0.0) for row in low_history)
    budget_summary = {
        "full_fidelity_definition": "completed full GNN candidate evaluations",
        "initial_seed_full_evals": seed_evals,
        "initial_expand_full_evals": expand_evals,
        "online_full_eval_budget": int(args.n_iter),
        "planned_total_full_evals": seed_evals + expand_evals + int(args.n_iter),
        "max_total_full_evals": args.max_total_full_evals,
        "completed_initial_full_evals": len(history),
        "low_fidelity_candidate_count": len(low_history),
        "low_fidelity_actual_epochs": low_actual_epochs,
        "low_fidelity_wall_seconds": low_wall,
        "low_fidelity_gpu_seconds": None,
        "low_fidelity_equivalent_full_evaluations": low_equivalent,
        "low_fidelity_counted_in_full_budget": False,
    }
    atomic_initialization_json_dump(budget_summary, paths["budget_summary.json"])
    atomic_initialization_json_dump(
        {
            "seed_derivation": SEED_DERIVATION,
            **candidate_evaluation_seed_provenance(),
            "search_seed": int(args.seed),
            "lhs_seed": int(args.seed),
            "wgmm_fit_seed": (
                None
                if args.wgmm_source == "checkpoint"
                else stable_seed(int(args.seed), "wgmm_fit_pool", pool_fingerprint)
            ),
            "python_hash_used": False,
        },
        paths["rng_provenance.json"],
    )
    _save_prediction_csv(prediction_records, args.output)
    return (
        history,
        X_obs,
        Y_obs,
        prediction_records,
        init_valid,
        predictor,
        len(history),
        {
            "initialization_config": initialization_config,
            "budget_summary": budget_summary,
            "candidate_pool_fingerprint": pool_fingerprint,
            "wgmm_source": wgmm_parameters.source,
            "initialization_method_id": initialization_config["initialization_method_id"],
            "wgmm_estimator_semantics": wgmm_parameters.estimator_semantics,
            "wgmm_parameter_fingerprint": wgmm_parameters.parameter_fingerprint,
            "wgmm_n_components": wgmm_parameters.n_components,
        },
    )


def run_wgmm_bo(
    vae: JointSpaceVAE,
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> None:
    """Run the unchanged Exact-GP online policy after WGMM-TED initialization."""

    run_started = time.monotonic()
    (
        history,
        X_obs,
        Y_obs,
        prediction_records,
        init_valid,
        predictor,
        step,
        initialization_context,
    ) = run_wgmm_two_stage_initialization(
        vae, data, in_ch, out_ch, args, device, logger,
    )
    if not Y_obs:
        raise RuntimeError("WGMM-TED initialization produced no full-fidelity observations")

    config = initialization_context["initialization_config"]
    config_fingerprint = str(config["initialization_config_fingerprint"])
    online_history_path = os.path.join(args.output, "wgmm_online_history.json")
    online_checkpoint_path = os.path.join(args.output, "accuracy_gp_wgmm_resume.pt")
    online_rng_path = os.path.join(args.output, "wgmm_online_rng.pt")
    online_state_path = os.path.join(args.output, "wgmm_online_state.json")
    online_history: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 7919)

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
    eval_times: list[float] = []
    update_times: list[float] = []
    online_valid_count = 0
    initial_history_count = len(history)
    elapsed_before_resume = 0.0

    def current_elapsed_seconds() -> float:
        return float(elapsed_before_resume + (time.monotonic() - run_started))

    def commit_online_state() -> None:
        atomic_initialization_json_dump(online_history, online_history_path)
        predictor.save(online_checkpoint_path)
        atomic_torch_save(generator.get_state().cpu(), online_rng_path)
        atomic_initialization_json_dump(
            {
                "format_version": 2,
                "candidate_evaluation_seed_scheme": CANDIDATE_EVALUATION_SEED_SCHEME,
                "initialization_config_fingerprint": config_fingerprint,
                "completed_online_evals": len(online_history),
                "gp_train_size": predictor.train_size,
                "gp_observation_count": int(predictor.train_observation_counts.sum().item()),
                "next_step": initial_history_count + len(online_history),
                "adaptive_sampling": bool(args.adaptive_sampling),
                "monitor_observed_online_evals": int(monitor.observed_results),
                "monitor_state": monitor.state_dict(),
                "elapsed_seconds_total": current_elapsed_seconds(),
            },
            online_state_path,
        )

    if not args.resume_initialization:
        commit_online_state()
    elif not os.path.exists(online_history_path):
        partial = [
            path for path in (online_checkpoint_path, online_rng_path, online_state_path)
            if os.path.exists(path)
        ]
        if partial:
            raise FileNotFoundError(
                "incomplete zero-step WGMM online resume transaction: " + ", ".join(partial)
            )
        commit_online_state()

    if args.resume_initialization and os.path.exists(online_history_path):
        required_resume = (online_checkpoint_path, online_rng_path, online_state_path)
        missing = [path for path in required_resume if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "incomplete WGMM online resume transaction; missing: " + ", ".join(missing)
            )
        raw_online = _read_json(online_history_path)
        state = _read_json(online_state_path)
        if not isinstance(raw_online, list) or not isinstance(state, dict):
            raise ValueError("WGMM online resume artifacts have invalid JSON structure")
        if int(state.get("format_version", -1)) != 2:
            raise ValueError("WGMM online resume state lacks complete convergence-monitor state")
        if state.get("candidate_evaluation_seed_scheme") != CANDIDATE_EVALUATION_SEED_SCHEME:
            raise ValueError(
                "WGMM online resume uses a legacy or missing "
                "candidate_evaluation_seed_scheme"
            )
        if state.get("initialization_config_fingerprint") != config_fingerprint:
            raise ValueError("WGMM online resume configuration fingerprint mismatch")
        if bool(state.get("adaptive_sampling")) != bool(args.adaptive_sampling):
            raise ValueError("WGMM online resume adaptive_sampling mode mismatch")
        completed_count = int(state.get("completed_online_evals", -1))
        _validate_online_resume_boundary(
            raw_history_count=len(raw_online),
            completed_count=completed_count,
            adaptive_sampling=bool(args.adaptive_sampling),
            n_iter=int(args.n_iter),
            max_bo_samples=int(args.max_bo_samples),
        )
        elapsed_before_resume = float(state.get("elapsed_seconds_total", -1.0))
        if not math.isfinite(elapsed_before_resume) or elapsed_before_resume < 0.0:
            raise ValueError("WGMM online resume elapsed time state is invalid")
        monitor_state = state.get("monitor_state")
        monitor.load_state_dict(monitor_state)
        if int(state.get("monitor_observed_online_evals", -1)) != completed_count:
            raise ValueError("WGMM online resume monitor observation count mismatch")
        if monitor.observed_results != completed_count:
            raise ValueError("WGMM online resume monitor state is not aligned with online history")
        predictor = AccuracyGPPredictor.load(
            online_checkpoint_path,
            device=device,
            expected={
                "arch_nz": ARCH_NZ,
                "hp_mode": args.hp_mode,
                "search_dim": _search_dim(args),
                "initialization_config_fingerprint": config_fingerprint,
            },
        )
        state_train_size = int(state.get("gp_train_size", -1))
        state_observation_count = int(state.get("gp_observation_count", -1))
        actual_observation_count = int(predictor.train_observation_counts.sum().item())
        checkpoint_has_pending_update = False
        if predictor.train_size != state_train_size or actual_observation_count != state_observation_count:
            if (
                len(raw_online) == completed_count + 1
                and actual_observation_count == state_observation_count + 1
                and predictor.train_size in (state_train_size, state_train_size + 1)
            ):
                checkpoint_has_pending_update = True
            else:
                raise ValueError("WGMM online resume GP state/count mismatch")
        generator.set_state(torch.as_tensor(_torch_load(online_rng_path, "cpu"), dtype=torch.uint8))
        seen_iterations: set[int] = set()
        for expected_iteration, raw in enumerate(raw_online):
            if not isinstance(raw, dict):
                raise ValueError("WGMM online history contains a non-object record")
            if raw.get("initialization_config_fingerprint") != config_fingerprint:
                raise ValueError("WGMM online history fingerprint mismatch")
            iteration = int(raw.get("online_iteration", -1))
            if iteration != expected_iteration or iteration in seen_iterations:
                raise ValueError("WGMM online history iteration sequence is not contiguous")
            if raw.get("evaluation_stage") != "online_bo" or raw.get("evaluation_fidelity") != "full":
                raise ValueError("WGMM online resume contains a non-online/full record")
            if int(raw.get("full_evaluation_index", -1)) != initial_history_count + iteration:
                raise ValueError("WGMM online resume full-evaluation indices are not contiguous")
            seen_iterations.add(iteration)
            row = copy.deepcopy(raw)
            canonical_z, _fingerprint, _expected_seed = _validate_candidate_evaluation_record(
                row,
                row["z_search"],
                search_seed=int(args.seed),
                hp_mode=args.hp_mode,
                z_bound=float(args.z_bound),
                evaluation_fidelity="full",
                context="WGMM online resume",
            )
            z = torch.tensor(canonical_z, dtype=torch.float32)
            history.append(row)
            X_obs.append(z)
            Y_obs.append(float(row["val_acc"]))
            prediction = copy.deepcopy(row)
            prediction["record_type"] = row.get("type")
            prediction_records.append(prediction)
            online_history.append(row)
            if expected_iteration >= completed_count:
                monitor.observe_bo_result(float(row["val_acc"]), prediction)
            eval_times.append(float(row.get("eval_seconds") or 0.0))
            update_times.append(float(row.get("gp_update_seconds") or 0.0))
            if expected_iteration < completed_count:
                online_valid_count += int(bool(row.get("gp_update_performed")))
        if len(raw_online) == completed_count + 1:
            pending = online_history[-1]
            pending_prediction = prediction_records[-1]
            safe = bool(pending.get("valid")) and not str(
                pending.get("gp_update_skipped_reason") or ""
            )
            if checkpoint_has_pending_update and not safe:
                raise ValueError("WGMM online resume checkpoint advanced for an unsafe record")
            update_started = time.monotonic()
            if safe and not checkpoint_has_pending_update:
                z = X_obs[-1]
                mask = [1.0] * ARCH_NZ + [
                    float(value) for value in pending["condition_mask_vector"]
                ]
                predictor.append_observation(
                    z,
                    float(pending["val_acc"]),
                    condition_mask=mask if predictor.use_conditional_kernel else None,
                )
                should_optimize = (
                    args.gp_update_mode == "warm_refit"
                    and (online_valid_count + 1) % int(args.gp_refit_every) == 0
                )
                predictor.refit(
                    optimize=should_optimize,
                    steps=int(args.gp_refit_steps) if should_optimize else None,
                )
            if safe:
                online_valid_count += 1
            update_seconds = time.monotonic() - update_started
            recovered_fields = {
                "gp_train_size_after": predictor.train_size,
                "gp_update_performed": bool(safe),
                "gp_update_seconds": (
                    float(update_seconds) if safe and not checkpoint_has_pending_update else 0.0
                ),
                "resume_recovered_pre_update_record": True,
                "resume_update_already_checkpointed": bool(checkpoint_has_pending_update),
            }
            pending.update(recovered_fields)
            pending_prediction.update(recovered_fields)
            update_times[-1] = float(recovered_fields["gp_update_seconds"])
            _save_prediction_csv(prediction_records, args.output)
            commit_online_state()
        step = len(history)
        logger.info("Resumed WGMM online BO at iteration %d", len(online_history))

    logger.info(
        "\n[Step 3] BO after %d WGMM-TED full initialization evaluations: n_iter=%d",
        len(history) - len(online_history), int(args.n_iter),
    )
    probe_seed = args.probe_pool_seed if args.probe_pool_seed is not None else int(args.seed) + 104729
    probe_norm = torch.tensor(
        _sample_unit_lhs(int(args.probe_pool_size), _search_dim(args), int(probe_seed)),
        dtype=torch.float32,
    )
    probe_raw = denormalize_search_vector(
        probe_norm, arch_nz=ARCH_NZ, hp_mode=args.hp_mode, z_bound=args.z_bound,
    ).float()
    probe_masks = (
        _candidate_masks(vae, probe_raw, args, device, logger)
        if predictor.use_conditional_kernel else None
    )
    best_acc = max(Y_obs)
    online_limit = int(args.max_bo_samples) if args.adaptive_sampling else int(args.n_iter)
    stop_reason = "fixed_iteration_complete"
    start_iteration = len(online_history)

    for it in range(start_iteration, online_limit):
        if args.adaptive_sampling:
            last_estimate = 0.0
            if eval_times or update_times:
                last_estimate = float(np.median(eval_times[-5:])) + float(np.median(update_times[-5:]))
            decision = monitor.stop_decision(it, current_elapsed_seconds(), last_estimate)
            if decision.deferred_reason is not None:
                logger.info("Adaptive stop deferred: %s", decision.deferred_reason)
            if decision.stop_reason is not None:
                stop_reason = decision.stop_reason
                break
        if args.online_candidate_strategy == "qlogei":
            candidate_selection_seed = stable_seed(int(args.seed), "acquisition", int(step))
            with isolated_rng(candidate_selection_seed, device):
                z_next, logei_value, condition_mask = optimize_acq(
                    predictor, best_acc, args, logger, vae, device, generator,
                )
        elif args.online_candidate_strategy == "random":
            z_next, condition_mask, candidate_selection_seed = sample_random_online_candidate(
                vae, args, device, logger, step,
            )
            logei_value = None
        else:
            raise ValueError(f"unsupported online candidate strategy: {args.online_candidate_strategy!r}")

        z_next = torch.tensor(
            clip_z_search_by_mode(z_next, args.hp_mode, ARCH_NZ, args.z_bound),
            dtype=torch.float32,
        )
        fingerprint = make_candidate_fingerprint(z_next.numpy())
        metadata = {
            "initialization_strategy": args.initial_selection_strategy,
            "evaluation_stage": "online_bo",
            "evaluation_fidelity": "full",
            "cluster_id": None,
            "cluster_responsibility": None,
            "candidate_pool_index": None,
            "candidate_fingerprint": fingerprint,
            "selection_rank": None,
            "ted_score": None,
            "quota_mode": None,
            "low_fidelity_used": False,
            "low_fidelity_record_id": None,
            "full_evaluation_index": len(history),
            "online_iteration": int(it),
            "initialization_config_fingerprint": config_fingerprint,
        }
        def persist_pre_update_record() -> None:
            pending_history = online_history + [copy.deepcopy(history[-1])]
            atomic_initialization_json_dump(pending_history, online_history_path)
            # qLogEI's local generator has already advanced; save that state with
            # the evaluated label before allowing the GP mutation to begin.
            atomic_torch_save(generator.get_state().cpu(), online_rng_path)

        acc, _valid, _, _safe = _append_eval(
            vae, z_next, data, in_ch, out_ch, args, device,
            history, X_obs, Y_obs, step, "bo", logger, predictor,
            prediction_records, "online_bo", logei_value, True,
            online_valid_count, condition_mask, best_acc=best_acc,
            convergence_monitor=monitor,
            candidate_selection_seed=candidate_selection_seed,
            record_metadata=metadata,
            pre_update_persist=persist_pre_update_record,
        )
        current_record = prediction_records[-1]
        online_history.append(copy.deepcopy(history[-1]))
        if current_record["gp_update_performed"]:
            online_valid_count += 1
        eval_times.append(float(current_record["eval_seconds"]))
        update_times.append(float(current_record["gp_update_seconds"]))
        if acc > best_acc:
            best_acc = acc
            logger.info("  iter %3d: <-- NEW BEST %.4f", it, best_acc)
        else:
            logger.info("  iter %3d: best=%.4f", it, best_acc)
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
                elapsed_seconds=current_elapsed_seconds(),
                eval_seconds=eval_times,
                gp_update_seconds=update_times,
            )
            _write_convergence_files(monitor.history, args.output)
            logger.info("GP convergence check: %s", json.dumps(check, ensure_ascii=False))

        # The state file inside this transaction is written last.
        commit_online_state()
        budget = dict(initialization_context["budget_summary"])
        budget["completed_online_full_evals"] = len(online_history)
        budget["completed_total_full_evals"] = len(history)
        atomic_initialization_json_dump(
            budget, os.path.join(args.output, "budget_summary.json"),
        )
        if online_samples % 10 == 0:
            _save(history, X_obs, Y_obs, args, logger, f"step{it}", predictor=predictor)
    else:
        if args.adaptive_sampling:
            stop_reason = "sample_budget_reached"

    online_bo_samples = len(online_history)
    final_decision = monitor.stop_decision(
        online_bo_samples,
        current_elapsed_seconds(),
        monitor.history[-1]["estimated_next_step_seconds"] if monitor.history else 0.0,
    )
    if args.adaptive_sampling and final_decision.stop_reason is not None:
        stop_reason = final_decision.stop_reason
    best_idx = int(np.argmax(np.asarray(Y_obs, dtype=np.float64)))
    best_z = X_obs[best_idx]
    best_cfg = decode_arch(
        vae, best_z[:ARCH_NZ], device, n_trials=10,
        decoder_seed=stable_seed(int(args.seed), "decoder", best_z[:ARCH_NZ]),
    )
    logger.info("WGMM-TED Phase4 best val_acc=%.4f config=%s", max(Y_obs), best_cfg)
    predictor.save(os.path.join(args.output, "accuracy_gp_online_final.pt"))
    summary = {
        "gp_init_mode": "scratch",
        **_surrogate_provenance(args, predictor),
        "initialization_strategy": args.initial_selection_strategy,
        "initialization_config_fingerprint": config_fingerprint,
        "used_previous_history": False,
        "used_offline_checkpoint": False,
        "converged": bool(final_decision.converged),
        "stop_reason": stop_reason,
        "offline_train_size": int(predictor.offline_train_size),
        "online_added": int(predictor.train_size - len(init_valid)),
        "scratch_init_samples": len(history) - online_bo_samples,
        "scratch_init_valid_samples": len(init_valid),
        "online_bo_samples": online_bo_samples,
        "total_evaluated_samples": len(history),
        "gp_train_size_final": predictor.train_size,
        "best_actual_val_acc": float(max(Y_obs)),
        "elapsed_seconds": current_elapsed_seconds(),
        "search_seed": int(args.seed),
        "seed_derivation": SEED_DERIVATION,
        **candidate_evaluation_seed_provenance(),
        "rng_isolation_enabled": True,
        "online_candidate_strategy": args.online_candidate_strategy,
        "initialization_source": "wgmm_clustered_ted",
        "replayed_initial_samples": int(len(history) - online_bo_samples) if args.resume_initialization else 0,
        "newly_evaluated_online_samples": int(online_bo_samples - start_iteration),
        "convergence_checks": len(monitor.history),
        "valid_convergence_checks": monitor.valid_convergence_checks,
        "stagnation_deferred_events": list(monitor.stagnation_deferred_events),
        "initialization": initialization_context,
        **monitor.current_prequential_metrics(),
    }
    atomic_json_dump(summary, os.path.join(args.output, "gp_metrics.json"))
    _write_convergence_files(monitor.history, args.output)
    _save(history, X_obs, Y_obs, args, logger, "final", predictor=predictor, summary=summary)
    _plot(history, args, logger)


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
    if args.initial_selection_strategy != "schur":
        if predictor is not None:
            raise ValueError("WGMM-clustered TED must not receive a preloaded GP predictor")
        run_wgmm_bo(vae, data, in_ch, out_ch, args, device, logger)
        return
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
        replay_provenance = _surrogate_provenance(args, predictor)
        for row in history:
            row.update(replay_provenance)
        for row in prediction_records:
            row.update(replay_provenance)
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
            **_surrogate_provenance(args, predictor),
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
            **candidate_evaluation_seed_provenance(),
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
        if getattr(args, "surrogate_type", "exact_gp") == "dkl_gp":
            init_update_seed = stable_seed(
                int(args.seed), "dkl_refit", int(predictor.train_size), 0,
            )
            predictor.refit(
                optimize=True,
                steps=int(args.dkl_refit_steps),
                training_seed=init_update_seed,
            )
        else:
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
        **_surrogate_provenance(args, predictor),
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
        **candidate_evaluation_seed_provenance(),
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
    if args.surrogate_type == "dkl_gp":
        validate_dkl_config(
            hidden_dim=args.dkl_hidden_dim,
            feature_dim=args.dkl_feature_dim,
            activation=args.dkl_activation,
            lr=args.dkl_lr,
            weight_decay=args.dkl_weight_decay,
            grad_clip=args.dkl_grad_clip,
            init_steps=args.dkl_init_steps,
            refit_steps=args.dkl_refit_steps,
            early_stopping_patience=args.dkl_early_stopping_patience,
            min_delta=args.dkl_min_delta,
        )
    if int(args.n_extra) < 0:
        raise ValueError("--n_extra must be non-negative")
    if int(args.min_bo_samples) > int(args.max_bo_samples):
        raise ValueError("--min_bo_samples cannot exceed --max_bo_samples")
    if float(args.max_wall_time_hours) <= 0.0:
        raise ValueError("--max_wall_time_hours must be positive")
    validate_frozen_init_configuration(args)
    validate_two_stage_initialization_config(args)
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
    logger.info(
        "surrogate_type=%s dkl=%s",
        args.surrogate_type,
        json.dumps(_surrogate_provenance(args), ensure_ascii=False),
    )
    logger.info(f"online_candidate_strategy={args.online_candidate_strategy}")
    logger.info(
        "initial_selection_strategy=%s initial_seed_evals=%d "
        "initial_expand_evals=%d max_total_full_evals=%s",
        args.initial_selection_strategy,
        _resolved_initial_seed_evals(args),
        int(args.initial_expand_evals),
        args.max_total_full_evals,
    )
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
        checkpoint_expected = {
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
        }
        if args.surrogate_type == "dkl_gp":
            checkpoint_expected.update(
                {
                    "feature_hidden_dims": [int(args.dkl_hidden_dim)],
                    "feature_dim": int(args.dkl_feature_dim),
                    "activation": args.dkl_activation,
                }
            )
            predictor = DKLAccuracyGPPredictor.load(
                args.gp_checkpoint,
                device=device,
                expected=checkpoint_expected,
            )
        else:
            predictor = AccuracyGPPredictor.load(
                args.gp_checkpoint,
                device=device,
                expected=checkpoint_expected,
            )
        holdout_size = 0 if predictor.holdout_Y is None else int(predictor.holdout_Y.numel())
        logger.info(
            "Accuracy GP loaded: source=%s surrogate_type=%s model=SingleTaskGP "
            "kernel=%s offline_train=%d holdout=%d",
            args.gp_checkpoint, args.surrogate_type, predictor.kernel_type,
            predictor.offline_train_size, holdout_size,
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
