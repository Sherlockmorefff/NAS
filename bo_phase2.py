"""Phase 2 arch-only Bayesian optimization warm start.

Phase2 intentionally remains architecture-only:
    - no GMM
    - no HP search
    - history hp_mode is always "arch_only"
    - search_dim is ARCH_NZ
    - z_arch is stored
    - z_search is None

Phase2 history is arch-only and should not be consumed by Phase4 GMM loaders
unless an explicit padding strategy is implemented.
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

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/mnt/project")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from eval_utils import train_and_eval_arch
from nas_space import JointSpaceVAE


ARCH_NZ = 12
ARCH_ONLY_HP_MODE = "arch_only"


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


def save_args_json(args, log_path: str) -> str:
    json_path = log_path.replace(".log", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="bo_phase2 arch-only")
    parser.add_argument("--checkpoint", type=str, default="results/joint_search/joint_model_global4_best.pth")
    parser.add_argument("--cora_root", type=str, default="/tmp/Cora")
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=40)
    parser.add_argument("--output", type=str, default="results/bo_phase2")
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--version", type=str, default="arch_only")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gcnii_alpha", type=float, default=0.1)
    parser.add_argument("--gcnii_theta", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--vae_hp_mode",
        type=str,
        default="global4",
        choices=["global4", "hybrid_cond7", "layer_cond19"],
        help="Only used to instantiate the checkpoint-compatible JointSpaceVAE HP branch.",
    )
    return parser.parse_args()


args = parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(args.output, exist_ok=True)
logger, log_filepath = setup_logger(args.log_dir, "bo_phase2", args.version)
save_args_json(args, log_filepath)


class ArchArgs:
    def __init__(self):
        self.max_n = 7
        self.num_vertex_type = 8
        self.START_TYPE = 0
        self.END_TYPE = 1
        self.hs = 501
        self.nz = ARCH_NZ
        self.bidirectional = True


def _torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_vae(checkpoint_path: str) -> JointSpaceVAE:
    model = JointSpaceVAE(ArchArgs(), hp_mode=args.vae_hp_mode).to(DEVICE)
    state = _torch_load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    logger.info(
        f"VAE loaded: {checkpoint_path} "
        f"(arch_nz={model.arch_nz}, vae_hp_mode={args.vae_hp_mode})"
    )
    return model


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


def _try_download_file(url, dest, timeout=30):
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


def load_cora(root):
    raw_dir = os.path.join(root, "raw")
    missing = [
        f
        for f in _CORA_FILES
        if not os.path.exists(os.path.join(raw_dir, f))
        or os.path.getsize(os.path.join(raw_dir, f)) < 10
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
    return dataset[0].to(DEVICE), dataset.num_features, dataset.num_classes


def decode_arch(vae, z_arch: torch.Tensor, n_trials: int = 5):
    results = []
    with torch.no_grad():
        for _ in range(n_trials):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(DEVICE))
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


def evaluate_z_arch(vae, z_arch, data, in_ch, out_ch):
    config = decode_arch(vae, z_arch)
    val_acc, is_valid = train_and_eval_arch(
        config,
        data,
        in_ch,
        out_ch,
        lr=args.lr,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        weight_decay=args.weight_decay,
        gcnii_alpha=args.gcnii_alpha,
        gcnii_theta=args.gcnii_theta,
        device=DEVICE,
        max_epochs=args.eval_epochs,
        patience=args.patience,
        track_test=False,
    )
    return val_acc, is_valid, config


def history_record(step: int, record_type: str, z_arch: torch.Tensor, val_acc: float, valid: bool, config: dict | None):
    return {
        "step": int(step),
        "type": record_type,
        "hp_mode": ARCH_ONLY_HP_MODE,
        "search_dim": ARCH_NZ,
        "z_search": None,
        "z_arch": z_arch.detach().cpu().float().flatten().tolist(),
        "val_acc": float(val_acc),
        "valid": bool(valid),
        "operations": [] if config is None else list(config.get("operations", [])),
        "edges": [] if config is None else [list(edge) for edge in config.get("edges", [])],
        "lr": float(args.lr),
        "dropout": float(args.dropout),
        "hidden_dim": int(args.hidden_dim),
        "l2": float(args.weight_decay),
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
        "condition_mask": {},
        "params": {
            "gcnii_alpha": float(args.gcnii_alpha),
            "gcnii_theta": float(args.gcnii_theta),
            "eval_epochs": int(args.eval_epochs),
            "patience": int(args.patience),
        },
    }


def _minmax_scale(X: torch.Tensor):
    X_min = X.min(dim=0).values
    X_range = (X.max(dim=0).values - X_min).clamp(min=1e-8)
    return (X - X_min) / X_range, X_min, X_range


def fit_gp_and_optimize_logei(X_obs, Y_obs, arch_nz, X_min, X_range):
    X = X_obs.double()
    Y = Y_obs.double()
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    log_ei = qLogExpectedImprovement(gp, best_f=Y.max())
    bounds = torch.zeros(2, arch_nz, dtype=torch.double)
    bounds[1] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidate, _ = optimize_acqf(
            log_ei,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=1024,
        )
    return candidate.squeeze(0).float() * X_range + X_min


def run_bo(vae, data, in_ch, out_ch):
    history = []
    X_obs = []
    Y_obs = []

    logger.info(f"\n[Step 1] Random init: {args.n_init} points arch_nz={ARCH_NZ}")
    logger.info(
        f"Phase2 is arch-only: hp_mode={ARCH_ONLY_HP_MODE}, search_dim={ARCH_NZ}"
    )
    logger.info(
        "Phase2 history is arch-only and should not be consumed by Phase4 GMM "
        "loaders unless an explicit padding strategy is implemented."
    )

    for i in tqdm(range(args.n_init), desc="Init"):
        z_arch = torch.randn(ARCH_NZ, device=DEVICE) * args.sigma
        acc, valid, config = evaluate_z_arch(vae, z_arch, data, in_ch, out_ch)
        X_obs.append(z_arch.cpu())
        Y_obs.append(float(acc))
        history.append(history_record(i, "init", z_arch, acc, valid, config))
        logger.info(
            f"  init {i:>2d}: val_acc={acc:.4f} "
            f"[{'valid' if valid else 'INVALID'}]"
        )

    if not any(h["valid"] for h in history):
        logger.error("All init points invalid.")
        return

    best_acc = max(Y_obs)
    logger.info(f"\nInit done. Best val_acc={best_acc:.4f}")

    logger.info(f"\n[Step 2] BO: {args.n_iter} iters")
    for it in range(args.n_iter):
        X_t = torch.stack(X_obs)
        Y_t = torch.tensor(Y_obs).unsqueeze(-1)
        Y_norm = (Y_t - Y_t.mean()) / Y_t.std().clamp(min=1e-6)
        X_norm, X_min, X_range = _minmax_scale(X_t)

        try:
            z_next = fit_gp_and_optimize_logei(
                X_norm,
                Y_norm,
                ARCH_NZ,
                X_min,
                X_range,
            ).to(DEVICE)
        except Exception as exc:
            logger.warning(f"  [iter {it}] GP optimize failed: {exc}, random fallback")
            z_next = torch.randn(ARCH_NZ, device=DEVICE) * args.sigma

        acc, valid, config = evaluate_z_arch(vae, z_next, data, in_ch, out_ch)
        X_obs.append(z_next.cpu())
        Y_obs.append(float(acc))

        if acc > best_acc:
            best_acc = float(acc)
            tag = "<-- NEW BEST"
        else:
            tag = f"(best={best_acc:.4f})"

        logger.info(
            f"  iter {it:>3d}: val_acc={acc:.4f} {tag}"
            f"{' [invalid]' if not valid else ''}"
        )
        history.append(
            history_record(args.n_init + it, "bo", z_next, acc, valid, config)
        )

        if (it + 1) % 10 == 0:
            _save(history, X_obs, Y_obs, f"step{it}")

    best_idx = Y_obs.index(max(Y_obs))
    best_z = X_obs[best_idx].to(DEVICE)
    best_cfg = decode_arch(vae, best_z, n_trials=10)

    logger.info("\n" + "=" * 55)
    logger.info("       Phase 2 arch-only BO Final Result")
    logger.info("=" * 55)
    logger.info(f"  Best val_acc : {max(Y_obs):.4f}")
    logger.info(f"  GNN config   : {best_cfg}")
    logger.info("=" * 55)

    _save(history, X_obs, Y_obs, "final")
    _plot(history)


def _save(history, X_obs, Y_obs, suffix):
    history_path = os.path.join(args.output, f"history_{suffix}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    best_z_path = os.path.join(args.output, f"best_z_arch_{suffix}.pt")
    torch.save(X_obs[Y_obs.index(max(Y_obs))], best_z_path)

    if suffix == "final":
        torch.save(
            X_obs[Y_obs.index(max(Y_obs))],
            os.path.join(args.output, "best_z_arch_final.pt"),
        )
    logger.info(f"  Saved: {history_path}, {best_z_path}")


def _plot(history):
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

        valid_steps = [s for s, v in zip(steps, valids) if v]
        valid_accs = [a for a, v in zip(accs, valids) if v]
        invalid_steps = [s for s, v in zip(steps, valids) if not v]
        invalid_accs = [a for a, v in zip(accs, valids) if not v]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(valid_steps, valid_accs, alpha=0.6, color="steelblue", label="Valid")
        ax.scatter(invalid_steps, invalid_accs, alpha=0.4, color="red", marker="x", label="Invalid")
        ax.plot(steps, bests, "r-", lw=2, label="Best so far")
        ax.axvline(x=args.n_init - 0.5, color="gray", ls="--", label="Init|BO")
        ax.set_xlabel("Step")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(f"Phase 2 arch-only BO best={max(bests):.4f}")
        ax.legend()
        ax.grid(alpha=0.3)

        path = os.path.join(args.output, "convergence_phase2_arch_only.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved: {path}")
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot")


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"bo_phase2.py arch-only -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"Device={DEVICE} arch_nz={ARCH_NZ} seed={args.seed}")
    logger.info(f"checkpoint={args.checkpoint}")

    vae = load_vae(args.checkpoint)
    data, in_ch, out_ch = load_cora(args.cora_root)
    logger.info(
        f"Cora: {in_ch} features, {out_ch} classes "
        f"train={data.train_mask.sum().item()} val={data.val_mask.sum().item()}"
    )

    run_bo(vae, data, in_ch, out_ch)
    elapsed = time.time() - start_time
    logger.info(f"\nRun complete. Total time: {elapsed / 60:.1f} min")
