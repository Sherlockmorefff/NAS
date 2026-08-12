"""Phase 3 joint architecture + HP Bayesian optimization.

Phase3 searches z_search = [z_arch, hp] and records unified history fields
for downstream Phase4 GMM loaders.
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

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Input.*not contained.*unit cube.*",
)

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from eval_utils import eval_z_search
from hp_modes import hp_dim_from_mode, hp_names_from_mode, validate_hp_mode
from nas_space import JointSpaceVAE


ARCH_NZ = 12


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
    parser = argparse.ArgumentParser(description="bo_phase3 hp_mode")
    parser.add_argument("--checkpoint", type=str, default="results/joint_search/joint_model_global4_best.pth")
    parser.add_argument("--warm_start", type=str, default="results/bo_phase2/best_z_arch_final.pt")
    parser.add_argument("--cora_root", type=str, default="/tmp/Cora")
    parser.add_argument("--n_init", type=int, default=15)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/bo_phase3")
    parser.add_argument("--sigma_arch", type=float, default=0.8)
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--novelty_w", type=float, default=0.0)
    parser.add_argument("--log_lr_min", type=float, default=-4.0)
    parser.add_argument("--log_lr_max", type=float, default=-1.5)
    parser.add_argument("--dropout_min", type=float, default=0.1)
    parser.add_argument("--dropout_max", type=float, default=0.6)
    parser.add_argument("--version", type=str, default="global4")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gcnii_alpha", type=float, default=0.1)
    parser.add_argument("--gcnii_theta", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--hp_mode",
        type=str,
        default="global4",
        choices=["global4", "hybrid_cond7", "layer_cond19"],
    )
    parser.add_argument(
        "--hp_dim",
        type=int,
        default=None,
        help="Deprecated compatibility guard. If provided, must match hp_mode.",
    )
    return parser.parse_args()


args = parse_args()
args.hp_mode = validate_hp_mode(args.hp_mode)
HP_DIM = hp_dim_from_mode(args.hp_mode)
if args.hp_dim is not None and args.hp_dim != HP_DIM:
    raise ValueError(f"--hp_dim={args.hp_dim} does not match hp_mode={args.hp_mode} hp_dim={HP_DIM}")

SEARCH_DIM = ARCH_NZ + HP_DIM
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(args.output, exist_ok=True)
logger, log_filepath = setup_logger(args.log_dir, "bo_phase3", args.version)
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


def load_vae(ckpt_path):
    model = JointSpaceVAE(
        ArchArgs(),
        hp_mode=args.hp_mode,
        hp_latent_dim=HP_DIM,
        hp_input_dim=HP_DIM,
    ).to(DEVICE)
    state = _torch_load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    logger.info(
        f"VAE loaded: {ckpt_path} "
        f"(arch_nz={ARCH_NZ}, hp_mode={args.hp_mode}, hp_dim={HP_DIM})"
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
]


def _dl(url, dest, timeout=30):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 10:
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def load_cora(root):
    raw = os.path.join(root, "raw")
    missing = [f for f in _CORA_FILES if not os.path.exists(os.path.join(raw, f))]
    if missing:
        os.makedirs(raw, exist_ok=True)
        for fname in missing:
            ok = any(_dl(f"{mirror}/{fname}", os.path.join(raw, fname)) for mirror in _MIRRORS)
            logger.info(f"  {fname}: {'OK' if ok else 'FAILED'}")
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    ds = Planetoid(root=pyg_root, name="Cora", transform=T.NormalizeFeatures())
    return ds[0].to(DEVICE), ds.num_features, ds.num_classes


def sample_hp_norm() -> torch.Tensor:
    return torch.rand(HP_DIM)


def clip_z_search(z_search: torch.Tensor) -> torch.Tensor:
    z = z_search.detach().cpu().float().flatten().clone()
    if z.numel() != SEARCH_DIM:
        raise ValueError(f"Expected z_search length {SEARCH_DIM}, got {z.numel()}")
    z[ARCH_NZ:] = z[ARCH_NZ:].clamp(0.0, 1.0)
    return z


def decode_arch(vae, z_arch: torch.Tensor, n_trials=5):
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


def eval_z(vae, z_search: torch.Tensor, data, in_ch, out_ch) -> dict:
    z_search = clip_z_search(z_search)
    result = eval_z_search(
        vae,
        z_search.to(DEVICE),
        data,
        in_ch,
        out_ch,
        arch_nz=ARCH_NZ,
        log_lr_min=args.log_lr_min,
        log_lr_max=args.log_lr_max,
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
        device=DEVICE,
        use_conditional_params=True,
        gcnii_alpha=args.gcnii_alpha,
        gcnii_theta=args.gcnii_theta,
        max_epochs=args.eval_epochs,
        patience=args.patience,
        hp_mode=args.hp_mode,
        return_hp=True,
    )
    return result


def history_record(step: int, record_type: str, z_search: torch.Tensor, result: dict, best_acc: float | None = None):
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
        "search_dim": SEARCH_DIM,
        "z_search": z_search.detach().cpu().float().flatten().tolist(),
        "z_arch": z_search[:ARCH_NZ].detach().cpu().float().flatten().tolist(),
        "val_acc": float(result.get("val_acc", 0.0)),
        "valid": bool(result.get("valid", False)),
        "operations": [] if config is None else list(config.get("operations", [])),
        "edges": [] if config is None else [list(edge) for edge in config.get("edges", [])],
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
        "params": params,
    }


def _minmax(X: torch.Tensor):
    xmin = X.min(0).values
    xrng = (X.max(0).values - xmin).clamp(min=1e-8)
    return (X - xmin) / xrng, xmin, xrng


def optimize_acq(X_obs, Y_obs):
    X_n, Xmin, Xrng = _minmax(X_obs)
    Y_n = (Y_obs - Y_obs.mean()) / Y_obs.std().clamp(min=1e-6)

    gp = SingleTaskGP(X_n.double(), Y_n.double())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    logei = qLogExpectedImprovement(gp, best_f=Y_n.double().max())
    bounds = torch.zeros(2, SEARCH_DIM, dtype=torch.double)
    bounds[1] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand, _ = optimize_acqf(
            logei,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

    return clip_z_search(cand.squeeze(0).float() * Xrng + Xmin)


def run_bo(vae, data, in_ch, out_ch):
    history = []
    X_obs = []
    Y_obs = []

    warm_pts = []
    if args.warm_start and os.path.exists(args.warm_start):
        z_arch_best = _torch_load(args.warm_start, map_location="cpu").float().flatten()
        if z_arch_best.shape[0] > ARCH_NZ:
            z_arch_best = z_arch_best[:ARCH_NZ]
        elif z_arch_best.shape[0] < ARCH_NZ:
            z_arch_best = torch.cat([z_arch_best, torch.zeros(ARCH_NZ - z_arch_best.shape[0])])

        n_warm = min(5, max(args.n_init // 3, 1))
        for _ in range(n_warm):
            z_arch = z_arch_best + torch.randn(ARCH_NZ) * 0.3
            warm_pts.append(torch.cat([z_arch, sample_hp_norm()]))
        logger.info(f"Warm start: {n_warm} points near Phase2 best")

    n_rand = args.n_init - len(warm_pts)
    rand_pts = [
        torch.cat([torch.randn(ARCH_NZ) * args.sigma_arch, sample_hp_norm()])
        for _ in range(n_rand)
    ]
    init_pts = warm_pts + rand_pts

    logger.info(f"\n[Step 1] Init: {args.n_init} points ({len(warm_pts)} warm + {n_rand} random)")
    logger.info(f"  hp_mode={args.hp_mode}")
    logger.info(f"  hp_dim={HP_DIM}")
    logger.info(f"  hp_names={hp_names_from_mode(args.hp_mode)}")
    logger.info(f"  ARCH_NZ={ARCH_NZ} SEARCH_DIM={SEARCH_DIM}")
    logger.info(f"  LR: 10^[{args.log_lr_min},{args.log_lr_max}] Dropout: [{args.dropout_min},{args.dropout_max}]")

    for i, z in enumerate(tqdm(init_pts, desc="Init")):
        z = clip_z_search(z)
        result = eval_z(vae, z, data, in_ch, out_ch)
        acc = float(result["val_acc"])
        X_obs.append(z)
        Y_obs.append(acc)
        history.append(history_record(i, "init", z, result))
        logger.info(
            f"  init {i:>2d}: val={acc:.4f} lr={result['lr']:.5f} "
            f"drop={result['dropout']:.3f} hidden={result['hidden_dim']} "
            f"l2={result['l2']:.1e} valid={result['valid']}"
        )

    if not any(h["valid"] for h in history):
        logger.error("All init points invalid.")
        return

    best_acc = max(Y_obs)
    logger.info(f"\nInit done. Best={best_acc:.4f}")
    logger.info(f"\n[Step 2] BO: {args.n_iter} iters SEARCH_DIM={SEARCH_DIM}")

    for it in range(args.n_iter):
        X_t = torch.stack(X_obs)
        Y_t = torch.tensor(Y_obs).unsqueeze(-1)
        try:
            z_next = optimize_acq(X_t, Y_t)
        except Exception as exc:
            logger.warning(f"  [iter {it}] acqf error: {exc}, random fallback")
            z_next = torch.cat([torch.randn(ARCH_NZ) * args.sigma_arch, sample_hp_norm()])
            z_next = clip_z_search(z_next)

        result = eval_z(vae, z_next, data, in_ch, out_ch)
        acc = float(result["val_acc"])
        X_obs.append(z_next)
        Y_obs.append(acc)

        if acc > best_acc:
            best_acc = acc
            tag = "<-- NEW BEST"
        else:
            tag = f"(best={best_acc:.4f})"

        logger.info(
            f"  iter {it:>3d}: val={acc:.4f} lr={result['lr']:.5f} "
            f"drop={result['dropout']:.3f} hidden={result['hidden_dim']} "
            f"l2={result['l2']:.1e} {tag}"
            f"{' [invalid]' if not result['valid'] else ''}"
        )
        history.append(history_record(args.n_init + it, "bo", z_next, result, best_acc=best_acc))

        if (it + 1) % 10 == 0:
            _save(history, X_obs, Y_obs, f"step{it}")

    best_idx = Y_obs.index(max(Y_obs))
    best_z = X_obs[best_idx]
    best_cfg = decode_arch(vae, best_z[:ARCH_NZ], n_trials=10)

    logger.info("\n" + "=" * 66)
    logger.info(f"          Phase 3 hp_mode={args.hp_mode} Final")
    logger.info("=" * 66)
    logger.info(f"  Best val_acc  : {max(Y_obs):.4f}")
    logger.info(f"  GNN config    : {best_cfg}")
    logger.info(f"  SEARCH_DIM    : {SEARCH_DIM}")
    logger.info("=" * 66)

    _save(history, X_obs, Y_obs, "final")
    _plot(history)
    _compare(history)


def _save(history, X_obs, Y_obs, suffix):
    history_path = os.path.join(args.output, f"history_{suffix}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    best_z_path = os.path.join(args.output, f"best_z_search_{suffix}.pt")
    best_z = X_obs[Y_obs.index(max(Y_obs))]
    torch.save(best_z, best_z_path)
    saved_paths = [history_path, best_z_path]

    if suffix == "final":
        best_z_final_path = os.path.join(args.output, "best_z_final.pt")
        torch.save(best_z, best_z_final_path)
        saved_paths.append(best_z_final_path)

    logger.info(f"  Saved: {', '.join(saved_paths)}")


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

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        valid_steps = [s for s, v in zip(steps, valids) if v]
        valid_accs = [a for a, v in zip(accs, valids) if v]
        invalid_steps = [s for s, v in zip(steps, valids) if not v]
        invalid_accs = [a for a, v in zip(accs, valids) if not v]
        ax.scatter(valid_steps, valid_accs, alpha=0.6, color="steelblue", label="Valid")
        ax.scatter(invalid_steps, invalid_accs, alpha=0.4, color="red", marker="x", label="Invalid")
        ax.plot(steps, bests, "r-", lw=2, label="Best so far")
        ax.axvline(x=args.n_init - 0.5, color="gray", ls="--", label="Init|BO")
        ax.set(xlabel="Step", ylabel="Val Accuracy", title=f"Phase3 {args.hp_mode}")
        ax.legend()
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        lrs_v = [h["lr"] for h in history if h.get("valid")]
        drs_v = [h["dropout"] for h in history if h.get("valid")]
        accs_v = [h["val_acc"] for h in history if h.get("valid")]
        sc = ax2.scatter(lrs_v, drs_v, c=accs_v, cmap="RdYlGn", alpha=0.7, s=60)
        plt.colorbar(sc, ax=ax2, label="Val Accuracy")
        ax2.set_xscale("log")
        ax2.set(xlabel="LR (log)", ylabel="Dropout", title="HP Landscape")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(args.output, f"convergence_phase3_{args.hp_mode}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved: {path}")
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot")


def _compare(h3):
    p2 = "results/bo_phase2/history_final.json"
    if not os.path.exists(p2):
        return
    with open(p2, "r", encoding="utf-8") as f:
        best2 = max(entry["val_acc"] for entry in json.load(f))
    best3 = max(entry["val_acc"] for entry in h3)
    logger.info(f"\n[Comparison] Phase2: {best2:.4f} Phase3: {best3:.4f} delta={best3 - best2:+.4f}")


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start_time = time.time()

    logger.info("=" * 70)
    logger.info(f"bo_phase3.py -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info(f"Device={DEVICE} ARCH_NZ={ARCH_NZ} SEARCH_DIM={SEARCH_DIM}")
    logger.info(f"hp_mode={args.hp_mode} hp_dim={HP_DIM}")
    logger.info(f"seed={args.seed} version={args.version}")

    vae = load_vae(args.checkpoint)
    data, in_ch, out_ch = load_cora(args.cora_root)
    logger.info(f"Cora: {in_ch} features, {out_ch} classes")
    logger.info(f"Search: z_arch({ARCH_NZ}) + hp({HP_DIM}) = {SEARCH_DIM}")

    run_bo(vae, data, in_ch, out_ch)
    elapsed = time.time() - start_time
    logger.info(f"\nRun complete. Total time: {elapsed / 60:.1f} min")
