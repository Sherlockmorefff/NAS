"""
bo_phase4_tpe.py
=================================================================
Optuna TPE version of Phase 4 conditional NAS + HP search.

This file intentionally keeps bo_phase4.py unchanged. It reuses the same
JointSpaceVAE decoder and eval_z_search() evaluator, but replaces the
GP/GPND acquisition loop with Optuna's tree-structured Parzen estimator.

Search vector:
  z_search = [z_arch(12), hp]

For hp_dim=17, HP is activated conditionally by decoded operator layer:
  [lr, dropout, gat_heads_i, sage_aggr_i, gin_eps_i], i=0..4
Inactive conditional parameters are left at their default normalized value.
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
import urllib.request
from datetime import datetime
from collections import Counter

import torch
import numpy as np

try:
    import optuna
except ImportError:
    print(
        "Optuna is required for bo_phase4_tpe.py. "
        "Install it in the NAS environment with: pip install optuna",
        file=sys.stderr,
    )
    raise SystemExit(1)

sys.path.insert(0, '/mnt/project')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from nas_space import JointSpaceVAE
from eval_utils import (
    eval_z_search,
    HP_DIM_LAYERWISE,
    MAX_OP_NODES,
    condition_mask_from_config,
    decode_layerwise_conditional_hp,
    norm_to_gat_heads,
    norm_to_sage_aggr,
)


# ============================================================
# Logging
# ============================================================
def setup_logger(log_dir: str, script_name: str, version: str):
    ts = datetime.now().strftime('%m%d_%H%M')
    log_subdir = os.path.join(log_dir, script_name)
    os.makedirs(log_subdir, exist_ok=True)

    log_path = os.path.join(log_subdir, f"train_{version}_{ts}.log")

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"日志文件: {log_path}")
    return logger, log_path


def save_args_json(args, log_path: str) -> str:
    json_path = log_path.replace('.log', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return json_path


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser(description='bo_phase4_tpe')
parser.add_argument('--checkpoint', type=str,
                    default='results/joint_search/joint_model_v3_ep100.pth')
parser.add_argument('--warm_start', type=str,
                    default='results/bo_phase2/best_z_arch_final.pt')
parser.add_argument('--cora_root', type=str, default='/tmp/Cora')
parser.add_argument('--output', type=str, default='results/bo_phase4_tpe')
parser.add_argument('--version', type=str, default='v3_tpe')
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--n_init', type=int, default=20,
                    help='TPE startup random trials.')
parser.add_argument('--n_iter', type=int, default=60,
                    help='TPE guided trials after startup.')
parser.add_argument('--n_trials', type=int, default=None,
                    help='Total trials. If omitted, n_init + n_iter is used.')
parser.add_argument('--n_warm_perturb', type=int, default=2,
                    help='Extra warm-start perturbations to enqueue.')
parser.add_argument('--warm_arch_noise', type=float, default=0.3)
parser.add_argument('--warm_hp_noise', type=float, default=0.1)

parser.add_argument('--hp_dim', type=int, default=4,
                    choices=[4, HP_DIM_LAYERWISE],
                    help='HP dimension: 4=v5 type-wise, 17=v6 layer-wise + GIN.')
parser.add_argument('--arch_nz', type=int, default=12)
parser.add_argument('--sigma_arch', type=float, default=0.8)
parser.add_argument('--z_bound', type=float, default=0.0,
                    help='Search each z_arch_i in [-z_bound, z_bound]. '
                         'If <=0, use 3 * sigma_arch.')

parser.add_argument('--eval_epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--decode_trials', type=int, default=5)
parser.add_argument('--log_lr_min', type=float, default=-4.0)
parser.add_argument('--log_lr_max', type=float, default=-1.5)
parser.add_argument('--dropout_min', type=float, default=0.1)
parser.add_argument('--dropout_max', type=float, default=0.6)
parser.add_argument('--gcnii_alpha', type=float, default=0.1)
parser.add_argument('--gcnii_theta', type=float, default=0.5)

parser.add_argument('--storage', type=str, default=None,
                    help='Optional Optuna storage, e.g. sqlite:///results/tpe.db')
parser.add_argument('--study_name', type=str, default=None)
parser.add_argument('--save_every', type=int, default=10)
parser.add_argument('--disable_multivariate', action='store_true')
parser.add_argument('--disable_group', action='store_true')
parser.add_argument('--constant_liar', action='store_true')

args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ARCH_NZ = args.arch_nz
HP_DIM = args.hp_dim
SEARCH_DIM = ARCH_NZ + HP_DIM
Z_BOUND = args.z_bound if args.z_bound > 0 else 3.0 * args.sigma_arch
N_TRIALS = args.n_trials if args.n_trials is not None else args.n_init + args.n_iter

os.makedirs(args.output, exist_ok=True)
logger, log_filepath = setup_logger(args.log_dir, 'bo_phase4_tpe', args.version)
save_args_json(args, log_filepath)


# ============================================================
# ArchArgs
# ============================================================
class ArchArgs:
    max_n = 7
    num_vertex_type = 8
    START_TYPE = 0
    END_TYPE = 1
    hs = 501
    nz = ARCH_NZ
    bidirectional = True


def _torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_vae(ckpt: str):
    model = JointSpaceVAE(
        ArchArgs(), hp_latent_dim=HP_DIM, hp_input_dim=HP_DIM).to(DEVICE)
    state = _torch_load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"VAE loaded: {ckpt}  (arch_nz={ARCH_NZ}, hp_dim={HP_DIM})")
    return model


# ============================================================
# Cora loading
# ============================================================
_CORA_FILES = [
    "ind.cora.x", "ind.cora.tx", "ind.cora.allx",
    "ind.cora.y", "ind.cora.ty", "ind.cora.ally",
    "ind.cora.graph", "ind.cora.test.index",
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


def load_cora(root: str):
    raw = os.path.join(root, "raw")
    missing = [f for f in _CORA_FILES if not os.path.exists(os.path.join(raw, f))]
    if missing:
        os.makedirs(raw, exist_ok=True)
        for f in missing:
            ok = any(_dl(f"{m}/{f}", os.path.join(raw, f)) for m in _MIRRORS)
            logger.info(f"  {f}: {'OK' if ok else 'FAILED'}")
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    ds = Planetoid(root=pyg_root, name='Cora', transform=T.NormalizeFeatures())
    return ds[0].to(DEVICE), ds.num_features, ds.num_classes


# ============================================================
# Decode / parameter helpers
# ============================================================
def decode_arch(vae, z_arch: torch.Tensor, n_trials: int = None):
    if n_trials is None:
        n_trials = args.decode_trials
    results = []
    with torch.no_grad():
        for _ in range(n_trials):
            graphs = vae.arch_vae.decode(z_arch.unsqueeze(0).to(DEVICE))
            g = graphs[0]
            ops = [
                vae.op_mapping.get(g.vs[i]['type'], 'Identity')
                for i in range(1, g.vcount() - 1)
            ]
            edges = g.get_edgelist()
            n_eff = sum(1 for op in ops if op != 'Identity')
            if n_eff > 0:
                key = (tuple(ops), tuple(edges))
                results.append((key, {
                    'effective_layers': n_eff,
                    'operations': ops,
                    'edges': edges,
                }))
    if not results:
        return None
    best_key = Counter(r[0] for r in results).most_common(1)[0][0]
    return next(cfg for key, cfg in results if key == best_key)


def z_to_hp(z_search: torch.Tensor, config: dict = None):
    log_lr_n = float(z_search[ARCH_NZ])
    drop_n = float(z_search[ARCH_NZ + 1])
    log_lr = log_lr_n * (args.log_lr_max - args.log_lr_min) + args.log_lr_min
    dropout = drop_n * (args.dropout_max - args.dropout_min) + args.dropout_min
    lr = float(np.clip(10 ** np.clip(log_lr, -6, 0), 1e-6, 1.0))
    dr = float(np.clip(dropout, 0.0, 0.9))

    if HP_DIM >= HP_DIM_LAYERWISE:
        gat_heads, sage_aggr, gin_eps = decode_layerwise_conditional_hp(
            z_search, ARCH_NZ, config)
        return lr, dr, gat_heads, sage_aggr, gin_eps

    mask = condition_mask_from_config(config)
    gat_heads = (
        norm_to_gat_heads(float(z_search[ARCH_NZ + 2]))
        if mask['gat_heads'] and z_search.shape[0] > ARCH_NZ + 2 else 1
    )
    sage_aggr = (
        norm_to_sage_aggr(float(z_search[ARCH_NZ + 3]))
        if mask['sage_aggr'] and z_search.shape[0] > ARCH_NZ + 3 else 'mean'
    )
    return lr, dr, gat_heads, sage_aggr, None


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _trial_to_z_search(trial, vae):
    z_arch = torch.tensor([
        trial.suggest_float(f"z_arch_{i}", -Z_BOUND, Z_BOUND)
        for i in range(ARCH_NZ)
    ], dtype=torch.float32, device=DEVICE)

    config = decode_arch(vae, z_arch)
    mask = condition_mask_from_config(config)

    hp = torch.zeros(HP_DIM, dtype=torch.float32, device=DEVICE)
    hp[0] = trial.suggest_float("lr_norm", 0.0, 1.0)
    hp[1] = trial.suggest_float("dropout_norm", 0.0, 1.0)

    if HP_DIM >= HP_DIM_LAYERWISE:
        for i in range(MAX_OP_NODES):
            if mask['gat_by_layer'][i]:
                hp[2 + i] = trial.suggest_float(f"gat_heads_norm_{i}", 0.0, 1.0)
            if mask['sage_by_layer'][i]:
                hp[2 + MAX_OP_NODES + i] = trial.suggest_float(
                    f"sage_aggr_norm_{i}", 0.0, 1.0)
            if mask['gin_by_layer'][i]:
                hp[2 + 2 * MAX_OP_NODES + i] = trial.suggest_float(
                    f"gin_eps_norm_{i}", 0.0, 1.0)
    else:
        if mask['gat_heads']:
            hp[2] = trial.suggest_float("gat_heads_norm", 0.0, 1.0)
        if mask['sage_aggr']:
            hp[3] = trial.suggest_float("sage_aggr_norm", 0.0, 1.0)

    return torch.cat([z_arch, hp]), config


def _vector_to_params(z_vec: torch.Tensor, vae) -> dict:
    z_vec = z_vec.detach().cpu().float().flatten()
    if z_vec.numel() < ARCH_NZ:
        z_vec = torch.cat([z_vec, torch.zeros(ARCH_NZ - z_vec.numel())])

    z_arch = z_vec[:ARCH_NZ].clone()
    params = {
        f"z_arch_{i}": float(np.clip(float(z_arch[i]), -Z_BOUND, Z_BOUND))
        for i in range(ARCH_NZ)
    }

    if z_vec.numel() <= ARCH_NZ:
        return params

    hp = torch.zeros(HP_DIM)
    n_hp = min(HP_DIM, z_vec.numel() - ARCH_NZ)
    hp[:n_hp] = z_vec[ARCH_NZ:ARCH_NZ + n_hp].clamp(0.0, 1.0)

    params["lr_norm"] = _clip01(hp[0])
    params["dropout_norm"] = _clip01(hp[1])

    config = decode_arch(vae, z_arch.to(DEVICE))
    mask = condition_mask_from_config(config)

    if HP_DIM >= HP_DIM_LAYERWISE:
        for i in range(MAX_OP_NODES):
            if mask['gat_by_layer'][i]:
                params[f"gat_heads_norm_{i}"] = _clip01(hp[2 + i])
            if mask['sage_by_layer'][i]:
                params[f"sage_aggr_norm_{i}"] = _clip01(hp[2 + MAX_OP_NODES + i])
            if mask['gin_by_layer'][i]:
                params[f"gin_eps_norm_{i}"] = _clip01(hp[2 + 2 * MAX_OP_NODES + i])
    else:
        if mask['gat_heads'] and HP_DIM > 2:
            params["gat_heads_norm"] = _clip01(hp[2])
        if mask['sage_aggr'] and HP_DIM > 3:
            params["sage_aggr_norm"] = _clip01(hp[3])
    return params


def _enqueue_trial(study, params):
    try:
        study.enqueue_trial(params, skip_if_exists=True)
    except TypeError:
        study.enqueue_trial(params)


def enqueue_warm_start(study, vae):
    if not args.warm_start or not os.path.exists(args.warm_start):
        logger.info("Warm start not found; TPE will start from sampler startup trials.")
        return

    z_best = _torch_load(args.warm_start, map_location='cpu').float().flatten()
    warm_vectors = [z_best]

    for _ in range(args.n_warm_perturb):
        z = z_best.clone()
        if z.numel() < ARCH_NZ:
            z = torch.cat([z, torch.zeros(ARCH_NZ - z.numel())])
        z[:ARCH_NZ] += torch.randn(ARCH_NZ) * args.warm_arch_noise
        if z.numel() >= SEARCH_DIM:
            z[ARCH_NZ:ARCH_NZ + HP_DIM] = (
                z[ARCH_NZ:ARCH_NZ + HP_DIM]
                + torch.randn(HP_DIM) * args.warm_hp_noise
            ).clamp(0.0, 1.0)
        warm_vectors.append(z)

    for z in warm_vectors:
        _enqueue_trial(study, _vector_to_params(z, vae))
    logger.info(f"Enqueued warm-start trials: {len(warm_vectors)} from {args.warm_start}")


# ============================================================
# Optuna
# ============================================================
def make_sampler():
    kwargs = {
        "n_startup_trials": args.n_init,
        "seed": args.seed,
    }
    if not args.disable_multivariate:
        kwargs["multivariate"] = True
    if not args.disable_group:
        kwargs["group"] = True
    if args.constant_liar:
        kwargs["constant_liar"] = True

    try:
        return optuna.samplers.TPESampler(**kwargs)
    except TypeError:
        logger.warning("Installed Optuna does not support all requested TPE options; "
                       "falling back to basic TPESampler.")
        return optuna.samplers.TPESampler(
            n_startup_trials=args.n_init,
            seed=args.seed,
        )


def ensure_storage_dir(storage: str):
    if not storage or not storage.startswith("sqlite:///"):
        return
    db_path = storage[len("sqlite:///"):]
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


def make_study():
    ensure_storage_dir(args.storage)
    study_name = args.study_name or args.version
    return optuna.create_study(
        direction="maximize",
        sampler=make_sampler(),
        storage=args.storage,
        study_name=study_name,
        load_if_exists=bool(args.storage),
    )


def make_objective(vae, data, in_ch, out_ch):
    def objective(trial):
        z_search, config = _trial_to_z_search(trial, vae)

        result = eval_z_search(
            vae, z_search, data, in_ch, out_ch,
            arch_nz=ARCH_NZ,
            log_lr_min=args.log_lr_min,
            log_lr_max=args.log_lr_max,
            dropout_min=args.dropout_min,
            dropout_max=args.dropout_max,
            device=DEVICE,
            has_hidden=True,
            has_l2=True,
            use_conditional_params=True,
            gcnii_alpha=args.gcnii_alpha,
            gcnii_theta=args.gcnii_theta,
            n_trials=args.decode_trials,
            max_epochs=args.eval_epochs,
            patience=args.patience,
            return_layerwise=HP_DIM >= HP_DIM_LAYERWISE,
        )
        if HP_DIM >= HP_DIM_LAYERWISE:
            acc, lr_v, dr_v, heads_v, aggr_v, gin_v, ok = result
        else:
            acc, lr_v, dr_v, heads_v, aggr_v, ok = result
            gin_v = None

        trial.set_user_attr("z_search", z_search.detach().cpu().tolist())
        trial.set_user_attr("valid", bool(ok))
        trial.set_user_attr("operations", None if config is None else config["operations"])
        trial.set_user_attr("edges", None if config is None else config["edges"])
        trial.set_user_attr("lr", float(lr_v))
        trial.set_user_attr("dropout", float(dr_v))
        trial.set_user_attr("gat_heads", heads_v)
        trial.set_user_attr("sage_aggr", aggr_v)
        trial.set_user_attr("gin_eps", gin_v)

        tag = "valid" if ok else "invalid"
        logger.info(
            f"  trial {trial.number:>3d}: val={acc:.4f}  {tag}  "
            f"lr={lr_v:.5f}  drop={dr_v:.3f}  "
            f"heads={heads_v}  sage={aggr_v}  gin={gin_v}"
        )
        return float(acc) if ok else 0.0

    return objective


# ============================================================
# Save / report
# ============================================================
def trial_to_history_entry(trial):
    attrs = trial.user_attrs
    return {
        "step": trial.number,
        "type": "tpe",
        "state": trial.state.name,
        "val_acc": trial.value,
        "valid": attrs.get("valid"),
        "operations": attrs.get("operations"),
        "edges": attrs.get("edges"),
        "lr": attrs.get("lr"),
        "dropout": attrs.get("dropout"),
        "gat_heads": attrs.get("gat_heads"),
        "sage_aggr": attrs.get("sage_aggr"),
        "gin_eps": attrs.get("gin_eps"),
        "params": dict(trial.params),
    }


def completed_trials(study):
    return [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def save_outputs(study, suffix: str):
    trials = completed_trials(study)
    history = [trial_to_history_entry(t) for t in trials]
    with open(os.path.join(args.output, f"history_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if not trials:
        logger.warning(f"No completed trials to save for suffix={suffix}")
        return

    best = max(trials, key=lambda t: t.value)
    z_search = torch.tensor(best.user_attrs["z_search"], dtype=torch.float32)
    torch.save(z_search, os.path.join(args.output, f"best_z_{suffix}.pt"))

    best_result = trial_to_history_entry(best)
    best_result["best_value"] = float(best.value)
    with open(os.path.join(args.output, f"best_result_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)

    if suffix == "final":
        with open(os.path.join(args.output, "best_result.json"), "w", encoding="utf-8") as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {suffix}")


def make_save_callback():
    def callback(study, trial):
        done = len(completed_trials(study))
        if args.save_every > 0 and done > 0 and done % args.save_every == 0:
            save_outputs(study, f"step{done}")
    return callback


def print_final(study, vae):
    trials = completed_trials(study)
    if not trials:
        logger.error("No completed TPE trials.")
        return

    best = max(trials, key=lambda t: t.value)
    z_search = torch.tensor(best.user_attrs["z_search"], dtype=torch.float32, device=DEVICE)
    best_cfg = decode_arch(vae, z_search[:ARCH_NZ], n_trials=max(args.decode_trials, 10))
    lr_b, dr_b, heads_b, aggr_b, gin_b = z_to_hp(z_search, best_cfg)

    logger.info("\n" + "=" * 70)
    logger.info(f"  Phase 4 TPE Final  SEARCH_DIM={SEARCH_DIM}")
    logger.info("=" * 70)
    logger.info(f"  Best val_acc  : {best.value:.4f}")
    logger.info(f"  Trial number  : {best.number}")
    logger.info(f"  GNN config    : {best_cfg}")
    logger.info(f"  LR={lr_b:.5f}  Dropout={dr_b:.3f}")
    logger.info(f"  GAT heads={heads_b}  SAGE aggr={aggr_b}  GIN eps={gin_b}")
    logger.info(f"  gcnii_alpha={args.gcnii_alpha}  gcnii_theta={args.gcnii_theta}")
    logger.info("=" * 70)


def run_tpe(vae, data, in_ch, out_ch):
    study = make_study()
    enqueue_warm_start(study, vae)
    already_done = len(completed_trials(study))
    remaining_trials = max(N_TRIALS - already_done, 0)

    logger.info(f"\n[TPE Search] target_trials={N_TRIALS}  "
                f"completed={already_done}  remaining={remaining_trials}  "
                f"n_startup={args.n_init}")
    logger.info(f"Search space: z_arch({ARCH_NZ}d in [-{Z_BOUND:.2f}, {Z_BOUND:.2f}]) "
                f"+ hp({HP_DIM}d) = {SEARCH_DIM}d")
    if HP_DIM >= HP_DIM_LAYERWISE:
        logger.info("Conditional HP: layer-wise GAT/SAGE/GIN parameters are suggested only when active.")
    else:
        logger.info("Conditional HP: type-wise GAT/SAGE parameters are suggested only when active.")

    if remaining_trials > 0:
        study.optimize(
            make_objective(vae, data, in_ch, out_ch),
            n_trials=remaining_trials,
            callbacks=[make_save_callback()],
            show_progress_bar=False,
        )
    else:
        logger.info("Target trial count already reached; only saving current best outputs.")

    save_outputs(study, "final")
    print_final(study, vae)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start_time = time.time()

    logger.info("=" * 70)
    logger.info(f"bo_phase4_tpe.py  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"arch_nz={ARCH_NZ}  hp_dim={HP_DIM}  search_dim={SEARCH_DIM}")
    logger.info(f"eval_epochs={args.eval_epochs}  patience={args.patience}")
    logger.info(f"TPE: multivariate={not args.disable_multivariate}  "
                f"group={not args.disable_group}  constant_liar={args.constant_liar}")

    vae = load_vae(args.checkpoint)
    data, in_ch, out_ch = load_cora(args.cora_root)
    logger.info(f"Cora: {in_ch} features  {out_ch} classes")

    run_tpe(vae, data, in_ch, out_ch)

    elapsed = time.time() - start_time
    logger.info(f"\n运行完成  总耗时: {elapsed / 60:.1f} min")
