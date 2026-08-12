"""Train JointSpaceVAE with hp_mode-aware HP reconstruction masks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hp_modes import hp_dim_from_mode, hp_names_from_mode, validate_hp_mode
from nas_space import JointSpaceVAE, condition_masks_from_graphs


def setup_logger(log_dir: str, script_name: str, version: str):
    ts = datetime.now().strftime("%m%d_%H%M")
    log_subdir = os.path.join(log_dir, script_name)
    os.makedirs(log_subdir, exist_ok=True)

    log_filename = f"train_{version}_{ts}.log"
    log_filepath = os.path.join(log_subdir, log_filename)

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_filepath, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_filepath


def save_args_json(args, log_filepath: str) -> str:
    json_path = log_filepath.replace(".log", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return json_path


def compute_loss(
    model,
    arch_out,
    mu_a,
    log_a,
    hp_out,
    mu_h,
    log_h,
    target_graphs,
    target_hps,
    hp_mask,
    beta,
    free_bits,
):
    arch_loss_tuple = model.arch_vae.loss(mu_a, log_a, target_graphs)

    if len(arch_loss_tuple) == 3:
        _, recon_arch, kl_arch_raw = arch_loss_tuple
    elif len(arch_loss_tuple) == 2:
        recon_arch, kl_arch_raw = arch_loss_tuple
    else:
        recon_arch = arch_loss_tuple[1]
        kl_arch_raw = arch_loss_tuple[2]

    recon_hp = torch.sum(((hp_out - target_hps) * hp_mask) ** 2)
    kl_hp_raw = -0.5 * torch.sum(1 + log_h - mu_h.pow(2) - log_h.exp())

    bs = target_hps.size(0)
    arch_nz = mu_a.size(1)
    hp_nz = mu_h.size(1)

    free_limit_arch = free_bits * bs * arch_nz
    free_limit_hp = free_bits * bs * hp_nz

    kl_arch = torch.clamp(kl_arch_raw - free_limit_arch, min=0.0)
    kl_hp = torch.clamp(kl_hp_raw - free_limit_hp, min=0.0)

    total_recon = recon_arch + recon_hp
    total_kl = kl_arch + kl_hp
    total_loss = total_recon + beta * total_kl

    return total_loss, recon_arch, recon_hp, kl_arch_raw, kl_hp_raw


def train_one_epoch(model, loader, optimizer, beta, args, device):
    model.train()
    total_loss = 0.0
    total_ra = 0.0
    total_rh = 0.0
    total_ka = 0.0
    total_kh = 0.0

    for g_batch, hp_batch in loader:
        hp_batch = hp_batch.to(device)
        hp_mask = condition_masks_from_graphs(
            g_batch,
            device=device,
            hp_mode=args.hp_mode,
        )
        hp_input = hp_batch * hp_mask

        optimizer.zero_grad()
        (arch_out, mu_a, log_a), (hp_out, mu_h, log_h) = model(g_batch, hp_input)

        loss, ra, rh, ka, kh = compute_loss(
            model=model,
            arch_out=arch_out,
            mu_a=mu_a,
            log_a=log_a,
            hp_out=hp_out,
            mu_h=mu_h,
            log_h=log_h,
            target_graphs=g_batch,
            target_hps=hp_batch,
            hp_mask=hp_mask,
            beta=beta,
            free_bits=args.free_bits,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_ra += float(ra.item())
        total_rh += float(rh.item())
        total_ka += float(ka.item())
        total_kh += float(kh.item())

    n = max(len(loader), 1)
    return total_loss / n, total_ra / n, total_rh / n, total_ka / n, total_kh / n


def evaluate_one_epoch(model, loader, beta, args, device):
    model.eval()
    total_loss = 0.0
    total_ra = 0.0
    total_rh = 0.0
    total_ka = 0.0
    total_kh = 0.0

    with torch.no_grad():
        for g_batch, hp_batch in loader:
            hp_batch = hp_batch.to(device)
            hp_mask = condition_masks_from_graphs(
                g_batch,
                device=device,
                hp_mode=args.hp_mode,
            )
            hp_input = hp_batch * hp_mask
            (arch_out, mu_a, log_a), (hp_out, mu_h, log_h) = model(g_batch, hp_input)
            loss, ra, rh, ka, kh = compute_loss(
                model=model,
                arch_out=arch_out,
                mu_a=mu_a,
                log_a=log_a,
                hp_out=hp_out,
                mu_h=mu_h,
                log_h=log_h,
                target_graphs=g_batch,
                target_hps=hp_batch,
                hp_mask=hp_mask,
                beta=beta,
                free_bits=args.free_bits,
            )
            total_loss += float(loss.item())
            total_ra += float(ra.item())
            total_rh += float(rh.item())
            total_ka += float(ka.item())
            total_kh += float(kh.item())

    n = max(len(loader), 1)
    return total_loss / n, total_ra / n, total_rh / n, total_ka / n, total_kh / n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train_joint hp_mode")
    parser.add_argument("--data", type=str, default="data/mini_gnn_dataset_global4.pkl")
    parser.add_argument("--checkpoint_dir", type=str, default="results/joint_search")
    parser.add_argument("--version", type=str, default="global4")
    parser.add_argument("--log_dir", type=str, default="logs/train_joint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_max", type=float, default=0.01)
    parser.add_argument("--free_bits", type=float, default=2.0)
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.hp_mode = validate_hp_mode(args.hp_mode)
    expected_hp_dim = hp_dim_from_mode(args.hp_mode)

    logger, log_path = setup_logger(args.log_dir, "train_joint", args.version)
    save_args_json(args, log_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info(f"train_joint hp_mode | Device: {device}")
    logger.info(f"data={args.data}")
    logger.info(f"hp_mode={args.hp_mode}")
    logger.info(f"hp_dim={expected_hp_dim}")
    logger.info(f"hp_names={hp_names_from_mode(args.hp_mode)}")
    logger.info("condition mask rule: global dims active; conditional dims active by decoded ops")
    logger.info("=" * 70)

    if not os.path.exists(args.data):
        logger.error(f"Dataset not found: {args.data}")
        return

    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    if not dataset:
        logger.error("Dataset is empty.")
        return

    inferred_hp_dim = len(dataset[0][2])
    if inferred_hp_dim != expected_hp_dim:
        logger.error(
            f"Dataset hp_dim={inferred_hp_dim} does not match "
            f"hp_mode={args.hp_mode} expected hp_dim={expected_hp_dim}"
        )
        return

    if args.hp_dim is not None and args.hp_dim != expected_hp_dim:
        logger.error(
            f"--hp_dim={args.hp_dim} does not match "
            f"hp_mode={args.hp_mode} expected hp_dim={expected_hp_dim}"
        )
        return

    logger.info(f"HP dim inferred from dataset: {inferred_hp_dim}")

    processed_data = []
    import igraph

    for types, adj, hp in dataset:
        g = igraph.Graph(directed=True)
        g.add_vertices(len(types))
        g.vs["type"] = types
        edges = [
            (i, j)
            for i in range(len(types))
            for j in range(len(types))
            if int(adj[i][j]) == 1
        ]
        g.add_edges(edges)
        processed_data.append((g, torch.tensor(hp, dtype=torch.float32)))

    indices = list(range(len(processed_data)))
    np.random.shuffle(indices)
    split = int(0.9 * len(processed_data))
    train_idx = indices[:split]
    val_idx = indices[split:]

    def collate_fn(batch):
        return [b[0] for b in batch], torch.stack([b[1] for b in batch])

    train_loader = DataLoader(
        Subset(processed_data, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(processed_data, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    class ArchArgs:
        max_n = 7
        num_vertex_type = 8
        nz = 12
        bidirectional = True
        hs = 501
        START_TYPE = 0
        END_TYPE = 1

    model = JointSpaceVAE(
        ArchArgs(),
        hp_mode=args.hp_mode,
        hp_latent_dim=expected_hp_dim,
        hp_input_dim=expected_hp_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = None

    for epoch in range(1, args.epochs + 1):
        if epoch <= 20:
            beta = 0.0
        elif epoch <= 80:
            beta = args.beta_max * (epoch - 20) / 60
        else:
            beta = args.beta_max

        loss, ra, rh, ka, kh = train_one_epoch(
            model,
            train_loader,
            optimizer,
            beta,
            args,
            device,
        )
        val_loss, val_ra, val_rh, val_ka, val_kh = evaluate_one_epoch(
            model,
            val_loader,
            beta,
            args,
            device,
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            logger.info(
                f"Epoch {epoch:>3d}/{args.epochs} | "
                f"train={loss:.2f} val={val_loss:.2f} | "
                f"RA={ra:.2f}/{val_ra:.2f} RH={rh:.4f}/{val_rh:.4f} | "
                f"KA={ka:.1f}/{val_ka:.1f} KH={kh:.1f}/{val_kh:.1f} | "
                f"Beta={beta:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(
                args.checkpoint_dir,
                f"joint_model_{args.version}_{args.hp_mode}_best.pth",
            )
            torch.save(model.state_dict(), best_path)

        if epoch == args.epochs:
            save_path = os.path.join(
                args.checkpoint_dir,
                f"joint_model_{args.version}_{args.hp_mode}_ep{epoch}.pth",
            )
            torch.save(model.state_dict(), save_path)
            logger.info(f"checkpoint path: {save_path}")

    if best_path:
        logger.info(f"best checkpoint path: {best_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
