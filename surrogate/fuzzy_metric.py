"""Frozen-code joint mapping and observed-label fuzzy metric objectives.

The caller supplies cached VAE codes and HP4 features transformed using the
observed training set. Only revealed validation accuracies may be passed to
``neighborhood_affinity``. Regression targets may be standardized separately.
No VAE, candidate decoding, GP, or GNN training is performed in this module.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class JointMetricModel(nn.Module):
    """Map frozen architecture codes and HP4 into an eight-dimensional space.

    The deterministic, nonzero initialization is isolated from the caller's RNG.
    Cached architecture codes are detached at the boundary: even a code with an
    attached VAE graph cannot send gradients back through this mapping.
    """

    def __init__(
        self,
        z_dim: int,
        hp_dim: int = 4,
        hp_embed_dim: int = 16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if z_dim <= 0 or hp_dim != 4 or hp_embed_dim <= 0:
            raise ValueError("z_dim and hp_embed_dim must be positive; hp_dim must be 4")
        self.z_dim = int(z_dim)
        self.hp_dim = int(hp_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.hp_encoder = nn.Sequential(nn.Linear(hp_dim, hp_embed_dim), nn.SiLU())
            self.mapping = nn.Sequential(
                nn.Linear(z_dim + hp_embed_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
                nn.Linear(32, 8),
            )
            self.regressor = nn.Linear(8, 1)

    def forward(
        self, z: torch.Tensor, hp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"z must have shape (n, {self.z_dim})")
        if hp.ndim != 2 or hp.shape != (z.shape[0], self.hp_dim):
            raise ValueError(f"hp must have shape (n, {self.hp_dim})")
        embedding = self.mapping(torch.cat((z.detach(), self.hp_encoder(hp)), dim=-1))
        return embedding, self.regressor(embedding).squeeze(-1)


def neighborhood_affinity(
    y: torch.Tensor,
    mode: str = "N2",
    epsilon: float = 0.01,
    k: int = 10,
) -> torch.Tensor:
    """Construct N0/N1/N2 solely from the supplied observed raw accuracies.

    N0 uses k nearest non-self neighbors, the nearest strictly positive distance
    as rho (zero when none exists), and the kth distance floored at 1e-6 as sigma.
    Its directed memberships are combined by a + b - a*b. N1/N2 use an absolute
    difference without row normalization or subtraction of estimated noise.
    N2 excludes the exact epsilon boundary. Diagonal memberships are zero.
    """
    labels = torch.as_tensor(y).detach().reshape(-1)
    if not labels.is_floating_point():
        labels = labels.to(torch.get_default_dtype())
    if not torch.isfinite(labels).all():
        raise ValueError("observed labels must be finite")
    mode = mode.upper()
    if mode not in {"N0", "N1", "N2"}:
        raise ValueError("mode must be N0, N1, or N2")
    if not math.isfinite(epsilon) or epsilon <= 0 or k < 1:
        raise ValueError("epsilon must be finite and positive; k must be positive")
    n = len(labels)
    delta = (labels[:, None] - labels[None, :]).abs()
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    if mode in {"N1", "N2"}:
        affinity = torch.exp(-4.0 * (delta / epsilon).square())
        if mode == "N2":
            affinity = affinity * (delta < epsilon)
        return affinity.masked_fill(eye, 0.0)
    if n <= 1:
        return torch.zeros_like(delta)
    distances = delta.masked_fill(eye, float("inf"))
    neighbors = torch.argsort(distances, dim=1, stable=True)[:, : min(k, n - 1)]
    nearest = distances.gather(1, neighbors)
    sigma = nearest[:, -1].clamp_min(1e-6)
    rho = distances.masked_fill(distances <= 0, float("inf")).min(dim=1).values
    rho = torch.where(torch.isfinite(rho), rho, torch.zeros_like(rho))
    directed = torch.zeros_like(delta)
    directed.scatter_(1, neighbors, torch.exp(-(nearest - rho[:, None]).clamp_min(0) / sigma[:, None]))
    return (directed + directed.T - directed * directed.T).masked_fill(eye, 0.0)


def metric_loss(
    embedding: torch.Tensor,
    prediction: torch.Tensor,
    y: torch.Tensor,
    affinity: torch.Tensor,
    lambda_f: float = 0.1,
    lambda_e: float = 0.001,
    pair_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Regression + all-unordered-pair fuzzy BCE + mean squared embedding norm.

    ``y`` contains only revealed regression labels, optionally standardized.
    ``affinity`` must have been computed using raw accuracies, not those scaled
    regression targets. ``pair_weights`` defaults to one; it may be an n-by-n
    symmetric matrix or a vector in ``torch.triu_indices(n, n, 1)`` order.
    The weighted pair loss is divided by total weight. With no pairs or zero
    total weight it is zero; regression and the size constraint remain active.
    """
    if embedding.ndim != 2 or len(embedding) == 0:
        raise ValueError("embedding must be a nonempty matrix")
    n = len(embedding)
    prediction = prediction.reshape(-1)
    targets = torch.as_tensor(y, dtype=embedding.dtype, device=embedding.device).detach().reshape(-1)
    memberships = torch.as_tensor(affinity, dtype=embedding.dtype, device=embedding.device).detach()
    if prediction.shape != (n,) or targets.shape != (n,) or memberships.shape != (n, n):
        raise ValueError("prediction, labels, and affinity must match the embedding count")
    if not torch.isfinite(targets).all() or not torch.isfinite(memberships).all():
        raise ValueError("targets and affinity must be finite")
    if (memberships < 0).any() or (memberships > 1).any():
        raise ValueError("affinity must lie in [0, 1]")
    if not torch.allclose(memberships, memberships.T):
        raise ValueError("affinity must be symmetric")
    if any(not math.isfinite(value) or value < 0 for value in (lambda_f, lambda_e)):
        raise ValueError("loss coefficients must be finite and nonnegative")
    regression = (prediction - targets).square().mean()
    scale = embedding.square().sum(dim=-1).mean()
    left, right = torch.triu_indices(n, n, offset=1, device=embedding.device)
    neighborhood = embedding.sum() * 0.0
    if left.numel():
        squared_distance = (embedding[left] - embedding[right]).square().sum(dim=-1)
        clip = max(torch.finfo(embedding.dtype).eps, 1e-7)
        q = (1.0 / (1.0 + squared_distance)).clamp(clip, 1.0 - clip)
        target = memberships[left, right]
        pair_loss = -(target * q.log() + (1.0 - target) * torch.log1p(-q))
        if pair_weights is None:
            weights = torch.ones_like(pair_loss)
        else:
            weights = torch.as_tensor(pair_weights, dtype=embedding.dtype, device=embedding.device).detach()
            if weights.shape == (n, n):
                if not torch.allclose(weights, weights.T):
                    raise ValueError("pair weight matrix must be symmetric")
                weights = weights[left, right]
            if weights.shape != pair_loss.shape:
                raise ValueError("pair_weights must be an n-by-n matrix or one value per unordered pair")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("pair weights must be finite and nonnegative")
        if weights.sum() > 0:
            neighborhood = (weights * pair_loss).sum() / weights.sum()
    total = regression + lambda_f * neighborhood + lambda_e * scale
    return {"total": total, "regression": regression, "neighborhood": neighborhood, "scale": scale}
