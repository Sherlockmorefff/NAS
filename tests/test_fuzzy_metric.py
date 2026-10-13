"""CPU engineering checks; these synthetic values are not a historical replay."""

import math

import pytest
import torch

from surrogate.fuzzy_metric import JointMetricModel, metric_loss, neighborhood_affinity


def test_absolute_difference_rejects_60_80_in_both_directions():
    y = torch.tensor([0.60, 0.80], dtype=torch.double)
    n2 = neighborhood_affinity(y, "N2", epsilon=0.05)
    assert torch.equal(n2, torch.zeros_like(n2))
    n1 = neighborhood_affinity(y, "N1", epsilon=0.05)
    assert n1[0, 1] == pytest.approx(math.exp(-64), abs=1e-38)
    assert n1[1, 0] == n1[0, 1]


def test_n2_strict_boundary_and_no_row_max_normalization():
    y = torch.tensor([0.0, 0.1, -0.1, 0.05], dtype=torch.double)
    affinity = neighborhood_affinity(y, "N2", epsilon=0.1)
    assert affinity[0, 1] == 0
    assert affinity[0, 2] == 0
    assert affinity[0, 3] == pytest.approx(math.exp(-1))
    assert torch.equal(affinity, affinity.T)
    assert torch.equal(affinity.diag(), torch.zeros(4, dtype=torch.double))
    assert affinity[0].max() < 1


def test_n0_nearest_positive_rho_kth_sigma_and_fuzzy_union():
    y = torch.tensor([0.0, 1.0, 3.0, 10.0], dtype=torch.double)
    affinity = neighborhood_affinity(y, "N0", k=2)
    # 0 -> 3: rho=1,sigma=3; 3 -> 0: rho=2,sigma=3.
    a, b = math.exp(-2 / 3), math.exp(-1 / 3)
    assert affinity[0, 2] == pytest.approx(a + b - a * b)
    assert affinity[0, 3] == 0
    assert affinity[0, 1] == 1
    assert torch.equal(affinity, affinity.T)


def test_n0_all_equal_and_duplicate_labels_are_finite():
    affinity = neighborhood_affinity(torch.ones(12), "N0")
    assert torch.isfinite(affinity).all()
    assert torch.equal(affinity, affinity.T)
    assert affinity.min() >= 0 and affinity.max() <= 1
    assert torch.equal(affinity.diag(), torch.zeros(12))
    duplicate = neighborhood_affinity(torch.tensor([0.0, 0.0, 0.1]), "N0", k=1)
    assert duplicate[0, 1] == 1
    assert torch.isfinite(duplicate).all()


def test_no_qualified_neighbors_still_has_regression_and_negative_pairs():
    e = torch.tensor([[0.0, 0.0], [0.2, 0.4]], dtype=torch.double, requires_grad=True)
    p = torch.tensor([0.0, 0.0], dtype=torch.double, requires_grad=True)
    y = torch.tensor([0.6, 0.8], dtype=torch.double)
    result = metric_loss(e, p, y, neighborhood_affinity(y, "N2", epsilon=0.05))
    assert result["regression"] > 0
    assert result["neighborhood"] > 0
    result["total"].backward()
    assert torch.isfinite(e.grad).all()
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


def test_single_observation_has_zero_pair_loss_and_active_regression():
    e = torch.ones(1, 8, requires_grad=True)
    p = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([1.0])
    result = metric_loss(e, p, y, neighborhood_affinity(y))
    assert result["neighborhood"] == 0
    assert result["regression"] == 1
    result["total"].backward()
    assert torch.isfinite(e.grad).all() and p.grad.item() != 0


def test_negative_pair_loss_gradient_separates_points():
    e = torch.tensor([[-0.2], [0.2]], dtype=torch.double, requires_grad=True)
    result = metric_loss(e, torch.zeros(2), torch.zeros(2), torch.zeros(2, 2), lambda_f=1, lambda_e=0)
    result["total"].backward()
    before = (e[0] - e[1]).square().sum()
    updated = e.detach() - 0.01 * e.grad
    assert (updated[0] - updated[1]).square().sum() > before


def test_representation_constraint_is_mean_squared_norm():
    e = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.double, requires_grad=True)
    result = metric_loss(e, torch.zeros(2), torch.zeros(2), torch.zeros(2, 2), lambda_f=0, lambda_e=0.001)
    assert result["scale"] == 12.5
    assert result["total"].detach().item() == pytest.approx(0.0125)
    result["total"].backward()
    assert torch.allclose(e.grad, e.detach() * 0.001)


def test_pair_weights_match_weighted_mean_and_ignore_zero_weight():
    e = torch.tensor([[0.0], [0.2], [0.7]], dtype=torch.double)
    affinity = torch.zeros(3, 3, dtype=torch.double)
    result = metric_loss(e, torch.zeros(3), torch.zeros(3), affinity, pair_weights=torch.tensor([1.0, 0.0, 0.0]))
    q = 1 / (1 + 0.2 ** 2)
    assert result["neighborhood"] == pytest.approx(-math.log(1 - q))
    zero = metric_loss(e, torch.zeros(3), torch.zeros(3), affinity, pair_weights=torch.zeros(3))
    assert zero["neighborhood"] == 0


@pytest.mark.parametrize("mode", ["N0", "N1", "N2"])
def test_joint_mapping_finite_gradients_and_frozen_vae_boundary(mode):
    generator = torch.Generator().manual_seed(20260909)
    vae = torch.nn.Linear(6, 12).double()
    original_vae = {name: tensor.detach().clone() for name, tensor in vae.state_dict().items()}
    inputs = torch.randn(15, 6, generator=generator, dtype=torch.double)
    z = vae(inputs)
    hp = torch.randn(15, 4, generator=generator, dtype=torch.double)
    y = torch.linspace(0.65, 0.85, 15, dtype=torch.double)
    model = JointMetricModel(12, seed=7).double()
    embedding, prediction = model(z, hp)
    assert embedding.shape == (15, 8) and prediction.shape == (15,)
    assert embedding.std() > 0
    loss = metric_loss(embedding, prediction, (y - y.mean()) / y.std(), neighborhood_affinity(y, mode))
    assert all(torch.isfinite(value) for value in loss.values())
    loss["total"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in vae.parameters())
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    assert all(torch.equal(value, original_vae[name]) for name, value in vae.state_dict().items())


def test_seed_is_reproducible_without_changing_callers_rng():
    state = torch.random.get_rng_state().clone()
    first = JointMetricModel(12, seed=17)
    assert torch.equal(state, torch.random.get_rng_state())
    second = JointMetricModel(12, seed=17)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


def test_colliding_embeddings_keep_bce_and_gradients_finite():
    e = torch.zeros(3, 8, requires_grad=True)
    p = torch.zeros(3, requires_grad=True)
    result = metric_loss(e, p, torch.ones(3), torch.zeros(3, 3))
    result["total"].backward()
    assert torch.isfinite(result["total"])
    assert torch.isfinite(e.grad).all() and torch.isfinite(p.grad).all()
