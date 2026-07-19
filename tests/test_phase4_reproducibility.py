from __future__ import annotations

import os
from pathlib import Path
import random
from types import SimpleNamespace
import subprocess
import sys

import pytest


torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("torch_geometric")

import eval_utils


ROOT = Path(__file__).resolve().parents[1]


def _numpy_states_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _rng_states():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_rng_states_equal(left, right) -> None:
    assert left[0] == right[0]
    assert _numpy_states_equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


def test_stable_seed_is_repeatable_and_sensitive_to_all_inputs():
    z = np.asarray([0.25, -1.5, 3.0], dtype=np.float32)
    expected = eval_utils.stable_seed(0, "decoder", z.tobytes())

    assert eval_utils.stable_seed(0, "decoder", z.tobytes()) == expected
    assert eval_utils.stable_seed(0, "decoder", z) == expected
    assert eval_utils.stable_seed(0, "decoder", torch.from_numpy(z)) == expected
    assert eval_utils.stable_seed(1, "decoder", z.tobytes()) != expected
    assert eval_utils.stable_seed(0, "candidate_eval", z.tobytes()) != expected
    changed = z.copy()
    changed[1] += np.float32(0.125)
    assert eval_utils.stable_seed(0, "decoder", changed.tobytes()) != expected
    assert 0 <= expected <= 2**32 - 1


def test_stable_seed_is_independent_of_python_hash_randomization():
    script = (
        "import numpy as np; "
        "from eval_utils import stable_seed; "
        "z=np.asarray([0.25,-1.5,3.0],dtype=np.float32); "
        "print(stable_seed(7,'decoder',z.tobytes()))"
    )
    outputs = []
    for hash_seed in ("1", "987654"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())
    assert outputs[0] == outputs[1]


def test_isolated_rng_repeats_sequences_and_restores_outer_state():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    before = _rng_states()

    samples = []
    for _ in range(2):
        with eval_utils.isolated_rng(1234, torch.device("cpu")):
            samples.append(
                (
                    [random.random() for _ in range(3)],
                    np.random.random(3),
                    torch.rand(3),
                )
            )
        _assert_rng_states_equal(_rng_states(), before)

    assert samples[0][0] == samples[1][0]
    assert np.array_equal(samples[0][1], samples[1][1])
    assert torch.equal(samples[0][2], samples[1][2])


def test_isolated_rng_restores_state_after_exception():
    random.seed(22)
    np.random.seed(22)
    torch.manual_seed(22)
    before = _rng_states()

    with pytest.raises(RuntimeError, match="boom"):
        with eval_utils.isolated_rng(99, torch.device("cpu")):
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("boom")

    _assert_rng_states_equal(_rng_states(), before)


class _VertexSeq:
    def __init__(self, types):
        self._types = list(types)

    def __getitem__(self, index):
        return {"type": self._types[index]}


class _FakeGraph:
    def __init__(self, types, edges):
        self.vs = _VertexSeq(types)
        self._edges = list(edges)

    def vcount(self):
        return len(self.vs._types)

    def get_edgelist(self):
        return list(self._edges)


class _FakeStochasticDecoder:
    def decode(self, z, stochastic=True):
        assert stochastic is True
        op_type = 2 + int(np.random.randint(0, 2))
        skip_edge = bool(torch.rand(()) < 0.5)
        reverse_edge = random.random() < 0.5
        edges = [(0, 1), (1, 2), (2, 3)]
        if skip_edge:
            edges.append((0, 2))
        if reverse_edge:
            edges.append((1, 3))
        return [_FakeGraph([0, op_type, op_type, 1], edges)]


def test_stochastic_decoder_is_repeatable_and_does_not_leak_rng():
    vae = SimpleNamespace(
        arch_vae=_FakeStochasticDecoder(),
        op_mapping={2: "GCNConv", 3: "GATConv"},
    )
    z = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float32)
    decoder_seed = eval_utils.stable_seed(5, "decoder", z)
    random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    before = _rng_states()

    first = eval_utils._decode_arch(
        vae, z, torch.device("cpu"), n_trials=9, decoder_seed=decoder_seed,
    )
    _assert_rng_states_equal(_rng_states(), before)
    second = eval_utils._decode_arch(
        vae, z, torch.device("cpu"), n_trials=9, decoder_seed=decoder_seed,
    )
    _assert_rng_states_equal(_rng_states(), before)

    assert first["operations"] == second["operations"]
    assert first["edges"] == second["edges"]


def _phase4_args(seed: int = 17):
    return SimpleNamespace(
        seed=seed,
        hp_mode="global4",
        z_bound=4.0,
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        eval_epochs=12,
        patience=4,
    )


def test_candidate_seed_propagation_and_history_metadata(monkeypatch):
    pytest.importorskip("botorch")
    import bo_phase4

    captured = {}

    def fake_eval_z_search(*args, **kwargs):
        captured.update(kwargs)
        return {
            "val_acc": 0.75,
            "valid": True,
            "config": {
                "effective_layers": 1,
                "operations": ["GCNConv"],
                "edges": [(0, 1), (1, 2)],
            },
            "hp": {
                "lr": 0.01,
                "dropout": 0.3,
                "hidden_dim": 64,
                "weight_decay": 5e-4,
                "gat_heads": 1,
                "sage_aggr": "mean",
                "gin_eps": 0.0,
            },
            "decoder_seed": kwargs["decoder_seed"],
            "candidate_eval_seed": kwargs["candidate_eval_seed"],
            "best_epoch": 6,
            "stopped_epoch": 10,
            "epochs_ran": 10,
        }

    monkeypatch.setattr(bo_phase4, "eval_z_search", fake_eval_z_search)
    args = _phase4_args(seed=17)
    z = torch.linspace(-0.8, 0.8, bo_phase4.ARCH_NZ + 4)

    z_tensor, result = bo_phase4.eval_candidate(
        object(), z, object(), 8, 3, args, torch.device("cpu"), step=9,
    )

    expected_decoder = eval_utils.stable_seed(17, "decoder", z_tensor[: bo_phase4.ARCH_NZ])
    expected_eval = eval_utils.stable_seed(17, "candidate_eval", 9)
    assert captured["decoder_seed"] == expected_decoder
    assert captured["candidate_eval_seed"] == expected_eval

    record = bo_phase4.history_record(9, "bo", z_tensor, result, args)
    assert record["search_seed"] == 17
    assert record["decoder_seed"] == expected_decoder
    assert record["candidate_eval_seed"] == expected_eval
    assert record["seed_derivation"] == "sha256_v1"
    assert record["best_epoch"] == 6
    assert record["stopped_epoch"] == 10
    assert record["epochs_ran"] == 10


def test_eval_z_search_passes_candidate_seed_to_training(monkeypatch):
    captured = {}
    config = {
        "effective_layers": 1,
        "operations": ["GCNConv"],
        "edges": [(0, 1), (1, 2)],
    }
    hp = {
        "lr": 0.01,
        "dropout": 0.3,
        "hidden_dim": 64,
        "weight_decay": 5e-4,
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
        "condition_mask": {},
        "condition_mask_vector": [1.0] * 16,
        "hp_norm": [0.5] * 4,
    }

    monkeypatch.setattr(eval_utils, "_decode_arch", lambda *args, **kwargs: config)
    monkeypatch.setattr(eval_utils, "decode_hp_by_mode", lambda **kwargs: hp)

    def fake_train_and_eval_arch(*args, **kwargs):
        captured.update(kwargs)
        return 0.8, True, {"best_epoch": 4, "stopped_epoch": 7, "epochs_ran": 7}

    monkeypatch.setattr(eval_utils, "train_and_eval_arch", fake_train_and_eval_arch)
    candidate_seed = eval_utils.stable_seed(3, "candidate_eval", 12)
    result = eval_utils.eval_z_search(
        object(),
        torch.zeros(16),
        object(),
        8,
        3,
        arch_nz=12,
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
        device=torch.device("cpu"),
        hp_mode="global4",
        return_hp=True,
        decoder_seed=101,
        candidate_eval_seed=candidate_seed,
    )

    assert captured["seed"] == candidate_seed
    assert captured["return_metadata"] is True
    assert result["candidate_eval_seed"] == candidate_seed
    assert result["best_epoch"] == 4
    assert result["epochs_ran"] == 7


class _TinyGNN(torch.nn.Module):
    def __init__(self, config, in_ch, out_ch, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x, edge_index):
        return self.linear(x)


def _toy_data():
    return SimpleNamespace(
        x=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        y=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, True]),
        test_mask=torch.tensor([False, False, True, True]),
    )


def test_training_metadata_is_opt_in_and_old_return_shape_is_unchanged(monkeypatch):
    monkeypatch.setattr(eval_utils, "DynamicGNN", _TinyGNN)
    kwargs = {
        "config": {
            "operations": ["GCNConv"],
            "effective_layers": 1,
            "edges": [(0, 1), (1, 2)],
        },
        "data": _toy_data(),
        "in_ch": 2,
        "out_ch": 2,
        "device": torch.device("cpu"),
        "max_epochs": 3,
        "patience": 2,
        "seed": 123,
    }

    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    before = _rng_states()
    legacy = eval_utils.train_and_eval_arch(**kwargs)
    _assert_rng_states_equal(_rng_states(), before)
    extended = eval_utils.train_and_eval_arch(**kwargs, return_metadata=True)
    _assert_rng_states_equal(_rng_states(), before)

    assert len(legacy) == 2
    assert extended[:2] == legacy
    metadata = extended[2]
    assert set(metadata) == {"best_epoch", "stopped_epoch", "epochs_ran"}
    assert metadata["epochs_ran"] == metadata["stopped_epoch"]
    assert 1 <= metadata["epochs_ran"] <= 3
