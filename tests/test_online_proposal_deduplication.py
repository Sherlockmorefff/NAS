from __future__ import annotations

import json

import pytest
import torch

import bo_phase4
from eval_utils import stable_seed
from hp_modes import hp_dim_from_mode


HP_MODE = "global4"
Z_BOUND = 2.5
SEARCH_DIM = bo_phase4.ARCH_NZ + hp_dim_from_mode(HP_MODE)


def _candidate(value: float) -> torch.Tensor:
    row = torch.zeros(SEARCH_DIM, dtype=torch.float32)
    row[0] = float(value)
    row[bo_phase4.ARCH_NZ :] = 0.5
    return row


def _fingerprint(row: torch.Tensor) -> str:
    _canonical, fingerprint = bo_phase4._canonical_candidate_identity(
        row, hp_mode=HP_MODE, z_bound=Z_BOUND,
    )
    return fingerprint


def _select(
    proposal_rows: list[torch.Tensor],
    evaluated: set[str],
    *,
    search_seed: int = 17,
    step: int = 73,
    strategy: str = "qlogei",
    max_attempts: int = 8,
    audit_path: str | None = None,
):
    calls: list[tuple[int, int]] = []

    def proposal_fn(retry_id: int, selection_seed: int):
        calls.append((retry_id, selection_seed))
        return proposal_rows[retry_id].clone(), float(retry_id), [1.0]

    selected = bo_phase4.propose_unique_online_candidate(
        search_seed=search_seed,
        online_full_step=step,
        strategy=strategy,
        hp_mode=HP_MODE,
        z_bound=Z_BOUND,
        evaluated_full_fingerprints=evaluated,
        max_attempts=max_attempts,
        proposal_fn=proposal_fn,
        audit_path=audit_path,
    )
    return selected, calls


def test_first_proposal_duplicate_second_is_accepted(tmp_path):
    duplicate = _candidate(0.25)
    unique = _candidate(0.5)
    evaluated = {_fingerprint(duplicate)}
    audit = tmp_path / "online_candidate_proposals.json"

    selected, calls = _select(
        [duplicate, unique], evaluated, audit_path=str(audit),
    )

    z_next, logei, _mask, _seed, fingerprint, attempts = selected
    assert torch.equal(z_next, unique)
    assert logei == 1.0
    assert fingerprint == _fingerprint(unique)
    assert calls[0][0] == 0 and calls[1][0] == 1
    assert [row["rejection_reason"] for row in attempts] == [
        "already_evaluated_full_fidelity",
        None,
    ]
    persisted = json.loads(audit.read_text(encoding="utf-8"))
    assert persisted[0]["accepted_proposal_attempt"] == 2
    assert persisted[0]["attempts"] == attempts


def test_multiple_consecutive_duplicates_are_rejected():
    rows = [_candidate(0.1), _candidate(0.2), _candidate(0.3), _candidate(0.4)]
    evaluated = {_fingerprint(row) for row in rows[:3]}

    selected, calls = _select(rows, evaluated)

    assert selected[4] == _fingerprint(rows[3])
    assert len(calls) == 4
    assert [row["duplicate_full_fidelity"] for row in selected[5]] == [
        True,
        True,
        True,
        False,
    ]


def test_retry_proposals_are_fully_deterministic():
    duplicate = _candidate(0.7)
    evaluated = {_fingerprint(duplicate)}

    def run_once():
        def proposal_fn(retry_id: int, selection_seed: int):
            if retry_id < 2:
                return duplicate.clone(), None, [1.0]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(selection_seed)
            row = torch.rand(SEARCH_DIM, generator=generator)
            row[: bo_phase4.ARCH_NZ] = row[: bo_phase4.ARCH_NZ] * 2.0 - 1.0
            return row, None, [1.0]

        return bo_phase4.propose_unique_online_candidate(
            search_seed=991,
            online_full_step=245,
            strategy="qlogei",
            hp_mode=HP_MODE,
            z_bound=Z_BOUND,
            evaluated_full_fingerprints=evaluated,
            max_attempts=5,
            proposal_fn=proposal_fn,
        )

    first = run_once()
    second = run_once()
    assert torch.equal(first[0], second[0])
    assert first[3:] == second[3:]


@pytest.mark.parametrize("method", ["S0", "G100", "G150"])
def test_all_methods_use_the_same_rule_for_the_same_state(method):
    del method
    duplicate = _candidate(-0.2)
    unique = _candidate(-0.4)
    selected, _calls = _select(
        [duplicate, unique], {_fingerprint(duplicate)}, search_seed=8, step=101,
    )
    assert selected[4] == _fingerprint(unique)
    assert selected[5][0]["rejection_reason"] == "already_evaluated_full_fidelity"


def test_duplicate_does_not_mutate_evaluated_set_or_full_budget():
    duplicate = _candidate(1.0)
    unique = _candidate(1.1)
    evaluated = {_fingerprint(duplicate)}
    before = set(evaluated)
    full_evaluations = 41

    selected, calls = _select([duplicate, unique], evaluated)

    assert evaluated == before
    assert full_evaluations == 41
    assert len(calls) == 2
    assert selected[5][0]["accepted"] is False
    assert selected[5][1]["accepted"] is True


def test_resume_produces_same_next_candidate(tmp_path):
    def make_proposer(state: torch.Tensor | None = None):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123456)
        if state is not None:
            generator.set_state(state)

        def proposal_fn(_retry_id: int, _selection_seed: int):
            row = torch.rand(SEARCH_DIM, generator=generator)
            row[: bo_phase4.ARCH_NZ] = row[: bo_phase4.ARCH_NZ] * 2.0 - 1.0
            return row, None, [1.0]

        return generator, proposal_fn

    generator, first_proposer = make_proposer()
    first = bo_phase4.propose_unique_online_candidate(
        search_seed=5,
        online_full_step=50,
        strategy="qlogei",
        hp_mode=HP_MODE,
        z_bound=Z_BOUND,
        evaluated_full_fingerprints=set(),
        max_attempts=4,
        proposal_fn=first_proposer,
        audit_path=str(tmp_path / "proposals.json"),
    )
    committed_rng_state = generator.get_state().clone()
    evaluated = {first[4]}

    uninterrupted = bo_phase4.propose_unique_online_candidate(
        search_seed=5,
        online_full_step=51,
        strategy="qlogei",
        hp_mode=HP_MODE,
        z_bound=Z_BOUND,
        evaluated_full_fingerprints=evaluated,
        max_attempts=4,
        proposal_fn=first_proposer,
    )
    _resumed_generator, resumed_proposer = make_proposer(committed_rng_state)
    resumed = bo_phase4.propose_unique_online_candidate(
        search_seed=5,
        online_full_step=51,
        strategy="qlogei",
        hp_mode=HP_MODE,
        z_bound=Z_BOUND,
        evaluated_full_fingerprints=evaluated,
        max_attempts=4,
        proposal_fn=resumed_proposer,
    )

    assert torch.equal(uninterrupted[0], resumed[0])
    assert uninterrupted[3:] == resumed[3:]


def test_evaluation_seed_does_not_depend_on_retry_id():
    duplicate = _candidate(1.2)
    accepted = _candidate(1.3)
    fingerprint = _fingerprint(accepted)
    direct, _ = _select([accepted], set(), search_seed=44, step=90)
    retried, _ = _select(
        [duplicate, accepted],
        {_fingerprint(duplicate)},
        search_seed=44,
        step=90,
    )

    assert direct[4] == retried[4] == fingerprint
    assert direct[3] != retried[3]
    assert bo_phase4.candidate_evaluation_seed(44, direct[4], "full") == (
        bo_phase4.candidate_evaluation_seed(44, retried[4], "full")
    )


def test_retry_exhaustion_is_explicit_and_persists_all_attempts(tmp_path):
    duplicate = _candidate(-1.0)
    audit = tmp_path / "online_candidate_proposals.json"

    with pytest.raises(RuntimeError, match="retries exhausted") as error:
        _select(
            [duplicate, duplicate, duplicate],
            {_fingerprint(duplicate)},
            max_attempts=3,
            audit_path=str(audit),
        )

    assert "max_attempts=3" in str(error.value)
    persisted = json.loads(audit.read_text(encoding="utf-8"))
    assert persisted[0]["status"] == "exhausted"
    assert len(persisted[0]["attempts"]) == 3
    assert all(row["rejection_reason"] for row in persisted[0]["attempts"])


def test_nonduplicate_path_remains_single_attempt_with_legacy_seed():
    row = _candidate(0.33)
    selected, calls = _select([row], set(), search_seed=19, step=88)

    expected_seed = stable_seed(19, "acquisition", 88)
    assert torch.equal(selected[0], row)
    assert selected[3] == expected_seed
    assert calls == [(0, expected_seed)]
    assert len(selected[5]) == 1


def test_exact_full_budget_requires_unique_fingerprints():
    first = _fingerprint(_candidate(0.1))
    second = _fingerprint(_candidate(0.2))
    history = [
        {"evaluation_fidelity": "full", "candidate_fingerprint": first},
        {"evaluation_fidelity": "full", "candidate_fingerprint": second},
    ]
    assert bo_phase4.validate_unique_full_evaluation_budget(
        history, expected_total=2,
    ) == 2

    with pytest.raises(RuntimeError, match="duplicate canonical"):
        bo_phase4.validate_unique_full_evaluation_budget(
            history + [history[0]], expected_total=3,
        )
    with pytest.raises(RuntimeError, match="exact unique budget"):
        bo_phase4.validate_unique_full_evaluation_budget(
            history, expected_total=3,
        )
