"""Discrete LayerNorm search-space identity and initialization helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from hp_modes import hp_dim_from_mode, validate_hp_mode
from initialization_wgmm_ted import candidate_fingerprint
from eval_utils import stable_seed


NORMALIZATION_NONE = "none"
NORMALIZATION_LAYER_NORM = "layer_norm"
NORMALIZATION_VALUES = {
    NORMALIZATION_NONE: 0,
    NORMALIZATION_LAYER_NORM: 1,
}
NORMALIZATION_SPACES = ("legacy", "none", "none_layernorm")
TRAINING_SEED_SCHEME = (
    "stable_seed('layernorm_search_training_v1', base_candidate_fingerprint, "
    "replicate_id)"
)


def has_normalization_dimension(normalization_space: str) -> bool:
    if normalization_space not in NORMALIZATION_SPACES:
        raise ValueError(
            f"normalization_space must be one of {NORMALIZATION_SPACES}, "
            f"got {normalization_space!r}"
        )
    return normalization_space == "none_layernorm"


def base_search_dim(arch_nz: int, hp_mode: str) -> int:
    return int(arch_nz) + int(hp_dim_from_mode(validate_hp_mode(hp_mode)))


def gp_search_dim(arch_nz: int, hp_mode: str, normalization_space: str) -> int:
    return base_search_dim(arch_nz, hp_mode) + int(
        has_normalization_dimension(normalization_space)
    )


def normalization_name(value: Any) -> str:
    if isinstance(value, str):
        if value not in NORMALIZATION_VALUES:
            raise ValueError("normalization must be exactly 'none' or 'layer_norm'")
        return value
    if isinstance(value, bool):
        raise ValueError("normalization must be exactly integer 0 or 1")
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return NORMALIZATION_NONE if int(value) == 0 else NORMALIZATION_LAYER_NORM
    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return NORMALIZATION_NONE if float(value) == 0.0 else NORMALIZATION_LAYER_NORM
    raise ValueError("normalization must be exactly 0/1 or none/layer_norm")


def normalization_value(value: Any) -> int:
    return NORMALIZATION_VALUES[normalization_name(value)]


def split_search_vector(
    vector: Any,
    *,
    arch_nz: int,
    hp_mode: str,
    normalization_space: str,
) -> tuple[np.ndarray, str]:
    raw = np.asarray(vector, dtype=np.float32).reshape(-1)
    expected_base = base_search_dim(arch_nz, hp_mode)
    expanded = has_normalization_dimension(normalization_space)
    expected = expected_base + int(expanded)
    if raw.shape != (expected,):
        raise ValueError(
            f"search vector must have length {expected} for "
            f"normalization_space={normalization_space}, got {raw.shape[0]}"
        )
    if not np.isfinite(raw).all():
        raise ValueError("search vector contains NaN or infinite values")
    if not expanded:
        return raw.copy(), NORMALIZATION_NONE
    name = normalization_name(float(raw[-1]))
    return raw[:-1].copy(), name


def compose_search_vector(
    base_vector: Any,
    normalization: Any,
    *,
    arch_nz: int,
    hp_mode: str,
    normalization_space: str,
) -> np.ndarray:
    base = np.asarray(base_vector, dtype=np.float32).reshape(-1)
    expected = base_search_dim(arch_nz, hp_mode)
    if base.shape != (expected,):
        raise ValueError(f"base search vector must have length {expected}")
    name = normalization_name(normalization)
    if not has_normalization_dimension(normalization_space):
        if name != NORMALIZATION_NONE:
            raise ValueError("baseline normalization_space only permits none")
        return base.copy()
    return np.concatenate(
        (base, np.asarray([normalization_value(name)], dtype=np.float32))
    )


def base_candidate_fingerprint(base_vector: Any) -> str:
    return candidate_fingerprint(np.asarray(base_vector, dtype=np.float32).reshape(-1))


def full_candidate_fingerprint(base_fingerprint: str, normalization: Any) -> str:
    if not isinstance(base_fingerprint, str) or len(base_fingerprint) != 64:
        raise ValueError("base_candidate_fingerprint must be a SHA-256 hex string")
    try:
        int(base_fingerprint, 16)
    except ValueError as exc:
        raise ValueError("base_candidate_fingerprint must be a SHA-256 hex string") from exc
    payload = json.dumps(
        {
            "base_candidate_fingerprint": base_fingerprint,
            "normalization": normalization_name(normalization),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_identity(base_vector: Any, normalization: Any) -> dict[str, Any]:
    base = base_candidate_fingerprint(base_vector)
    name = normalization_name(normalization)
    return {
        "base_candidate_fingerprint": base,
        "full_candidate_fingerprint": full_candidate_fingerprint(base, name),
        "normalization": name,
        "normalization_value": normalization_value(name),
    }


def matched_training_seed(base_fingerprint: str, replicate_id: int = 0) -> int:
    if isinstance(replicate_id, bool) or int(replicate_id) < 0:
        raise ValueError("replicate_id must be a non-negative integer")
    return stable_seed(
        0,
        "layernorm_search_training_v1",
        str(base_fingerprint),
        int(replicate_id),
    )


def balanced_initialization_assignment(
    base_fingerprints: Sequence[str],
    *,
    search_seed: int,
) -> dict[str, str]:
    fingerprints = [str(value) for value in base_fingerprints]
    if len(fingerprints) == 0 or len(fingerprints) % 2:
        raise ValueError("balanced assignment requires a positive even candidate count")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("balanced assignment requires unique base fingerprints")
    ranked = sorted(
        fingerprints,
        key=lambda fingerprint: (
            stable_seed(
                int(search_seed),
                "layernorm_initial_assignment_v1",
                fingerprint,
            ),
            fingerprint,
        ),
    )
    layer_norm = set(ranked[: len(ranked) // 2])
    return {
        fingerprint: (
            NORMALIZATION_LAYER_NORM
            if fingerprint in layer_norm
            else NORMALIZATION_NONE
        )
        for fingerprint in fingerprints
    }


def top_normalization_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    valid = [row for row in records if bool(row.get("valid", True))]
    ordered = sorted(
        valid,
        key=lambda row: (
            -float(row["val_acc"]),
            str(row["full_candidate_fingerprint"]),
        ),
    )
    top = ordered[: min(int(top_k), len(ordered))]
    total_ln = sum(normalization_name(row["normalization"]) == NORMALIZATION_LAYER_NORM for row in valid)
    top_ln = sum(normalization_name(row["normalization"]) == NORMALIZATION_LAYER_NORM for row in top)
    return {
        "candidate_count": len(valid),
        "layernorm_count": total_ln,
        "layernorm_fraction": None if not valid else total_ln / len(valid),
        "top_k": len(top),
        "top_layernorm_count": top_ln,
        "top_layernorm_fraction": None if not top else top_ln / len(top),
    }


def validate_resume_normalization_space(
    saved: Mapping[str, Any], expected_space: str
) -> None:
    has_normalization_dimension(expected_space)
    actual = saved.get("normalization_space", "legacy")
    if actual != expected_space:
        raise ValueError(
            "resume normalization_space mismatch: "
            f"expected {expected_space!r}, got {actual!r}"
        )
