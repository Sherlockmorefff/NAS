"""Weighted diagonal Gaussian mixture initialization for NAS search.

This module is intentionally independent of sklearn. It provides a small,
deterministic, numpy-only diagonal-covariance GMM implementation plus helpers
for loading Phase3/Phase4 history vectors and sampling warm-start candidates.

Sampling order follows the blueprint:

1. Read X_raw, y from history.
2. Filter by hp_mode/search_dim.
3. Keep the top fraction by y.
4. Standardize top X.
5. Fit WeightedDiagonalGMM in standardized space.
6. Sample in standardized space.
7. Inverse-standardize.
8. Clip arch dimensions to [-z_bound, z_bound] and HP dimensions to [0, 1].
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Optional

import numpy as np

from hp_modes import hp_dim_from_mode, validate_hp_mode


def _log(logger, level: str, message: str) -> None:
    if logger is None:
        print(message)
        return
    fn = getattr(logger, level, None)
    if fn is None:
        print(message)
    else:
        fn(message)


def _logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    return X


def _finite_sample_weight(sample_weight: Optional[np.ndarray], n: int) -> np.ndarray:
    if sample_weight is None:
        weight = np.ones(n, dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weight.shape[0] != n:
            raise ValueError(f"sample_weight length {weight.shape[0]} != n_samples {n}")
        weight = np.where(np.isfinite(weight), weight, 0.0)
        weight = np.maximum(weight, 0.0)
        if float(weight.sum()) <= 0.0:
            weight = np.ones(n, dtype=np.float64)
    return weight


class WeightedDiagonalGMM:
    def __init__(
        self,
        n_components: int = 4,
        max_iter: int = 100,
        tol: float = 1e-4,
        var_floor: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int = 42,
    ):
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if var_floor <= 0:
            raise ValueError("var_floor must be positive")
        if reg_covar < 0:
            raise ValueError("reg_covar must be non-negative")

        self.requested_n_components = int(n_components)
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.var_floor = float(var_floor)
        self.reg_covar = float(reg_covar)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf

    def _weighted_unique_indices(self, sample_weight: np.ndarray, n_components: int) -> np.ndarray:
        n = sample_weight.shape[0]
        positive = np.flatnonzero(sample_weight > 0.0)
        chosen: list[int] = []

        if positive.size > 0:
            probs = sample_weight[positive].astype(np.float64)
            probs = probs / probs.sum()
            take = min(n_components, positive.size)
            chosen.extend(self.rng.choice(positive, size=take, replace=False, p=probs).tolist())

        if len(chosen) < n_components:
            remaining = np.array([idx for idx in range(n) if idx not in set(chosen)], dtype=int)
            if remaining.size == 0:
                remaining = np.arange(n, dtype=int)
            take = n_components - len(chosen)
            replace = remaining.size < take
            chosen.extend(self.rng.choice(remaining, size=take, replace=replace).tolist())

        return np.asarray(chosen[:n_components], dtype=int)

    def _estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.vars_ is None:
            raise RuntimeError("GMM is not fitted")
        X = _as_2d_float(X)
        diff = X[:, None, :] - self.means_[None, :, :]
        log_det = np.sum(np.log(self.vars_), axis=1)
        quad = np.sum((diff * diff) / self.vars_[None, :, :], axis=2)
        d = X.shape[1]
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det[None, :] + quad)

    def _estimate_log_weights(self) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("GMM is not fitted")
        return np.log(np.maximum(self.weights_, 1e-300))

    def fit(self, X: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = _as_2d_float(X)
        finite_rows = np.isfinite(X).all(axis=1)
        if not finite_rows.all():
            X = X[finite_rows]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)[finite_rows]
        if X.shape[0] == 0:
            raise ValueError("No finite rows available for GMM fit")

        n, d = X.shape
        self.n_components = min(self.requested_n_components, n)
        k = self.n_components

        sw = _finite_sample_weight(sample_weight, n)
        sw_sum = float(sw.sum())
        if sw_sum <= 0.0:
            sw = np.ones(n, dtype=np.float64)
            sw_sum = float(n)

        weighted_mean = (sw[:, None] * X).sum(axis=0) / sw_sum
        weighted_var = (sw[:, None] * (X - weighted_mean) ** 2).sum(axis=0) / sw_sum
        weighted_var = np.maximum(weighted_var + self.reg_covar, self.var_floor)

        init_idx = self._weighted_unique_indices(sw, k)
        self.means_ = X[init_idx].copy()
        self.vars_ = np.tile(weighted_var[None, :], (k, 1))
        self.weights_ = np.full(k, 1.0 / k, dtype=np.float64)

        prev_lower = -np.inf
        eps = 1e-12

        for iteration in range(1, self.max_iter + 1):
            log_prob = self._estimate_log_prob(X)
            log_joint = log_prob + self._estimate_log_weights()[None, :]
            log_norm = _logsumexp(log_joint, axis=1)
            resp = np.exp(log_joint - log_norm[:, None])
            resp = np.where(np.isfinite(resp), resp, 0.0)
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp = resp / np.maximum(resp_sum, eps)

            weighted_resp = resp * sw[:, None]
            nk = weighted_resp.sum(axis=0)

            collapsed = nk <= eps
            reinit_idx = None
            if collapsed.any():
                reinit_idx = self._weighted_unique_indices(sw, int(collapsed.sum()))
                nk[collapsed] = eps

            self.weights_ = nk / max(float(nk.sum()), eps)
            self.weights_ = np.maximum(self.weights_, eps)
            self.weights_ = self.weights_ / self.weights_.sum()

            self.means_ = (weighted_resp.T @ X) / nk[:, None]
            second_moment = (weighted_resp.T @ (X * X)) / nk[:, None]
            self.vars_ = np.maximum(second_moment - self.means_ * self.means_ + self.reg_covar, self.var_floor)

            if collapsed.any() and reinit_idx is not None:
                self.means_[collapsed] = X[reinit_idx]
                self.vars_[collapsed] = weighted_var

            self.means_ = np.where(np.isfinite(self.means_), self.means_, weighted_mean[None, :])
            self.vars_ = np.where(np.isfinite(self.vars_), self.vars_, weighted_var[None, :])
            self.vars_ = np.maximum(self.vars_, self.var_floor)

            log_prob = self._estimate_log_prob(X)
            log_joint = log_prob + self._estimate_log_weights()[None, :]
            lower = float(np.sum(sw * _logsumexp(log_joint, axis=1)) / sw_sum)

            self.n_iter_ = iteration
            self.lower_bound_ = lower
            if iteration > 1 and abs(lower - prev_lower) < self.tol:
                self.converged_ = True
                break
            prev_lower = lower

        if not self.converged_:
            self.converged_ = self.n_iter_ < self.max_iter

        assert self.weights_ is not None
        assert self.means_ is not None
        assert self.vars_ is not None
        self.weights_ = np.nan_to_num(self.weights_, nan=1.0 / k, posinf=1.0 / k, neginf=1.0 / k)
        self.weights_ = np.maximum(self.weights_, eps)
        self.weights_ = self.weights_ / self.weights_.sum()
        self.means_ = np.nan_to_num(self.means_, nan=0.0, posinf=0.0, neginf=0.0)
        self.vars_ = np.maximum(np.nan_to_num(self.vars_, nan=self.var_floor, posinf=self.var_floor, neginf=self.var_floor), self.var_floor)
        return self

    def sample(self, n_samples: int, std_scale: float = 1.0) -> np.ndarray:
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if self.weights_ is None or self.means_ is None or self.vars_ is None:
            raise RuntimeError("GMM must be fitted before sampling")
        if n_samples == 0:
            return np.empty((0, self.means_.shape[1]), dtype=np.float64)

        std_scale = float(std_scale)
        if not math.isfinite(std_scale) or std_scale < 0.0:
            raise ValueError("std_scale must be finite and non-negative")

        comp = self.rng.choice(self.n_components, size=n_samples, p=self.weights_)
        mean = self.means_[comp]
        std = np.sqrt(np.maximum(self.vars_[comp], self.var_floor)) * std_scale
        samples = mean + self.rng.normal(size=mean.shape) * std
        return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return deterministic posterior component responsibilities."""

        X = _as_2d_float(X)
        log_joint = self._estimate_log_prob(X) + self._estimate_log_weights()[None, :]
        log_norm = _logsumexp(log_joint, axis=1, keepdims=True)
        responsibilities = np.exp(log_joint - log_norm)
        responsibilities = np.where(np.isfinite(responsibilities), responsibilities, 0.0)
        row_sum = responsibilities.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0.0):
            raise RuntimeError("GMM produced an empty responsibility row")
        responsibilities = responsibilities / row_sum
        if not np.isfinite(responsibilities).all():
            raise RuntimeError("GMM produced non-finite responsibilities")
        return responsibilities


def softmax_weights(y: np.ndarray, temperature: float = 8.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.empty((0,), dtype=np.float64)

    finite = np.isfinite(y)
    if not finite.any():
        return np.full(y.size, 1.0 / y.size, dtype=np.float64)

    y_clean = np.where(finite, y, np.min(y[finite])).astype(np.float64, copy=False)
    y_min = float(np.min(y_clean))
    y_max = float(np.max(y_clean))
    if y_max > y_min:
        y_norm = (y_clean - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(y_clean, dtype=np.float64)

    try:
        temp = float(temperature)
    except (TypeError, ValueError):
        temp = 1.0
    if not math.isfinite(temp) or temp <= 0.0:
        temp = 1.0

    logits = temp * y_norm
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(y.size, 1.0 / y.size, dtype=np.float64)

    weights = weights / total
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(y.size, 1.0 / y.size, dtype=np.float64)
    return weights / total


def standardize_fit(X: np.ndarray):
    X = _as_2d_float(X)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    X_std = standardize_apply(X, mean, std)
    return X_std, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = _as_2d_float(X)
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(std, dtype=np.float64).reshape(1, -1)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def standardize_inverse(X_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X_std = _as_2d_float(X_std)
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(std, dtype=np.float64).reshape(1, -1)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return np.nan_to_num(X_std * std + mean, nan=0.0, posinf=0.0, neginf=0.0)


def _expand_history_paths(paths) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    expanded: list[str] = []
    for path in paths:
        if path is None:
            continue
        path = os.fspath(path)
        matches = glob.glob(path)
        if not matches:
            matches = [path]
        for match in matches:
            if os.path.isdir(match):
                expanded.extend(sorted(glob.glob(os.path.join(match, "*.json"))))
            else:
                expanded.append(match)
    return list(dict.fromkeys(expanded))


def _history_records_from_json(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("history", "records", "trials", "results"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
        if any(key in obj for key in ("z_search", "val_acc", "value", "best_value")):
            return [obj]
    return []


def _value_from_record(record: dict):
    for key in ("val_acc", "value", "best_value"):
        if key in record:
            try:
                value = float(record[key])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def _infer_mode_from_search_dim(dim: int, logger) -> Optional[str]:
    mapping = {
        16: "global4",
        19: "hybrid_cond7",
        31: "layer_cond19",
    }
    if dim == 29:
        _log(logger, "warning", "Skipping legacy search_dim=29 layer_cond17 history; not a main hp_mode.")
        return None
    mode = mapping.get(int(dim))
    if mode is None:
        _log(logger, "warning", f"Skipping history row with unknown search_dim={dim}.")
        return None
    _log(logger, "warning", f"History row missing hp_mode; inferred {mode} from search_dim={dim}.")
    return mode


def load_history_vectors(paths, hp_mode, search_dim, logger=None):
    """Load valid z_search vectors and objective values from history JSON files.

    Only rows satisfying all of the following are retained:
        valid is True
        finite val_acc/value/best_value
        hp_mode matches the requested hp_mode
        z_search exists and has length search_dim
    """

    hp_mode = validate_hp_mode(hp_mode)
    search_dim = int(search_dim)
    history_paths = _expand_history_paths(paths)

    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    n_loaded = 0
    n_bad_json = 0
    n_missing_path = 0
    n_skipped = 0

    for path in history_paths:
        if not os.path.exists(path):
            n_missing_path += 1
            _log(logger, "warning", f"History path not found: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as exc:
            n_bad_json += 1
            _log(logger, "warning", f"Failed to read history JSON {path}: {exc}")
            continue

        for record in _history_records_from_json(obj):
            if not isinstance(record, dict):
                n_skipped += 1
                continue
            n_loaded += 1
            if record.get("valid") is not True:
                n_skipped += 1
                continue

            y = _value_from_record(record)
            if y is None:
                n_skipped += 1
                continue

            z_search = record.get("z_search")
            if z_search is None:
                n_skipped += 1
                continue
            try:
                z = np.asarray(z_search, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                n_skipped += 1
                continue

            row_mode = record.get("hp_mode")
            if row_mode is None:
                row_search_dim = int(record.get("search_dim", z.shape[0]))
                row_mode = _infer_mode_from_search_dim(row_search_dim, logger)
                if row_mode is None:
                    n_skipped += 1
                    continue

            if z.shape[0] != search_dim or not np.isfinite(z).all():
                n_skipped += 1
                continue

            if row_mode != hp_mode:
                n_skipped += 1
                continue

            X_rows.append(z.tolist())
            y_rows.append(float(y))

    if X_rows:
        X = np.asarray(X_rows, dtype=np.float64)
        y = np.asarray(y_rows, dtype=np.float64)
    else:
        X = np.empty((0, search_dim), dtype=np.float64)
        y = np.empty((0,), dtype=np.float64)

    meta = {
        "hp_mode": hp_mode,
        "search_dim": search_dim,
        "n_loaded": n_loaded,
        "n_valid": int(X.shape[0]),
        "n_skipped": n_skipped,
        "n_bad_json": n_bad_json,
        "n_missing_path": n_missing_path,
        "history_paths": history_paths,
    }
    _log(
        logger,
        "info",
        f"Loaded GMM history vectors: n_loaded={n_loaded}, n_valid={X.shape[0]}, hp_mode={hp_mode}",
    )
    return X, y, meta


def _get_arg(args, name: str, default):
    return getattr(args, name, default) if args is not None else default


def fit_and_sample_gmm_init(X, y, hp_mode, args, logger=None):
    """Fit weighted diagonal GMM on top history vectors and sample candidates."""

    hp_mode = validate_hp_mode(hp_mode)
    X = _as_2d_float(X)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != y length {y.shape[0]}")

    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[finite]
    y = y[finite]

    search_dim = int(X.shape[1]) if X.ndim == 2 else int(_get_arg(args, "search_dim", 0))
    hp_dim = hp_dim_from_mode(hp_mode)
    arch_nz = int(_get_arg(args, "arch_nz", search_dim - hp_dim))
    z_bound = float(_get_arg(args, "z_bound", 2.5))
    gmm_init_trials = int(_get_arg(args, "gmm_init_trials", 0))
    top_frac = float(_get_arg(args, "gmm_top_frac", 0.3))
    n_components = int(_get_arg(args, "gmm_n_components", 4))
    weight_temp = float(_get_arg(args, "gmm_weight_temp", 8.0))
    min_samples = int(_get_arg(args, "gmm_min_samples", 20))
    var_floor = float(_get_arg(args, "gmm_var_floor", 1e-4))
    sample_std_scale = float(_get_arg(args, "gmm_sample_std_scale", 1.0))
    random_state = int(_get_arg(args, "seed", 42))
    history_paths = list(_get_arg(args, "gmm_init_history", []) or [])

    n_valid = int(X.shape[0])
    summary = {
        "hp_mode": hp_mode,
        "search_dim": search_dim,
        "n_loaded": n_valid,
        "n_valid": n_valid,
        "n_used_top": 0,
        "n_components": 0,
        "top_frac": top_frac,
        "y_min": float(np.min(y)) if y.size else None,
        "y_max": float(np.max(y)) if y.size else None,
        "y_mean": float(np.mean(y)) if y.size else None,
        "sampled_count": 0,
        "history_paths": history_paths,
    }

    if gmm_init_trials <= 0:
        _log(logger, "info", "Skipping GMM init: gmm_init_trials <= 0.")
        return [], summary
    if n_valid < min_samples:
        _log(logger, "warning", f"Skipping GMM init: n_valid={n_valid} < gmm_min_samples={min_samples}.")
        return [], summary
    if search_dim <= 0:
        _log(logger, "warning", "Skipping GMM init: empty search_dim.")
        return [], summary

    top_frac = min(max(top_frac, 1e-6), 1.0)
    n_top = max(1, int(math.ceil(n_valid * top_frac)))
    order = np.argsort(y)[::-1]
    top_idx = order[:n_top]
    X_top = X[top_idx]
    y_top = y[top_idx]

    X_std, mean, std = standardize_fit(X_top)
    weights = softmax_weights(y_top, temperature=weight_temp)

    gmm = WeightedDiagonalGMM(
        n_components=n_components,
        max_iter=int(_get_arg(args, "gmm_max_iter", 100)),
        tol=float(_get_arg(args, "gmm_tol", 1e-4)),
        var_floor=var_floor,
        reg_covar=float(_get_arg(args, "gmm_reg_covar", 1e-6)),
        random_state=random_state,
    ).fit(X_std, sample_weight=weights)

    sampled_std = gmm.sample(gmm_init_trials, std_scale=sample_std_scale)
    sampled_raw = standardize_inverse(sampled_std, mean, std)

    if arch_nz < 0 or arch_nz > search_dim:
        raise ValueError(f"Invalid arch_nz={arch_nz} for search_dim={search_dim}")
    sampled_raw[:, :arch_nz] = np.clip(sampled_raw[:, :arch_nz], -z_bound, z_bound)
    sampled_raw[:, arch_nz:] = np.clip(sampled_raw[:, arch_nz:], 0.0, 1.0)
    sampled_raw = np.nan_to_num(sampled_raw, nan=0.0, posinf=0.0, neginf=0.0)

    sampled_z_list = [row.astype(np.float64, copy=True) for row in sampled_raw]
    summary.update(
        {
            "n_used_top": int(n_top),
            "n_components": int(gmm.n_components),
            "sampled_count": int(len(sampled_z_list)),
        }
    )
    _log(
        logger,
        "info",
        f"GMM init sampled {len(sampled_z_list)} candidates from top {n_top}/{n_valid} history rows.",
    )
    return sampled_z_list, summary


def _self_test() -> None:
    y = np.array([0.7, 0.8, 0.9])
    w = softmax_weights(y, temperature=8.0)
    assert w.shape == (3,)
    assert np.isfinite(w).all()
    assert abs(w.sum() - 1.0) < 1e-8
    assert w[2] > w[1] > w[0]

    y = np.array([0.8, 0.8, 0.8])
    w = softmax_weights(y, temperature=8.0)
    assert np.allclose(w, np.ones(3) / 3)

    y = np.array([0.7, np.nan, np.inf, 0.9])
    w = softmax_weights(y, temperature=8.0)
    assert w.shape == (4,)
    assert np.isfinite(w).all()
    assert abs(w.sum() - 1.0) < 1e-8

    y = np.array([])
    w = softmax_weights(y)
    assert w.shape == (0,)

    rng = np.random.default_rng(123)
    X = rng.normal(size=(64, 16))
    X[:, 12:] = np.clip(rng.random(size=(64, 4)), 0.0, 1.0)
    y = 0.5 + 0.1 * rng.normal(size=64) + 0.2 * X[:, 0]
    weights = softmax_weights(y, temperature=8.0)
    X_std, mean, std = standardize_fit(X)
    gmm = WeightedDiagonalGMM(n_components=4, random_state=7).fit(X_std, sample_weight=weights)
    samples_std = gmm.sample(10)
    samples = standardize_inverse(samples_std, mean, std)
    assert samples.shape == (10, X.shape[1])
    assert np.isfinite(samples).all()

    class Args:
        arch_nz = 12
        z_bound = 2.5
        gmm_init_trials = 8
        gmm_top_frac = 0.3
        gmm_n_components = 4
        gmm_weight_temp = 8.0
        gmm_min_samples = 20
        gmm_var_floor = 1e-4
        gmm_sample_std_scale = 1.0
        seed = 42
        gmm_init_history = []

    sampled, summary = fit_and_sample_gmm_init(X, y, "global4", Args(), logger=None)
    assert len(sampled) == 8
    assert summary["sampled_count"] == 8
    arr = np.asarray(sampled)
    assert arr.shape == (8, 16)
    assert np.isfinite(arr).all()
    assert np.all(arr[:, :12] <= Args.z_bound)
    assert np.all(arr[:, :12] >= -Args.z_bound)
    assert np.all(arr[:, 12:] >= 0.0)
    assert np.all(arr[:, 12:] <= 1.0)
    print("weighted_diag_gmm_init OK")


if __name__ == "__main__":
    _self_test()
