"""Reusable offline-trained and online-updated GP accuracy predictor."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from hp_modes import hp_dim_from_mode, validate_hp_mode
from surrogate.checkpoint_io import atomic_torch_save, torch_load_compat


X_TRANSFORM_VERSION = "fixed_search_domain_v1"
CHECKPOINT_FORMAT_VERSION = 1


def _expected_dim(arch_nz: int, hp_mode: str) -> int:
    return int(arch_nz) + int(hp_dim_from_mode(validate_hp_mode(hp_mode)))


def _as_float_tensor(value: Any) -> tuple[torch.Tensor, bool, np.dtype[Any] | None]:
    if isinstance(value, torch.Tensor):
        tensor = value
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.get_default_dtype())
        return tensor, True, None
    array = np.asarray(value)
    original_dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else None
    tensor = torch.as_tensor(array, dtype=torch.float64 if original_dtype == np.float64 else torch.float32)
    return tensor, False, original_dtype


def _restore_type(tensor: torch.Tensor, was_tensor: bool, dtype: np.dtype[Any] | None) -> Any:
    if was_tensor:
        return tensor
    result = tensor.detach().cpu().numpy()
    return result.astype(dtype, copy=False) if dtype is not None else result


def normalize_search_vector(
    z_search: Any,
    *,
    arch_nz: int,
    hp_mode: str,
    z_bound: float,
) -> Any:
    """Map raw ``[z_arch, hp]`` vectors into the fixed unit search domain."""

    tensor, was_tensor, dtype = _as_float_tensor(z_search)
    expected = _expected_dim(arch_nz, hp_mode)
    if tensor.ndim == 0 or tensor.shape[-1] != expected:
        got = None if tensor.ndim == 0 else tensor.shape[-1]
        raise ValueError(f"z_search last dimension must be {expected} for hp_mode={hp_mode}, got {got}")
    if not math.isfinite(float(z_bound)) or float(z_bound) <= 0.0:
        raise ValueError(f"z_bound must be positive and finite, got {z_bound}")
    if not torch.isfinite(tensor).all():
        raise ValueError("z_search contains NaN or infinite values")
    arch = tensor[..., : int(arch_nz)].clamp(-float(z_bound), float(z_bound))
    hp = tensor[..., int(arch_nz) :].clamp(0.0, 1.0)
    normalized = torch.cat(((arch + float(z_bound)) / (2.0 * float(z_bound)), hp), dim=-1)
    return _restore_type(normalized, was_tensor, dtype)


def denormalize_search_vector(
    z_normalized: Any,
    *,
    arch_nz: int,
    hp_mode: str,
    z_bound: float,
) -> Any:
    """Map fixed-domain normalized vectors back into raw search coordinates."""

    tensor, was_tensor, dtype = _as_float_tensor(z_normalized)
    expected = _expected_dim(arch_nz, hp_mode)
    if tensor.ndim == 0 or tensor.shape[-1] != expected:
        got = None if tensor.ndim == 0 else tensor.shape[-1]
        raise ValueError(f"normalized vector last dimension must be {expected}, got {got}")
    if not math.isfinite(float(z_bound)) or float(z_bound) <= 0.0:
        raise ValueError(f"z_bound must be positive and finite, got {z_bound}")
    if not torch.isfinite(tensor).all():
        raise ValueError("normalized vector contains NaN or infinite values")
    clipped = tensor.clamp(0.0, 1.0)
    arch = clipped[..., : int(arch_nz)] * (2.0 * float(z_bound)) - float(z_bound)
    raw = torch.cat((arch, clipped[..., int(arch_nz) :]), dim=-1)
    return _restore_type(raw, was_tensor, dtype)


class ConditionalMaskedKernel(Kernel):
    """RBF kernel on augmented ``[features, masks]`` inputs."""

    has_lengthscale = True

    def __init__(self, feature_dim: int, **kwargs: Any) -> None:
        feature_dim = int(feature_dim)
        # GPyTorch validates ard_num_dims against the actual model-input
        # dimension.  Conditional inputs are augmented as [features, masks], so
        # validation must see 2 * feature_dim even though the custom distance
        # below only consumes feature lengthscales.
        super().__init__(ard_num_dims=2 * feature_dim, **kwargs)
        self.feature_dim = feature_dim

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise ValueError("ConditionalMaskedKernel does not support last_dim_is_batch")
        x2 = x1 if x2 is None else x2
        dim = self.feature_dim
        if x1.shape[-1] != 2 * dim or x2.shape[-1] != 2 * dim:
            raise ValueError(f"conditional inputs must have last dimension {2 * dim}")
        feat1, mask1 = x1[..., :dim], x1[..., dim:].clamp(0.0, 1.0)
        feat2, mask2 = x2[..., :dim], x2[..., dim:].clamp(0.0, 1.0)
        diff = feat1.unsqueeze(-2) - feat2.unsqueeze(-3)
        pair_mask = torch.sqrt((mask1.unsqueeze(-2) * mask2.unsqueeze(-3)).clamp_min(0.0))
        lengthscale = self.lengthscale.reshape(-1)[:dim].clamp_min(1e-8)
        lengthscale = lengthscale.view(*([1] * (diff.dim() - 1)), dim)
        covariance = torch.exp(-0.5 * torch.sum(torch.square(diff * pair_mask / lengthscale), dim=-1))
        return covariance.diagonal(dim1=-2, dim2=-1) if diag else covariance


class AccuracyGPPredictor:
    """Exact GP predictor with fixed transforms and checkpoint-safe online updates."""

    def __init__(
        self,
        *,
        arch_nz: int,
        hp_mode: str,
        z_bound: float,
        use_conditional_kernel: bool = False,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        metadata: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.arch_nz = int(arch_nz)
        self.hp_mode = validate_hp_mode(hp_mode)
        self.search_dim = _expected_dim(self.arch_nz, self.hp_mode)
        self.z_bound = float(z_bound)
        if not math.isfinite(self.z_bound) or self.z_bound <= 0.0:
            raise ValueError("z_bound must be positive and finite")
        self.use_conditional_kernel = bool(use_conditional_kernel)
        self.condition_mask_dim = self.search_dim if self.use_conditional_kernel else 0
        self.kernel_type = "conditional_masked_rbf" if self.use_conditional_kernel else "default_single_task_gp"
        self.y_mean = float(y_mean)
        self.y_std = max(float(y_std), 1e-8)
        if not math.isfinite(self.y_mean) or not math.isfinite(self.y_std):
            raise ValueError("y_mean and y_std must be finite")
        self.metadata = dict(metadata or {})
        self.device = torch.device(device)
        self.train_X_norm = torch.empty(0, self.search_dim, dtype=torch.double, device=self.device)
        self.train_Y_norm = torch.empty(0, 1, dtype=torch.double, device=self.device)
        self.train_condition_masks: torch.Tensor | None = None
        self.model: SingleTaskGP | None = None
        self.holdout_X_raw: torch.Tensor | None = None
        self.holdout_Y: torch.Tensor | None = None
        self.holdout_condition_masks: torch.Tensor | None = None
        self.offline_train_size = 0

    @property
    def train_size(self) -> int:
        return int(self.train_X_norm.shape[0])

    @property
    def normalized_bounds(self) -> torch.Tensor:
        return torch.stack(
            (
                torch.zeros(self.search_dim, dtype=torch.double, device=self.device),
                torch.ones(self.search_dim, dtype=torch.double, device=self.device),
            )
        )

    @property
    def likelihood(self) -> Any:
        if self.model is None:
            raise RuntimeError("GP predictor is not fitted")
        return self.model.likelihood

    def _validate_masks(self, masks: Any, count: int) -> torch.Tensor:
        if masks is None:
            raise ValueError("condition masks are required for a conditional GP")
        tensor = torch.as_tensor(masks, dtype=torch.double, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (count, self.search_dim):
            raise ValueError(
                f"condition masks must have shape {(count, self.search_dim)}, got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("condition masks contain NaN or infinite values")
        return tensor.clamp(0.0, 1.0)

    def _model_inputs(self, X_norm: torch.Tensor, condition_masks: Any = None) -> torch.Tensor:
        if not self.use_conditional_kernel:
            return X_norm
        if condition_masks is None:
            raise ValueError("condition_masks are required by this conditional GP checkpoint")
        masks = self._validate_masks(condition_masks, X_norm.shape[0])
        return torch.cat((X_norm, masks), dim=-1)

    def _build_model(self) -> SingleTaskGP:
        if self.train_size < 2:
            raise ValueError("at least two training samples are required for an Exact GP")
        train_input = self._model_inputs(self.train_X_norm, self.train_condition_masks)
        covar_module = None
        if self.use_conditional_kernel:
            covar_module = ScaleKernel(ConditionalMaskedKernel(feature_dim=self.search_dim)).to(
                device=self.device, dtype=torch.double
            )
        model = SingleTaskGP(
            train_input,
            self.train_Y_norm,
            covar_module=covar_module,
            input_transform=None,
            outcome_transform=None,
        )
        return model.to(device=self.device, dtype=torch.double)

    @staticmethod
    def _optimize_model(model: SingleTaskGP, steps: int | None = None) -> None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if steps is None:
            fit_gpytorch_mll(mll)
            return
        if int(steps) <= 0:
            raise ValueError("refit steps must be positive")
        model.train()
        model.likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        targets = model.train_targets
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            output = model(*model.train_inputs)
            loss = -mll(output, targets).sum()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite marginal likelihood during GP refit")
            loss.backward()
            optimizer.step()

    @classmethod
    def fit_offline(
        cls,
        train_X_raw: Any,
        train_Y: Any,
        *,
        arch_nz: int,
        hp_mode: str,
        z_bound: float,
        use_conditional_kernel: bool = False,
        condition_masks: Any = None,
        metadata: dict[str, Any] | None = None,
        holdout_X_raw: Any = None,
        holdout_Y: Any = None,
        holdout_condition_masks: Any = None,
        device: str | torch.device = "cpu",
        fit_steps: int | None = None,
    ) -> "AccuracyGPPredictor":
        raw = torch.as_tensor(train_X_raw, dtype=torch.double, device=device)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        targets = torch.as_tensor(train_Y, dtype=torch.double, device=device).reshape(-1, 1)
        if raw.shape[0] != targets.shape[0] or raw.shape[0] < 2:
            raise ValueError("offline GP requires at least two paired X/Y samples")
        if not torch.isfinite(targets).all():
            raise ValueError("train_Y contains NaN or infinite values")
        y_mean = float(targets.mean())
        y_std = max(float(targets.std(unbiased=False)), 1e-8)
        predictor = cls(
            arch_nz=arch_nz,
            hp_mode=hp_mode,
            z_bound=z_bound,
            use_conditional_kernel=use_conditional_kernel,
            y_mean=y_mean,
            y_std=y_std,
            metadata=metadata,
            device=device,
        )
        predictor.train_X_norm = normalize_search_vector(
            raw, arch_nz=arch_nz, hp_mode=hp_mode, z_bound=z_bound
        ).to(dtype=torch.double, device=predictor.device)
        predictor.train_Y_norm = (targets.to(predictor.device) - y_mean) / y_std
        if use_conditional_kernel:
            predictor.train_condition_masks = predictor._validate_masks(condition_masks, raw.shape[0])
        predictor.model = predictor._build_model()
        predictor._optimize_model(predictor.model, steps=fit_steps)
        predictor.model.eval()
        predictor.model.likelihood.eval()
        predictor.offline_train_size = predictor.train_size
        if holdout_X_raw is not None:
            predictor._set_holdout(
                holdout_X_raw,
                holdout_Y,
                holdout_condition_masks if use_conditional_kernel else None,
            )
        return predictor

    def _set_holdout(self, X_raw: Any, y: Any, condition_masks: Any = None) -> None:
        if y is None:
            raise ValueError("holdout_Y is required when holdout_X_raw is present")
        holdout_X = torch.as_tensor(X_raw, dtype=torch.double).cpu()
        if holdout_X.ndim == 1:
            holdout_X = holdout_X.unsqueeze(0)
        if holdout_X.ndim != 2 or holdout_X.shape[1] != self.search_dim:
            raise ValueError(
                f"holdout_X must have shape (n, {self.search_dim}), got {tuple(holdout_X.shape)}"
            )
        holdout_y = torch.as_tensor(y, dtype=torch.double).reshape(-1).cpu()
        if holdout_X.shape[0] != holdout_y.shape[0]:
            raise ValueError("holdout X/Y lengths differ")
        if not torch.isfinite(holdout_X).all() or not torch.isfinite(holdout_y).all():
            raise ValueError("holdout data contains NaN or infinite values")
        self.holdout_X_raw = holdout_X
        self.holdout_Y = holdout_y
        if self.use_conditional_kernel:
            self.holdout_condition_masks = self._validate_masks(condition_masks, holdout_X.shape[0]).cpu()
        else:
            self.holdout_condition_masks = None
        self._check_holdout_data()

    def _normalized_raw(self, X_raw: Any) -> torch.Tensor:
        raw = torch.as_tensor(X_raw, dtype=torch.double, device=self.device)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        normalized = normalize_search_vector(
            raw, arch_nz=self.arch_nz, hp_mode=self.hp_mode, z_bound=self.z_bound
        )
        return normalized.to(device=self.device, dtype=torch.double)

    def predict_batch(self, X_raw: Any, condition_masks: Any = None) -> list[dict[str, float]]:
        if self.model is None:
            raise RuntimeError("GP predictor is not fitted")
        X_norm = self._normalized_raw(X_raw)
        model_input = self._model_inputs(X_norm, condition_masks)
        self.model.eval()
        self.model.likelihood.eval()
        with torch.no_grad():
            posterior = self.model.posterior(model_input)
            means = posterior.mean.reshape(-1) * self.y_std + self.y_mean
            stds = posterior.variance.clamp_min(1e-16).sqrt().reshape(-1) * self.y_std
        results = []
        for mean_value, std_value in zip(means, stds):
            mean = float(mean_value.detach().cpu())
            std = float(std_value.detach().cpu())
            results.append(
                {"mean": mean, "std": std, "lower_95": mean - 1.96 * std, "upper_95": mean + 1.96 * std}
            )
        return results

    def predict(self, X_raw: Any, condition_mask: Any = None) -> dict[str, float]:
        results = self.predict_batch(X_raw, condition_masks=condition_mask)
        if len(results) != 1:
            raise ValueError(f"predict expects one vector, received {len(results)}")
        return results[0]

    def make_logei(self, best_f: float) -> qLogExpectedImprovement:
        if self.model is None:
            raise RuntimeError("GP predictor is not fitted")
        best_normalized = (float(best_f) - self.y_mean) / self.y_std
        self.model.eval()
        self.model.likelihood.eval()
        return qLogExpectedImprovement(self.model, best_f=best_normalized)

    def acquisition_inputs(self, X_raw: Any, condition_masks: Any = None) -> torch.Tensor:
        """Return the exact transformed model inputs used for LogEI scoring."""

        return self._model_inputs(self._normalized_raw(X_raw), condition_masks)

    def append_observation(self, X_raw: Any, y: float, condition_mask: Any = None) -> None:
        self.append_observations(X_raw, [y], condition_masks=condition_mask)

    def append_observations(self, X_raw: Any, y: Any, condition_masks: Any = None) -> None:
        X_norm = self._normalized_raw(X_raw)
        targets = torch.as_tensor(y, dtype=torch.double, device=self.device).reshape(-1, 1)
        if X_norm.shape[0] != targets.shape[0]:
            raise ValueError("appended X/Y lengths differ")
        if not torch.isfinite(targets).all():
            raise ValueError("appended targets contain NaN or infinite values")
        if self.holdout_X_raw is not None:
            holdout_norm = self._normalized_raw(self.holdout_X_raw)
            overlaps = torch.isclose(
                X_norm.unsqueeze(1), holdout_norm.unsqueeze(0), rtol=0.0, atol=1e-7
            ).all(dim=-1).any(dim=1)
            if bool(overlaps.any()):
                raise ValueError("refusing to add a fixed holdout sample to GP training data")
        masks = None
        if self.use_conditional_kernel:
            masks = self._validate_masks(condition_masks, X_norm.shape[0])
        self.train_X_norm = torch.cat((self.train_X_norm, X_norm), dim=0)
        self.train_Y_norm = torch.cat((self.train_Y_norm, (targets - self.y_mean) / self.y_std), dim=0)
        if self.use_conditional_kernel:
            if self.train_condition_masks is None:
                self.train_condition_masks = masks
            else:
                self.train_condition_masks = torch.cat((self.train_condition_masks, masks), dim=0)
        self._check_training_data()

    def is_holdout_point(self, X_raw: Any) -> bool:
        """Return whether any supplied vector exactly matches the fixed holdout."""

        if self.holdout_X_raw is None or self.holdout_X_raw.numel() == 0:
            return False
        X_norm = self._normalized_raw(X_raw)
        holdout_norm = self._normalized_raw(self.holdout_X_raw)
        overlaps = torch.isclose(
            X_norm.unsqueeze(1), holdout_norm.unsqueeze(0), rtol=0.0, atol=1e-7
        ).all(dim=-1).any(dim=1)
        return bool(overlaps.any())

    def _check_training_data(self) -> None:
        if self.train_X_norm.shape != (self.train_size, self.search_dim):
            raise RuntimeError("GP train_X shape invariant failed")
        if self.train_Y_norm.shape != (self.train_size, 1):
            raise RuntimeError("GP train_Y shape invariant failed")
        if not torch.isfinite(self.train_X_norm).all() or not torch.isfinite(self.train_Y_norm).all():
            raise RuntimeError("GP training data contains non-finite values")
        if self.use_conditional_kernel and (
            self.train_condition_masks is None
            or self.train_condition_masks.shape != self.train_X_norm.shape
        ):
            raise RuntimeError("conditional GP mask/train_X shape invariant failed")
        if not self.use_conditional_kernel and self.train_condition_masks is not None:
            raise RuntimeError("standard GP checkpoint unexpectedly has training masks")

    def _check_holdout_data(self) -> None:
        if self.holdout_X_raw is None:
            return
        if self.holdout_X_raw.ndim != 2 or self.holdout_X_raw.shape[1] != self.search_dim:
            raise RuntimeError("GP holdout_X shape invariant failed")
        if self.holdout_Y is None or self.holdout_Y.shape != (self.holdout_X_raw.shape[0],):
            raise RuntimeError("GP holdout_Y shape invariant failed")
        if not torch.isfinite(self.holdout_X_raw).all() or not torch.isfinite(self.holdout_Y).all():
            raise RuntimeError("GP holdout data contains non-finite values")
        if self.use_conditional_kernel:
            if (
                self.holdout_condition_masks is None
                or self.holdout_condition_masks.shape != self.holdout_X_raw.shape
            ):
                raise RuntimeError("conditional GP holdout mask shape invariant failed")
        elif self.holdout_condition_masks is not None:
            raise RuntimeError("standard GP checkpoint unexpectedly has holdout masks")
        holdout_norm = self._normalized_raw(self.holdout_X_raw)
        overlaps = torch.isclose(
            self.train_X_norm.unsqueeze(1), holdout_norm.unsqueeze(0), rtol=0.0, atol=1e-7
        ).all(dim=-1).any()
        if bool(overlaps):
            raise RuntimeError("GP checkpoint leaks fixed holdout vectors into training data")

    def refit(self, *, optimize: bool = True, steps: int | None = None) -> None:
        """Rebuild the Exact GP, optionally warm-optimizing its hyperparameters."""

        self._check_training_data()
        old_state = None if self.model is None else self.model.state_dict()
        model = self._build_model()
        if old_state is not None:
            model.load_state_dict(old_state, strict=True)
        if optimize:
            self._optimize_model(model, steps=steps)
        model.eval()
        model.likelihood.eval()
        self.model = model
        with torch.no_grad():
            probe = self._model_inputs(self.train_X_norm[-1:], None if not self.use_conditional_kernel else self.train_condition_masks[-1:])
            posterior = self.model.posterior(probe)
            if not torch.isfinite(posterior.mean).all() or not torch.isfinite(posterior.variance).all():
                raise RuntimeError("GP posterior became non-finite after update")

    def checkpoint_payload(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted GP")
        self._check_training_data()
        self._check_holdout_data()
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_type": "SingleTaskGP",
            "kernel_type": self.kernel_type,
            "use_conditional_kernel": self.use_conditional_kernel,
            "condition_mask_dim": self.condition_mask_dim,
            "train_X_norm": self.train_X_norm.detach().cpu(),
            "train_Y_norm": self.train_Y_norm.detach().cpu(),
            "train_condition_masks": None if self.train_condition_masks is None else self.train_condition_masks.detach().cpu(),
            "model_state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "arch_nz": self.arch_nz,
            "hp_mode": self.hp_mode,
            "search_dim": self.search_dim,
            "z_bound": self.z_bound,
            "x_transform": X_TRANSFORM_VERSION,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "offline_train_size": self.offline_train_size,
            "holdout_X_raw": None if self.holdout_X_raw is None else self.holdout_X_raw.detach().cpu(),
            "holdout_Y": None if self.holdout_Y is None else self.holdout_Y.detach().cpu(),
            "holdout_condition_masks": (
                None if self.holdout_condition_masks is None
                else self.holdout_condition_masks.detach().cpu()
            ),
        }
        for key, default in {
            "dataset": "unknown",
            "metric": "val_acc",
            "eval_epochs": None,
            "patience": None,
            "vae_checkpoint": "",
            "vae_version": "",
            "seed": None,
        }.items():
            payload[key] = self.metadata.get(key, default)
        for key, value in self.metadata.items():
            if key not in payload:
                payload[key] = value
        return payload

    def save(self, path: str | Path) -> None:
        atomic_torch_save(self.checkpoint_payload(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        expected: dict[str, Any] | None = None,
    ) -> "AccuracyGPPredictor":
        payload = torch_load_compat(path, map_location=device)
        if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported or legacy GP checkpoint: {path}")
        required = {
            "model_type", "kernel_type", "use_conditional_kernel", "condition_mask_dim", "train_X_norm",
            "train_Y_norm", "model_state_dict", "arch_nz", "hp_mode", "search_dim",
            "z_bound", "x_transform", "y_mean", "y_std", "offline_train_size",
            "dataset", "metric", "eval_epochs", "patience", "vae_checkpoint",
            "vae_version", "seed",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"GP checkpoint is missing required fields: {missing}")
        if payload["x_transform"] != X_TRANSFORM_VERSION:
            raise ValueError(f"unsupported GP input transform: {payload['x_transform']}")
        if payload["model_type"] != "SingleTaskGP":
            raise ValueError(f"unsupported GP model_type: {payload['model_type']!r}")
        expected_kernel = (
            "conditional_masked_rbf" if payload["use_conditional_kernel"]
            else "default_single_task_gp"
        )
        if payload["kernel_type"] != expected_kernel:
            raise ValueError(
                f"GP kernel metadata is inconsistent: expected {expected_kernel!r}, "
                f"got {payload['kernel_type']!r}"
            )
        expected_mask_dim = int(payload["search_dim"]) if payload["use_conditional_kernel"] else 0
        if int(payload.get("condition_mask_dim", -1)) != expected_mask_dim:
            raise ValueError("GP checkpoint condition_mask_dim is inconsistent")
        if not payload["use_conditional_kernel"] and payload.get("train_condition_masks") is not None:
            raise ValueError("standard GP checkpoint must not contain training masks")
        if not payload["use_conditional_kernel"] and payload.get("holdout_condition_masks") is not None:
            raise ValueError("standard GP checkpoint must not contain holdout masks")
        actual_expected = dict(expected or {})
        for key, wanted in actual_expected.items():
            if wanted is None:
                continue
            actual = payload.get(key)
            same = math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-9) if key == "z_bound" and actual is not None else actual == wanted
            if not same:
                raise ValueError(f"GP checkpoint metadata mismatch for {key}: expected {wanted!r}, got {actual!r}")
        structural_keys = {
            "format_version", "model_type", "kernel_type", "use_conditional_kernel",
            "condition_mask_dim", "train_X_norm", "train_Y_norm", "train_condition_masks",
            "model_state_dict", "arch_nz", "hp_mode", "search_dim", "z_bound",
            "x_transform", "y_mean", "y_std", "offline_train_size", "holdout_X_raw",
            "holdout_Y", "holdout_condition_masks",
        }
        predictor = cls(
            arch_nz=int(payload["arch_nz"]), hp_mode=str(payload["hp_mode"]),
            z_bound=float(payload["z_bound"]),
            use_conditional_kernel=bool(payload["use_conditional_kernel"]),
            y_mean=float(payload["y_mean"]), y_std=float(payload["y_std"]),
            metadata={key: value for key, value in payload.items() if key not in structural_keys},
            device=device,
        )
        if int(payload["search_dim"]) != predictor.search_dim:
            raise ValueError("GP checkpoint search_dim is inconsistent with arch_nz and hp_mode")
        predictor.train_X_norm = torch.as_tensor(payload["train_X_norm"], dtype=torch.double, device=device)
        predictor.train_Y_norm = torch.as_tensor(payload["train_Y_norm"], dtype=torch.double, device=device)
        if predictor.use_conditional_kernel:
            if payload.get("train_condition_masks") is None:
                raise ValueError("conditional GP checkpoint has no training masks")
            predictor.train_condition_masks = torch.as_tensor(payload["train_condition_masks"], dtype=torch.double, device=device)
        predictor._check_training_data()
        predictor.model = predictor._build_model()
        predictor.model.load_state_dict(payload["model_state_dict"], strict=True)
        predictor.model.eval()
        predictor.model.likelihood.eval()
        predictor.offline_train_size = int(payload["offline_train_size"])
        if payload.get("holdout_X_raw") is not None:
            predictor._set_holdout(
                payload["holdout_X_raw"],
                payload.get("holdout_Y"),
                payload.get("holdout_condition_masks"),
            )
        return predictor
