"""Small single-output deep-kernel GP for the Phase4 NAS search space.

The deep-kernel-learning design follows Wilson et al., *Deep Kernel Learning*
(AISTATS 2016, https://proceedings.mlr.press/v51/wilson16.html).  This module is
an independent, deliberately small, single-output implementation for this NAS
project.  It does not copy or port source code from BOOM-Explorer.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any

import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from eval_utils import isolated_rng
from surrogate.accuracy_gp import (
    GP_NOISE_FLOOR,
    X_TRANSFORM_VERSION,
    AccuracyGPPredictor,
    _safe_gpytorch_context,
    normalize_search_vector,
)
from surrogate.checkpoint_io import atomic_torch_save, torch_load_compat


DKL_CHECKPOINT_FORMAT_VERSION = 1
DKL_MODEL_TYPE = "DKLSingleTaskGP"
DKL_KERNEL_TYPE = "small_feature_rbf"
VALID_FEATURE_DIMS = (4, 8)
VALID_ACTIVATIONS = ("silu", "relu")


class SmallFeatureExtractor(torch.nn.Module):
    """One-hidden-layer feature map used by the small DKL kernel."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        feature_dim: int = 8,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        feature_dim = int(feature_dim)
        activation = str(activation).lower()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if feature_dim not in VALID_FEATURE_DIMS:
            raise ValueError(f"feature_dim must be one of {VALID_FEATURE_DIMS}, got {feature_dim}")
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {VALID_ACTIVATIONS}, got {activation!r}")
        activation_module: torch.nn.Module
        activation_module = torch.nn.SiLU() if activation == "silu" else torch.nn.ReLU()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.activation_name = activation
        activation_code = 1.0 if activation == "silu" else 2.0
        self.register_buffer(
            "_architecture_signature",
            torch.tensor(
                [float(input_dim), float(hidden_dim), float(feature_dim), activation_code],
                dtype=torch.double,
            ),
        )
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            activation_module,
            torch.nn.Linear(hidden_dim, feature_dim),
        )
        self.to(dtype=torch.double)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature input last dimension must be {self.input_dim}, got {inputs.shape[-1]}"
            )
        return self.network(inputs.to(dtype=torch.double))


class DeepFeatureKernel(Kernel):
    """Shared learned feature map followed by a scaled ARD RBF kernel."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        feature_dim: int = 8,
        activation: str = "silu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_extractor = SmallFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            activation=activation,
        )
        self.latent_kernel = ScaleKernel(RBFKernel(ard_num_dims=int(feature_dim)))
        self.to(dtype=torch.double)

    @property
    def feature_dim(self) -> int:
        return self.feature_extractor.feature_dim

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ):
        if last_dim_is_batch:
            raise ValueError("DeepFeatureKernel does not support last_dim_is_batch")
        x2 = x1 if x2 is None else x2
        features1 = self.feature_extractor(x1)
        features2 = self.feature_extractor(x2)
        return self.latent_kernel(
            features1,
            features2,
            diag=diag,
            last_dim_is_batch=False,
            **params,
        )


def validate_dkl_config(
    *,
    hidden_dim: int,
    feature_dim: int,
    activation: str,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    init_steps: int,
    refit_steps: int,
    early_stopping_patience: int,
    min_delta: float,
) -> None:
    """Validate every public DKL training option with explicit errors."""

    if int(hidden_dim) <= 0:
        raise ValueError("dkl_hidden_dim must be positive")
    if int(feature_dim) not in VALID_FEATURE_DIMS:
        raise ValueError(f"dkl_feature_dim must be one of {VALID_FEATURE_DIMS}")
    if str(activation).lower() not in VALID_ACTIVATIONS:
        raise ValueError(f"dkl_activation must be one of {VALID_ACTIVATIONS}")
    for name, value in {
        "dkl_init_steps": init_steps,
        "dkl_refit_steps": refit_steps,
        "dkl_early_stopping_patience": early_stopping_patience,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    finite_values = {
        "dkl_lr": lr,
        "dkl_weight_decay": weight_decay,
        "dkl_grad_clip": grad_clip,
        "dkl_min_delta": min_delta,
    }
    for name, value in finite_values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(lr) <= 0.0:
        raise ValueError("dkl_lr must be positive")
    if float(weight_decay) < 0.0:
        raise ValueError("dkl_weight_decay must be non-negative")
    if float(grad_clip) <= 0.0:
        raise ValueError("dkl_grad_clip must be positive")
    if float(min_delta) < 0.0:
        raise ValueError("dkl_min_delta must be non-negative")


class DKLAccuracyGPPredictor(AccuracyGPPredictor):
    """Exact single-output GP whose covariance uses a supervised feature map."""

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
        hidden_dim: int = 32,
        feature_dim: int = 8,
        activation: str = "silu",
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
        init_steps: int = 200,
        refit_steps: int = 50,
        early_stopping_patience: int = 25,
        min_delta: float = 1e-5,
    ) -> None:
        validate_dkl_config(
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            activation=activation,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            init_steps=init_steps,
            refit_steps=refit_steps,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
        )
        super().__init__(
            arch_nz=arch_nz,
            hp_mode=hp_mode,
            z_bound=z_bound,
            use_conditional_kernel=use_conditional_kernel,
            y_mean=y_mean,
            y_std=y_std,
            metadata=metadata,
            device=device,
        )
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.activation = str(activation).lower()
        self.dkl_lr = float(lr)
        self.dkl_weight_decay = float(weight_decay)
        self.dkl_grad_clip = float(grad_clip)
        self.dkl_init_steps = int(init_steps)
        self.dkl_refit_steps = int(refit_steps)
        self.dkl_early_stopping_patience = int(early_stopping_patience)
        self.dkl_min_delta = float(min_delta)
        self.feature_input_dim = self.search_dim * (2 if self.use_conditional_kernel else 1)
        self.kernel_type = DKL_KERNEL_TYPE
        self.mask_mode = "masked_features_and_mask" if self.use_conditional_kernel else "none"
        self.training_summary: dict[str, Any] = {
            "best_step": None,
            "best_loss": None,
            "steps_ran": 0,
            "early_stopped": False,
        }

    def _validate_masks(self, masks: Any, count: int) -> torch.Tensor:
        tensor = super()._validate_masks(masks, count)
        original = torch.as_tensor(masks, dtype=torch.double, device=self.device)
        if not bool(((original >= 0.0) & (original <= 1.0)).all()):
            raise ValueError("condition masks must contain values in [0, 1]")
        return tensor

    def _model_inputs(self, X_norm: torch.Tensor, condition_masks: Any = None) -> torch.Tensor:
        if not self.use_conditional_kernel:
            return X_norm
        masks = self._validate_masks(condition_masks, X_norm.shape[0])
        return torch.cat((X_norm * masks, masks), dim=-1)

    def _build_model(self) -> SingleTaskGP:
        if self.train_size < 2:
            raise ValueError("at least two training samples are required for a DKL GP")
        train_input = self._model_inputs(self.train_X_norm, self.train_condition_masks)
        covar_module = DeepFeatureKernel(
            input_dim=self.feature_input_dim,
            hidden_dim=self.hidden_dim,
            feature_dim=self.feature_dim,
            activation=self.activation,
        ).to(device=self.device, dtype=torch.double)
        model = SingleTaskGP(
            train_input,
            self.train_Y_norm,
            covar_module=covar_module,
            input_transform=None,
            outcome_transform=None,
        ).to(device=self.device, dtype=torch.double)
        self._ensure_likelihood_noise_floor(model, GP_NOISE_FLOOR)
        return model

    def _optimizer_groups(self, model: SingleTaskGP) -> list[dict[str, Any]]:
        feature_parameters = list(model.covar_module.feature_extractor.parameters())
        feature_ids = {id(parameter) for parameter in feature_parameters}
        other_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in feature_ids
        ]
        if len(feature_ids) != len(feature_parameters):
            raise RuntimeError("duplicate feature-extractor parameters in DKL optimizer")
        if feature_ids.intersection(id(parameter) for parameter in other_parameters):
            raise RuntimeError("DKL optimizer parameter groups overlap")
        return [
            {
                "params": feature_parameters,
                "lr": self.dkl_lr,
                "weight_decay": self.dkl_weight_decay,
            },
            {"params": other_parameters, "lr": self.dkl_lr, "weight_decay": 0.0},
        ]

    @staticmethod
    def _training_hyperparameters(model: SingleTaskGP) -> dict[str, Any]:
        latent_kernel = model.covar_module.latent_kernel
        return {
            "likelihood_noise": float(model.likelihood.noise.detach().reshape(-1)[0].cpu()),
            "lengthscale": latent_kernel.base_kernel.lengthscale.detach().reshape(-1).cpu().tolist(),
            "outputscale": float(latent_kernel.outputscale.detach().reshape(-1)[0].cpu()),
        }

    @staticmethod
    def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
        if not parameters:
            return 0.0
        squared = torch.zeros((), dtype=torch.double, device=parameters[0].device)
        for parameter in parameters:
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().double().square().sum()
        return float(squared.sqrt().cpu())

    def _optimize_model(
        self,
        model: SingleTaskGP,
        steps: int | None = None,
        *,
        noise_floor: float = GP_NOISE_FLOOR,
    ) -> None:
        steps = self.dkl_refit_steps if steps is None else int(steps)
        if steps <= 0:
            raise ValueError("refit steps must be positive")
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = torch.optim.Adam(self._optimizer_groups(model))
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        best_step: int | None = None
        stale_steps = 0
        steps_ran = 0
        early_stopped = False
        initial_loss: float | None = None
        final_loss: float | None = None
        final_gradient_norm: float | None = None
        final_feature_gradient_norm: float | None = None
        model.train()
        model.likelihood.train()
        self._ensure_likelihood_noise_floor(model, noise_floor)
        initial_hyperparameters = self._training_hyperparameters(model)
        with _safe_gpytorch_context(max(GP_NOISE_FLOOR, float(noise_floor))):
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                output = model(*model.train_inputs)
                loss = -mll(output, model.train_targets).sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite marginal likelihood during DKL refit")
                loss.backward()
                parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
                feature_parameters = [
                    parameter
                    for parameter in model.covar_module.feature_extractor.parameters()
                    if parameter.grad is not None
                ]
                final_gradient_norm = self._gradient_norm(parameters)
                final_feature_gradient_norm = self._gradient_norm(feature_parameters)
                torch.nn.utils.clip_grad_norm_(parameters, self.dkl_grad_clip)
                if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
                    raise RuntimeError("non-finite gradient during DKL refit")
                optimizer.step()
                self._ensure_likelihood_noise_floor(model, noise_floor)
                steps_ran = step + 1
                loss_value = float(loss.detach().cpu())
                if initial_loss is None:
                    initial_loss = loss_value
                final_loss = loss_value
                if best_state is None or best_loss - loss_value > self.dkl_min_delta:
                    best_loss = loss_value
                    best_step = step
                    best_state = self._clone_state_dict(model)
                    stale_steps = 0
                else:
                    stale_steps += 1
                    if stale_steps >= self.dkl_early_stopping_patience:
                        early_stopped = True
                        break
        if best_state is None or best_step is None:
            raise RuntimeError("DKL optimization produced no finite training state")
        model.load_state_dict(best_state, strict=True)
        self._ensure_likelihood_noise_floor(model, noise_floor)
        final_hyperparameters = self._training_hyperparameters(model)
        self.training_summary = {
            "requested_steps": int(steps),
            "best_step": int(best_step),
            "best_loss": float(best_loss),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "steps_ran": int(steps_ran),
            "early_stopped": bool(early_stopped),
            "gradient_norm": final_gradient_norm,
            "feature_gradient_norm": final_feature_gradient_norm,
            "likelihood_noise_initial": initial_hyperparameters["likelihood_noise"],
            "likelihood_noise_final": final_hyperparameters["likelihood_noise"],
            "lengthscale_initial": initial_hyperparameters["lengthscale"],
            "lengthscale_final": final_hyperparameters["lengthscale"],
            "outputscale_initial": initial_hyperparameters["outputscale"],
            "outputscale_final": final_hyperparameters["outputscale"],
        }
        self.metadata["dkl_training_summary"] = dict(self.training_summary)

    def refit(
        self,
        *,
        optimize: bool = True,
        steps: int | None = None,
        training_seed: int | None = None,
    ) -> None:
        context = nullcontext() if training_seed is None else isolated_rng(training_seed, self.device)
        with context:
            super().refit(optimize=optimize, steps=steps)
        if training_seed is not None:
            self.metadata["dkl_training_seed"] = int(training_seed)

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
        training_seed: int | None = None,
        **dkl_config: Any,
    ) -> "DKLAccuracyGPPredictor":
        raw = torch.as_tensor(train_X_raw, dtype=torch.double, device=device)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        targets = torch.as_tensor(train_Y, dtype=torch.double, device=device).reshape(-1, 1)
        if raw.shape[0] != targets.shape[0] or raw.shape[0] < 2:
            raise ValueError("offline DKL GP requires at least two paired X/Y samples")
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
            **dkl_config,
        )
        predictor.train_X_norm = normalize_search_vector(
            raw, arch_nz=arch_nz, hp_mode=hp_mode, z_bound=z_bound
        ).to(dtype=torch.double, device=predictor.device)
        predictor.train_Y_norm = (targets.to(predictor.device) - y_mean) / y_std
        predictor.train_observation_counts = torch.ones(
            predictor.train_X_norm.shape[0], dtype=torch.long, device=predictor.device
        )
        if use_conditional_kernel:
            predictor.train_condition_masks = predictor._validate_masks(condition_masks, raw.shape[0])
        predictor._aggregate_duplicate_training_rows()
        predictor.refit(
            optimize=True,
            steps=predictor.dkl_init_steps if fit_steps is None else int(fit_steps),
            training_seed=training_seed,
        )
        predictor.offline_train_size = predictor.train_size
        if holdout_X_raw is not None:
            predictor._set_holdout(
                holdout_X_raw,
                holdout_Y,
                holdout_condition_masks if use_conditional_kernel else None,
            )
        return predictor

    def checkpoint_payload(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted DKL GP")
        self._check_training_data()
        self._check_holdout_data()
        payload = {
            "format_version": DKL_CHECKPOINT_FORMAT_VERSION,
            "surrogate_type": "dkl_gp",
            "model_type": DKL_MODEL_TYPE,
            "kernel_type": DKL_KERNEL_TYPE,
            "feature_input_dim": self.feature_input_dim,
            "feature_hidden_dims": [self.hidden_dim],
            "feature_dim": self.feature_dim,
            "activation": self.activation,
            "mask_mode": self.mask_mode,
            "use_conditional_kernel": self.use_conditional_kernel,
            "condition_mask_dim": self.condition_mask_dim,
            "arch_nz": self.arch_nz,
            "hp_mode": self.hp_mode,
            "search_dim": self.search_dim,
            "z_bound": self.z_bound,
            "x_transform": X_TRANSFORM_VERSION,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "train_X_norm": self.train_X_norm.detach().cpu(),
            "train_Y_norm": self.train_Y_norm.detach().cpu(),
            "train_observation_counts": self.train_observation_counts.detach().cpu(),
            "train_condition_masks": None if self.train_condition_masks is None else self.train_condition_masks.detach().cpu(),
            "model_state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "offline_train_size": self.offline_train_size,
            "holdout_X_raw": None if self.holdout_X_raw is None else self.holdout_X_raw.detach().cpu(),
            "holdout_Y": None if self.holdout_Y is None else self.holdout_Y.detach().cpu(),
            "holdout_condition_masks": None if self.holdout_condition_masks is None else self.holdout_condition_masks.detach().cpu(),
            "training_configuration": {
                "lr": self.dkl_lr,
                "weight_decay": self.dkl_weight_decay,
                "grad_clip": self.dkl_grad_clip,
                "init_steps": self.dkl_init_steps,
                "refit_steps": self.dkl_refit_steps,
                "early_stopping_patience": self.dkl_early_stopping_patience,
                "min_delta": self.dkl_min_delta,
            },
            "training_summary": dict(self.training_summary),
        }
        for key, default in {
            "dataset": "unknown",
            "metric": "val_acc",
            "eval_epochs": None,
            "patience": None,
            "vae_checkpoint": "",
            "vae_version": "",
            "seed": None,
            "seed_derivation": None,
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
    ) -> "DKLAccuracyGPPredictor":
        payload = torch_load_compat(path, map_location=device)
        actual_type = payload.get("surrogate_type", "exact_gp") if isinstance(payload, dict) else None
        if actual_type != "dkl_gp":
            raise ValueError(
                f"surrogate checkpoint type mismatch: expected 'dkl_gp', actual {actual_type!r}"
            )
        if payload.get("format_version") != DKL_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported DKL checkpoint format: {payload.get('format_version')!r}")
        required = {
            "model_type", "kernel_type", "feature_input_dim", "feature_hidden_dims",
            "feature_dim", "activation", "mask_mode", "use_conditional_kernel",
            "condition_mask_dim", "arch_nz", "hp_mode", "search_dim", "z_bound",
            "x_transform", "y_mean", "y_std", "train_X_norm", "train_Y_norm",
            "train_observation_counts", "train_condition_masks", "model_state_dict",
            "offline_train_size", "holdout_X_raw", "holdout_Y",
            "holdout_condition_masks", "training_configuration", "training_summary",
            "dataset", "metric", "eval_epochs", "patience", "vae_checkpoint",
            "vae_version", "seed",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"DKL checkpoint is missing required fields: {missing}")
        if payload["model_type"] != DKL_MODEL_TYPE or payload["kernel_type"] != DKL_KERNEL_TYPE:
            raise ValueError("DKL checkpoint model/kernel type is inconsistent")
        if payload["x_transform"] != X_TRANSFORM_VERSION:
            raise ValueError(f"unsupported DKL input transform: {payload['x_transform']}")
        expected_values = dict(expected or {})
        for key, wanted in expected_values.items():
            if wanted is None:
                continue
            actual = payload.get(key)
            same = (
                math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-9)
                if key == "z_bound" and actual is not None else actual == wanted
            )
            if not same:
                raise ValueError(
                    f"DKL checkpoint metadata mismatch for {key}: expected {wanted!r}, got {actual!r}"
                )
        configuration = payload["training_configuration"]
        if not isinstance(configuration, dict):
            raise ValueError("DKL checkpoint training_configuration must be a mapping")
        hidden_dims = payload["feature_hidden_dims"]
        if not isinstance(hidden_dims, list) or len(hidden_dims) != 1:
            raise ValueError("DKL checkpoint must contain exactly one feature hidden dimension")
        structural_metadata = required | {"format_version", "surrogate_type"}
        provenance_keys = {
            "dataset", "metric", "eval_epochs", "patience", "vae_checkpoint",
            "vae_version", "seed", "seed_derivation", "dkl_training_seed",
        }
        predictor = cls(
            arch_nz=int(payload["arch_nz"]),
            hp_mode=str(payload["hp_mode"]),
            z_bound=float(payload["z_bound"]),
            use_conditional_kernel=bool(payload["use_conditional_kernel"]),
            y_mean=float(payload["y_mean"]),
            y_std=float(payload["y_std"]),
            metadata={
                key: value for key, value in payload.items()
                if key not in structural_metadata or key in provenance_keys
            },
            device=device,
            hidden_dim=int(hidden_dims[0]),
            feature_dim=int(payload["feature_dim"]),
            activation=str(payload["activation"]),
            lr=float(configuration["lr"]),
            weight_decay=float(configuration["weight_decay"]),
            grad_clip=float(configuration["grad_clip"]),
            init_steps=int(configuration["init_steps"]),
            refit_steps=int(configuration["refit_steps"]),
            early_stopping_patience=int(configuration["early_stopping_patience"]),
            min_delta=float(configuration["min_delta"]),
        )
        expected_feature_input = predictor.search_dim * (2 if predictor.use_conditional_kernel else 1)
        structural_checks = {
            "search_dim": predictor.search_dim,
            "feature_input_dim": expected_feature_input,
            "condition_mask_dim": predictor.search_dim if predictor.use_conditional_kernel else 0,
            "mask_mode": predictor.mask_mode,
        }
        for key, wanted in structural_checks.items():
            if payload.get(key) != wanted:
                raise ValueError(
                    f"DKL checkpoint structural mismatch for {key}: expected {wanted!r}, "
                    f"got {payload.get(key)!r}"
                )
        predictor.train_X_norm = torch.as_tensor(payload["train_X_norm"], dtype=torch.double, device=device)
        predictor.train_Y_norm = torch.as_tensor(payload["train_Y_norm"], dtype=torch.double, device=device)
        predictor.train_observation_counts = torch.as_tensor(
            payload["train_observation_counts"], dtype=torch.long, device=device
        ).reshape(-1)
        if predictor.use_conditional_kernel:
            if payload["train_condition_masks"] is None:
                raise ValueError("conditional DKL checkpoint has no training masks")
            predictor.train_condition_masks = torch.as_tensor(
                payload["train_condition_masks"], dtype=torch.double, device=device
            )
        elif payload["train_condition_masks"] is not None:
            raise ValueError("non-conditional DKL checkpoint unexpectedly contains training masks")
        predictor._check_training_data()
        predictor.model = predictor._build_model()
        signature_key = "covar_module.feature_extractor._architecture_signature"
        checkpoint_signature = payload["model_state_dict"].get(signature_key)
        expected_signature = predictor.model.state_dict()[signature_key]
        if checkpoint_signature is None or not torch.equal(
            torch.as_tensor(checkpoint_signature, dtype=torch.double).cpu(),
            expected_signature.detach().cpu(),
        ):
            raise ValueError(
                "DKL checkpoint feature architecture metadata does not match its model state"
            )
        try:
            predictor.model.load_state_dict(payload["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(f"DKL checkpoint state is incompatible with its feature metadata: {exc}") from exc
        predictor._ensure_likelihood_noise_floor(predictor.model, GP_NOISE_FLOOR)
        predictor.model.eval()
        predictor.model.likelihood.eval()
        predictor.offline_train_size = int(payload["offline_train_size"])
        predictor.training_summary = dict(payload["training_summary"])
        if payload["holdout_X_raw"] is not None:
            predictor._set_holdout(
                payload["holdout_X_raw"],
                payload["holdout_Y"],
                payload["holdout_condition_masks"],
            )
        elif payload["holdout_Y"] is not None or payload["holdout_condition_masks"] is not None:
            raise ValueError("DKL checkpoint has holdout metadata without holdout_X_raw")
        return predictor
