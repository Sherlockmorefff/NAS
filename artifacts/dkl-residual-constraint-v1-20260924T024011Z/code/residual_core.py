"""Selected-data RESIDUAL16 models and fixed, restorable optimizer blocks.

No source files, query labels, historical scores, gate decisions, or scientific
selection logic are accessed here. The frozen ExactGP is reused unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time

import gpytorch
import numpy as np
import torch
from torch import nn

from stage01_core import ExactGP, cpu_setup, tensor_hash
from stage2_core import Budget as _OldBudget, BudgetExceeded, RawBranch as _OldRaw
from stage2_core import numerical_settings

CAP = math.sqrt(0.5)
SPECTRAL_TOLERANCE = 1e-10
REGRESSION_OBJECTIVE = 'standardized_y_pure_mse_v1'


class Budget(_OldBudget):
    def __init__(self, path=None, limit=1200):
        super().__init__(path, limit)

    def as_dict(self):
        value = super().as_dict()
        value.pop('diagnostics_charged_as_training', None)
        value['accounting_policy'] = 'training includes projection; model diagnostics charged separately; verification separately counted'
        return value


def rng_state():
    return {'torch': torch.get_rng_state().clone(), 'numpy': copy.deepcopy(np.random.get_state()),
            'python': random.getstate()}


def set_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _isolate_gp_transforms(gp):
    """Detach GPyTorch's default Softplus singletons before any mode changes."""
    count = 0
    for module in list(gp.modules()):
        for name, child in list(module._modules.items()):
            if isinstance(child, nn.Softplus):
                setattr(module, name, copy.deepcopy(child))
                count += 1
    return count


@torch.no_grad()
def project_weight(weight):
    """Exact singular-value clipping, never uniform scaling or power iteration."""
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    before = float(singular.max())
    changed = before > CAP
    correction = 0.0
    if changed:
        projected = (u * singular.clamp(max=CAP).unsqueeze(0)) @ vh
        correction = float((projected - weight).norm())
        weight.copy_(projected)
    return {'pre_spectral_norm': before, 'post_spectral_norm_from_svd': min(before, CAP),
            'projected': changed, 'frobenius_correction': correction}


class ResidualMapper(nn.Module):
    def __init__(self, seed, initialize=True):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            # Creation order is scientifically fixed, including W2's consumed
            # default-initialization draws before its weights are zeroed.
            self.W1 = nn.Linear(16, 8, bias=True, device='cpu', dtype=torch.float64)
            self.W2 = nn.Linear(8, 16, bias=False, device='cpu', dtype=torch.float64)
            self.head = nn.Linear(16, 1, bias=True, device='cpu', dtype=torch.float64)
        self.relu = nn.ReLU()
        if initialize:
            with torch.no_grad():
                self.W1.bias.zero_()
                self.W2.weight.zero_()
                project_weight(self.W1.weight)

    @property
    def regressor(self):
        return self.head

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 16:
            raise ValueError('RESIDUAL16 requires x[n,16]')
        h = x + self.W2(self.relu(self.W1(x)))
        return h, self.head(h).squeeze(-1)


def spectral_stats(model, enforce=True):
    if not isinstance(model, ResidualBranch):
        return {'constraint_mode': 'none', 'w1_spectral_norm': None, 'w2_spectral_norm': None,
                'residual_lipschitz_upper_bound': None}
    with torch.no_grad():
        one = float(torch.linalg.svdvals(model.mapper.W1.weight).max())
        two = float(torch.linalg.svdvals(model.mapper.W2.weight).max())
    if not math.isfinite(one + two):
        raise FloatingPointError('Nonfinite residual spectral norms')
    bounded = one <= CAP + SPECTRAL_TOLERANCE and two <= CAP + SPECTRAL_TOLERANCE and one * two <= .5 + SPECTRAL_TOLERANCE
    if enforce and model.constraint_mode == 'bounded' and not bounded:
        raise FloatingPointError('Saved/projected bounded residual violates the fixed spectral cap')
    return {'constraint_mode': model.constraint_mode, 'w1_spectral_norm': one, 'w2_spectral_norm': two,
            'residual_lipschitz_upper_bound': one * two, 'cap': CAP,
            'spectral_tolerance': SPECTRAL_TOLERANCE, 'within_bounded_limits': bounded}


@torch.no_grad()
def project_mapping(model):
    if not isinstance(model, ResidualBranch) or model.constraint_mode != 'bounded':
        raise ValueError('Projection is authorized only for a bounded residual mapping update')
    one, two = project_weight(model.mapper.W1.weight), project_weight(model.mapper.W2.weight)
    return {'projection_evaluated': True, 'w1': one, 'w2': two,
            'projection_applied': one['projected'] or two['projected'],
            'matrices_projected': int(one['projected']) + int(two['projected']),
            'projection_correction_l2': math.hypot(one['frobenius_correction'], two['frobenius_correction']),
            'w1_spectral_norm': one['post_spectral_norm_from_svd'],
            'w2_spectral_norm': two['post_spectral_norm_from_svd'],
            'residual_lipschitz_upper_bound': one['post_spectral_norm_from_svd'] * two['post_spectral_norm_from_svd']}


class ResidualBranch:
    def __init__(self, x, y, known, seed, constraint_mode='bounded', _initialize=True):
        if constraint_mode not in ('bounded', 'unbounded'):
            raise ValueError('Unknown residual constraint mode')
        self.constraint_mode = constraint_mode
        self.x, self.y, self.known = (v.detach().clone().double() for v in (x, y, known))
        if self.x.shape != (len(y), 16) or self.y.ndim != 1 or self.known.shape != self.y.shape:
            raise ValueError('Aligned selected x16/y/known-variance tensors required')
        if not all(bool(torch.isfinite(v).all()) for v in (self.x, self.y, self.known)) or bool((self.known < 0).any()):
            raise ValueError('Invalid selected training tensors')
        self.mapper = ResidualMapper(seed, initialize=_initialize)
        with torch.no_grad():
            h, _ = self.encode(self.x)
        self.gp = ExactGP(h.detach().clone(), self.y, self.known)
        self.detached_constraint_transforms = _isolate_gp_transforms(self.gp)
        self.cache_version = None

    def encode(self, x):
        return self.mapper(x)

    def mapping_params(self):
        return list(self.mapper.W1.parameters()) + list(self.mapper.W2.parameters())

    def configuration(self):
        return {'model_type': 'RESIDUAL16', 'structure': 'x+W2(ReLU(W1(x)));W1=Linear16x8_bias;W2=Linear8x16_no_bias;head=Linear16x1_bias',
                'constraint_mode': self.constraint_mode, 'spectral_cap': CAP,
                'residual_lipschitz_cap': .5, 'spectral_tolerance': SPECTRAL_TOLERANCE,
                'regression_objective': REGRESSION_OBJECTIVE}

    def configuration_hash(self):
        return _digest(self.configuration())

    def state(self):
        return {**{'mapper.' + k: v for k, v in self.mapper.state_dict().items()},
                **{'gp.' + k: v for k, v in self.gp.state_dict().items()},
                'data.x': self.x, 'data.y': self.y, 'data.known': self.known,
                'likelihood.fixed_noise': self.gp.likelihood.noise_covar.noise}

    def fingerprint(self):
        return tensor_hash(self.state())

    def synchronize_history(self):
        # No mode mutations: the mapping has no dropout or batch normalization.
        with torch.no_grad():
            history, _ = self.encode(self.x)
        self.gp.set_train_data(inputs=history.detach().clone(), targets=self.y, strict=False)
        self.gp.prediction_strategy = None
        self.cache_version = None

    def clone(self, constraint_mode=None):
        incoming = rng_state()
        try:
            result = ResidualBranch(self.x, self.y, self.known, 0,
                                    self.constraint_mode if constraint_mode is None else constraint_mode,
                                    _initialize=False)
            result.mapper.load_state_dict(copy.deepcopy(self.mapper.state_dict()))
            result.gp.load_state_dict(copy.deepcopy(self.gp.state_dict()))
            result.synchronize_history()
            spectral_stats(result, enforce=True)
            return result
        finally:
            set_rng(incoming)

    def objective(self, mode, mll=None, fixed_embedding=None):
        """The sole objective implementation, shared by updates and diagnostics."""
        if mode not in ('regression', 'gp', 'joint'):
            raise ValueError('Unknown objective mode')
        if mode == 'regression':
            _, pred = self.encode(self.x)
            loss = (pred - self.y).square().mean()
            return loss, {'regression_mse': float(loss.detach()), 'data_nll': None,
                          'regression_objective': REGRESSION_OBJECTIVE}
        h = fixed_embedding if fixed_embedding is not None else self.encode(self.x)[0]
        if mode == 'gp':
            h = h.detach()
        if not self.gp.training:
            self.gp.train()
        self.gp.set_train_data(inputs=h, targets=self.y, strict=False)
        mll = mll or gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        with numerical_settings():
            loss = -mll(self.gp(h), self.y)
        return loss, {'regression_mse': None, 'data_nll': float(loss.detach())}

    def predict(self, query):
        with torch.no_grad(), numerical_settings():
            self.synchronize_history()
            self.mapper.eval()
            self.gp.eval()
            self.gp.likelihood.eval()
            h, _ = self.encode(query)
            posterior = self.gp(h)
            mean, variance = posterior.mean.clone(), posterior.lazy_covariance_matrix.diagonal().clone()
        if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()) or bool((variance < -1e-8).any()):
            raise FloatingPointError('Nonfinite prediction or raw standardized latent variance below -1e-8')
        self.cache_version = self.fingerprint()
        return mean, variance

    def kernel(self):
        with torch.no_grad(), numerical_settings():
            h, _ = self.encode(self.x)
            return self.gp.covar_module(h).to_dense().clone()

    def snapshot(self):
        return snapshot(self)


class RawBranch(_OldRaw):
    def __init__(self, x, y, known, seed=None):
        super().__init__(x, y, known)
        self.detached_constraint_transforms = _isolate_gp_transforms(self.gp)
        self.constraint_mode = 'none'

    def configuration(self):
        return {'model_type': 'RAW16', 'structure': 'standardized_input16_direct_GP', 'constraint_mode': 'none'}

    def configuration_hash(self):
        return _digest(self.configuration())

    def clone(self):
        incoming = rng_state()
        try:
            result = RawBranch(self.x, self.y, self.known)
            result.gp.load_state_dict(copy.deepcopy(self.gp.state_dict()))
            result.synchronize_history()
            return result
        finally:
            set_rng(incoming)

    def predict(self, query):
        self.synchronize_history()
        return super().predict(query)

    def objective(self, mode, mll=None, fixed_embedding=None):
        if mode != 'gp':
            raise ValueError('RAW16 permits GP-only optimization')
        if not self.gp.training:
            self.gp.train()
        self.gp.set_train_data(inputs=self.x, targets=self.y, strict=False)
        mll = mll or gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        with numerical_settings():
            loss = -mll(self.gp(self.x), self.y)
        return loss, {'data_nll': float(loss.detach()), 'regression_mse': None}

    def snapshot(self):
        return snapshot(self)


def snapshot(model):
    # Saving must also preserve an INVALID attempt after a failed projection.
    # Constraint enforcement belongs to updates/diagnostics/restoration; do not
    # replace the original failure with an inability to serialize its weights.
    with torch.no_grad():
        history, _ = model.encode(model.x)
    state = {'model_type': model.configuration()['model_type'], 'model_kind': model.configuration()['model_type'],
             'configuration': model.configuration(), 'configuration_hash': model.configuration_hash(),
             'constraint_mode': model.constraint_mode, 'gp': copy.deepcopy(model.gp.state_dict()),
             'x': model.x.clone(), 'y': model.y.clone(), 'known': model.known.clone(),
             'history_embedding': history.detach().clone(), 'fingerprint': model.fingerprint(), 'rng': rng_state()}
    if hasattr(model, 'mapper'):
        state['mapper'] = copy.deepcopy(model.mapper.state_dict())
    return state


def restore(state):
    if state['model_type'] == 'RESIDUAL16':
        model = ResidualBranch(state['x'], state['y'], state['known'], 0,
                               state['configuration']['constraint_mode'], _initialize=False)
        model.mapper.load_state_dict(state['mapper'])
    elif state['model_type'] == 'RAW16':
        model = RawBranch(state['x'], state['y'], state['known'])
    else:
        raise ValueError('Unsupported model type; old MAPPED8 is not RESIDUAL16')
    if model.configuration() != state['configuration'] or model.configuration_hash() != state['configuration_hash']:
        raise ValueError('Model architecture/constraint/objective configuration changed')
    if state['constraint_mode'] != model.constraint_mode:
        raise ValueError('Saved top-level constraint mode disagrees with configuration')
    model.gp.load_state_dict(state['gp'])
    spectral_stats(model, enforce=True)
    model.synchronize_history()
    if model.fingerprint() != state['fingerprint'] or not torch.equal(model.gp.train_inputs[0], state['history_embedding']):
        raise ValueError('Restored parameters or current history embedding differ')
    set_rng(state['rng'])
    return model


def _groups(model, mode):
    if mode not in ('regression', 'gp', 'joint') or isinstance(model, RawBranch) and mode != 'gp':
        raise ValueError('Invalid model training mode')
    named, groups = [], []
    if hasattr(model, 'mapper'):
        model.mapper.train(mode in ('regression', 'joint'))
        for p in model.mapper.parameters():
            p.requires_grad_(mode == 'regression')
        for p in model.mapping_params():
            p.requires_grad_(mode in ('regression', 'joint'))
        model.mapper.zero_grad(set_to_none=True)
        if mode in ('regression', 'joint'):
            pairs = [('mapper.' + k, p) for k, p in model.mapper.named_parameters()
                     if mode == 'regression' or not k.startswith('head.')]
            named.append([k for k, _ in pairs])
            groups.append({'params': [p for _, p in pairs], 'lr': 1e-3, 'base_lr': 1e-3, 'name': 'mapping_head' if mode == 'regression' else 'mapping'})
    model.gp.train()
    for p in model.gp.parameters():
        p.requires_grad_(mode != 'regression')
    model.gp.zero_grad(set_to_none=True)
    if mode != 'regression':
        pairs = [('gp.' + k, p) for k, p in model.gp.named_parameters()]
        named.append([k for k, _ in pairs])
        groups.append({'params': [p for _, p in pairs], 'lr': 1e-2, 'base_lr': 1e-2, 'name': 'gp'})
    return groups, named


def _grad_norm(params):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in params if p.grad is not None))


class BlockSession:
    def __init__(self, model, mode, planned_steps, block, resume=None):
        if planned_steps <= 0:
            raise ValueError('Planned steps must be positive')
        self.model, self.mode, self.planned_steps, self.block = model, mode, int(planned_steps), block
        groups, self.ordered_parameter_names = _groups(model, mode)
        self.optimizer = torch.optim.Adam(groups, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        self.completed, self.training_seconds, self.last_update = 0, 0., None
        self.budget_updates_after_last_step = None
        self.projection_steps, self.matrices_projected, self.projection_correction_l2_sum = 0, 0, 0.
        if resume is not None:
            schedule = resume['scheduler']
            if (schedule['block'], schedule['mode'], schedule['planned_steps']) != (block, mode, planned_steps):
                raise ValueError('Resume block/mode/planned schedule changed')
            if model.fingerprint() != resume['fingerprint'] or model.configuration_hash() != resume['configuration_hash']:
                raise ValueError('Resume requires matching restored mathematical and configuration state')
            if resume['ordered_parameter_names'] != self.ordered_parameter_names:
                raise ValueError('Adam parameter-name order changed')
            self.optimizer.load_state_dict(copy.deepcopy(resume['optimizer']))
            if not exact(self.optimizer.state_dict(), resume['optimizer']):
                raise ValueError('Adam moments/metadata changed on restore')
            self.completed = int(schedule['completed'])
            if not 0 <= self.completed <= self.planned_steps:
                raise ValueError('Invalid resumed completed step count')
            account = resume['accounting']
            if account['completed_updates'] != self.completed:
                raise ValueError('Restored block update accounting mismatch')
            self.training_seconds = account['block_training_seconds']
            self.budget_updates_after_last_step = account['budget_updates_after_last_step']
            self.projection_steps = account['projection_steps']
            self.matrices_projected = account['matrices_projected']
            self.projection_correction_l2_sum = account['projection_correction_l2_sum']
            self.last_update = copy.deepcopy(resume['last_update'])
            set_rng(resume['rng'])
        self._all_params = [p for group in self.optimizer.param_groups for p in group['params']]
        self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.gp.likelihood, model.gp)
        self._fixed_embedding = None
        if mode == 'gp':
            with torch.no_grad():
                self._fixed_embedding = model.encode(model.x)[0].detach().clone()

    @property
    def lrs(self):
        return {group['name']: float(group['lr']) for group in self.optimizer.param_groups}

    @classmethod
    def from_snapshot(cls, state):
        schedule = state['scheduler']
        return cls(restore(state), schedule['mode'], schedule['planned_steps'], schedule['block'], resume=state)

    def step(self, budget):
        budget.check()
        if self.completed >= self.planned_steps:
            raise RuntimeError('Fixed block endpoint already reached')
        tick, updated = time.monotonic(), 0
        try:
            for group in self.optimizer.param_groups:
                group['lr'] = group['base_lr'] * (.1 if self.completed >= int(.8 * self.planned_steps) else 1.)
            self.optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for p in self.model.mapping_params()] if hasattr(self.model, 'mapper') else []
            loss, components = self.model.objective(self.mode, self._mll, self._fixed_embedding)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            mapping_gradient = _grad_norm(self.model.mapping_params())
            gp_gradient = _grad_norm(self.model.gp.parameters())
            gradient = float(torch.nn.utils.clip_grad_norm_(self._all_params, 5., error_if_nonfinite=True))
            self.optimizer.step()
            # Count a performed Adam update even if the subsequent projection
            # itself fails; such a failure is INVALID and cannot be hidden.
            self.completed += 1
            updated = 1
            projection = {'projection_evaluated': False, 'projection_applied': False, 'matrices_projected': 0,
                          'projection_correction_l2': 0., 'w1_spectral_norm': None,
                          'w2_spectral_norm': None, 'residual_lipschitz_upper_bound': None}
            if self.mode in ('regression', 'joint') and self.model.constraint_mode == 'bounded':
                projection = project_mapping(self.model)
            self.projection_steps += int(projection['projection_applied'])
            self.matrices_projected += projection['matrices_projected']
            self.projection_correction_l2_sum += projection['projection_correction_l2']
            change = math.sqrt(sum(float((p.detach() - old).square().sum())
                                   for p, old in zip(self.model.mapping_params(), before)))
            self.model.cache_version = None
            self.last_update = {'block': self.block, 'mode': self.mode, 'step': self.completed,
                'planned_steps': self.planned_steps, 'loss_before_update': float(loss.detach()), **components,
                'gradient_before_clip': gradient, 'mapping_gradient_before_clip': mapping_gradient,
                'gp_gradient_before_clip': gp_gradient, 'actual_mapping_parameter_change_l2': change,
                'constraint_mode': self.model.constraint_mode, 'projection': projection, 'learning_rates': self.lrs,
                'projection_steps_cumulative': self.projection_steps,
                'projection_step_fraction': self.projection_steps / self.completed,
                'matrices_projected_cumulative': self.matrices_projected,
                'projection_correction_l2_sum': self.projection_correction_l2_sum}
        except Exception:
            budget.failed_training_attempts += 1
            raise
        finally:
            elapsed = time.monotonic() - tick
            self.training_seconds += elapsed
            budget.charge(elapsed, kind='training', updates=updated)
            self.budget_updates_after_last_step = budget.updates
        self.last_update.update(step_training_seconds=elapsed, block_training_seconds=self.training_seconds,
                                budget_total_seconds=budget.seconds, budget_formal_updates=budget.updates)
        budget.check()
        return self.last_update

    def snapshot(self):
        state = snapshot(self.model)
        state.update(optimizer=copy.deepcopy(self.optimizer.state_dict()),
            ordered_parameter_names=copy.deepcopy(self.ordered_parameter_names), last_update=copy.deepcopy(self.last_update),
            scheduler={'block': self.block, 'mode': self.mode, 'planned_steps': self.planned_steps,
                       'completed': self.completed, 'policy': 'first80base_last20base_div10',
                       'next_lr_stage': 'last20' if self.completed >= int(.8 * self.planned_steps) else 'first80'},
            accounting={'block_training_seconds': self.training_seconds, 'completed_updates': self.completed,
                        'budget_updates_after_last_step': self.budget_updates_after_last_step,
                        'projection_steps': self.projection_steps, 'matrices_projected': self.matrices_projected,
                        'projection_correction_l2_sum': self.projection_correction_l2_sum})
        return state


def check_forks(common):
    names = ['R_FROZEN', 'R_REG', 'R_DKL_C', 'R_DKL_U']
    branches = {name: common.clone(constraint_mode='unbounded' if name == 'R_DKL_U' else 'bounded') for name in names}
    models = [common] + list(branches.values())
    stores, modules = [], []
    for model in models:
        tensors = list(model.state().values()) + list(model.gp.train_inputs) + [model.gp.train_targets]
        stores.append({v.untyped_storage().data_ptr() for v in tensors if v.numel()})
        modules.append({id(m) for root in (model.mapper, model.gp) for m in root.modules()})
    overlaps = [{'left': a, 'right': b, 'tensor_storages': len(stores[a] & stores[b]), 'modules': len(modules[a] & modules[b])}
                for a in range(len(models)) for b in range(a + 1, len(models)) if stores[a] & stores[b] or modules[a] & modules[b]]
    same = all(model.fingerprint() == common.fingerprint() for model in branches.values())
    checks = {'pass': same and not overlaps, 'same_mathematical_state': same,
              'common_state_hash': common.fingerprint(), 'cross_model_overlaps': overlaps,
              'branch_configuration_hashes': {name: model.configuration_hash() for name, model in branches.items()}}
    if not checks['pass']:
        raise RuntimeError('Four residual branches are not exactly initialized or mutable-state isolated')
    return {'branches': branches, 'checks': checks}
