"""Latent posterior adapter, with no data-source or oracle imports."""
import numpy as np
import torch
import gpytorch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from surrogate.accuracy_gp import StableQLogExpectedImprovement
from analyse.structured_kernel_phase1.gp import NumericalPolicy


class FrozenAdapter(Model):
    def __init__(self, fitted_gp, jitter_index=0):
        super().__init__()
        self.gp = fitted_gp
        self.gp.eval()
        self.gp.likelihood.eval()
        self.policy = NumericalPolicy(jitter_index)

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return torch.Size([])

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
        if X.dtype != torch.float64 or X.device.type != 'cpu' or X.ndim < 2 or X.shape[-2] != 1:
            raise ValueError('Expected CPU float64 q=1 inputs')
        if observation_noise is not False or posterior_transform is not None or output_indices not in (None,[0]):
            raise ValueError('Only the untransformed latent posterior is permitted')
        def evaluate():
            self.gp.eval()
            distribution = self.gp(X)
            # Materialize inside the strict phase1 numerical policy; the outer
            # production wrapper cannot alter conditioning jitter/noise.
            covariance = distribution.lazy_covariance_matrix.to_dense()
            mean = distribution.mean
            if not bool(torch.isfinite(mean).all() and torch.isfinite(covariance).all()):
                raise FloatingPointError('Nonfinite posterior')
            if bool((covariance.diagonal(dim1=-2,dim2=-1) < 0).any()):
                raise FloatingPointError('Negative latent variance; no clipping allowed')
            return GPyTorchPosterior(gpytorch.distributions.MultivariateNormal(mean, covariance))
        return self.policy.evaluate(evaluate, self.gp, 'acquisition_posterior')


def mc_seed(seed, step):
    if not 1 <= step <= 20:
        raise ValueError('qLogEI step must be 1..20')
    return int(np.random.SeedSequence([seed,step,1]).generate_state(1,dtype=np.uint32)[0])


def choose(ids, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if len(ids) != len(scores) or not len(ids) or not np.isfinite(scores).all():
        raise ValueError('Invalid finite-pool scores')
    best = scores.max()
    return min(cid for cid,value in zip(ids,scores) if abs(value-best) <= 1e-12)


def enumerate_scores(adapter, X, best_f, seed, step, chunk_size=64):
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=mc_seed(seed,step))
    acquisition = StableQLogExpectedImprovement(model=adapter,best_f=float(best_f),sampler=sampler,
        fat=True,tau_max=.01,tau_relu=1e-6,eta=.001)
    values = []
    with torch.no_grad():
        for chunk in X.split(chunk_size):
            values.append(acquisition(chunk.unsqueeze(-2)).detach())
    result = torch.cat(values)
    if result.shape != (len(X),) or not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Invalid acquisition result')
    return result.numpy(), sampler.base_samples.detach().clone()
