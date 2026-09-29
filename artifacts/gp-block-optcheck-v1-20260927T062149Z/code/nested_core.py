"""Dense, deterministic nested Matérn GPs. No data paths or query observations.

All public numerical entry points require a single persistent Budget. Objective
requests (including repeated parameters) are distinct from accepted iterations.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
import numpy as np
import scipy
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

METHODS = ('A_REFIT', 'B_REFIT', 'C_REFIT')
TRAIN_KEYS = {'ids', 'architectures', 'x_standardized16', 'y_standardized', 'known_variance_standardized'}
QUERY_KEYS = {'ids', 'architectures', 'x_standardized16'}
FORMULA_VERSION = 'dense_matern52_nested_closed_boundary_v1'
TIE = 1e-10
LOW_SIGNAL, LOW_NOISE, LOW_LENGTH = math.log(1e-8), math.log(1e-6), math.log(1e-4)
UPPER_A = math.log1p(1e8)
SCOPES = ('synthetic', 'source_validation', 'optimization', 'diagnostic', 'restore', 'case_audit', 'setup')


class BudgetExceeded(RuntimeError): pass
class SyntheticLimit(RuntimeError): pass
class RequestLimit(RuntimeError): pass
class IterationLimit(RuntimeError): pass
class NumericalFailure(RuntimeError): pass


def _finite(value, label):
    if not np.isfinite(value).all():
        raise NumericalFailure('Nonfinite ' + label)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


class Budget:
    """Wall time includes failed operations; nesting charges only elapsed remainder."""
    def __init__(self, path, limit=600):
        if limit != 600:
            raise ValueError('Frozen model limit is 600 seconds')
        self.path = Path(path) if path is not None else None
        old = json.loads(self.path.read_text()) if self.path and self.path.exists() else None
        if old is not None and (old.get('accounting_version') != 'nested_gp_v1' or old['limit'] != limit):
            raise ValueError('Incompatible model ledger')
        self.limit = float(limit)
        self.seconds_by_scope = copy.deepcopy(old['seconds_by_scope']) if old else dict.fromkeys(SCOPES, 0.)
        self.counts = copy.deepcopy(old['counts']) if old else {s: {'nll_gradient_requests': 0, 'successful_requests': 0,
            'failed_requests': 0, 'cholesky_calls': 0, 'prediction_calls': 0, 'kernel_calls': 0,
            'model_constructions': 0, 'solver_calls': 0, 'accepted_iterations': 0} for s in SCOPES}
        self.operations = copy.deepcopy(old['operations']) if old else {}
        self.methods = copy.deepcopy(old['methods']) if old else {m: {'optimization_requests': 0, 'accepted_iterations': 0} for m in METHODS}
        self.overrun_seconds = float(old.get('overrun_seconds', 0)) if old else 0.
        self.persist()

    @property
    def seconds(self): return math.fsum(self.seconds_by_scope.values())
    @property
    def synthetic_requests(self): return self.counts['synthetic']['nll_gradient_requests']
    @property
    def optimization_requests(self): return self.counts['optimization']['nll_gradient_requests']
    @property
    def nll_requests(self): return sum(v['nll_gradient_requests'] for v in self.counts.values())
    def as_dict(self):
        return {'accounting_version': 'nested_gp_v1', 'limit': self.limit, 'seconds': self.seconds,
                'seconds_by_scope': copy.deepcopy(self.seconds_by_scope), 'counts': copy.deepcopy(self.counts),
                'methods': copy.deepcopy(self.methods), 'operations': copy.deepcopy(self.operations),
                'synthetic_nll_gradient_requests': self.synthetic_requests, 'synthetic_request_limit': 300,
                'optimization_nll_gradient_requests': self.optimization_requests, 'optimization_request_limit': 18000,
                'total_nll_gradient_requests': self.nll_requests, 'overrun_seconds': self.overrun_seconds,
                'adam_updates': 0, 'automatic_resume': False, 'clock': 'time.monotonic'}
    def persist(self):
        if self.path: _atomic(self.path, self.as_dict())
    def check(self):
        if self.seconds >= self.limit: raise BudgetExceeded('Model budget exhausted')
    def charge(self, seconds, scope='setup'):
        if scope not in SCOPES or not math.isfinite(seconds) or seconds < 0: raise ValueError('Invalid charge')
        self.seconds_by_scope[scope] += float(seconds)
        self.overrun_seconds = max(self.overrun_seconds, self.seconds - self.limit, 0.)
        self.persist()
    @contextmanager
    def measure(self, operation, scope='setup'):
        self.check(); before = self.seconds; started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            self.charge(max(0., elapsed - (self.seconds - before)), scope)
            cell = self.operations.setdefault(operation, {}).setdefault(scope, {'calls': 0, 'wall_seconds': 0.})
            cell['calls'] += 1; cell['wall_seconds'] += elapsed
            cell['max_wall_seconds'] = max(cell.get('max_wall_seconds',0.),elapsed)
            self.persist()
        self.check()
    def before_request(self, scope, method):
        self.check()
        if scope not in SCOPES: raise ValueError('Unknown request scope')
        if scope == 'synthetic' and self.synthetic_requests >= 300: raise SyntheticLimit('300 synthetic requests exhausted')
        if scope == 'optimization' and self.optimization_requests >= 18000: raise RequestLimit('18000 formal requests exhausted')
        self.counts[scope]['nll_gradient_requests'] += 1
        if scope == 'optimization': self.methods[method]['optimization_requests'] += 1
        self.persist()
    def count(self, scope, name):
        self.counts[scope][name] += 1


def cpu_setup(budget, scope='setup'):
    with budget.measure('cpu_setup', scope):
        np.random.seed(0)
        from threadpoolctl import threadpool_limits, threadpool_info
        threadpool_limits(limits=1)
        return {'numpy': np.__version__, 'scipy': scipy.__version__, 'threadpools': threadpool_info(),
                'dtype': 'float64', 'rng_seed': 0, 'device': 'cpu'}


def bounds(method):
    if method not in METHODS: raise ValueError('Unknown method')
    base = [(None, None), (LOW_SIGNAL, None), (LOW_NOISE, None)]
    return base + ([(0., UPPER_A), (LOW_LENGTH, None)] if method == 'C_REFIT' else [(LOW_LENGTH, None)])


def common_theta(method):
    if method not in ('A_REFIT', 'B_REFIT'): raise ValueError('C has no additional common start')
    return [0., 0., math.log(.01), 0.]


def parent_theta(method, physical):
    result = [float(physical['constant_mean']), math.log(physical['outputscale']),
              math.log(physical['extra_noise_variance_standardized'])]
    if method == 'C_REFIT':
        result += [math.log1p(1 / physical['lengthscale_architecture'] ** 2), math.log(physical['lengthscale_hp'])]
    elif method in ('A_REFIT', 'B_REFIT'):
        result += [math.log(physical['lengthscale_hp'])]
    else: raise ValueError('Unknown method')
    return result


def embed_into_C(method, theta):
    if method not in ('A_REFIT', 'B_REFIT') or len(theta) != 4: raise ValueError('Expected A/B source')
    return list(map(float, theta[:3])) + [math.log1p(math.exp(-2 * theta[3])) if method == 'A_REFIT' else 0., float(theta[3])]


def _coordinates(method, theta):
    p = np.array(theta, dtype=np.float64, copy=True)
    if p.shape != (len(bounds(method)),): raise ValueError('Wrong coordinate count')
    _finite(p, 'coordinates')
    for value, (low, high) in zip(p, bounds(method)):
        if low is not None and value < low or high is not None and value > high:
            raise NumericalFailure('Infeasible coordinate')
    return p


def physical_parameters(method, theta):
    p = _coordinates(method, theta)
    signal, noise = math.exp(p[1]), math.exp(p[2])
    ell_h = math.exp(p[-1]); beta = math.expm1(p[3]) if method == 'C_REFIT' else (0. if method == 'B_REFIT' else math.exp(-2*p[3]))
    return {'constant_mean': float(p[0]), 'outputscale': signal, 'extra_noise_variance_standardized': noise,
            'lengthscale_hp': ell_h, 'lengthscale_architecture': None if beta == 0 else 1 / math.sqrt(beta),
            'beta_architecture': beta, 'architecture_disabled': beta == 0,
            'a': float(p[3]) if method == 'C_REFIT' else None}


def _distances(x, z):
    delta = x[:, None, :] - z[None, :, :]
    return np.sum(delta[:, :, :12] ** 2, axis=2), np.sum(delta[:, :, 12:] ** 2, axis=2)


def _kernel(method, p, da, dh, derivatives=False):
    signal = np.exp(p[1]); hp = np.exp(-2*p[-1]) * dh
    if method == 'A_REFIT': s = np.exp(-2*p[3]) * (da + dh)
    elif method == 'B_REFIT': s = hp
    else: s = np.expm1(p[3]) * da + hp
    t = np.sqrt(5*s); et = np.exp(-t)
    k = signal * (1 + t + t*t/3) * et
    _finite(k, 'kernel')
    if not derivatives: return k
    derivative_s = -(5/6)*signal*(1+t)*et
    d = [k]
    if method == 'C_REFIT': d += [derivative_s*np.exp(p[3])*da, derivative_s*(-2*hp)]
    else: d += [derivative_s*(-2*s)]
    return k, d


class DenseGP:
    """Construct via make_model: constructor performs no uncharged numerical work."""
    def __init__(self, method, training, budget, scope):
        if set(training) != TRAIN_KEYS: raise ValueError('Training interface is an exact whitelist')
        if method not in METHODS: raise ValueError('Unknown method')
        self.method = method
        self.ids = list(training['ids']); self.architectures = list(training['architectures'])
        self.x = np.array(training['x_standardized16'], dtype=np.float64, copy=True)
        self.y = np.array(training['y_standardized'], dtype=np.float64, copy=True)
        self.known = np.array(training['known_variance_standardized'], dtype=np.float64, copy=True)
        n = len(self.ids)
        if self.x.shape != (n, 16) or self.y.shape != (n,) or self.known.shape != (n,) or len(self.architectures) != n or len(set(self.ids)) != n or n < 2:
            raise ValueError('Invalid training shapes/identities')
        for v in (self.x, self.y, self.known): _finite(v, 'training input'); v.setflags(write=False)
        if np.any(self.known < 0): raise ValueError('Known variance must be nonnegative')
        self.da, self.dh = _distances(self.x, self.x)
        self.da.setflags(write=False); self.dh.setflags(write=False)
        self.n = n; self.training_identity = digest(training)
        budget.count(scope, 'model_constructions')


def make_model(method, training, budget, scope='setup'):
    with budget.measure('model_' + method, scope): return DenseGP(method, training, budget, scope)


def _factor(model, p, budget, scope):
    kernel, derivatives = _kernel(model.method, p, model.da, model.dh, True)
    ky = kernel + np.diag(model.known + np.exp(p[2]))
    budget.count(scope, 'cholesky_calls')
    try: chol = cholesky(ky, lower=True, check_finite=True)
    except Exception as error: raise NumericalFailure('Cholesky failed; jitter is forbidden') from error
    return kernel, derivatives, chol


def evaluate(model, theta, budget, scope='optimization'):
    """Every invocation is one joint f/g request, including duplicate theta."""
    budget.before_request(scope, model.method)
    try:
        with budget.measure('nll_gradient_' + model.method, scope):
            p = _coordinates(model.method, theta)
            _, derivatives, chol = _factor(model, p, budget, scope)
            residual = model.y - p[0]
            alpha = cho_solve((chol, True), residual, check_finite=False)
            value = (.5*np.dot(residual, alpha) + np.log(np.diag(chol)).sum() + .5*model.n*math.log(2*math.pi)) / model.n
            inverse = cho_solve((chol, True), np.eye(model.n), check_finite=False)
            w = (inverse - np.outer(alpha, alpha)) / (2*model.n)
            gradient = [-alpha.sum()/model.n, np.sum(w*derivatives[0]), np.exp(p[2])*np.trace(w)]
            gradient += [np.sum(w*d) for d in derivatives[1:]]
            _finite(value, 'NLL'); _finite(gradient, 'gradient')
            result = (float(value), np.array(gradient, dtype=np.float64))
        budget.count(scope, 'successful_requests'); budget.persist()
        return result
    except BaseException:
        budget.count(scope, 'failed_requests'); budget.persist(); raise


def independent_evaluate(model, theta, budget, scope='diagnostic'):
    """Independent loops, NumPy Cholesky/triangular solves, and analytic gradient.

    Does not use _kernel, cached distances, _factor, or the production objective.
    Finite differences in preflight separately test both implementations.
    """
    budget.before_request(scope, model.method)
    try:
        with budget.measure('independent_nll_gradient_' + model.method, scope):
            p = _coordinates(model.method, theta); n = model.n; d = len(p)
            cov = np.empty((n, n)); derivatives = np.zeros((d, n, n))
            signal, noise, hp_beta = math.exp(p[1]), math.exp(p[2]), math.exp(-2*p[-1])
            arch_beta = (math.expm1(p[3]) if model.method == 'C_REFIT' else hp_beta if model.method == 'A_REFIT' else 0.)
            for i in range(n):
                for j in range(i+1):
                    da = math.fsum((float(model.x[i,k])-float(model.x[j,k]))**2 for k in range(12))
                    dh = math.fsum((float(model.x[i,k])-float(model.x[j,k]))**2 for k in range(12,16))
                    distance = arch_beta*da + hp_beta*dh; radius = math.sqrt(5*distance)
                    entry = signal*(1+radius+radius*radius/3)*math.exp(-radius)
                    ds = -5*signal*(1+radius)*math.exp(-radius)/6
                    cov[i,j] = cov[j,i] = entry
                    derivatives[1,i,j] = derivatives[1,j,i] = entry
                    if model.method == 'C_REFIT':
                        derivatives[3,i,j] = derivatives[3,j,i] = ds*math.exp(p[3])*da
                        derivatives[4,i,j] = derivatives[4,j,i] = -2*ds*hp_beta*dh
                    else: derivatives[3,i,j] = derivatives[3,j,i] = -2*ds*distance
            cov += np.diag(model.known + noise); derivatives[2] = noise*np.eye(n)
            budget.count(scope, 'cholesky_calls')
            try: lower = np.linalg.cholesky(cov)
            except Exception as error: raise NumericalFailure('Independent Cholesky failed') from error
            residual = model.y-p[0]
            alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, residual))
            identity_solution = np.linalg.solve(lower.T, np.linalg.solve(lower, np.eye(n)))
            value = (.5*float(residual@alpha) + math.fsum(math.log(float(lower[i,i])) for i in range(n)) + .5*n*math.log(2*math.pi))/n
            w = (identity_solution-np.outer(alpha, alpha))/(2*n)
            gradient = np.array([-math.fsum(alpha)/n] + [math.fsum((w*derivatives[k]).ravel()) for k in range(1,d)])
            _finite(value, 'independent objective'); _finite(gradient, 'independent gradient')
        budget.count(scope, 'successful_requests'); budget.persist()
        return float(value), gradient
    except BaseException:
        budget.count(scope, 'failed_requests'); budget.persist(); raise


def _query(query):
    if set(query) != QUERY_KEYS: raise ValueError('Prediction interface is an exact whitelist; labels/SE forbidden')
    ids = list(query['ids']); x = np.array(query['x_standardized16'], dtype=np.float64, copy=True)
    if x.shape != (len(ids), 16) or len(query['architectures']) != len(ids) or len(set(ids)) != len(ids): raise ValueError('Query shape/identity')
    _finite(x, 'query input'); return x


def kernels(model, theta, query, budget, scope='diagnostic'):
    with budget.measure('kernels_' + model.method, scope):
        p = _coordinates(model.method, theta); q = _query(query)
        result = {'TT': _kernel(model.method,p,model.da,model.dh),
                  'QT': _kernel(model.method,p,*_distances(q,model.x)), 'QQ': _kernel(model.method,p,*_distances(q,q))}
        budget.count(scope, 'kernel_calls'); return result


def independent_prediction(model, theta, query, scaler, budget, scope='diagnostic'):
    """Scalar closed-form covariance and NumPy solves, independent of _kernel."""
    with budget.measure('independent_predict_' + model.method, scope):
        p=_coordinates(model.method,theta);q=_query(query);joined=np.concatenate((model.x,q));n=model.n
        signal,noise=math.exp(p[1]),math.exp(p[2]);hp=math.exp(-2*p[-1])
        arch=math.expm1(p[3]) if model.method=='C_REFIT' else hp if model.method=='A_REFIT' else 0.
        full=np.empty((len(joined),len(joined)))
        for i in range(len(joined)):
            for j in range(i+1):
                da=math.fsum((float(joined[i,k])-float(joined[j,k]))**2 for k in range(12))
                dh=math.fsum((float(joined[i,k])-float(joined[j,k]))**2 for k in range(12,16))
                radius=math.sqrt(5*(arch*da+hp*dh))
                full[i,j]=full[j,i]=signal*(1+radius+radius**2/3)*math.exp(-radius)
        budget.count(scope,'cholesky_calls')
        try:lower=np.linalg.cholesky(full[:n,:n]+np.diag(model.known+noise))
        except Exception as error:raise NumericalFailure('Independent posterior Cholesky failed') from error
        solve=lambda rhs:np.linalg.solve(lower.T,np.linalg.solve(lower,rhs))
        mean=p[0]+full[n:,:n]@solve(model.y-p[0]);cov=full[n:,n:]-full[n:,:n]@solve(full[:n,n:])
        _finite(mean,'independent posterior mean');_finite(cov,'independent posterior covariance')
        budget.count(scope,'prediction_calls');budget.count(scope,'kernel_calls')
        return {'mean_standardized':mean.tolist(),'latent_covariance_standardized':cov.tolist(),
                'mean_raw':(float(scaler['y_mean'])+float(scaler['y_scale'])*mean).tolist(),
                'latent_covariance_raw':(float(scaler['y_scale'])**2*cov).tolist(),
                'TT':full[:n,:n].tolist(),'QT':full[n:,:n].tolist(),'QQ':full[n:,n:].tolist()}


def predict(model, theta, query, scaler, budget, scope='diagnostic'):
    with budget.measure('predict_' + model.method, scope):
        p = _coordinates(model.method, theta); q = _query(query)
        center, scale = float(scaler['y_mean']), float(scaler['y_scale'])
        if not math.isfinite(center) or not math.isfinite(scale) or scale < 1e-8: raise ValueError('Invalid frozen y scaler')
        _, _, chol = _factor(model,p,budget,scope)
        qt = _kernel(model.method,p,*_distances(q,model.x)); qq = _kernel(model.method,p,*_distances(q,q))
        alpha = cho_solve((chol, True), model.y-p[0], check_finite=False)
        mean = p[0] + qt@alpha; projected = solve_triangular(chol,qt.T,lower=True,check_finite=False)
        covariance = qq-projected.T@projected
        _finite(mean,'prediction mean'); _finite(covariance,'latent covariance')
        symmetry = float(np.max(np.abs(covariance-covariance.T)))
        eigenvalue = float(np.linalg.eigvalsh((covariance+covariance.T)/2).min())
        diagonal = np.diag(covariance).copy()
        if symmetry > 1e-10 or eigenvalue < -1e-8 or diagonal.min() < -1e-8: raise NumericalFailure('Invalid latent covariance')
        raw = center+scale*mean; order = sorted(range(len(raw)), key=lambda i:(-float(raw[i]),query['ids'][i]))
        budget.count(scope, 'prediction_calls')
        return {'candidate_ids': list(query['ids']), 'architecture_fingerprints': list(query['architectures']),
                'mean_standardized': mean.tolist(), 'mean_raw': raw.tolist(),
                'latent_covariance_standardized': covariance.tolist(), 'latent_covariance_raw': (scale**2*covariance).tolist(),
                'latent_variance_standardized': diagonal.tolist(), 'latent_variance_raw': (scale**2*diagonal).tolist(),
                'model_variance_raw': (scale**2*(np.maximum(diagonal,0)+math.exp(p[2]))).tolist(),
                'extra_noise_variance_standardized': math.exp(p[2]), 'negative_roundoff_variance_count': int(np.sum(diagonal<0)),
                'latent_covariance_symmetry_max_abs': symmetry, 'latent_covariance_min_eigenvalue': eigenvalue,
                'y_center': center, 'y_scale': scale, 'ranking_ids': [query['ids'][i] for i in order],
                'top5_ids': [query['ids'][i] for i in order[:5]], 'query_labels_or_SE_present': False}


def select_candidates(trajectories):
    candidates = [state for trace in trajectories for state in trace['accepted']]
    if not candidates: raise ValueError('No accepted candidates')
    minimum = min(c['nll'] for c in candidates)
    chosen = min((c for c in candidates if c['nll']-minimum <= TIE), key=lambda c:(c['start_index'],c['iteration']))
    return {**copy.deepcopy(chosen), 'numerical_minimum_nll': minimum, 'selected_minus_numerical_minimum': chosen['nll']-minimum,
            'selection_rule': 'min training NLL; <=1e-10 tie by start index then accepted iteration', 'candidate_count': len(candidates)}


def run_trajectory(model, x0, start_index, max_requests, max_iterations, budget, scope='optimization', request_log=None, source=None):
    if max_requests < 1 or max_iterations < 1: raise ValueError('Positive hard limits required')
    if scope == 'optimization' and (max_requests,max_iterations) != ((200,150) if model.method=='C_REFIT' else (300,250)):
        raise ValueError('Formal per-start request/iteration limits are frozen')
    x0 = list(map(float,x0)); requests=[]; accepted=[]; latest={}; refused=0
    path = Path(request_log) if request_log is not None else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise ValueError('Request log already exists; no resume')
    def emit(record):
        if path:
            with path.open('a') as stream: stream.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
    def key(x): return np.asarray(x,dtype=np.float64).tobytes()
    def fg(x):
        nonlocal refused
        if len(requests) >= max_requests:
            refused += 1; raise RequestLimit('Per-trajectory joint request cap')
        request={'request':len(requests)+1,'coordinates':np.asarray(x,dtype=np.float64).tolist(),'status':'STARTED'}
        requests.append(request); emit({'event':'request_started',**request})
        try:
            before_request_seconds=budget.seconds
            value,gradient=evaluate(model,x,budget,scope)
            request.update(nll=value,gradient=gradient.tolist(),status='FINITE',model_seconds=budget.seconds-before_request_seconds)
            latest[key(x)]=request; emit({'event':'request_finished',**request})
            return value,gradient
        except BaseException as error:
            request.update(status='FAILED',error=repr(error));emit({'event':'request_failed',**request});raise
    def keep(x,iteration):
        request=latest.get(key(x))
        if request is None: raise NumericalFailure('Accepted optimizer state lacks an evaluated request')
        candidate={k:copy.deepcopy(request[k]) for k in ('coordinates','nll','gradient','request')}
        candidate.update(start_index=int(start_index),iteration=iteration,kind='initial' if iteration==0 else 'accepted_iteration')
        accepted.append(candidate);emit({'event':'candidate_accepted',**candidate})
    def callback(xk):
        iteration=len(accepted)
        if iteration>max_iterations: raise NumericalFailure('Accepted iteration limit was bypassed')
        keep(xk,iteration);budget.count(scope,'accepted_iterations')
        if scope=='optimization':budget.methods[model.method]['accepted_iterations']+=1
        budget.persist()
        if iteration==max_iterations:raise IterationLimit('Hard accepted-iteration limit')
    before=budget.seconds
    with budget.measure('solver_'+model.method,scope):
        budget.count(scope,'solver_calls')
        fg(x0);keep(x0,0)
        result=None
        try:
            result=minimize(fg,np.array(x0),method='L-BFGS-B',jac=True,bounds=bounds(model.method),callback=callback,
                            options={'ftol':1e-12,'gtol':1e-6,'maxcor':10,'maxls':20,'maxfun':max_requests,'maxiter':max_iterations})
            code='CONVERGED' if result.success else 'FINITE_SOLVER_TERMINATION'
            message=str(result.message)
        except RequestLimit as error:code='REQUEST_LIMIT';message=str(error)
        except IterationLimit as error:code='ITERATION_LIMIT';message=str(error)
        returned=None
        if result is not None:
            request=latest.get(key(result.x))
            if request is None:raise NumericalFailure('Returned optimizer coordinates lack request identity')
            returned={k:copy.deepcopy(request[k]) for k in ('coordinates','nll','gradient','request')}
            returned['accepted_candidate']=any(c['coordinates']==returned['coordinates'] for c in accepted)
        trace={'method':model.method,'start_index':int(start_index),'source':copy.deepcopy(source),'initial_coordinates':x0,
               'max_requests':max_requests,'max_iterations':max_iterations,'actual_requests':len(requests),
               'accepted_iterations':len(accepted)-1,'refused_over_limit_requests':refused,'requests':requests,'accepted':accepted,
               'termination':code,'message':message,'scipy_success':bool(result.success) if result is not None else None,
               'scipy_status':int(result.status) if result is not None else None,'scipy_nfev':int(result.nfev) if result is not None else None,
               'scipy_nit':int(result.nit) if result is not None else None,'returned':returned,'fresh_hessian_history':True}
        trace['best']=select_candidates([trace])
    trace['model_seconds']=budget.seconds-before
    return trace


def projected_gradient(theta, gradient, coordinate_bounds):
    result=[]
    for x,g,(low,high) in zip(theta,gradient,coordinate_bounds):
        if low is not None and abs(x-low)<=1e-12*max(1.,abs(low)):g=min(g,0.)
        if high is not None and abs(x-high)<=1e-12*max(1.,abs(high)):g=max(g,0.)
        result.append(float(g))
    return result


def check_gradient_consistency(selected_gradient, independent_gradient):
    """Formal pre-freeze audit guard; uses the already prescribed gradient tolerance."""
    if len(selected_gradient)!=len(independent_gradient):raise NumericalFailure('Independent gradient dimension differs')
    differences=[]
    for a,b in zip(selected_gradient,independent_gradient):
        a,b=float(a),float(b)
        if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-4):
            raise NumericalFailure('Independent gradient disagrees with selected accepted-state gradient')
        differences.append(abs(a-b))
    return {'pass':True,'maximum_absolute_error':max(differences,default=0.),'atol':1e-6,'rtol':1e-4,
            'selected_gradient':list(map(float,selected_gradient))}


def diagnose_selected(model, selected, known_feasible, budget, scope='diagnostic'):
    with budget.measure('selected_diagnostic_'+model.method,scope):
        theta=selected['coordinates'];value,gradient=independent_evaluate(model,theta,budget,scope)
        if not np.isclose(value,selected['nll'],atol=1e-10,rtol=1e-8):raise NumericalFailure('Independent selected NLL mismatch')
        gradient_consistency=check_gradient_consistency(selected['gradient'],gradient)
        cb=bounds(model.method);pg=projected_gradient(theta,gradient,cb);probes=[]
        for k,(low,high) in enumerate(cb):
            for direction in (-1,1):
                probe=list(theta);probe[k]+=direction*1e-3
                if low is not None:probe[k]=max(probe[k],low)
                if high is not None:probe[k]=min(probe[k],high)
                rec={'coordinate':k,'direction':direction,'probe_coordinates':probe,'skipped_no_displacement':probe[k]==theta[k]}
                if not rec['skipped_no_displacement']:
                    rec['nll']=evaluate(model,probe,budget,scope)[0];rec['improvement']=value-rec['nll']
                probes.append(rec)
        norm=max(map(abs,pg));dominance=all(value-r['nll']<=1e-6 for r in known_feasible)
        return {'method':model.method,'selected_nll':value,'coordinates':list(theta),'independent_gradient':gradient.tolist(),
                'bounds':cb,'projected_gradient':pg,'projected_gradient_infinity_norm':norm,'probes':probes,
                'selected_gradient_consistency':gradient_consistency,
                'known_feasible_nlls':copy.deepcopy(known_feasible),'KNOWN_FEASIBLE_DOMINANCE_PASS':dominance,
                'LOCAL_STATIONARITY_PASS':norm<=1e-5,'LOCAL_PROBE_PASS':all(r.get('improvement',0)<=1e-6 for r in probes),
                'architecture_disabled':model.method=='C_REFIT' and theta[3]==0.,'probes_returned_to_optimizer':False,
                'independent_gradient_method':'separate pair loops and NumPy Cholesky; no production kernel/objective/cache'}


def snapshot(model, selected, scaler, budget, identity=None, scope='diagnostic'):
    with budget.measure('snapshot_'+model.method,scope):
        theta=selected['coordinates']
        result={'method':model.method,'formula_version':FORMULA_VERSION,'coordinates':list(theta),'bounds':bounds(model.method),
                'physical_parameters':physical_parameters(model.method,theta),'selected':copy.deepcopy(selected),
                'training_identity':model.training_identity,'train_candidate_ids':list(model.ids),'scaler':copy.deepcopy(scaler),
                'identity':copy.deepcopy(identity),'model_seed':None,'execution_rng_seed':0,'resume_allowed':False}
        result['state_identity']=digest(result);return result


def parent_verify(model, theta, parent_prediction, query, scaler, budget, parent_training_kernel=None):
    """One charged reconstruction of a saved parent; never reads query truth."""
    scope='source_validation'
    with budget.measure('parent_verify_'+model.method,scope):
        if parent_prediction['candidate_ids']!=query['ids']:raise NumericalFailure('Parent query ordering differs')
        value,gradient=evaluate(model,theta,budget,scope)
        expected_nll=parent_prediction.get('endpoint_nll_after_update',parent_prediction.get('negative_mll'))
        if expected_nll is None or not np.isclose(value,expected_nll,atol=1e-10,rtol=1e-8):raise NumericalFailure('Parent NLL mismatch')
        pred=predict(model,theta,query,scaler,budget,scope);ks=kernels(model,theta,query,budget,scope)
        differences={'nll':abs(value-float(expected_nll))}
        for field in ('mean_standardized','mean_raw','latent_covariance_standardized','latent_covariance_raw'):
            got=np.asarray(pred[field]);expected=np.asarray(parent_prediction[field]);differences[field]=float(np.max(np.abs(got-expected)))
            if not np.allclose(got,expected,atol=1e-10,rtol=1e-8):raise NumericalFailure('Parent posterior mismatch: '+field)
        if parent_training_kernel is not None:
            differences['training_kernel']=float(np.max(np.abs(ks['TT']-np.asarray(parent_training_kernel))))
            if not np.allclose(ks['TT'],parent_training_kernel,atol=1e-10,rtol=1e-8):raise NumericalFailure('Parent training kernel mismatch')
        q=_query(query);correlations=ks['QT']/math.exp(theta[1]);neighbors=[]
        for i,cid in enumerate(query['ids']):
            indices=sorted(range(model.n),key=lambda j:(-float(correlations[i,j]),model.ids[j]))[:5]
            neighbors.append({'candidate_id':cid,'maximum_normalized_kernel_correlation':float(correlations[i,indices[0]]),
                'neighbors':[{'training_candidate_id':model.ids[j],'architecture_fingerprint':model.architectures[j],
                              'training_mean_raw':float(scaler['y_mean']+scaler['y_scale']*model.y[j]),
                              'normalized_kernel_correlation':float(correlations[i,j]),
                              'hp_distance_standardized':float(np.linalg.norm(q[i,12:]-model.x[j,12:]))} for j in indices]})
        return {'pass':True,'method':model.method,'nll':value,'gradient':gradient.tolist(),'coordinates':list(theta),
                'physical_parameters':physical_parameters(model.method,theta),'maximum_absolute_differences':differences,
                'training_kernel_checked':parent_training_kernel is not None,'neighbors':neighbors,
                'query_labels_or_SE_read':False,'new_optimization':False}


def restore_check(saved, training, query, prediction, budget, scope='restore'):
    with budget.measure('restore_check_'+saved['method'],scope):
        state=copy.deepcopy(saved);identity=state.pop('state_identity')
        if digest(state)!=identity or saved['formula_version']!=FORMULA_VERSION:raise NumericalFailure('Saved scalar state identity differs')
        model=make_model(saved['method'],training,budget,scope)
        if model.training_identity!=saved['training_identity']:raise NumericalFailure('Restore training identity differs')
        value,grad=independent_evaluate(model,saved['coordinates'],budget,scope)
        if not np.isclose(value,saved['selected']['nll'],atol=1e-10,rtol=1e-8):raise NumericalFailure('Restored selected NLL differs')
        restored=independent_prediction(model,saved['coordinates'],query,saved['scaler'],budget,scope)
        differences={}
        for k in ('mean_standardized','mean_raw','latent_covariance_standardized','latent_covariance_raw'):
            a=np.asarray(restored[k]);b=np.asarray(prediction[k]);differences[k]=float(np.max(np.abs(a-b)))
            if not np.allclose(a,b,atol=1e-10,rtol=1e-8):raise NumericalFailure('Restored prediction differs: '+k)
        return {'pass':True,'method':saved['method'],'state_identity':identity,'nll':value,'independent_gradient':grad.tolist(),
                'maximum_absolute_differences':differences,'optimizer_calls':0,'numerical_restore_not_resume':True}
