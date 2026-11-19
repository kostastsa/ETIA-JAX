"""Nonparametric CI tests on dense n x n matrices (GPU-friendly).

* ``CMIknn``  tigramite CMIknn: Frenzel-Pompe kNN estimator of CMI on rank
  data, max-norm. ETIA uses ``significance='fixed_thres'``: dependent iff
  CMI > alpha. A local shuffle test is also given (approximate).
* ``GPDC``    tigramite GPDC: GP regression (RBF + white noise, ML-II
  hyperparameters) of x and y on Z, distance correlation of the residuals
  on uniform marginals, p-value from a simulated null distribution.
* ``CCI``     Tetrad IndTestConditionalCorrelation (Ramsey 2014): kernel
  regression residuals, then correlations of basis functions of the two
  residuals.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax.scipy.stats import norm

from .base import CITest, normal_sf_two_sided, as_index_batch


# --------------------------------------------------------------------------
# CMIknn
# --------------------------------------------------------------------------
def _cheb(A):
    """Max-norm distance matrix of the rows of A (n, d)."""
    return jnp.max(jnp.abs(A[:, None, :] - A[None, :, :]), axis=-1)


@partial(jax.jit, static_argnums=(2, 3))
def _cmi_knn(R, idx, knn, dz):
    """CMI estimate for columns idx = (x, y, z...) of the rank data R."""
    X = R[:, idx]
    d_x = jnp.abs(X[:, 0:1] - X[:, 0:1].T)
    d_y = jnp.abs(X[:, 1:2] - X[:, 1:2].T)
    d_z = _cheb(X[:, 2:]) if dz > 0 else jnp.zeros_like(d_x)
    d_xyz = jnp.maximum(jnp.maximum(d_x, d_y), d_z)
    # distance to the knn-th neighbour (the point itself is the 0-th)
    eps = -jax.lax.top_k(-d_xyz, knn + 1)[0][:, knn]
    eps = eps * 0.999999999
    k_xz = jnp.sum(jnp.maximum(d_x, d_z) <= eps[:, None], axis=1)
    k_yz = jnp.sum(jnp.maximum(d_y, d_z) <= eps[:, None], axis=1)
    k_z = jnp.sum(d_z <= eps[:, None], axis=1) if dz > 0 else jnp.full(X.shape[0], X.shape[0])
    return digamma(knn) - jnp.mean(digamma(k_xz) + digamma(k_yz) - digamma(k_z))


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _cmi_knn_shuffle(R, idx, key, knn, dz, n_perm, shuffle_neighbors):
    """Local permutation null: x_i is replaced by x_j for a random j among the
    ``shuffle_neighbors`` nearest neighbours of i in Z (approximation of
    tigramite's restricted permutation, which draws without replacement)."""
    val = _cmi_knn(R, idx, knn, dz)
    Zd = _cheb(R[:, idx[2:]])
    nb = jax.lax.top_k(-Zd, shuffle_neighbors)[1]        # (n, s)

    def one(k):
        pick = jax.random.randint(k, (R.shape[0],), 0, shuffle_neighbors)
        perm = nb[jnp.arange(R.shape[0]), pick]
        Rp = R.at[:, idx[0]].set(R[perm, idx[0]])
        return _cmi_knn(Rp, idx, knn, dz)

    nulls = jax.lax.map(one, jax.random.split(key, n_perm))
    return val, jnp.mean(nulls >= val)


class CMIknn(CITest):
    name = "CMIknn"

    def __init__(self, data, kinds=None, knn=0.2, significance="fixed_thres",
                 sig_samples=500, shuffle_neighbors=5, seed=0):
        super().__init__(data, kinds)
        rng = np.random.default_rng(seed)
        X = self.data + 1e-6 * self.data.std(0) * rng.random(self.data.shape)  # break ties
        self.R = jnp.asarray(np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64))
        self.knn = max(1, int(knn * self.n)) if knn < 1 else int(knn)
        self.significance = significance
        self.fixed_threshold = significance == "fixed_thres"
        self.sig_samples, self.shuffle_neighbors = sig_samples, shuffle_neighbors
        self.key = jax.random.key(seed)

    def test(self, x, y, Z=()):
        self.n_calls += 1
        idx = jnp.asarray([x, y, *Z], dtype=jnp.int32)
        if self.fixed_threshold:
            v = float(_cmi_knn(self.R, idx, self.knn, len(Z)))
            return v, float("nan")
        if not Z:
            raise ValueError("the shuffle test needs a non-empty Z; use fixed_thres")
        self.key, k = jax.random.split(self.key)
        v, p = _cmi_knn_shuffle(self.R, idx, k, self.knn, len(Z), self.sig_samples,
                                self.shuffle_neighbors)
        return float(v), float(p)


# --------------------------------------------------------------------------
# GPDC
# --------------------------------------------------------------------------
def _standardize(A):
    s = A.std(0)
    return (A - A.mean(0)) / jnp.where(s > 0, s, 1.0)


def _gp_resid(Z, y, n_steps=15):
    """Residual of a GP regression (RBF + white kernel, as tigramite's
    sklearn default). ML-II hyperparameters (log length, log noise), start
    at (0, 0), bounds [1e-5, 1e5] as sklearn. There are only two parameters,
    so a damped Newton method with the exact 2 x 2 Hessian and backtracking
    converges in a few steps."""
    d2 = jnp.sum((Z[:, None, :] - Z[None, :, :]) ** 2, -1)
    lo, hi = np.log(1e-5), np.log(1e5)
    eye = jnp.eye(len(y))

    def parts(theta):
        ell, sn = jnp.exp(theta)
        K = jnp.exp(-0.5 * d2 / ell ** 2)
        L = jnp.linalg.cholesky(K + (sn + 1e-10) * eye)
        a = jax.scipy.linalg.cho_solve((L, True), y)
        return 0.5 * y @ a + jnp.sum(jnp.log(jnp.diag(L))), K, a

    nll = lambda t: parts(t)[0]
    grad, hess = jax.grad(nll), jax.hessian(nll)

    def step(theta, _):
        f0, g, H = nll(theta), grad(theta), hess(theta)
        # Newton step if H is positive definite, else a gradient step
        w, V = jnp.linalg.eigh(H)
        Hd = (V * jnp.maximum(jnp.abs(w), 1e-3)) @ V.T
        delta = jnp.linalg.solve(Hd, g)

        def try_t(carry, _):
            t, best = carry
            cand = jnp.clip(theta - t * delta, lo, hi)
            ok = nll(cand) <= f0 - 1e-4 * t * (g @ delta)
            best = jnp.where((best == 0) & ok, t, best)
            return (t * 0.5, best), None

        (_, t), _ = jax.lax.scan(try_t, (1.0, 0.0), None, length=8)
        return jnp.clip(theta - t * delta, lo, hi), None

    theta, _ = jax.lax.scan(step, jnp.zeros(2), None, length=n_steps)
    _, K, a = parts(theta)
    return y - K @ a


def _uniform(v):
    """tigramite ``_trafo2uniform``: rank / n (distinct values)."""
    n = v.shape[0]
    return (jnp.argsort(jnp.argsort(v)) + 1.0) / n


def _dcor(x, y):
    """Distance correlation (V-statistic), as dcor.distance_correlation."""
    def centred(v):
        D = jnp.abs(v[:, None] - v[None, :])
        return D - D.mean(0) - D.mean(1)[:, None] + D.mean()
    A, B = centred(x), centred(y)
    dcov = jnp.mean(A * B)
    dvx, dvy = jnp.mean(A * A), jnp.mean(B * B)
    return jnp.sqrt(jnp.maximum(dcov, 0.0) / jnp.sqrt(dvx * dvy))


@partial(jax.jit, static_argnums=(2,))
def _gpdc_stat(X, idx, dz):
    A = _standardize(X[:, idx])
    if dz > 0:
        rx = _gp_resid(A[:, 2:], A[:, 0])
        ry = _gp_resid(A[:, 2:], A[:, 1])
    else:
        rx, ry = A[:, 0], A[:, 1]
    return _dcor(_uniform(rx), _uniform(ry))


@partial(jax.jit, static_argnums=(1, 2))
def _dcor_null(key, n, n_samples):
    def one(k):
        u = jax.random.uniform(k, (2, n))
        return _dcor(_uniform(u[0]), _uniform(u[1]))
    return jnp.sort(jax.lax.map(one, jax.random.split(key, n_samples), batch_size=25))


@partial(jax.jit, static_argnums=(2,))
def _gpdc_batch(X, idx, dz):
    return jax.vmap(lambda ix: _gpdc_stat.__wrapped__(X, ix, dz))(idx)


class GPDC(CITest):
    """p = fraction of the null distances correlations > the statistic
    (tigramite 'analytic' significance, null of 500 samples for this n)."""
    name = "GPDC"

    def __init__(self, data, kinds=None, null_samples=500, seed=0):
        super().__init__(data, kinds)
        self.X = jnp.asarray(self.data)
        self.null = _dcor_null(jax.random.key(seed), self.n, null_samples)

    def test(self, x, y, Z=()):
        self.n_calls += 1
        v = _gpdc_stat(self.X, jnp.asarray([x, y, *Z], dtype=jnp.int32), len(Z))
        return float(v), float(jnp.mean(self.null > jnp.abs(v)))

    def test_batch(self, xs, ys, Zs):
        idx = jnp.asarray(as_index_batch(xs, ys, Zs))
        self.n_calls += len(idx)
        v = _gpdc_batch(self.X, idx, idx.shape[1] - 2)
        p = jnp.mean(self.null[None, :] > jnp.abs(v)[:, None], axis=1)
        return np.asarray(v), np.asarray(p)


# --------------------------------------------------------------------------
# CCI (Ramsey 2014), Tetrad 7.6.5 ConditionalCorrelationIndependence
# --------------------------------------------------------------------------
def _tetrad_scale(c):
    """Tetrad ``scale``: min + (d - min) / (max - min); max starts at
    Double.MIN_VALUE (kept for exactness)."""
    mx = jnp.maximum(c.max(), 5e-324)
    mn = c.min()
    return mn + (c - mn) / (mx - mn)


def _mad_bandwidth(c):
    n = c.shape[0]
    med = jnp.median(c)
    mad = jnp.median(jnp.abs(c - med))
    return 1.4826 * mad * ((4.0 / 3.0) / n) ** 0.2


def _std_pop(v):
    v = v - v.mean()
    return v / jnp.sqrt(jnp.mean(v * v))


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _cci(X, idx, dz, n_functions, sample_size, compat):
    """Score of Tetrad's CCI for idx = (x, y, z...) (z in Tetrad name order)."""
    def residuals(target):
        cols = jnp.concatenate([idx[target:target + 1], idx[2:]])
        D = jax.vmap(_tetrad_scale, in_axes=1, out_axes=1)(X[:, cols])   # (n, 1 + dz)
        t = D[:, 0]
        if dz == 0:
            return t
        n = D.shape[0]
        h = jnp.max(jax.vmap(_mad_bandwidth, in_axes=1)(D))
        Zr = D[:, 1:]
        # neighbour sets: union over z of rank windows [q - R, q + R]
        rk = jnp.argsort(jnp.argsort(Zr, axis=0, stable=True), axis=0)    # (n, dz)
        m = jnp.min(jnp.abs(rk[:, None, :] - rk[None, :, :]), axis=-1)   # (n, n)
        ss = sample_size if sample_size <= n else int(np.ceil(0.8 * (1 + dz)))
        R = jnp.sort(m, axis=1)[:, ss - 1]
        near = m <= R[:, None]
        # Tetrad 7.6.5 uses rows 0..dz-1 of (x, z...) for the distance: x and
        # all z but the last (compat); the correct distance uses the z rows.
        Dd = D[:, :dz] if compat else Zr
        d = jnp.sqrt(jnp.sum(((Dd[:, None, :] - Dd[None, :, :]) / 2.0) ** 2, -1))
        K = jnp.where(near, jnp.exp(-(d / h) ** 2), 0.0)
        tw = K.sum(1)
        tw = jnp.where(tw == 0, 1.0, tw)
        r = t - (K @ t) / tw
        return jnp.where(jnp.isnan(r), 0.0, r)

    rx, ry = residuals(0), residuals(1)
    n = rx.shape[0]
    powers = jnp.arange(1, n_functions + 1)
    Fx = jax.vmap(lambda q: _std_pop(rx ** q), out_axes=1)(powers)     # (n, m)
    Fy = jax.vmap(lambda q: _std_pop(ry ** q), out_axes=1)(powers)
    # Tetrad: data standardized with divisor n, but StatUtils.covariance
    # divides by n - 1 (compat); the correct correlation divides by n.
    r = (Fx.T @ Fy) / ((n - 1) if compat else n)
    m22 = ((Fx ** 2).T @ (Fy ** 2)) / n
    z = 0.5 * jnp.sqrt(n) * (jnp.log1p(r) - jnp.log1p(-r)) / jnp.sqrt(m22)
    z = jnp.where(jnp.isfinite(z), jnp.abs(z), -jnp.inf)
    return jnp.max(z)


class CCI(CITest):
    """Conditional correlation independence (Ramsey 2014), as Tetrad
    IndTestConditionalCorrelation with its defaults: 10 polynomial basis
    functions, Gaussian kernel, width 1, 100 kernel-regression neighbours.
    Statistic: max over the basis pairs of the nonparametric Fisher z of the
    two kernel-regression residuals; p = 2 (1 - Phi(score)).

    ``tetrad_compat=True`` reproduces two Tetrad 7.6.5 defects: the kernel
    distance uses x and all but the last conditioning variable, and the
    correlation of the standardized basis values divides by n - 1.

    The p-value of the maximum over the m^2 basis pairs has no correction
    for multiple comparisons (as in Tetrad), so the test rejects too often
    under independence. ``bonferroni=True`` multiplies it by m^2.
    """
    name = "cci"

    def __init__(self, data, kinds=None, n_functions=10, sample_size=100, tetrad_compat=False,
                 bonferroni=False):
        super().__init__(data, kinds)
        self.bonferroni = bool(bonferroni)
        self.X = jnp.asarray(self.data)
        self.n_functions, self.sample_size = int(n_functions), int(sample_size)
        self.tetrad_compat = bool(tetrad_compat)

    def test(self, x, y, Z=()):
        self.n_calls += 1
        Z = sorted(Z, key=lambda v: f"X{v + 1}")       # Tetrad sorts Z by node name
        s = _cci(self.X, jnp.asarray([x, y, *Z], dtype=jnp.int32), len(Z), self.n_functions,
                 self.sample_size, self.tetrad_compat)
        s = float(s)
        p = float(normal_sf_two_sided(s))
        if self.bonferroni:                      # correct for the m^2 basis pairs
            p = min(1.0, p * self.n_functions ** 2)
        return s, p
