"""Linear-Gaussian CI tests.

* ``FisherZ``       Tetrad IndTestFisherZ.
* ``ParCorr``       tigramite ParCorr (Student t, dof = n - 2 - |Z|).
* ``RobustParCorr`` tigramite RobustParCorr (normal scores, then ParCorr).
* ``ParCorrWLS``    tigramite ParCorrWLS with the default
                    "time-dependent heteroskedasticity" (window 10).
* ``DGLrt``         Tetrad IndTestDegenerateGaussianLrt (one-hot embedding
                    of discrete variables, Gaussian likelihood-ratio test).
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from .base import (CITest, partial_corr_from_cov, t_sf_two_sided, chi2_sf,
                   normal_sf_two_sided, as_index_batch, pad_batch)


@jax.jit
def _fisher_z(C, idx, n):
    r = jax.vmap(partial_corr_from_cov, in_axes=(None, 0))(C, idx)
    k = idx.shape[1] - 2
    z = 0.5 * jnp.sqrt(n - k - 3.0) * jnp.log((1.0 + r) / (1.0 - r))
    return z, normal_sf_two_sided(z)


@jax.jit
def _par_corr(C, idx, n):
    r = jax.vmap(partial_corr_from_cov, in_axes=(None, 0))(C, idx)
    dof = n - idx.shape[1]
    t = r * jnp.sqrt(dof / (1.0 - r * r))
    return r, t_sf_two_sided(t, dof)


class _CovTest(CITest):
    """Tests that need only the covariance matrix of (transformed) data."""
    _kernel = None

    def _transform(self, X):
        return X

    def __init__(self, data, kinds=None):
        super().__init__(data, kinds)
        X = self._transform(self.data)
        self.C = jnp.asarray(np.corrcoef(X, rowvar=False))

    def test_batch(self, xs, ys, Zs):
        idx, B = pad_batch(as_index_batch(xs, ys, Zs))
        self.n_calls += B
        s, p = type(self)._kernel(self.C, jnp.asarray(idx), float(self.n))
        return np.asarray(s)[:B], np.asarray(p)[:B]

    def test(self, x, y, Z=()):
        s, p = self.test_batch([x], [y], [list(Z)])
        return float(s[0]), float(p[0])


class FisherZ(_CovTest):
    """Statistic: Fisher z of the partial correlation; two-sided normal p-value."""
    name = "FisherZ"
    _kernel = staticmethod(_fisher_z)


class ParCorr(_CovTest):
    """Statistic: partial correlation; Student t p-value, dof = n - 2 - |Z|."""
    name = "ParCor"
    _kernel = staticmethod(_par_corr)


def normal_scores(X, thres=1e-5):
    """tigramite ``trafo2normal``: empirical CDF (rank / n), clipped, then the
    standard normal quantile."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    U = np.empty_like(X)
    grid = np.linspace(1.0 / n, 1.0, n)
    for j in range(X.shape[1]):
        U[:, j] = np.interp(X[:, j], np.sort(X[:, j]), grid)
    U[U == 0.0] = thres
    U[U == 1.0] = 1.0 - thres
    return np.asarray(norm.ppf(jnp.asarray(U)))


class RobustParCorr(_CovTest):
    """ParCorr on normal scores (tigramite RobustParCorr)."""
    name = "RobustParCor"
    _kernel = staticmethod(_par_corr)

    def _transform(self, X):
        return normal_scores(X)


# --------------------------------------------------------------------------
# ParCorrWLS
# --------------------------------------------------------------------------
def _lstsq_resid(Z, y):
    """Residual of OLS of y on Z (no intercept, as in tigramite)."""
    beta = jnp.linalg.lstsq(Z, y)[0]
    return y - Z @ beta


def _wls_resid_one(X, idx, target, window):
    """tigramite ParCorrWLS residual of column idx[target] given idx[2:]."""
    y = X[:, idx[target]]
    Z = X[:, idx[2:]]
    has_z = idx.shape[0] > 2
    r = jnp.abs(_lstsq_resid(Z, y)) if has_z else jnp.abs(y)
    ma = jnp.convolve(r, jnp.ones(window), mode="valid") / window
    std = jnp.concatenate([jnp.ones(window - 1), ma])
    w = 1.0 / std
    if has_z:
        beta = jnp.linalg.lstsq(Z * w[:, None], y * w)[0]
        return (y - Z @ beta) * w
    return y * w


@partial(jax.jit, static_argnums=(2,))
def _par_corr_wls(X, idx, window):
    def one(ix):
        rx = _wls_resid_one(X, ix, 0, window)
        ry = _wls_resid_one(X, ix, 1, window)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        r = (rx @ ry) / jnp.sqrt((rx @ rx) * (ry @ ry))
        return jnp.clip(r, -1 + 1e-15, 1 - 1e-15)
    r = jax.vmap(one)(idx)
    n = X.shape[0]
    dof = n - idx.shape[1]
    t = r * jnp.sqrt(dof / (1.0 - r * r))
    return r, t_sf_two_sided(t, dof)


class ParCorrWLS(CITest):
    """Weighted ParCorr. The weights are 1 / (moving average of |OLS
    residual|) over ``window`` samples in the given sample order."""
    name = "ParCorrWLS"

    def __init__(self, data, kinds=None, window_size=10):
        super().__init__(data, kinds)
        self.window = int(window_size)
        self.X = jnp.asarray(self.data)

    def test_batch(self, xs, ys, Zs):
        idx, B = pad_batch(as_index_batch(xs, ys, Zs))
        self.n_calls += B
        s, p = _par_corr_wls(self.X, jnp.asarray(idx), self.window)
        return np.asarray(s)[:B], np.asarray(p)[:B]

    def test(self, x, y, Z=()):
        s, p = self.test_batch([x], [y], [list(Z)])
        return float(s[0]), float(p[0])


# --------------------------------------------------------------------------
# Degenerate Gaussian LRT
# --------------------------------------------------------------------------
def embed(data, is_discrete, drop="first"):
    """Columns of the embedded data and, per variable, the list of its
    embedded column indices. A discrete variable becomes indicator columns:

    * ``drop="first"``: one per category except the smallest category;
    * ``drop="last_seen"``: Tetrad 7.6.5, one per category except the
      category that appears first latest in the row order;
    * ``drop=None``: one per category (full one-hot).
    """
    cols, groups = [], []
    for j in range(data.shape[1]):
        if is_discrete[j]:
            cats = list(np.unique(data[:, j]))
            if drop == "first":
                cats = cats[1:]
            elif drop == "last_seen":
                _, first = np.unique(data[:, j], return_index=True)
                cats = [c for _, c in sorted(zip(first, cats))][:-1]
            g = []
            for c in cats:
                g.append(len(cols))
                cols.append((data[:, j] == c).astype(np.float64))
            groups.append(g)
        else:
            groups.append([len(cols)])
            cols.append(data[:, j])
    return np.column_stack(cols), groups


L2PE = float(np.log(2.0 * np.pi * np.e))


class DGLrt(CITest):
    """Degenerate Gaussian likelihood-ratio test (Andrews et al. 2019),
    Tetrad IndTestDegenerateGaussianLrt.

    Discrete variables are embedded as indicator columns. For child ``c``
    with parent set P (embedded columns B, and A = c + B):
    ``lik = N (logdet S_B - logdet S_A) + log(2 pi e) (|B| - |A|)``,
    ``dof = (|A| (|A| + 1) - |B| (|B| + 1)) / 2``. The test takes y as the
    child: ``stat = lik(y | x, Z) - lik(y | Z)`` (times 2 in Tetrad).

    * ``tetrad_compat=True`` (Tetrad 7.6.5): the indicator of the category
      that appears first latest is dropped, and the covariance uses a mean
      of sum / (n - 1) with a divisor n, and the statistic is
      -2 (lik0 - lik1), which is twice the likelihood ratio because ``lik``
      is already 2 x log-likelihood (Tetrad defects, kept for exactness).
    * default: the indicator of the smallest category is dropped, and the
      usual covariance (mean = sum / n), and the likelihood ratio
      statistic -(lik0 - lik1).
    """
    symmetric = False                     # y is the child of the likelihood ratio
    name = "dg_lrt"
    data_types = ("continuous", "discrete", "mixed")

    def __init__(self, data, kinds=None, tetrad_compat=False):
        super().__init__(data, kinds)
        self.tetrad_compat = tetrad_compat
        E, self.groups = embed(self.data, self.is_discrete, "last_seen" if tetrad_compat else "first")
        E = jnp.asarray(E)
        n = E.shape[0]
        mu = E.sum(0) / (n - 1) if tetrad_compat else E.mean(0)
        C = E - mu
        S = np.asarray((C.T @ C) / n)
        # Each variable gets a block of W columns: its embedded columns, then
        # independent unit-variance columns. A unit block adds log 1 = 0 to a
        # log-determinant, so all queries of one |Z| have the same shape.
        self.width = np.array([len(g) for g in self.groups])
        W = int(self.width.max())
        m = self.p * W
        Sp = np.eye(m)
        cols = np.full(m, -1)
        for v, g in enumerate(self.groups):
            cols[v * W: v * W + len(g)] = g
        real = cols >= 0
        Sp[np.ix_(real, real)] = S[np.ix_(cols[real], cols[real])]
        self.W, self.Sp = W, jnp.asarray(Sp)
        self.S = jnp.asarray(S)

    def _lik(self, child, parents):
        """Unbatched reference (used by the tests)."""
        B = [c for q in parents for c in self.groups[q]]
        A = list(self.groups[child]) + B
        def logdet(cols):
            if not cols:
                return 0.0
            M = self.S[jnp.asarray(cols)[:, None], jnp.asarray(cols)[None, :]]
            return float(jnp.log(jnp.abs(jnp.linalg.det(M))))
        lik = self.n * (logdet(B) - logdet(A)) + L2PE * (len(B) - len(A))
        dof = (len(A) * (len(A) + 1) - len(B) * (len(B) + 1)) / 2.0
        return lik, dof

    def test_batch(self, xs, ys, Zs):
        idx, B = pad_batch(as_index_batch(xs, ys, Zs))
        self.n_calls += B
        ld = np.asarray(_dg_logdets(self.Sp, jnp.asarray(idx), self.W))      # (B', 4)
        idx, ld = idx[:B], ld[:B]
        w = self.width
        dz = w[idx[:, 2:]].sum(1) if idx.shape[1] > 2 else np.zeros(B, int)
        dx, dy = w[idx[:, 0]], w[idx[:, 1]]
        # ret0: child y, parents Z (B0 = Z, A0 = y + Z); ret1: parents x + Z
        l0 = self.n * (ld[:, 0] - ld[:, 1]) + L2PE * (-dy)
        l1 = self.n * (ld[:, 2] - ld[:, 3]) + L2PE * (-dy)
        dof = lambda a, b: (a * (a + 1) - b * (b + 1)) / 2.0
        ddof = dof(dy + dx + dz, dx + dz) - dof(dy + dz, dz)
        stat = (-2.0 if self.tetrad_compat else -1.0) * (l0 - l1)
        ok = (ddof > 0) & np.isfinite(stat)
        p = np.where(ok, np.asarray(chi2_sf(jnp.asarray(stat), jnp.asarray(np.maximum(ddof, 1)))), np.nan)
        return stat, p

    def test(self, x, y, Z=()):
        s, p = self.test_batch([x], [y], [sorted(Z)])
        return float(s[0]), float(p[0])


@partial(jax.jit, static_argnums=(2,))
def _dg_logdets(Sp, idx, W):
    """log|det| of the four blocks of each query (x, y, Z...):
    Z, y + Z, x + Z, y + x + Z (each variable is a block of W columns)."""
    def cols(vs):
        return (vs[:, None] * W + jnp.arange(W)[None, :]).reshape(-1)

    def logdet(c):
        if c.shape[0] == 0:
            return 0.0
        return jnp.linalg.slogdet(Sp[c[:, None], c[None, :]])[1]

    def one(ix):
        x, y, Z = ix[0:1], ix[1:2], ix[2:]
        return jnp.stack([logdet(cols(Z)), logdet(cols(jnp.concatenate([y, Z]))),
                          logdet(cols(jnp.concatenate([x, Z]))),
                          logdet(cols(jnp.concatenate([y, x, Z])))])
    return jax.vmap(one)(idx)
