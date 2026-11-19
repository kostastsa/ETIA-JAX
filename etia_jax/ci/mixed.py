"""CI tests for mixed (continuous + discrete) data.

* ``RegressionCI``  tigramite RegressionCI: deviance difference of two
  regressions (linear for a continuous target, multinomial logistic with the
  sklearn default L2 penalty C = 1 for a discrete target).
* ``CGLrt``  Tetrad IndTestConditionalGaussianLrt (discretize = True):
  conditional Gaussian likelihood of x given (y, Z) against x given Z.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .base import CITest, chi2_sf, as_index_batch, pad_batch


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def one_hot(col):
    cats = np.unique(col)
    return (col[:, None] == cats[None, :]).astype(np.float64)


def design(data, cols, is_discrete):
    """Features of ``cols``: continuous as is, discrete as full one-hot."""
    parts = [one_hot(data[:, c]) if is_discrete[c] else data[:, [c]] for c in cols]
    return np.column_stack(parts) if parts else np.ones((len(data), 1))


def linear_deviance(X, y):
    """-2 log-likelihood of OLS with intercept (sklearn LinearRegression)."""
    X1 = jnp.column_stack([jnp.ones(len(y)), X])
    beta = jnp.linalg.lstsq(X1, y)[0]
    rss = jnp.sum((y - X1 @ beta) ** 2)
    n = len(y)
    return float(n * jnp.log(2 * jnp.pi) + n * jnp.log(rss / n) + n), X.shape[1] + 1


@jax.jit
def _logistic_fit(X, Y):
    """Multinomial logistic regression, L2 penalty 0.5 * ||W||^2 (C = 1, as
    sklearn), intercept not penalized. Damped Newton with the exact Hessian
    and backtracking. Returns the summed log-loss at the optimum."""
    d, K = X.shape[1], Y.shape[1]
    P = d * K + K

    def nll(theta):
        W, b = theta[: d * K].reshape(d, K), theta[d * K:]
        logits = X @ W + b
        return -jnp.sum(Y * (logits - logsumexp(logits, axis=1, keepdims=True)))

    def loss(theta):
        W = theta[: d * K]
        return nll(theta) + 0.5 * jnp.sum(W * W)

    grad, hess = jax.grad(loss), jax.hessian(loss)

    def newton(state):
        theta, it, _ = state
        g, H = grad(theta), hess(theta)
        # the loss is flat along a common shift of the intercepts: tiny ridge
        delta = jnp.linalg.solve(H + 1e-9 * jnp.eye(P), g)
        f0, slope = loss(theta), g @ delta

        def shrink(ts):
            return ts * 0.5

        def too_big(ts):
            return (loss(theta - ts * delta) > f0 - 1e-4 * ts * slope) & (ts > 1e-6)

        t = jax.lax.while_loop(too_big, shrink, 1.0)
        return theta - t * delta, it + 1, jnp.max(jnp.abs(g))

    def go_on(state):
        _, it, gmax = state
        return (it < 50) & (gmax > 1e-8)

    theta, _, _ = jax.lax.while_loop(go_on, newton, (jnp.zeros(P), 0, jnp.inf))
    return nll(theta)


def logistic_deviance(X, y, width=None):
    """2 * log-loss (sum) of the penalized multinomial logistic fit.

    With ``width``, X is padded with zero columns to that width: a zero
    column gets a zero weight under the L2 penalty, so the fit does not
    change, and the fit compiles once per (width, classes)."""
    Y = one_hot(y)
    dof = X.shape[1] + 1
    if Y.shape[1] < 2:
        return 0.0, dof
    if width is not None and X.shape[1] < width:
        X = np.column_stack([X, np.zeros((len(X), width - X.shape[1]))])
    return float(2.0 * _logistic_fit(jnp.asarray(X), jnp.asarray(Y))), dof


def dagostino_k2(x):
    """scipy.stats.normaltest statistic (D'Agostino and Pearson K^2)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    m = x - x.mean()
    m2, m3, m4 = (m ** 2).mean(), (m ** 3).mean(), (m ** 4).mean()
    b2 = m3 / m2 ** 1.5
    # skewtest
    y = b2 * np.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
    beta2 = (3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3)) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    W2 = -1 + np.sqrt(2 * (beta2 - 1))
    delta = 1 / np.sqrt(0.5 * np.log(W2))
    alpha = np.sqrt(2.0 / (W2 - 1))
    y = np.where(y == 0, 1, y)
    Zs = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))
    # kurtosistest
    b2k = m4 / m2 ** 2
    E = 3.0 * (n - 1) / (n + 1)
    varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.) * (n + 3) * (n + 5))
    xk = (b2k - E) / np.sqrt(varb2)
    sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * np.sqrt((6.0 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
    A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + np.sqrt(1 + 4.0 / (sqrtbeta1 ** 2)))
    term1 = 1 - 2 / (9.0 * A)
    denom = 1 + xk * np.sqrt(2 / (A - 4.0))
    term2 = np.sign(denom) * np.where(denom == 0.0, np.nan, np.power((1 - 2.0 / A) / np.abs(denom), 1 / 3.0))
    Zk = (term1 - term2) / np.sqrt(2 / (9.0 * A))
    return Zs ** 2 + Zk ** 2


def entropy(col):
    _, c = np.unique(col, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------
# RegressionCI
# --------------------------------------------------------------------------
class RegressionCI(CITest):
    name = "RegressionCI"
    data_types = ("continuous", "discrete", "mixed")

    def __init__(self, data, kinds=None):
        super().__init__(data, kinds)
        # widest possible design: every column, one-hot for the discrete ones
        self.width = sum(len(np.unique(self.data[:, j])) if self.is_discrete[j] else 1
                         for j in range(self.p)) + 1

    def test(self, x, y, Z=()):
        self.n_calls += 1
        D, disc = self.data, self.is_discrete
        Z = list(Z)
        if Z:
            z_feat = design(D, Z, disc)
        else:
            z_feat = np.ones((self.n, 1))      # intercept-only design, as in tigramite
        xd, yd = disc[x], disc[y]
        if not xd and not yd:
            if dagostino_k2(D[:, x]) >= dagostino_k2(D[:, y]):
                dep, other = y, x
            else:
                dep, other = x, y
            fit = linear_deviance
        elif xd and not yd:
            dep, other, fit = y, x, linear_deviance
        elif yd and not xd:
            dep, other, fit = x, y, linear_deviance
        else:
            dep, other = (y, x) if entropy(D[:, x]) >= entropy(D[:, y]) else (x, y)
            fit = logistic_deviance
        rest = np.column_stack([design(D, [other], disc), z_feat])
        target = D[:, dep] if fit is linear_deviance else D[:, dep].astype(int)
        if fit is logistic_deviance:
            dev1, dof1 = fit(z_feat, target, self.width)
            dev2, dof2 = fit(rest, target, self.width)
        else:
            dev1, dof1 = fit(z_feat, target)
            dev2, dof2 = fit(rest, target)
        stat = dev1 - dev2
        dof = dof2 - dof1
        return float(stat), float(chi2_sf(stat, dof))


# --------------------------------------------------------------------------
# Conditional Gaussian LRT (Tetrad 7.6.5 ConditionalGaussianLikelihood)
# --------------------------------------------------------------------------
LOG2PI = float(np.log(2.0 * np.pi))


def tetrad_equal_freq(col, k=3):
    """Tetrad Discretizer: breakpoints sorted[(n // k) * (i + 1)], bin j iff
    value < breakpoint j (first such j), else the last bin."""
    srt = np.sort(col)
    m = len(col) // k
    cuts = srt[[m * (i + 1) for i in range(k - 1)]]
    return np.searchsorted(cuts, col, side="right")


def _cells(codes):
    """Dense cell ids (n,) from the rows of an integer code matrix (n, m)."""
    if codes.shape[1] == 0:
        return np.zeros(len(codes), dtype=np.int32), 1
    _, inv = np.unique(codes, axis=0, return_inverse=True)
    inv = inv.ravel().astype(np.int32)
    return inv, int(inv.max()) + 1


@partial(jax.jit, static_argnums=(2, 4, 5))
def _joint_ll(Xc, cell, n_cells, n_total, has_A, min_cell):
    """sum over cells (a >= min_cell) of a log(a / N) (if A is not empty)
    plus a * (-0.5 log|S| - 0.5 k (1 + log 2 pi)), S = cell covariance (n - 1)."""
    cnt = jax.ops.segment_sum(jnp.ones(len(cell)), cell, n_cells)
    ok = cnt >= min_cell
    c1 = jnp.sum(jnp.where(ok, cnt * jnp.log(jnp.maximum(cnt, 1.0) / n_total), 0.0)) if has_A else 0.0
    k = Xc.shape[1]
    if k == 0:
        return c1
    s1 = jax.ops.segment_sum(Xc, cell, n_cells)                                  # (C, k)
    s2 = jax.ops.segment_sum(Xc[:, :, None] * Xc[:, None, :], cell, n_cells)     # (C, k, k)
    mean = s1 / jnp.maximum(cnt, 1.0)[:, None]
    S = (s2 - cnt[:, None, None] * mean[:, :, None] * mean[:, None, :]) \
        / jnp.maximum(cnt - 1.0, 1.0)[:, None, None]
    det = jnp.linalg.det(S)
    okg = ok & (cnt >= k) & (det != 0) & jnp.isfinite(det)
    gl = -0.5 * jnp.log(jnp.abs(jnp.where(okg, det, 1.0))) - 0.5 * k * (1.0 + LOG2PI)
    return c1 + jnp.sum(jnp.where(okg, cnt * gl, 0.0))


class CGLrt(CITest):
    """Conditional Gaussian LRT (Andrews, Ramsey and Cooper 2018), as Tetrad
    IndTestConditionalGaussianLrt(data, alpha, discretize=True).

    L(child | P) = J(child + P) - J(P), where J(X, A) is the joint
    conditional Gaussian log-likelihood with continuous X and discrete A
    (cells of A with fewer than 4 rows are skipped). With ``discretize``,
    a discrete child moves its continuous parents into A after an
    equal-frequency discretization into 3 bins. y is the child:
    stat = -2 (L(y | Z) - L(y | x, Z)), dof from f(A) h(X) + f(A).
    """
    symmetric = False                     # y is the child of the likelihood ratio
    name = "cg_lrt"
    data_types = ("continuous", "discrete", "mixed")

    def __init__(self, data, kinds=None, discretize=True, n_bins=3, min_cell=4):
        super().__init__(data, kinds)
        self.discretize, self.n_bins, self.min_cell = discretize, n_bins, min_cell
        self.codes = np.zeros(self.data.shape, dtype=np.int32)
        self.n_cat = np.zeros(self.p, dtype=np.int64)
        for j in range(self.p):
            if self.is_discrete[j]:
                u, inv = np.unique(self.data[:, j], return_inverse=True)
                self.codes[:, j], self.n_cat[j] = inv, len(u)
            else:
                self.codes[:, j], self.n_cat[j] = tetrad_equal_freq(self.data[:, j], n_bins), n_bins
        self.X = jnp.asarray(self.data)
        self.codes_j = jnp.asarray(self.codes)

    def _joint(self, X, A):
        """Unbatched reference of J(X, A) (used by the tests)."""
        cell, n_cells = _cells(self.codes[:, A])
        n_cells = 1 << max(0, int(n_cells - 1).bit_length())   # empty cells: count 0, skipped
        ll = float(_joint_ll(self.X[:, X], jnp.asarray(cell), n_cells, float(self.n), bool(A), self.min_cell))
        fA = int(np.prod(self.n_cat[A])) if A else 1
        hX = len(X) * (len(X) + 1) // 2
        return ll, fA * hX + fA

    def _parts(self, child, parents):
        """(X1, A1, X0, A0) of L(child | parents) = J(X1, A1) - J(X0, A0)."""
        X0 = [q for q in parents if not self.is_discrete[q]]
        A0 = [q for q in parents if self.is_discrete[q]]
        if self.discretize and self.is_discrete[child]:
            A0, X0 = A0 + X0, []
        X1, A1 = (X0, A0 + [child]) if self.is_discrete[child] else (X0 + [child], A0)
        return X1, A1, X0, A0

    def loglik(self, child, parents):
        X1, A1, X0, A0 = self._parts(child, parents)
        l1, d1 = self._joint(X1, A1)
        l0, d0 = self._joint(X0, A0)
        return l1 - l0, d1 - d0

    def _dof(self, X, A):
        fA = int(np.prod(self.n_cat[list(A)])) if len(A) else 1
        return fA * (len(X) * (len(X) + 1) // 2) + fA

    def test_batch(self, xs, ys, Zs):
        """Queries grouped by the shapes of their four joint likelihoods; one
        vmapped call for each group."""
        idx = as_index_batch(xs, ys, Zs)
        self.n_calls += len(idx)
        out_s, out_p = np.empty(len(idx)), np.empty(len(idx))
        jobs = {}
        for q, row in enumerate(idx):
            x, y, Z = int(row[0]), int(row[1]), sorted(int(z) for z in row[2:])
            parts = self._parts(y, Z) + self._parts(y, [x] + Z)      # 8 lists: 4 x (X, A)
            shape = tuple(len(v) for v in parts)
            jobs.setdefault(shape, []).append((q, parts))
        r = int(self.n_cat.max())
        for shape, items in jobs.items():
            arrs = [np.array([it[1][k] for it in items], dtype=np.int32).reshape(len(items), shape[k])
                    for k in range(8)]
            J = []
            for k in range(0, 8, 2):                                 # J(X, A) for the 4 pairs
                Xi, B = pad_batch(arrs[k], min_size=1)
                Ai, _ = pad_batch(arrs[k + 1], min_size=1)
                J.append(np.asarray(_cg_joint_batch(self.X, self.codes_j, jnp.asarray(Xi), jnp.asarray(Ai),
                                                    r, float(self.n), self.min_cell))[:B])
            l0 = J[0] - J[1]                                         # L(y | Z)
            l1 = J[2] - J[3]                                         # L(y | x, Z)
            for t, (q, parts) in enumerate(items):
                d0 = self._dof(parts[0], parts[1]) - self._dof(parts[2], parts[3])
                d1 = self._dof(parts[4], parts[5]) - self._dof(parts[6], parts[7])
                stat = -2.0 * (l0[t] - l1[t])
                ddof = d1 - d0
                out_s[q] = stat
                out_p[q] = (float(chi2_sf(stat, ddof)) if ddof > 0 and np.isfinite(stat) else np.nan)
        return out_s, out_p

    def test(self, x, y, Z=()):
        s, p = self.test_batch([x], [y], [sorted(Z)])
        return float(s[0]), float(p[0])


@partial(jax.jit, static_argnums=(4, 6))
def _cg_joint_batch(X, codes, Xidx, Aidx, r, n_total, min_cell):
    """J(X, A) of Tetrad's conditional Gaussian likelihood for a batch of
    (X columns, A columns): cells of A (mixed radix r), a multinomial part
    and a Gaussian part per cell (cells with < min_cell rows are skipped)."""
    kX, kA = Xidx.shape[1], Aidx.shape[1]
    n_cells = r ** kA

    def one(xi, ai):
        cell = jnp.zeros(X.shape[0], dtype=jnp.int32)
        for j in range(kA):
            cell = cell * r + codes[:, ai[j]]
        cnt = jax.ops.segment_sum(jnp.ones(X.shape[0]), cell, n_cells)
        ok = cnt >= min_cell
        c1 = jnp.sum(jnp.where(ok, cnt * jnp.log(jnp.maximum(cnt, 1.0) / n_total), 0.0)) if kA else 0.0
        if kX == 0:
            return c1
        Xc = X[:, xi]
        s1 = jax.ops.segment_sum(Xc, cell, n_cells)
        s2 = jax.ops.segment_sum(Xc[:, :, None] * Xc[:, None, :], cell, n_cells)
        mean = s1 / jnp.maximum(cnt, 1.0)[:, None]
        S = (s2 - cnt[:, None, None] * mean[:, :, None] * mean[:, None, :]) \
            / jnp.maximum(cnt - 1.0, 1.0)[:, None, None]
        det = jnp.linalg.det(S)
        okg = ok & (cnt >= kX) & (det != 0) & jnp.isfinite(det)
        gl = -0.5 * jnp.log(jnp.abs(jnp.where(okg, det, 1.0))) - 0.5 * kX * (1.0 + LOG2PI)
        return c1 + jnp.sum(jnp.where(okg, cnt * gl, 0.0))

    return jax.vmap(one)(Xidx, Aidx)
