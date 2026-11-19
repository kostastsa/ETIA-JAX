"""Base class and shared helpers for conditional independence (CI) tests.

A CI test is made once from the data and then answers queries
``X _||_ Y | Z`` given as column indices.

* ``test(x, y, Z)`` -> ``(stat, pvalue)`` for one query.
* ``test_batch(xs, ys, Zs)`` -> arrays ``(stat, pvalue)`` for B queries
  with the same conditioning-set size (``Zs`` has shape ``(B, k)``).
  The default loops over ``test``; fast tests override it with one
  ``jax.vmap`` call.
* ``independent(x, y, Z, alpha)``: the decision. For most tests it is
  ``pvalue > alpha``. Tests with ``significance="fixed_thres"`` compare the
  statistic with ``alpha`` (as tigramite does).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, gammaincc
from jax.scipy.stats import norm


class CITest:
    name = "base"
    data_types = ("continuous",)          # data kinds the test accepts
    fixed_threshold = False

    def __init__(self, data, kinds=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.n, self.p = self.data.shape
        self.kinds = list(kinds) if kinds is not None else ["continuous"] * self.p
        self.is_discrete = np.array([k == "discrete" for k in self.kinds])
        self.n_calls = 0

    # -- interface -------------------------------------------------------
    def test(self, x, y, Z=()):
        raise NotImplementedError

    def test_batch(self, xs, ys, Zs):
        xs, ys = np.asarray(xs), np.asarray(ys)
        Zs = np.asarray(Zs, dtype=int).reshape(len(xs), -1)
        out = [self.test(int(x), int(y), [int(z) for z in Z]) for x, y, Z in zip(xs, ys, Zs)]
        s, p = zip(*out) if out else ((), ())
        return np.array(s, dtype=float), np.array(p, dtype=float)

    def independent(self, x, y, Z, alpha):
        stat, pval = self.test(x, y, Z)
        if self.fixed_threshold:
            return bool(stat <= alpha), stat, pval
        return bool(pval > alpha), stat, pval

    def __call__(self, x, y, Z=()):
        """p-value only (the oracle interface)."""
        return self.test(x, y, Z)[1]


# -- distributions (jit-safe) ----------------------------------------------
def t_sf_two_sided(t, dof):
    """2 * P(T > |t|) for Student t with ``dof`` degrees of freedom."""
    x = dof / (dof + t * t)
    return betainc(dof / 2.0, 0.5, x)


def chi2_sf(x, dof):
    return gammaincc(dof / 2.0, jnp.maximum(x, 0.0) / 2.0)


def normal_sf_two_sided(z):
    return 2.0 * norm.sf(jnp.abs(z))


# -- helpers -----------------------------------------------------------------
def partial_corr_from_cov(C, idx):
    """Partial correlation of idx[0], idx[1] given idx[2:] from a
    covariance (or correlation) matrix ``C``. ``idx`` is a (k + 2,) array."""
    sub = C[idx[:, None], idx[None, :]]
    P = jnp.linalg.inv(sub)
    r = -P[0, 1] / jnp.sqrt(P[0, 0] * P[1, 1])
    return jnp.clip(r, -1.0 + 1e-15, 1.0 - 1e-15)


def as_index_batch(xs, ys, Zs):
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    Zs = np.asarray(Zs, dtype=np.int32).reshape(len(xs), -1)
    return np.concatenate([xs[:, None], ys[:, None], Zs], axis=1)


def ranks(x):
    """Ranks 1..n along axis 0 (ties broken by order)."""
    return jnp.argsort(jnp.argsort(x, axis=0), axis=0) + 1
