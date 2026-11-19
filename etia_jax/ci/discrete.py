"""CI tests for discrete data, on stratified contingency tables.

* ``ChiSquare``  Tetrad IndTestChiSquare (Pearson X^2).
* ``GSquare``    Tetrad IndTestGSquare (likelihood ratio G^2).
* ``Gsquared``   tigramite Gsquared (G^2; same table and dof rule).
* ``CMIsymb``    tigramite CMIsymb (plug-in CMI, local shuffle test).

For one query, the table has shape (S, rx, ry), with S = number of strata of
Z (mixed radix code). Degrees of freedom per stratum:
(rows with non-zero margin - 1) * (columns with non-zero margin - 1).
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy

from .base import CITest, chi2_sf, as_index_batch


def _codes(data):
    """Map each column to codes 0..r-1; return (codes, arities)."""
    codes = np.empty(data.shape, dtype=np.int32)
    ar = np.empty(data.shape[1], dtype=np.int32)
    for j in range(data.shape[1]):
        u, inv = np.unique(data[:, j], return_inverse=True)
        codes[:, j], ar[j] = inv, len(u)
    return codes, ar


@partial(jax.jit, static_argnums=(2, 3, 4))
def _tables(D, idx, rx, ry, n_strata):
    """Contingency tables (B, S, rx, ry) for the queries in ``idx``."""
    def one(ix):
        x, y, Z = D[:, ix[0]], D[:, ix[1]], D[:, ix[2:]]
        s = jnp.zeros(D.shape[0], dtype=jnp.int32)
        for k in range(Z.shape[1]):          # radix = max arity = rx
            s = s * rx + Z[:, k]
        cell = (s * rx + x) * ry + y
        return jnp.bincount(cell, length=n_strata * rx * ry).reshape(n_strata, rx, ry)
    return jax.vmap(one)(idx)


def _g2_stats(T):
    """G^2, X^2 and dof from tables (B, S, rx, ry)."""
    T = T.astype(jnp.float64)
    row = T.sum(3, keepdims=True)
    col = T.sum(2, keepdims=True)
    tot = T.sum((2, 3), keepdims=True)
    E = jnp.where(tot > 0, row * col / jnp.where(tot > 0, tot, 1.0), 0.0)
    g2 = 2.0 * jnp.sum(xlogy(T, T) - xlogy(T, jnp.where(E > 0, E, 1.0)), axis=(1, 2, 3))
    x2 = jnp.sum(jnp.where(E > 0, (T - E) ** 2 / jnp.where(E > 0, E, 1.0), 0.0), axis=(1, 2, 3))
    nr = (row[..., 0] > 0).sum(2)            # (B, S)
    nc = (col[:, :, 0, :] > 0).sum(2)
    present = tot[..., 0, 0] > 0
    dof = jnp.sum(jnp.where(present, jnp.maximum(nr - 1, 0) * jnp.maximum(nc - 1, 0), 0), axis=1)
    return g2, x2, dof


class _TableTest(CITest):
    data_types = ("discrete",)

    def __init__(self, data, kinds=None):
        if kinds is None:
            kinds = ["discrete"] * np.asarray(data).shape[1]
        super().__init__(data, kinds)
        codes, self.arity = _codes(self.data)
        self.D = jnp.asarray(codes)
        self.r = int(self.arity.max())

    def tables(self, idx):
        """Tables for an index batch with a common |Z|; uses the max arity."""
        k = idx.shape[1] - 2
        return _tables(self.D, jnp.asarray(idx), self.r, self.r, self.r ** k)

    def test(self, x, y, Z=()):
        s, p = self.test_batch([x], [y], [list(Z)])
        return float(s[0]), float(p[0])


def _tetrad_stats(T, ax, ay):
    """Tetrad 7.6.5 ChiSquareTest (minCountPerCell = 1) on tables (B, S, r, r).

    ``ax``, ``ay`` (B,) are the numbers of categories of x and y. In each
    stratum, a row is dropped if its margin < ay, a column if its margin < ax;
    a stratum is skipped if its total < ax * ay, or if fewer than two rows or
    columns are left. Expected counts use the full stratum total.
    Returns X^2, G^2 and dof (dof = 0: no valid stratum).
    """
    T = T.astype(jnp.float64)
    r = T.shape[-1]
    axf, ayf = ax[:, None, None].astype(jnp.float64), ay[:, None, None].astype(jnp.float64)
    in_x = (jnp.arange(r)[None, None, :] < ax[:, None, None])    # (B, 1, r) valid rows
    in_y = (jnp.arange(r)[None, None, :] < ay[:, None, None])
    row = T.sum(3)                                                 # (B, S, r)
    col = T.sum(2)
    tot = T.sum((2, 3))                                            # (B, S)
    keep_r = in_x & (row >= ayf) & (row > 0)
    keep_c = in_y & (col >= axf) & (col > 0)
    nr, nc = keep_r.sum(2), keep_c.sum(2)
    valid = (tot >= (ax * ay)[:, None]) & (nr > 1) & (nc > 1) & (tot > 0)
    E = row[..., :, None] * col[..., None, :] / jnp.where(tot > 0, tot, 1.0)[..., None, None]
    cell = keep_r[..., :, None] & keep_c[..., None, :] & valid[..., None, None]
    Es = jnp.where(cell, E, 1.0)
    x2 = jnp.sum(jnp.where(cell, (T - E) ** 2 / Es, 0.0), axis=(1, 2, 3))
    g2 = jnp.sum(jnp.where(cell & (T > 0), 2.0 * T * jnp.log(jnp.where(T > 0, T, 1.0) / Es), 0.0),
                 axis=(1, 2, 3))
    dof = jnp.sum(jnp.where(valid, (nr - 1) * (nc - 1), 0), axis=1)
    return x2, g2, dof


class _TetradTableTest(_TableTest):
    _which = None

    def test_batch(self, xs, ys, Zs):
        idx = as_index_batch(xs, ys, Zs)
        self.n_calls += len(idx)
        ar = jnp.asarray(self.arity)
        x2, g2, dof = _tetrad_stats(self.tables(idx), ar[idx[:, 0]], ar[idx[:, 1]])
        stat = x2 if self._which == "x2" else g2
        # no valid stratum: Tetrad returns p = NaN and "independent"
        p = jnp.where(dof > 0, chi2_sf(stat, jnp.maximum(dof, 1)), jnp.nan)
        return np.asarray(stat), np.asarray(p)

    def independent(self, x, y, Z, alpha):
        stat, pval = self.test(x, y, Z)
        return bool(np.isnan(pval) or pval > alpha), stat, pval


class GSquare(_TetradTableTest):
    """Tetrad IndTestGSquare (G^2 with Tetrad's cell rules)."""
    name = "gsquare"
    _which = "g2"


class ChiSquare(_TetradTableTest):
    """Tetrad IndTestChiSquare (Pearson X^2 with Tetrad's cell rules)."""
    name = "chisquare"
    _which = "x2"


class Gsquared(_TableTest):
    """tigramite Gsquared: G^2 with dof = max(dof, 1)."""
    name = "Gsquared"

    def test_batch(self, xs, ys, Zs):
        idx = as_index_batch(xs, ys, Zs)
        self.n_calls += len(idx)
        g2, _, dof = _g2_stats(self.tables(idx))
        p = chi2_sf(g2, jnp.maximum(dof, 1))
        return np.asarray(g2), np.asarray(p)


# --------------------------------------------------------------------------
# CMIsymb
# --------------------------------------------------------------------------
def _cmi_from_table(T):
    """Plug-in conditional mutual information (nats) of a table (S, rx, ry)."""
    T = T.astype(jnp.float64)
    n = T.sum()
    pxyz = T / n
    pz = T.sum((1, 2), keepdims=True) / n
    pxz = T.sum(2, keepdims=True) / n
    pyz = T.sum(1, keepdims=True) / n
    num = pz * pxyz
    den = pxz * pyz
    return jnp.sum(jnp.where(T > 0, pxyz * (jnp.log(jnp.where(num > 0, num, 1.0))
                                            - jnp.log(jnp.where(den > 0, den, 1.0))), 0.0))


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _cmisymb(D, idx, key, r, n_strata, n_perm, radix):
    x, y, Z = D[:, idx[0]], D[:, idx[1]], D[:, idx[2:]]
    s = jnp.zeros(D.shape[0], dtype=jnp.int32)
    for k in range(Z.shape[1]):
        s = s * radix + Z[:, k]

    def table(xv):
        cell = (s * r + xv) * r + y
        return jnp.bincount(cell, length=n_strata * r * r).reshape(n_strata, r, r)

    val = _cmi_from_table(table(x))
    order0 = jnp.argsort(s, stable=True)

    def null(k):
        # local permutation: x is shuffled only within each stratum of Z
        u = jax.random.uniform(k, s.shape)
        order1 = jnp.lexsort((u, s))
        xp = jnp.zeros_like(x).at[order0].set(x[order1])
        return _cmi_from_table(table(xp))

    nulls = jax.lax.map(null, jax.random.split(key, n_perm), batch_size=64)
    return val, jnp.mean(nulls >= val)


class CMIsymb(_TableTest):
    """tigramite CMIsymb with significance='shuffle_test' (sig_samples=500):
    p = fraction of within-stratum shuffles with CMI >= the observed CMI."""
    name = "CMIsymb"

    def __init__(self, data, kinds=None, sig_samples=500, seed=0):
        super().__init__(data, kinds)
        self.sig_samples = int(sig_samples)
        self.key = jax.random.key(seed)

    def test(self, x, y, Z=()):
        self.n_calls += 1
        self.key, k = jax.random.split(self.key)
        idx = jnp.asarray([x, y, *Z], dtype=jnp.int32)
        v, p = _cmisymb(self.D, idx, k, self.r, self.r ** len(Z), self.sig_samples, self.r)
        return float(v), float(p)
