"""BDeu and discrete BIC (Tetrad 7.6.5 BdeuScore and DiscreteBicScore)."""
import numpy as np
from scipy.special import gammaln

from .base import Score


class _CountScore(Score):
    data_types = ("discrete",)

    def __init__(self, data, kinds=None, penalty_discount=1.0, structure_prior=0.0, tetrad_compat=False):
        if kinds is None:
            kinds = ["discrete"] * np.shape(data)[1]
        super().__init__(data, kinds, penalty_discount, structure_prior, tetrad_compat)
        self.codes = np.zeros(self.data.shape, dtype=np.int64)
        self.n_cat = np.zeros(self.p, dtype=np.int64)
        for j in range(self.p):
            u, inv = np.unique(self.data[:, j], return_inverse=True)
            self.codes[:, j], self.n_cat[j] = inv.ravel(), len(u)

    def _counts(self, i, parents):
        """n_jk (r, c): rows = parent configurations (mixed radix, the first
        parent is the most significant digit), columns = values of i."""
        c = int(self.n_cat[i])
        row = np.zeros(self.n, dtype=np.int64)
        r = 1
        for q in parents:
            row = row * self.n_cat[q] + self.codes[:, q]
            r *= int(self.n_cat[q])
        n_jk = np.bincount(row * c + self.codes[:, i], minlength=r * c).reshape(r, c)
        return n_jk, r, c


class BdeuScore(_CountScore):
    """BDeu (sample prior 1 by default) with Tetrad's structure prior
    ``k ln(e / v) + (v - k) ln(1 - e / v)``. Tetrad 7.6.5 uses v = N - 1
    with N the number of rows (a defect, ``tetrad_compat=True``); the
    default uses v = p - 1, the number of other variables."""
    name = "bdeu"

    def __init__(self, data, kinds=None, penalty_discount=1.0, structure_prior=0.0, tetrad_compat=False,
                 sample_prior=1.0):
        super().__init__(data, kinds, penalty_discount, structure_prior, tetrad_compat)
        self.sample_prior = float(sample_prior)

    def _prior(self, k, N):
        e = self.structure_prior
        if e == 0:
            return 0.0
        v = (N - 1) if self.tetrad_compat else (self.p - 1)
        return k * np.log(e / v) + (v - k) * np.log(1.0 - e / v)

    def _local(self, i, parents):
        n_jk, r, c = self._counts(i, parents)
        n_j = n_jk.sum(1)
        cell, row = self.sample_prior / (c * r), self.sample_prior / r
        s = self._prior(len(parents), int(n_jk.sum()))
        s += -gammaln(row + n_j).sum() + gammaln(cell + n_jk).sum()
        s += r * gammaln(row) - c * r * gammaln(cell)
        return float(s) if np.isfinite(s) else np.nan


class DiscreteBicScore(_CountScore):
    """Discrete BIC: ``2 lik - c (attested rows) (r_i - 1) ln n`` plus a
    structure prior term. Tetrad 7.6.5 adds ``2 x (minus the log prior)``,
    which makes more parents better (a sign defect, ``tetrad_compat=True``);
    the default adds 2 x the log prior. Tetrad's getMaxDegree is 1000."""
    name = "discrete_bic"

    def max_degree(self):
        return 1000

    def _log_prior(self, k):
        if abs(self.structure_prior) <= 0:
            return 0.0
        q = self.structure_prior / self.p
        return k * np.log(q) + (self.p - k) * np.log(1.0 - q)

    def _local(self, i, parents):
        n_jk, r, c = self._counts(i, parents)
        n_j = n_jk.sum(1, keepdims=True)
        ok = (n_jk > 0) & (n_j > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lik = np.sum(np.where(ok, n_jk * np.log(n_jk / np.maximum(n_j, 1)), 0.0))
        params = int((n_j > 0).sum()) * (c - 1)
        lp = self._log_prior(len(parents))
        s = 2 * lik - self.penalty_discount * params * np.log(self.n) + 2 * (-lp if self.tetrad_compat else lp)
        return float(s) if np.isfinite(s) else np.nan
