"""SEM BIC and degenerate Gaussian BIC (Tetrad 7.6.5 SemBicScore and
DegenerateGaussianScore)."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .base import Score
from ..ci.base import pad_batch
from ..ci.gaussian import embed


@jax.jit
def _resid_var(S, idx):
    """For each row (i, P...) of idx: the residual variance of i given P,
    b*' S b* with b* = (1, -S_PP^-1 S_Pi), as Tetrad's getVarRy."""
    def one(ix):
        C = S[ix[:, None], ix[None, :]]
        if ix.shape[0] == 1:
            return C[0, 0]
        b = jnp.linalg.solve(C[1:, 1:], C[1:, 0])
        bs = jnp.concatenate([jnp.ones(1), -b])
        return bs @ C @ bs
    return jax.vmap(one)(idx)


def _resid_var_np(S, idx):
    """The same as _resid_var, with numpy (for small batches: no dispatch cost)."""
    C = S[idx[:, :, None], idx[:, None, :]]
    if idx.shape[1] == 1:
        return C[:, 0, 0]
    with np.errstate(all="ignore"):
        try:
            b = np.linalg.solve(C[:, 1:, 1:], C[:, 1:, 0:1])[:, :, 0]
        except np.linalg.LinAlgError:              # a singular set: one at a time
            out = np.empty(len(idx))
            for t in range(len(idx)):
                try:
                    bt = np.linalg.solve(C[t, 1:, 1:], C[t, 1:, 0])
                    bs = np.concatenate([[1.0], -bt])
                    out[t] = bs @ C[t] @ bs
                except np.linalg.LinAlgError:
                    out[t] = np.nan
            return out
        bs = np.concatenate([np.ones((len(idx), 1)), -b], axis=1)
        return np.einsum("bi,bij,bj->b", bs, C, bs)


class SemBicScore(Score):
    """Linear Gaussian BIC (Tetrad SemBicScore, rule CHICKERING):
    ``-(n / 2) ln var(i | P) - c (k / 2) ln n - prior(k)``, with the
    unbiased covariance matrix. A singular S_PP gives NaN."""
    name = "sem_bic_score"
    jax_min_batch = 256

    def __init__(self, data, kinds=None, penalty_discount=1.0, structure_prior=0.0, tetrad_compat=False, cov=None):
        super().__init__(data, kinds, penalty_discount, structure_prior, tetrad_compat)
        S = np.cov(self.data, rowvar=False) if cov is None else np.asarray(cov)
        self.S_np = np.atleast_2d(np.asarray(S, dtype=np.float64))
        self.S = jnp.asarray(self.S_np)
        self.log_n = float(np.log(self.n))

    def _prior(self, k):
        if abs(self.structure_prior) <= 0:
            return 0.0
        q = self.structure_prior / self.p
        return -(k * np.log(q) + (self.p - k) * np.log(1.0 - q))

    def _local_batch(self, i, P):
        m, k = P.shape
        idx = np.concatenate([np.full((m, 1), i, np.int32), P], axis=1)
        if m >= self.jax_min_batch:                  # large batch: one jitted, vmapped call
            idx_p, B = pad_batch(idx, min_size=8)
            var = np.asarray(_resid_var(self.S, jnp.asarray(idx_p)))[:B]
        else:                                        # small batch: stacked LAPACK solves
            var = _resid_var_np(self.S_np, idx)
        with np.errstate(invalid="ignore", divide="ignore"):
            lik = -(self.n / 2.0) * np.log(var)
        out = lik - self.penalty_discount * (k / 2.0) * self.log_n - self._prior(k)
        return np.where(np.isfinite(out), out, np.nan)


class DegenerateGaussianScore(Score):
    """Degenerate Gaussian BIC (Andrews et al. 2019; Tetrad
    DegenerateGaussianScore). Each discrete variable becomes indicator
    columns (one category dropped). The local score of i is the sum of the
    SEM BIC scores of its columns a_1, a_2, ..., each given the columns of
    the parents and the earlier columns of i."""
    name = "dg_bic"
    data_types = ("continuous", "discrete", "mixed")

    def __init__(self, data, kinds=None, penalty_discount=1.0, structure_prior=0.0, tetrad_compat=False):
        super().__init__(data, kinds, penalty_discount, structure_prior, tetrad_compat)
        E, self.groups = embed(self.data, self.is_discrete, "last_seen" if tetrad_compat else "first")
        self.bic = SemBicScore(E, penalty_discount=penalty_discount)

    def _local(self, i, parents):
        B = [c for q in parents for c in self.groups[q]]
        s = 0.0
        for a in self.groups[i]:
            s += self.bic.local_score(a, B)
            B = B + [a]
        return s

    def _local_batch(self, i, P):
        Bs = [[c for q in row for c in self.groups[int(q)]] for row in P]
        total = np.zeros(len(P))
        for a in self.groups[i]:
            total += self.bic.local_scores(a, Bs)
            Bs = [b + [a] for b in Bs]
        return total
