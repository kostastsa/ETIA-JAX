"""Conditional Gaussian BIC (Tetrad 7.6.5 ConditionalGaussianScore)."""
import jax.numpy as jnp
import numpy as np

from .base import Score
from ..ci.base import pad_batch
from ..ci.mixed import CGLrt, _cg_joint_batch


class ConditionalGaussianScore(Score):
    """``2 (lik + prior) - c dof ln n``, where lik and dof are the conditional
    Gaussian log-likelihood of i given its parents and its number of
    parameters (the same likelihood as the CG-LRT test: discretize = True,
    3 bins, cells with fewer than 4 rows skipped)."""
    name = "cg_bic"
    data_types = ("continuous", "discrete", "mixed")

    def __init__(self, data, kinds=None, penalty_discount=1.0, structure_prior=0.0, tetrad_compat=False):
        super().__init__(data, kinds, penalty_discount, structure_prior, tetrad_compat)
        self.lik = CGLrt(self.data, self.kinds)

    def _prior(self, k):
        if self.structure_prior <= 0:
            return 0.0
        m = self.p - 1
        q = self.structure_prior / m
        return k * np.log(q) + (m - k) * np.log(1.0 - q)

    def _local_batch(self, i, P):
        """All the parent sets of one shape in one vmapped call (the kernel of CG-LRT)."""
        L = self.lik
        out = np.empty(len(P))
        jobs = {}
        for q, row in enumerate(P):
            parts = L._parts(i, [int(v) for v in row])            # X1, A1, X0, A0
            jobs.setdefault(tuple(len(v) for v in parts), []).append((q, parts))
        r = int(L.n_cat.max())
        for shape, items in jobs.items():
            arrs = [np.array([it[1][k] for it in items], dtype=np.int32).reshape(len(items), shape[k])
                    for k in range(4)]
            J = []
            for k in (0, 2):
                Xi, B = pad_batch(arrs[k], min_size=1)
                Ai, _ = pad_batch(arrs[k + 1], min_size=1)
                J.append(np.asarray(_cg_joint_batch(L.X, L.codes_j, jnp.asarray(Xi), jnp.asarray(Ai),
                                                    r, float(L.n), L.min_cell))[:B])
            for t, (q, parts) in enumerate(items):
                lik = J[0][t] - J[1][t]
                dof = L._dof(parts[0], parts[1]) - L._dof(parts[2], parts[3])
                sc = 2.0 * (lik + self._prior(len(P[q]))) - self.penalty_discount * dof * np.log(self.n)
                out[q] = sc if np.isfinite(sc) else -np.inf
        return out

    def _local(self, i, parents):
        lik, dof = self.lik.loglik(i, list(parents))
        s = 2.0 * (lik + self._prior(len(parents))) - self.penalty_discount * dof * np.log(self.n)
        return float(s) if np.isfinite(s) else -np.inf
