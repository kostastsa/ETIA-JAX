"""The base class of the scores (increment 5).

A score gives ``local_score(i, parents)``: the score of node i with the
given parents. FGES and GFCI use ``local_score_diff(x, y, Z)``: the change of
the score of y when x is added to its parents Z. The scores are
decomposable, and a cache keeps each local score (the key is the sorted
parent set, because all the scores do not depend on the parent order).
"""
import math

import numpy as np


class Score:
    name = "base"
    data_types = ("continuous",)

    def __init__(self, data, kinds=None, penalty_discount=1.0, structure_prior=0.0, tetrad_compat=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.n, self.p = self.data.shape
        self.kinds = list(kinds) if kinds is not None else ["continuous"] * self.p
        self.is_discrete = np.array([k == "discrete" for k in self.kinds])
        self.penalty_discount = float(penalty_discount)
        self.structure_prior = float(structure_prior)
        self.tetrad_compat = bool(tetrad_compat)
        self.cache = {}
        self.n_calls = 0                 # local scores computed (not from the cache)

    # -- to implement ----------------------------------------------------
    def _local_batch(self, i, P):
        """Local scores of child i for the parent sets P (an (m, k) int array)."""
        return np.array([self._local(i, tuple(int(v) for v in row)) for row in P])

    def _local(self, i, parents):
        raise NotImplementedError

    def max_degree(self):
        """Tetrad's getMaxDegree(): FGES does not add an edge at a node with
        this degree (Tetrad 7.6.5: ceil(ln n) for most scores)."""
        return int(math.ceil(math.log(self.n)))

    # -- interface -------------------------------------------------------
    def local_score(self, i, parents=()):
        key = (int(i), tuple(sorted(int(v) for v in parents)))
        r = self.cache.get(key)
        if r is None:
            r = float(self._local_batch(key[0], np.array([key[1]], dtype=np.int32).reshape(1, len(key[1])))[0])
            self.n_calls += 1
            self.cache[key] = r
        return r

    def local_scores(self, i, parent_sets):
        """Local scores of child i for many parent sets (one batched call
        for each size of the parent sets that are not in the cache)."""
        keys = [(int(i), tuple(sorted(int(v) for v in P))) for P in parent_sets]
        need = {}
        for k in keys:
            if k not in self.cache:
                need.setdefault(len(k[1]), {})[k] = None
        for size, ks in need.items():
            ks = list(ks)
            P = np.array([k[1] for k in ks], dtype=np.int32).reshape(len(ks), size)
            vals = self._local_batch(int(i), P)
            self.n_calls += len(ks)
            for k, v in zip(ks, vals):
                self.cache[k] = float(v)
        return np.array([self.cache[k] for k in keys])

    def local_score_diff(self, x, y, Z=()):
        Z = tuple(Z)
        return self.local_score(y, Z + (x,)) - self.local_score(y, Z)

    def local_score_diffs(self, x, y, Zs):
        """local_score_diff(x, y, Z) for many sets Z, batched."""
        Zs = [tuple(Z) for Z in Zs]
        with_x = self.local_scores(y, [Z + (x,) for Z in Zs])
        without = self.local_scores(y, Zs)
        return with_x - without
