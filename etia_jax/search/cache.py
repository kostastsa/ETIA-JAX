"""A cache of CI results for the searches (PC, CPC, FCI, ...).

The key is the pair (x, y) and the set Z. For a symmetric test the key is the
unordered pair {x, y}. For an asymmetric test (CG-LRT, DG-LRT: y is the child)
each direction has its own answer. This is also what Tetrad 7.6.5 does in
practice: its IndependenceFact has a symmetric equals() but hashCode() = 1
and a directional compareTo(), so in the tree bin of its ConcurrentHashMap a
lookup of the reverse direction misses (except by chance).

``prefetch`` computes many queries in one batched call (``test_batch``) when
the test has a vectorized path. A search asks for the queries that it may
need at the start of a level, then replays its sequential logic from the
cache. This gives the same result as one query at a time.
"""
import numpy as np

from ..ci.base import CITest


def _key(x, y, Z, symmetric=True):
    if symmetric:
        return (min(x, y), max(x, y), tuple(sorted(Z)))
    return (x, y, tuple(sorted(Z)))


class CICache:
    def __init__(self, ci, alpha):
        self.ci, self.alpha = ci, float(alpha)
        self.facts = {}       # key -> (independent, stat, p), see _key
        self.store = {}       # (x, y, Z) -> (independent, stat, p): computed, directed
        self.n_tests = 0      # queries answered (as Tetrad counts them)
        self.n_computed = 0   # queries computed (with the extra queries of prefetch)
        tb = getattr(type(ci), "test_batch", None)
        self.batched = tb is not None and tb is not CITest.test_batch
        self.symmetric = getattr(ci, "symmetric", True)

    def _directed(self, x, y, Z):
        return (x, y, tuple(sorted(Z)))

    def prefetch(self, queries):
        """Compute the queries (x, y, Z) that are not known yet, in one batched
        call per size of Z. Without a batched path, do nothing (lazy)."""
        if not self.batched:
            return
        need = {}
        for x, y, Z in queries:
            if _key(x, y, Z, self.symmetric) in self.facts:
                continue
            d = self._directed(x, y, Z)
            if d in self.store or (self.symmetric and self._directed(y, x, Z) in self.store):
                continue
            need.setdefault(len(Z), {})[d] = None
        for k, qs in need.items():
            qs = list(qs)
            xs = np.array([q[0] for q in qs], dtype=np.int32)
            ys = np.array([q[1] for q in qs], dtype=np.int32)
            Zs = np.array([q[2] for q in qs], dtype=np.int32).reshape(len(qs), k)
            stats, ps = self.ci.test_batch(xs, ys, Zs)
            self.n_computed += len(qs)
            for q, s, p in zip(qs, stats, ps):
                self.store[q] = (self.ci.decide(float(s), float(p), self.alpha), float(s), float(p))

    def query(self, x, y, Z):
        """(independent, stat, p) of x _||_ y | Z, as Tetrad's cached test."""
        k = _key(x, y, Z, self.symmetric)
        r = self.facts.get(k)
        if r is None:
            d = self._directed(x, y, Z)
            r = self.store.get(d)
            if r is None and self.symmetric:
                r = self.store.get(self._directed(y, x, Z))
            if r is None:
                r = self.ci.independent(x, y, list(d[2]), self.alpha)
                r = (bool(r[0]), float(r[1]), float(r[2]))
                self.n_computed += 1
            self.facts[k] = r
        self.n_tests += 1
        return r

    def independent(self, x, y, Z):
        return self.query(x, y, Z)[0]
