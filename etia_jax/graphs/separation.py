"""d-separation in DAGs (Bayes-ball / reachability, Koller & Friedman Alg. 3.1)."""
import numpy as np
from .dag import ancestors_of


def d_connected_set(A, x, Z):
    """Nodes d-connected to ``x`` given ``Z`` in DAG ``A`` (excluding Z)."""
    A = np.asarray(A) != 0
    n = len(A)
    Z = set(int(z) for z in Z)
    anZ = ancestors_of(A, Z) if Z else np.zeros(n, dtype=bool)
    # direction: 0 = arrived from a child (moving up), 1 = from a parent (down)
    visited = set()
    reach = set()
    stack = [(x, 0)]
    while stack:
        v, d = stack.pop()
        if (v, d) in visited:
            continue
        visited.add((v, d))
        if v not in Z:
            reach.add(v)
        if d == 0 and v not in Z:
            stack.extend((p, 0) for p in np.flatnonzero(A[:, v]))
            stack.extend((c, 1) for c in np.flatnonzero(A[v]))
        elif d == 1:
            if v not in Z:
                stack.extend((c, 1) for c in np.flatnonzero(A[v]))
            if anZ[v]:
                stack.extend((p, 0) for p in np.flatnonzero(A[:, v]))
    reach.discard(x)
    return reach


def d_separated(A, x, y, Z=()):
    """True iff ``x`` and ``y`` are d-separated given ``Z`` in DAG ``A``."""
    if x == y:
        raise ValueError("x and y must differ")
    if x in Z or y in Z:
        raise ValueError("x and y must not be in Z")
    return y not in d_connected_set(A, x, Z)


class DSepOracle:
    """Conditional independence oracle on the observed nodes of a DAG.

    ``observed`` maps observed index -> DAG node. Calls use observed indices.
    """

    def __init__(self, A, observed=None):
        self.A = np.asarray(A) != 0
        n = len(self.A)
        self.observed = np.arange(n) if observed is None else np.asarray(observed)
        self.n_calls = 0

    def __call__(self, x, y, Z=()):
        """Return p-value-like 1.0 (independent) or 0.0 (dependent)."""
        self.n_calls += 1
        o = self.observed
        return 1.0 if d_separated(self.A, int(o[x]), int(o[y]), [int(o[z]) for z in Z]) else 0.0
