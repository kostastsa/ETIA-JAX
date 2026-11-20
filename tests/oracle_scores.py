"""A d-separation "score" for the oracle tests of FGES and GFCI: adding x to
the parents Z of y changes the score by +1 if x and y are d-connected
given Z, and by -1 if they are d-separated."""
import numpy as np

from etia_jax.graphs.separation import d_separated


class DSepScore:
    def __init__(self, A, observed=None):
        self.A = np.asarray(A) != 0
        self.o = np.arange(len(A)) if observed is None else np.asarray(observed)
        self.p = len(self.o)
        self.n_calls = 0

    def max_degree(self):
        return 1000

    def local_scores(self, i, sets):
        return np.zeros(len(sets))

    def local_score_diff(self, x, y, Z=()):
        self.n_calls += 1
        o = self.o
        return -1.0 if d_separated(self.A, int(o[x]), int(o[y]), [int(o[z]) for z in Z]) else 1.0

    def local_score_diffs(self, x, y, Zs):
        return np.array([self.local_score_diff(x, y, Z) for Z in Zs])
