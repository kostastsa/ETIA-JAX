"""GFCI (increment 5): Greedy FCI (Ogarrio, Spirtes and Ramsey 2016).

GFCI starts from the CPDAG of FGES, removes the extra edges with CI tests,
copies the colliders of the CPDAG, and orients the rest with the FCI rules.

* ``tetrad_compat=True``: Tetrad 7.6.5 ``GFci(test, score)`` as ETIA runs it:
  FGES with faithfulness assumed (two rounds) and no degree limit; the
  separating sets of ``SepsetsMinP`` (see ``_minp_sepset`` for its
  defects); no Possible-D-SEP step; the final orientation of FCI.
* default: the algorithm of the paper: FGES (default mode, faithfulness
  assumed), the extra-edge removal with the first separating set among the
  subsets of the adjacents, the GFCI collider rule, the Possible-D-SEP step
  of FCI, the GFCI collider rule again, and Zhang's rules.
"""
from itertools import combinations

import numpy as np

from ..graphs.encoding import NONE, CIRCLE, ARROW
from ..graphs.orientation import fci_orient
from .cache import CICache
from .fci import _TetradFci, _pds_step, _possible_dsep
from .fges import _Fges
from .tetrad_graph import TGraph, CIRCLE as C, ARROW as A


class GfciResult:
    def __init__(self, G, cpdag, cache, score, **extra):
        self.graph = G
        self.cpdag = cpdag
        self.n_tests = cache.n_tests
        self.n_computed = cache.n_computed
        self.n_scores = score.n_calls
        for k, v in extra.items():
            setattr(self, k, v)


def _sublists(n):
    return [c for k in range(n + 1) for c in combinations(range(n), k)]


# ------------------------------------------------------------- Tetrad 7.6.5
class _TetradGfci(_TetradFci):
    def minp_sepset(self, g, x, y):
        """SepsetFinder.getSepsetContainingMinP(graph, x, y, null, test, -1).

        Tetrad 7.6.5 takes the subset S of adj(x) \\ {y} with the smallest
        p-value of x _||_ y | S (the first one if there are ties), and
        returns it only if that p-value is above alpha: all the subsets must
        show independence. Its second search uses adj(x) again, not adj(y)
        (a defect), so adj(y) is never searched."""
        adjx = [n for n in g.get_adjacent_nodes(x) if n != y]
        subsets = [tuple(adjx[i] for i in c) for c in _sublists(len(adjx))]
        self.cache.prefetch([(x, y, S) for S in subsets])
        best, best_p = None, None
        for S in subsets:
            pv = self.cache.query(x, y, S)[2]
            if pv != pv:                                   # NaN: larger than all (Double.compareTo)
                if best is None:
                    best, best_p = S, pv
                continue
            if best_p is None or best_p != best_p or pv < best_p:
                best, best_p = S, pv
        if best is not None and best_p > self.cache.alpha:
            return set(best)
        return None

    def fges_step(self, score):
        fg = _Fges(score, self.p, compat=True, max_degree=-1)
        g = fg.search(faithfulness_assumed=True).g          # the FGES graph becomes the PAG
        cpdag = TGraph(self.p)                             # new EdgeListGraph(graph)
        for e in g.get_edges():
            cpdag.add_edge(e)
        self.cpdag_marks = cpdag.marks()
        return g, cpdag

    def removal_step(self, g, cpdag):
        """GraphUtils.gfciExtraEdgeRemovalStep."""
        for b in range(self.p):
            adj = cpdag.get_adjacent_nodes(b)
            if len(adj) < 2:
                continue
            for i, j in combinations(range(len(adj)), 2):
                a, c = adj[i], adj[j]
                if g.is_adjacent(a, c) and cpdag.is_adjacent(a, c):
                    if self.minp_sepset(g, a, c) is not None:
                        g.remove_edge(g.get_edge(a, c))

    def r0_step(self, g, cpdag):
        """GraphUtils.gfciR0."""
        g.reorient_all_with(C)
        for y in range(self.p):
            adj = g.get_adjacent_nodes(y)
            if len(adj) < 2:
                continue
            for i, j in combinations(range(len(adj)), 2):
                x, z = adj[i], adj[j]
                unshielded = g.is_adjacent(x, y) and g.is_adjacent(y, z) and not g.is_adjacent(x, z)
                cpdag_collider = (x != z and cpdag.is_adjacent(x, y) and cpdag.is_adjacent(y, z)
                                  and not cpdag.is_adjacent(x, z) and cpdag.is_def_collider(x, y, z))
                if unshielded and cpdag_collider:
                    if self.arrowhead_allowed(g, x, y) and self.arrowhead_allowed(g, z, y):
                        g.set_endpoint(x, y, A)
                        g.set_endpoint(z, y, A)
                elif cpdag.is_adjacent(x, z):
                    if self.arrowhead_allowed(g, x, y) and self.arrowhead_allowed(g, z, y):
                        S = self.minp_sepset(g, x, z)
                        if S is not None:
                            e = g.get_edge(x, z)              # pag.removeEdge(x, z): no edge, no change
                            if e is not None:
                                g.remove_edge(e)
                            if y not in S:
                                g.set_endpoint(x, y, A)
                                g.set_endpoint(z, y, A)

    def search(self, score):
        g, cpdag = self.fges_step(score)
        self.removal_step(g, cpdag)
        self.r0_step(g, cpdag)
        self.final_orientation(g)
        return g


# ---------------------------------------------------------------- textbook
def _gfci_textbook(cache, score, p):
    fg = _Fges(score, p, compat=False, max_degree=-1)
    cp = fg.search(faithfulness_assumed=True).marks()
    adj0 = cp != NONE
    G = np.where(adj0, CIRCLE, NONE).astype(np.int8)
    sepset = {}
    # extra-edge removal: the first separating set among subsets of the
    # adjacents of x or of y in the CPDAG (stable: the CPDAG adjacencies)
    for x in range(p):
        for y in range(x + 1, p):
            if not adj0[x, y]:
                continue
            sides = [[v for v in np.flatnonzero(adj0[x]) if v != y], [v for v in np.flatnonzero(adj0[y]) if v != x]]
            qs = [(x, y, tuple(side[i] for i in c)) for side in sides for c in _sublists(len(side))]
            cache.prefetch(qs)
            for _, _, S in qs:
                if cache.independent(x, y, S):
                    G[x, y] = G[y, x] = NONE
                    sepset[(x, y)] = S
                    break

    def cpdag_collider(a, b, c):
        return (adj0[a, b] and adj0[b, c] and not adj0[a, c]
                and cp[a, b] == ARROW and cp[c, b] == ARROW)

    def r0():
        for b in range(p):
            nb = np.flatnonzero(G[b] != NONE)
            for a, c in combinations(nb, 2):
                if G[a, c] != NONE:
                    continue
                S = sepset.get((min(a, c), max(a, c)))
                if cpdag_collider(a, b, c) or (S is not None and b not in S):
                    G[a, b] = G[c, b] = ARROW

    r0()
    for x, y, Z in _pds_step(cache, G):
        G[x, y] = G[y, x] = NONE
        sepset[(x, y)] = tuple(Z)
    G[G != NONE] = CIRCLE
    r0()

    def find_sepset(d, c):
        """A separating set of a pair that FGES did not make adjacent (GFCI
        keeps none): the first one among the subsets of the adjacents of d or
        of c, then among the subsets of Possible-D-SEP."""
        x, y = min(d, c), max(d, c)
        sides = [[v for v in np.flatnonzero(G[x] != NONE) if v != y], [v for v in np.flatnonzero(G[y] != NONE) if v != x]]
        sides += [_possible_dsep(G, x, y), _possible_dsep(G, y, x)]
        for side in sides:
            qs = [(x, y, tuple(side[i] for i in cmb)) for cmb in _sublists(len(side))]
            cache.prefetch(qs)
            for _, _, S in qs:
                if cache.independent(x, y, S):
                    return S
        return None

    def is_noncollider(d, a, b, c):
        key = (min(d, c), max(d, c))
        if key not in sepset:
            sepset[key] = find_sepset(d, c)
        S = sepset[key]
        return S is not None and b in S

    fci_orient(G, is_noncollider, complete=True)
    sepset = {k: v for k, v in sepset.items() if v is not None}
    return G, cp, sepset


def gfci(ci, score, alpha=0.05, tetrad_compat=False):
    """GFCI with a CI test ``ci`` and a score ``score`` (etia_jax.scores).
    Returns a GfciResult with the PAG (mark matrix) in ``graph`` and the
    FGES CPDAG in ``cpdag``."""
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    if tetrad_compat:
        t = _TetradGfci(cache, p)
        g = t.search(score)
        return GfciResult(g.marks(), t.cpdag_marks, cache, score)
    G, cp, sep = _gfci_textbook(cache, score, p)
    return GfciResult(G, cp, cache, score, sepset=sep)
