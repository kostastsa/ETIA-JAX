"""Tetrad 7.6.5 ``GraphTransforms.dagFromCpdag`` and ``magFromPag`` (increment 9).

ETIA's Tetrad wrapper returns, next to the CPDAG or PAG, one random member
of the class: ``dagFromCpdag`` when the configuration has
``causal_sufficiency=True``, else ``magFromPag``. The OCT of ETIA takes the
Markov boundaries from this random graph. Tetrad draws the random numbers
in Java (``Collections.shuffle``, ``RandomUtil``) without a seed that Python
can set, so ETIA's results change from run to run. These ports draw from a
NumPy ``Generator``: the distribution is the distribution of Tetrad, the
numbers are not.

``magFromPag`` (``transormPagIntoRandomMag``):
  1. the ordered node pairs of all edges, shuffled;
  2. for each pair (a, b) with a circle at b: the mark at b becomes a tail
     or an arrow (probability 1/2 each);
  3. after each change, ``FciOrient.finalOrientation`` (R1-R10) runs with an
     m-separation test on the graph itself (``MsepTest``: the descendant
     map comes from the graph at the start of the run, the edges are the
     current edges).
  A circle can become a tail at both ends of an edge, so the "MAG" can
  have undirected edges.

``dagFromCpdag`` (``transformCpdagIntoRandomDag``): the undirected edges,
shuffled; the first edge x --- y that is still undirected, with y not an
ancestor of x, gets a random direction, then Meek's rules run
(``meekPreventCycles=true``), and the scan starts again.
"""
from collections import deque

import numpy as np

from ..graphs.encoding import NONE, CIRCLE, ARROW as MA, TAIL as MT
from .tetrad_graph import TGraph, ARROW, TAIL, CIRCLE as CI
from .pcalg import meek_tetrad
from .fci import _TetradFci

_EP = {1: CI, 2: ARROW, 3: TAIL}


def to_tgraph(G):
    p = len(G)
    g = TGraph(p)
    for a in range(p):
        for b in range(a + 1, p):
            if G[a, b] != NONE:
                g.add_edge(g.edge(a, b, _EP[int(G[b, a])], _EP[int(G[a, b])]))
    return g


class _MsepTest:
    """Tetrad MsepTest(graph): m-separation in the live graph, with the
    descendant map of the graph at construction and a cache of facts."""

    def __init__(self, g):
        self.g = g
        self.desc = {v: self._descendants(v) for v in range(g.p)}
        self.facts = {}
        self.n_tests = 0

    def _descendants(self, b):
        out, q = {b}, deque([b])
        while q:
            t = q.popleft()
            for e in self.g.edge_lists[t]:
                c = e.distal(t)
                if e.proximal_endpoint(t) == TAIL and e.proximal_endpoint(c) == ARROW and c not in out:
                    out.add(c)
                    q.append(c)
        return out

    def independent(self, x, y, Z):
        key = (x, y, frozenset(Z))
        r = self.facts.get(key)
        if r is None:
            r = self.facts[key] = not self._m_connected(x, y, set(Z))
        return r

    def _m_connected(self, x, y, Z):
        g = self.g
        if x == y:
            return True
        edges = lambda v: [(e.n1, e.n2, e.e1, e.e2) for e in g.edge_lists[v]]
        prox = lambda e, v: e[2] if e[0] == v else e[3]
        dist = lambda e, v: e[1] if e[0] == v else e[0]
        Q, V = deque(), set()
        for e in edges(x):
            if dist(e, x) == y:
                return True
            Q.append((e, x))
            V.add((e, x))
        while Q:
            e1, a = Q.popleft()
            b = dist(e1, a)
            for e2 in edges(b):
                c = dist(e2, b)
                if c == a:
                    continue
                collider = prox(e1, b) == ARROW and prox(e2, b) == ARROW
                anc = any(z in Z for z in self.desc[b])
                if ((not collider) and b not in Z) or (collider and anc):
                    if c == y:
                        return True
                    if prox(e1, b) == ARROW:
                        if e2[2] == TAIL and e2[3] == TAIL:
                            e2 = (b, c, TAIL, ARROW)
                        elif e2[2] == CI and e2[3] == CI:
                            e2 = (b, c, CI, ARROW)
                    u = (e2, b)
                    if u not in V:
                        V.add(u)
                        Q.append(u)
        return False


def tetrad_mag_from_pag(G, rng):
    """``GraphTransforms.magFromPag`` with the random numbers of ``rng``."""
    g = to_tgraph(G)
    pairs = []
    for e in g.get_edges():
        pairs.append((e.n1, e.n2))
        pairs.append((e.n2, e.n1))
    order = rng.permutation(len(pairs))
    for k in order:
        a, b = pairs[k]
        if g.get_endpoint(a, b) == CI:
            g.set_endpoint(a, b, TAIL if rng.random() < 0.5 else ARROW)
            orient = _TetradFci(_MsepTest(g), g.p)
            orient.final_orientation(g)
    return g.marks()


def _is_ancestor(G, y, x):
    """A directed path y -> ... -> x (or y == x)."""
    seen, q = {x}, deque([x])
    while q:
        t = q.popleft()
        if t == y:
            return True
        for c in np.flatnonzero((G[:, t] == MT) & (G[t, :] == MA)):
            if c not in seen:
                seen.add(int(c))
                q.append(int(c))
    return False


def tetrad_dag_from_cpdag(G, rng):
    """``GraphTransforms.dagFromCpdag`` with the random numbers of ``rng``."""
    G = np.array(G, dtype=np.int8)
    und = [(a, b) for a in range(len(G)) for b in range(a + 1, len(G)) if G[a, b] == MT and G[b, a] == MT]
    und = [und[k] for k in rng.permutation(len(und))]
    while True:
        for x, y in und:
            if not (G[x, y] == MT and G[y, x] == MT):
                continue
            if not _is_ancestor(G, y, x):
                if rng.random() < 0.5:
                    G[x, y], G[y, x] = MA, MT
                else:
                    G[y, x], G[x, y] = MA, MT
                meek_tetrad(G, prevent_cycles=True, revert=False)
                break
        else:
            return G
