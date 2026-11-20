"""FGES (increment 5): Fast Greedy Equivalence Search (Ramsey et al. 2017).

Tetrad 7.6.5 ``Fges`` with ``Bes`` and ``MeekRules``, as ETIA runs it
(``new Fges(score)``, one thread, faithfulness not assumed):

1. Effect edges: each pair (x, y) with localScoreDiff(x, y) > 0.
2. Three rounds of a forward step (FES) and a backward step (BES). The
   forward step looks at the effect edges in round 1, at the nodes of
   possible non-colliders in round 2 ("cover non-colliders"), and at all
   the pairs in round 3 ("allow unfaithfulness").
3. Each step keeps a sorted set of candidate operators (by score change,
   then by creation index), applies the best valid one, goes back to the
   CPDAG with the Meek rules, and scores the operators again near the
   changed nodes.

The local score of a node is not added for an edge at a node whose degree
is at the score's getMaxDegree() (ceil(ln n) for most scores).

Two modes:

* ``tetrad_compat=True``: the iteration orders of Tetrad (Java HashSet
  orders of nodes and edges in ``TGraph``), and Tetrad's degree limit.
  This gives the same CPDAG as ETIA.
* default: the same algorithm with index orders and no degree limit.
"""
import heapq
from itertools import combinations

import numpy as np

from .tetrad_graph import TGraph, JHashSet, JNode, ARROW, TAIL

NONE_LIMIT = 1000


# --------------------------------------------------------------- node sets
class _JSet:
    """java.util.HashSet<Node> (Java iteration order)."""
    __slots__ = ("h", "s")

    def __init__(self, h, items=(), cap=None):
        self.h, self.s = h, JHashSet(cap)
        for i in items:
            self.add(i)

    @classmethod
    def of_collection(cls, h, items):
        """new HashSet<>(collection): the copy constructor's capacity."""
        items = list(items)
        s = cls(h, cap=max(int(len(items) / 0.75) + 1, 16))
        for i in items:
            s.add(i)
        return s

    def add(self, i):
        self.s.add(JNode(i, self.h[i]))

    def discard(self, i):
        self.s.remove(JNode(i, self.h[i]))

    def __contains__(self, i):
        return JNode(i, self.h[i]) in self.s

    def __iter__(self):
        return (n.i for n in self.s)

    def __len__(self):
        return len(self.s)


class _ISet:
    """A node set that iterates in index order (default mode)."""
    __slots__ = ("s",)

    def __init__(self, h=None, items=(), cap=None):
        self.s = set(items)

    @classmethod
    def of_collection(cls, h, items):
        return cls(h, items)

    def add(self, i):
        self.s.add(i)

    def discard(self, i):
        self.s.discard(i)

    def __contains__(self, i):
        return i in self.s

    def __iter__(self):
        return iter(sorted(self.s))

    def __len__(self):
        return len(self.s)


def _undirected(e):
    return e is not None and e.e1 == TAIL and e.e2 == TAIL


def _points_towards(e, node):
    """Edge.pointsTowards(node): an arrow at node and a tail (or circle) at the other end."""
    return e is not None and e.proximal_endpoint(node) == ARROW and e.distal_endpoint(node) != ARROW


class _Arrow:
    __slots__ = ("bump", "a", "b", "h_or_t", "t_neighbors", "na_yx", "parents", "index")

    def __init__(self, bump, a, b, h_or_t, t_neighbors, na_yx, parents, index):
        self.bump, self.a, self.b = bump, a, b
        self.h_or_t, self.t_neighbors, self.na_yx, self.parents = h_or_t, t_neighbors, na_yx, parents
        self.index = index


class _SortedArrows:
    """ConcurrentSkipListSet<Arrow>: the largest bump first, then the
    smallest index."""

    def __init__(self):
        self.q = []

    def add(self, arrow):
        heapq.heappush(self.q, (-arrow.bump, arrow.index, arrow))

    def pop_first(self):
        return heapq.heappop(self.q)[2]

    def __bool__(self):
        return bool(self.q)


# ------------------------------------------------------------------ the graph
class _Graph:
    """The search graph: a TGraph, with Java orders (compat) or index orders."""

    def __init__(self, p, compat):
        self.g = TGraph(p)
        self.p, self.compat = p, compat
        self.Set = _JSet if compat else _ISet
        self.h = self.g.h

    def new_set(self, items=()):
        return self.Set(self.h, items)

    def set_of(self, items):
        return self.Set.of_collection(self.h, items)

    # orders
    def adj(self, n):
        a = self.g.get_adjacent_nodes(n)
        return a if self.compat else sorted(a)

    def parents(self, n):
        a = self.g.get_parents(n)
        return a if self.compat else sorted(a)

    def edges_of(self, n):
        es = self.g.get_edges_of(n)
        return es if self.compat else sorted(es, key=lambda e: e.distal(n))

    def edges(self):
        es = self.g.get_edges()
        return es if self.compat else sorted(es, key=lambda e: (min(e.n1, e.n2), max(e.n1, e.n2)))

    # queries
    def is_adj(self, a, b):
        return self.g.is_adjacent(a, b)

    def edge(self, a, b):
        return self.g.get_edge(a, b)

    def degree(self, n):
        return len(self.g.edge_lists[n])

    def is_parent_of(self, a, b):
        return self.g.is_parent_of(a, b)

    def is_directed(self, a, b):
        """Paths.isDirected(a, b): the edge a -> b."""
        return _points_towards(self.g.get_edge(a, b), b)

    def is_undirected(self, a, b):
        return _undirected(self.g.get_edge(a, b))

    def children(self, n):
        return [e.distal(n) for e in self.g.edge_lists[n] if _points_towards(e, e.distal(n))]

    def exists_directed_path(self, a, b):
        seen, todo = {a}, [a]
        while todo:
            t = todo.pop()
            for c in self.children(t):
                if c == b:
                    return True
                if c not in seen:
                    seen.add(c)
                    todo.append(c)
        return False

    def common_adjacents(self, x, y):
        """new HashSet<>(adj(x)) retainAll adj(y)."""
        s = self.set_of(self.adj(x))
        ay = set(self.g.get_adjacent_nodes(y))
        for n in list(s):
            if n not in ay:
                s.discard(n)
        return s

    # changes
    def remove(self, a, b):
        e = self.g.get_edge(a, b)
        if e is not None:
            self.g.remove_edge(e)

    def add_directed(self, a, b):
        self.g.add_directed_edge(a, b)

    def add_undirected(self, a, b):
        self.g.add_undirected_edge(a, b)

    def marks(self):
        return self.g.marks()


# ------------------------------------------------------------- Meek rules
def meek_orient_implied(G, prevent_cycles):
    """MeekRules.orientImplied (Tetrad 7.6.5, no knowledge: R1 to R3, with
    the revert to unshielded colliders). Returns the set of visited nodes."""
    visited = G.new_set()

    def direct(a, c):
        e = G.edge(a, c)
        if not _undirected(e):
            return False
        G.g.remove_edge(e)
        if prevent_cycles and G.exists_directed_path(c, a):
            G.add_directed(c, a)
        else:
            G.add_directed(a, c)
        visited.add(a)
        visited.add(c)
        return True

    def r1(b, c):
        for a in G.parents(b):
            if G.is_adj(c, a):
                continue
            if direct(b, c):
                return True
        return False

    def r2(a, c):
        oriented = False
        for b in G.common_adjacents(a, c):
            if G.is_directed(a, b) and G.is_directed(b, c):
                if direct(a, c):
                    oriented = True
            if G.is_directed(c, b) and G.is_directed(b, a):
                if direct(c, a):
                    oriented = True
        return oriented

    def r3(d, a):
        nodes = list(G.common_adjacents(a, d))
        if len(nodes) < 2:
            return False
        oriented = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                b, c = nodes[i], nodes[j]
                if not G.is_adj(b, c):
                    if (G.is_undirected(d, a) and G.is_undirected(d, b) and G.is_undirected(d, c)
                            and G.is_directed(b, a) and G.is_directed(c, a)):
                        if direct(d, a):
                            oriented = True
        return oriented

    # revert to the unshielded colliders
    for y in range(G.p):
        parents = G.parents(y)
        for p_ in parents:
            if any(p_ != q and not G.is_adj(p_, q) for q in parents):
                continue
            G.remove(p_, y)
            G.add_undirected(p_, y)
            visited.add(p_)
            visited.add(y)

    oriented = True
    while oriented:
        oriented = False
        for e in G.edges():
            if not _undirected(e):
                continue
            x, y = e.n1, e.n2
            if r1(x, y) or r1(y, x) or r2(x, y) or r2(y, x) or r3(x, y) or r3(y, x):
                oriented = True
    return visited


# ------------------------------------------------------------------- FGES
class _Fges:
    def __init__(self, score, p, compat, max_degree=None):
        self.score, self.p, self.compat = score, p, compat
        self.G = _Graph(p, compat)
        self.effect = _Graph(p, compat)
        if max_degree is None:
            max_degree = score.max_degree() if compat else -1
        self.max_degree = max_degree
        self.sorted_arrows = _SortedArrows()
        self.arrows_map = {}
        self.arrow_index = 0
        self.mode = "heuristic"
        self.n_inserts = self.n_deletes = 0

    # --------------------------------------------------------- helpers
    def na_yx(self, x, y):
        G = self.G
        s = G.new_set()
        for z in G.adj(y):
            if z == x:
                continue
            if not _undirected(G.edge(y, z)):
                continue
            if not G.is_adj(z, x):
                continue
            s.add(z)
        return s

    def t_neighbors(self, x, y):
        G = self.G
        out = []
        for e in G.edges_of(y):
            if not _undirected(e):
                continue
            z = e.distal(y)
            if G.is_adj(z, x):
                continue
            out.append(z)
        return out

    def is_clique(self, nodes):
        nodes = list(nodes)
        return all(self.G.is_adj(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)))

    def semidirected_ok(self, frm, to, cond):
        """semidirectedPathCondition: no semidirected path frm ... to outside cond."""
        G = self.G
        Q, V = [frm], {frm}
        while Q:
            t = Q.pop(0)
            if t in cond:
                continue
            if t == to:
                return False
            for e in G.g.edge_lists[t]:
                if t == e.n1 and e.e1 == TAIL:
                    c = e.n2
                elif t == e.n2 and e.e2 == TAIL:
                    c = e.n1
                else:
                    continue
                if c not in V:
                    V.add(c)
                    Q.append(c)
        return True

    # ------------------------------------------------------- forward
    def add_arrow(self, a, b, h_or_t, t_nb, na_yx, parents, bump):
        self.sorted_arrows.add(_Arrow(bump, a, b, h_or_t, t_nb, na_yx, parents, self.arrow_index))
        self.arrow_index += 1

    def initialize_effect_edges(self):
        sc, p = self.score, self.p
        for i in range(p):
            sc.local_scores(i, [()] + [(j,) for j in range(i + 1, p)])
        for i in range(p):
            y = i
            for j in range(i + 1, p):
                x = j
                bump = sc.local_score_diff(x, y)
                if bump > 0:
                    self.effect.add_undirected(x, y)
                    self.add_arrow(x, y, (), frozenset(), frozenset(), frozenset(), bump)
                    self.add_arrow(y, x, (), frozenset(), frozenset(), frozenset(), bump)

    def reevaluate_forward(self, nodes):
        G = self.G
        for y in list(nodes):
            if self.mode == "heuristic":
                adj = self.effect.adj(y)
            elif self.mode == "cover":
                g = G.new_set()
                for n in G.adj(y):
                    for m in G.adj(n):
                        if G.is_adj(y, m):
                            continue
                        if G.g.is_def_collider(m, n, y):
                            continue
                        g.add(m)
                adj = list(g)
            else:
                adj = list(range(self.p))
            for x in adj:
                self.calculate_arrows_forward(x, y)

    def calculate_arrows_forward(self, a, b):
        G = self.G
        if a == b or G.is_adj(a, b):
            return
        na_yx = self.na_yx(a, b)
        t_nb = self.t_neighbors(a, b)
        parents = frozenset(G.g.get_parents(b))
        config = (frozenset(t_nb), frozenset(na_yx), parents)
        if self.arrows_map.get((a, b)) == config:
            return
        self.arrows_map[(a, b)] = config
        base = set(na_yx) | parents
        Ts = [c for k in range(len(t_nb) + 1) for c in combinations(range(len(t_nb)), k)]
        Zs = [tuple(base | {t_nb[i] for i in c}) for c in Ts]
        bumps = self.score.local_score_diffs(a, b, Zs)
        max_bump, max_c = -np.inf, None
        for c, bump in zip(Ts, bumps):
            if bump > max_bump:
                max_bump, max_c = bump, c
        if max_bump > 0:
            T = G.new_set(t_nb[i] for i in max_c)              # GraphUtils.asSet(choice, TNeighbors)
            self.add_arrow(a, b, T, frozenset(t_nb), frozenset(na_yx), parents, float(max_bump))

    def valid_insert(self, x, y, T, na_yx):
        union = set(T) | set(na_yx)
        return self.is_clique(union) and self.semidirected_ok(y, x, union)

    def insert(self, x, y, T):
        G = self.G
        G.add_directed(x, y)
        for t in T:
            G.remove(t, y)
            G.add_directed(t, y)
        self.n_inserts += 1

    def fes(self):
        G = self.G
        md = NONE_LIMIT if self.max_degree == -1 else self.max_degree
        self.reevaluate_forward(G.set_of(range(self.p)))
        while self.sorted_arrows:
            ar = self.sorted_arrows.pop_first()
            x, y = ar.a, ar.b
            if G.is_adj(x, y):
                continue
            if G.degree(x) > md - 1 or G.degree(y) > md - 1:
                continue
            na = self.na_yx(x, y)
            if frozenset(na) != ar.na_yx:
                continue
            if frozenset(self.t_neighbors(x, y)) != ar.t_neighbors:
                continue
            if frozenset(G.g.get_parents(y)) != ar.parents:
                continue
            if not self.valid_insert(x, y, ar.h_or_t, na):
                continue
            self.insert(x, y, ar.h_or_t)
            process = meek_orient_implied(G, prevent_cycles=True)
            process.add(x)
            process.add(y)
            for n in G.common_adjacents(x, y):
                process.add(n)
            self.reevaluate_forward(G.set_of(list(process)))

    # ------------------------------------------------------ backward
    def bes(self):
        """Bes.bes (Tetrad 7.6.5): a new sorted set and config map each call."""
        G = self.G
        back, cfg, idx = _SortedArrows(), {}, [0]

        def calc(a, b):
            na = self.na_yx(a, b)
            parents = frozenset(G.g.get_parents(b))
            config = (frozenset(na), parents)
            if cfg.get((a, b)) == config:
                return
            cfg[(a, b)] = config
            lst = list(na)
            Cs = [c for k in range(len(lst) + 1) for c in combinations(range(len(lst)), k)]
            Zs = []
            for c in Cs:
                s = {lst[i] for i in c} | parents
                s.discard(a)
                Zs.append(tuple(s))
            bumps = -self.score.local_score_diffs(a, b, Zs)
            max_bump, max_c = -np.inf, None
            for c, bump in zip(Cs, bumps):
                if bump > max_bump:
                    max_bump, max_c = bump, c
            if max_bump > 0:
                H = G.set_of(list(na))                       # new HashSet<>(naYX) removeAll complement
                for i in max_c:
                    H.discard(lst[i])
                back.add(_Arrow(float(max_bump), a, b, H, None, frozenset(na), parents, idx[0]))
                idx[0] += 1

        def reevaluate(to_process):
            nodes = list(to_process)
            for r in nodes:
                for w in nodes:
                    e = G.edge(w, r)
                    if e is None:
                        continue
                    if _points_towards(e, r):
                        calc(w, r)
                    elif _points_towards(e, w):
                        calc(r, w)
                    else:
                        calc(w, r)
                        calc(r, w)

        reevaluate(G.set_of(range(self.p)))
        while back:
            ar = back.pop_first()
            x, y = ar.a, ar.b
            if not G.is_adj(x, y):
                continue
            if _points_towards(G.edge(x, y), x):
                continue
            if frozenset(self.na_yx(x, y)) != ar.na_yx:
                continue
            if frozenset(G.g.get_parents(y)) != ar.parents:
                continue
            diff = set(ar.na_yx) - set(ar.h_or_t)
            if not self.is_clique(diff):
                continue
            # delete
            G.remove(x, y)
            for h in ar.h_or_t:
                if G.is_parent_of(h, y) or G.is_parent_of(h, x):
                    continue
                G.remove(y, h)
                G.add_directed(y, h)
                if _undirected(G.edge(x, h)):
                    G.remove(x, h)
                    G.add_directed(x, h)
            self.n_deletes += 1
            process = meek_orient_implied(G, prevent_cycles=False)
            process.add(x)
            process.add(y)
            for n in G.adj(x):
                process.add(n)
            for n in G.adj(y):
                process.add(n)
            reevaluate(G.set_of(list(process)))

    # --------------------------------------------------------- search
    def search(self, faithfulness_assumed=False):
        self.initialize_effect_edges()
        self.mode = "heuristic"
        self.fes()
        self.bes()
        self.mode = "cover"
        self.fes()
        self.bes()
        if not faithfulness_assumed:
            self.mode = "all"
            self.fes()
            self.bes()
        return self.G


class FgesResult:
    def __init__(self, G, score, search):
        self.graph = G
        self.n_scores = score.n_calls
        self.n_inserts, self.n_deletes = search.n_inserts, search.n_deletes


def fges(score, tetrad_compat=False, faithfulness_assumed=False, max_degree=None):
    """FGES. ``score``: a Score of etia_jax.scores. Returns an FgesResult with
    the CPDAG (mark matrix; undirected edges are tail-tail) in ``graph``."""
    s = _Fges(score, score.p, tetrad_compat, max_degree)
    G = s.search(faithfulness_assumed)
    return FgesResult(G.marks(), score, s)
