"""SVAR-FCI and SVAR-GFCI (increment 6): FCI and GFCI for time series with
"similar pairs" (Malinsky and Spirtes 2018).

The data is ETIA's lagged table (X1..Xp, X1:1..Xp:1, ...) with the tier
knowledge of ``Knowledge.time_lagged``. A pair of nodes (x, y) has "similar
pairs": the pairs with the same variables and the same lag difference at the
other lags. Tetrad 7.6.5 uses them in three places:

* ``SvarFas``: an adjacency (depth 0) is copied to the similar pairs, and an
  edge removal is copied to them (the copies are not tested).
* ``SvarSetEndpointStrategy``: a mark that FciOrient sets is copied to the
  similar pairs (where they have a circle) -- only when tier(x) >= tier(y),
  as in Tetrad.
* ``SvarFci.removeSimilarPairs``: an edge that the possible-m-sep step removes
  is removed at the similar pairs.

``tetrad_compat=True`` follows Tetrad 7.6.5 (``SvarFci(test)`` and
``SvarGfci(test, score)`` with ETIA's knowledge).
"""
from itertools import combinations

from .cache import CICache
from .fci import _TetradFci, _sublists, node_compare_key
from .gfci import _TetradGfci, GfciResult
from .pcalg import SearchResult, Skeleton
from .tetrad_graph import TGraph, CIRCLE as C
from .fges import _undirected, _points_towards


def _no_lag(name):
    return name.split(":", 1)[0]


class _SvarTiers:
    """The parts of Knowledge that the similar-pair code uses: getTier(i)
    (the names of tier i, sorted as Java Strings), isInWhichTier, and
    test.getVariable(name)."""

    def __init__(self, kn, names):
        self.kn, self.names = kn, names
        self.index = {n: i for i, n in enumerate(names)}
        self.sorted_tiers = [sorted(names[v] for v in t) for t in kn.tiers]

    def similar_pairs(self, x, y, only_if_x_later=False):
        """returnSimilarPairs(x, y): the (x1, y1) at the other lags. With
        ``only_if_x_later`` (orientSimilarPairs), nothing when tier(x) < tier(y)."""
        kn, T = self.kn, self.sorted_tiers
        tx, ty = kn.tier(x), kn.tier(y)
        if tx < 0 or ty < 0:
            return []
        if only_if_x_later and tx < ty:
            return []
        diff = abs(tx - ty)
        tier_x, tier_y = T[tx], T[ty]
        nx, ny = _no_lag(self.names[x]), _no_lag(self.names[y])
        ix = next((i for i, n in enumerate(tier_x) if _no_lag(n) == nx), -1)
        iy = next((i for i, n in enumerate(tier_y) if _no_lag(n) == ny), -1)
        out = []
        for i in range(len(T) - diff):
            if len(T[i]) == 1:
                continue
            t1, t2 = (T[i + diff], T[i]) if tx >= ty else (T[i], T[i + diff])
            A, B = t1[ix], t2[iy]
            if A == B:
                continue
            if A == tier_x[ix] and B == tier_y[iy]:
                continue
            if B == tier_x[ix] and A == tier_y[iy]:
                continue
            out.append((self.index[A], self.index[B]))
        return out


def svar_fas(cache, p, kn, names, depth=-1):
    """SvarFas.search (Tetrad 7.6.5). Returns a Skeleton (the separating sets
    of the tested pairs only; SvarFci does not use them for orientation)."""
    st = _SvarTiers(kn, names)
    key = lambda v: node_compare_key(names[v])
    adj = {v: set() for v in range(p)}
    sk = Skeleton(p)
    sk.adj = {v: [] for v in range(p)}
    d_max = 1000 if depth == -1 else depth
    # depth 0
    sim = set()
    cache.prefetch([(i, j, ()) for i in range(p) for j in range(i + 1, p)])
    for i in range(p):
        for j in range(i + 1, p):
            if (i, j) in sim or (j, i) in sim:
                continue
            pairs = st.similar_pairs(i, j)
            if cache.independent(i, j, ()):
                sk.sepset[(i, j)] = ()
                for a, b in pairs:
                    sk.sepset[(min(a, b), max(a, b))] = ()
            elif not (kn.is_forbidden(i, j) and kn.is_forbidden(j, i)):
                adj[i].add(j)
                adj[j].add(i)
                for a, b in pairs:
                    adj[a].add(b)
                    adj[b].add(a)
            sim.update(pairs)

    def free_degree():
        return max((len(v) - 1 for v in adj.values() if v), default=0)

    more = free_degree() > 0
    d = 1
    while more and d <= d_max:
        queries = []
        for x in range(p):
            for y in adj[x]:
                ppx = [v for v in adj[x] if v != y and kn.possible_parent(v, x)]
                if len(ppx) >= d:
                    queries.extend((x, y, Z) for Z in combinations(sorted(ppx, key=key), d))
        cache.prefetch(queries)
        for x in range(p):
            for y in sorted(adj[x], key=key):                 # the TreeSet at the start of x
                ppx = [v for v in sorted(adj[x], key=key) if v != y and kn.possible_parent(v, x)]
                if len(ppx) < d:
                    continue
                for Z in combinations(ppx, d):
                    if cache.independent(x, y, Z):
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sk.sepset[(min(x, y), max(x, y))] = tuple(Z)
                        for a, b in st.similar_pairs(x, y):
                            adj[a].discard(b)
                            adj[b].discard(a)
                        break
        more = free_degree() > d
        d += 1
    for v in range(p):
        sk.adj[v] = sorted(adj[v])
    return sk


class _TetradSvarFci(_TetradFci):
    """SvarFci.search: SvarFas, R0 with the similar-pair endpoint strategy,
    the possible-m-sep removal (SepsetsPossibleMsep, with similar pairs),
    R0 again, and the final orientation (completeRuleSetUsed = false)."""

    def __init__(self, cache, p, depth=-1, knowledge=None):
        super().__init__(cache, p, depth, knowledge, orient_knowledge=True)
        self.st = _SvarTiers(self.kn, self.names)
        self.complete = False

    def set_ep(self, g, a, b, ep):
        g.set_endpoint(a, b, ep)
        for x1, y1 in self.st.similar_pairs(a, b, only_if_x_later=True):
            if g.is_adjacent(x1, y1) and g.get_endpoint(x1, y1) == C:
                g.set_endpoint(x1, y1, ep)

    def sepset_possible_msep(self, g, x, y):
        """SepsetsPossibleMsep.getSepset: x's side, then y's side."""
        for a, b in ((x, y), (y, x)):
            S = self._cond_set(g, a, b)
            if S is not None:
                return S
        return None

    def _cond_set(self, g, a, b):
        pm = self.possible_msep(g, a, b)
        d = len(pm) if self.depth == -1 else min(self.depth, len(pm))
        top = max(self.kn.tier(a), self.kn.tier(b))
        cands = [tuple(pm[i] for i in ch) for ch in _sublists(len(pm), d) if len(ch) >= 1]
        cands = [S for S in cands if all(self.kn.tier(j) <= top for j in S)]
        self.cache.prefetch([(a, b, S) for S in cands])
        for S in cands:
            if self.indep(a, b, S):
                return set(S)
        return None

    def remove_edge_nodes(self, g, a, b):
        e = g.get_edge(a, b)
        if e is not None:
            g.remove_edge(e)

    def search_svar(self, skeleton):
        g = TGraph(self.p, self.names)
        for i in range(self.p):
            for j in range(i + 1, self.p):
                if skeleton.adjacent(i, j):
                    g.add_undirected_edge(i, j)
        g.reorient_all_with(C)
        self.rule_r0(g)
        for e in list(g.get_edges()):
            x, y = e.n1, e.n2
            S = self.sepset_possible_msep(g, x, y)
            if S is not None:
                self.remove_edge_nodes(g, x, y)
                for a, b in self.st.similar_pairs(x, y):
                    self.remove_edge_nodes(g, a, b)
        g.reorient_all_with(C)
        self.rule_r0(g)
        self.final_orientation(g)
        return g


def svarfci(ci, alpha=0.05, depth=-1, tetrad_compat=False, knowledge=None):
    """SVAR-FCI on ETIA's lagged table with ``knowledge`` (Knowledge.time_lagged)."""
    if knowledge is None or knowledge.is_empty():
        raise ValueError("svarfci needs the tier knowledge of the lagged table")
    if not tetrad_compat:                     # default: FCI with the tiers (no similar-pair copies)
        from .fci import fci
        return fci(ci, alpha=alpha, depth=depth, knowledge=knowledge)
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    t = _TetradSvarFci(cache, p, depth, knowledge)
    sk = svar_fas(cache, p, t.kn, t.names, depth)
    g = t.search_svar(sk)
    return SearchResult(g.marks(), sk, cache)


# ------------------------------------------------------------- SvarFges
class _ArrowSet:
    """ConcurrentSkipListSet<Arrow> (largest bump first, then smallest index)
    with removal (clearArrow)."""

    def __init__(self, jitter=None):
        import heapq
        self._h, self._hq, self.n = [], heapq, 0
        self.jitter = jitter                 # a function arrow -> relative change of the sort key

    def add(self, arrow):
        key = arrow.bump if self.jitter is None else arrow.bump * (1.0 + self.jitter(arrow))
        self._hq.heappush(self._h, (-key, arrow.index, arrow))
        self.n += 1

    def remove(self, arrow):
        if not arrow.dead:
            arrow.dead = True
            self.n -= 1

    def pop_first(self):
        while self._h:
            _, _, a = self._hq.heappop(self._h)
            if not a.dead:
                a.dead = True
                self.n -= 1
                return a
        return None

    def __bool__(self):
        return self.n > 0


class _SArrow:
    __slots__ = ("bump", "a", "b", "h_or_t", "na_yx", "index", "dead", "comp")

    def __init__(self, bump, a, b, h_or_t, na_yx, index):
        self.bump, self.a, self.b, self.h_or_t, self.na_yx, self.index = bump, a, b, h_or_t, na_yx, index
        self.dead = False


class _SvarFges:
    """Tetrad 7.6.5 SvarFges (as SvarGfci runs it: faithfulness assumed, the
    score's degree limit, the knowledge of the lagged table). Each edge that
    the search adds, removes or orients is copied to its similar pairs."""

    def __init__(self, score, p, kn, names, tie_seed=None):
        """``tie_seed``: None for Tetrad's order of equal score changes (the
        creation index). An integer changes the order of score changes that
        differ by less than 1e-11 (relative): Tetrad orders these by its own
        rounding, which ETIA-JAX cannot copy (the tests try some seeds)."""
        from .fges import _Graph
        import numpy as _np
        self._tie_rng = None if tie_seed is None else _np.random.default_rng(tie_seed)
        self._tie_u = {}
        self.score, self.p, self.kn, self.names = score, p, kn, names
        self.G = _Graph(p, True, names)
        self.effect = _Graph(p, True, names)
        self.st = _SvarTiers(kn, names)
        md = score.max_degree()
        self.max_indegree = 1000 if md == -1 else md
        self.arrow_index = 0
        self.removed = set()
        self.mode = "heuristic"

    # ---------------------------------------------------- bookkeeping
    def _jitter(self, ar):
        # one value per computation: the two arrows of an empty-graph pair
        # share their bump exactly (Tetrad: index order), so they share it
        k = ar.comp
        if k not in self._tie_u:
            self._tie_u[k] = 1e-11 * (2.0 * self._tie_rng.random() - 1.0)
        return self._tie_u[k]

    def add_arrow(self, a, b, na_yx, h_or_t, bump, comp=None):
        ar = _SArrow(bump, a, b, h_or_t, na_yx, self.arrow_index)
        ar.comp = self.arrow_index if comp is None else comp
        self.arrow_index += 1
        self.sorted.add(ar)
        self.lookup.setdefault((a, b), []).append(ar)

    def clear_arrow(self, x, y):
        for ar in self.lookup.pop((x, y), []):
            self.sorted.remove(ar)

    def reset(self):
        self.sorted = _ArrowSet(None if self._tie_rng is None else self._jitter)
        self.lookup, self.neighbors = {}, {}

    # ---------------------------------------------------- graph helpers
    def neighbors_of(self, y):
        G = self.G
        return frozenset(e.distal(y) for e in G.g.edge_lists[y] if _undirected(e))

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
        s = G.new_set()
        for e in G.edges_of(y):
            if not _undirected(e):
                continue
            z = e.distal(y)
            if G.is_adj(z, x):
                continue
            s.add(z)
        return s

    def is_clique(self, nodes):
        nodes = list(nodes)
        return all(self.G.is_adj(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)))

    def indegree(self, b):
        return len(self.G.g.get_parents(b))

    def score_change(self, y, parents, x):
        """scoreGraphChange(y, parents, x): NaN if x is in parents."""
        if x in parents:
            return float("nan")
        return self.score.local_score_diff(x, y, tuple(parents))

    def add_similar(self, x, y):
        for a, b in self.st.similar_pairs(x, y):
            self.G.add_directed(a, b)

    def remove_similar(self, x, y):
        for a, b in self.st.similar_pairs(x, y):
            e = self.G.edge(a, b)
            if e is not None:
                self.G.g.remove_edge(e)
            self.removed.add(frozenset((a, b)))

    def semidirected_path(self, frm, to, cond):
        """existsUnblockedSemiDirectedPath(frm, to, cond, -1)."""
        G = self.G
        Q, V = [frm], {frm}
        while Q:
            t = Q.pop(0)
            if t == to:
                return True
            for u in G.adj(t):
                e = G.edge(t, u)
                if t == e.n1 and e.e1 == "TAIL":
                    c = e.n2
                elif t == e.n2 and e.e2 == "TAIL":
                    c = e.n1
                else:
                    continue
                if c in cond:
                    continue
                if c == to:
                    return True
                if c not in V:
                    V.add(c)
                    Q.append(c)
        return False

    def meek(self):
        from .fges import meek_orient_implied
        return meek_orient_implied(self.G, prevent_cycles=False, knowledge=self.kn)

    # ---------------------------------------------------- forward
    def init_from_empty(self):
        self.reset()
        sc = self.score
        for i in range(self.p):
            sc.local_scores(i, [()] + [(j,) for j in range(i + 1, self.p)])
        for i in range(self.p):
            y = i
            self.neighbors[y] = frozenset()
            for j in range(i + 1, self.p):
                x = j
                bump = sc.local_score_diff(x, y)
                if bump > 0:
                    self.effect.add_undirected(x, y)
                    e = frozenset()
                    comp = ("init", x, y)
                    self.add_arrow(x, y, e, e, bump, comp)
                    self.add_arrow(y, x, e, e, bump, comp)

    def init_two_step(self):
        self.reset()
        G = self.G
        for y in range(self.p):
            g = G.new_set()
            for n in G.adj(y):
                for m in G.adj(n):
                    if G.is_adj(y, m):
                        continue
                    if G.g.is_def_collider(m, n, y):
                        continue
                    g.add(m)
            for x in list(g):
                if frozenset((x, y)) in self.removed:
                    continue
                self.calc_forward(x, y)

    def calc_forward(self, a, b):
        G = self.G
        if self.mode == "heuristic" and not self.effect.is_adj(a, b):
            return
        self.neighbors[b] = self.neighbors_of(b)
        if self.kn.is_forbidden(a, b):
            return
        na = self.na_yx(a, b)
        if not self.is_clique(na):
            return
        tn = list(self.t_neighbors(a, b))
        mx = min(len(tn), self.max_indegree - self.indegree(b))
        parents = set(G.g.get_parents(b))
        prev = [frozenset()]
        for i in range(mx + 1):
            new = []
            stop = False
            for ch in combinations(range(len(tn)), i):
                T = G.new_set(tn[k] for k in ch)
                union = frozenset(na) | frozenset(T)
                if not any(c <= union for c in prev):
                    stop = True
                    break
                if not self.is_clique(union):
                    continue
                new.append(union)
                bump = self.score_change(b, set(na) | set(T) | parents, a)
                if bump > 0:
                    self.add_arrow(a, b, frozenset(na), T, bump)
            if stop:
                break
            prev = new

    def reevaluate_forward(self, nodes):
        G = self.G
        for x in list(nodes):
            if self.mode == "heuristic":
                adj = self.effect.adj(x)
            elif self.mode == "cover":
                g = G.new_set()
                for n in G.adj(x):
                    for m in G.adj(n):
                        if G.is_adj(x, m):
                            continue
                        if G.g.is_def_collider(m, n, x):
                            continue
                        g.add(m)
                adj = list(g)
            else:
                adj = [v for v in range(self.p) if v != x]
            for w in adj:
                if w == x:
                    continue
                if not G.is_adj(w, x):
                    self.clear_arrow(w, x)
                    self.calc_forward(w, x)

    def insert(self, x, y, T):
        G = self.G
        if G.is_adj(x, y):
            return False
        G.add_directed(x, y)
        self.add_similar(x, y)
        for t in T:
            G.remove(t, y)
            self.remove_similar(t, y)
            G.add_directed(t, y)
            self.add_similar(t, y)
        return True

    def valid_insert(self, x, y, T, na):
        if self.kn.is_forbidden(x, y) or any(self.kn.is_forbidden(t, y) for t in T):
            return False
        union = set(T) | set(na)
        return self.is_clique(union) and not self.semidirected_path(y, x, union)

    def changed_nodes(self, visited):
        G = self.G
        out = G.new_set()
        for v in visited:
            if self.neighbors_of(v) != self.neighbors.get(v):
                out.add(v)
        return out

    def fes(self):
        while self.sorted:
            ar = self.sorted.pop_first()
            x, y = ar.a, ar.b
            if self.G.is_adj(x, y):
                continue
            if ar.na_yx != frozenset(self.na_yx(x, y)):
                continue
            if not set(self.t_neighbors(x, y)) >= set(ar.h_or_t):
                continue
            if not self.valid_insert(x, y, ar.h_or_t, self.na_yx(x, y)):
                continue
            if not self.insert(x, y, ar.h_or_t):
                continue
            visited = self.meek()
            todo = self.changed_nodes(visited)
            todo.add(x)
            todo.add(y)
            self.reevaluate_forward(todo)

    # ---------------------------------------------------- backward
    def calc_backward(self, a, b):
        G = self.G
        na = self.na_yx(a, b)
        lst = list(na)
        parents = set(G.g.get_parents(b))
        for i in range(len(lst) + 1):
            for ch in combinations(range(len(lst)), i):
                diff = {lst[k] for k in ch}
                h = G.set_of(lst)
                for v in diff:
                    h.discard(v)
                if any(self.kn.is_forbidden(v, b) for v in h):     # invalidSetByKnowledge(b, h)
                    continue
                s = (diff | parents) - {a}
                bump = -self.score_change(b, s, a)
                if bump > 0:
                    self.add_arrow(a, b, frozenset(na), h, bump)

    def init_backward(self):
        G = self.G
        for e in G.edges():
            x, y = e.n1, e.n2
            self.clear_arrow(x, y)
            self.clear_arrow(y, x)
            if _points_towards(e, y):
                self.calc_backward(x, y)
            elif _points_towards(e, x):
                self.calc_backward(y, x)
            else:
                self.calc_backward(x, y)
                self.calc_backward(y, x)
            self.neighbors[x] = self.neighbors_of(x)
            self.neighbors[y] = self.neighbors_of(y)

    def reevaluate_backward(self, todo):
        G = self.G
        for r in list(todo):
            self.neighbors[r] = self.neighbors_of(r)
            for w in G.adj(r):
                e = G.edge(w, r)
                if e is None:
                    continue
                if _points_towards(e, r):
                    self.clear_arrow(w, r)
                    self.clear_arrow(r, w)
                    self.calc_backward(w, r)
                elif _undirected(G.edge(w, r)):
                    self.clear_arrow(w, r)
                    self.clear_arrow(r, w)
                    self.calc_backward(w, r)
                    self.calc_backward(r, w)

    def delete(self, x, y, H, na):
        G = self.G
        old = G.edge(x, y)
        if old is not None:
            G.g.remove_edge(old)
        self.removed.add(frozenset((x, y)))
        self.remove_similar(x, y)
        for h in H:
            if G.is_parent_of(h, y) or G.is_parent_of(h, x):
                continue
            oyh = G.edge(y, h)
            if oyh is not None:
                G.g.remove_edge(oyh)
            G.add_directed(y, h)
            self.remove_similar(y, h)
            self.add_similar(y, h)
            oxh = G.edge(x, h)
            if _undirected(oxh):
                G.g.remove_edge(oxh)
                G.add_directed(x, h)
                self.remove_similar(x, h)
                self.add_similar(x, h)
        return True

    def bes(self):
        self.reset()
        self.init_backward()
        G = self.G
        while self.sorted:
            ar = self.sorted.pop_first()
            x, y = ar.a, ar.b
            if ar.na_yx != frozenset(self.na_yx(x, y)):
                continue
            if not G.is_adj(x, y):
                continue
            if _points_towards(G.edge(x, y), x):
                continue
            if any(self.kn.is_forbidden(x, h) or self.kn.is_forbidden(y, h) for h in ar.h_or_t):
                continue
            if not self.is_clique(set(ar.na_yx) - set(ar.h_or_t)):
                continue
            self.delete(x, y, ar.h_or_t, ar.na_yx)
            self.clear_arrow(x, y)
            visited = self.meek()
            todo = self.changed_nodes(visited)
            todo.add(x)
            todo.add(y)
            for v in G.common_adjacents(x, y):
                todo.add(v)
            self.reevaluate_backward(todo)
        self.meek()

    def search(self):
        self.init_from_empty()
        self.mode = "heuristic"
        self.fes()
        self.bes()
        self.mode = "cover"
        self.init_two_step()
        self.fes()
        self.bes()
        return self.G


class _TetradSvarGfci(_TetradSvarFci):
    """SvarGfci.search: SvarFges, the removal of extra edges with SepsetsMinP
    on the FGES graph (and their similar pairs), modifiedR0, and the final
    orientation with the similar-pair endpoint strategy."""

    def minp_sepset(self, g, x, y):
        return _TetradGfci.minp_sepset(self, g, x, y)

    def search_svargfci(self, score):
        fg = _SvarFges(score, self.p, self.kn, self.names, getattr(self, "tie_seed", None))
        g = fg.search().g
        fges_g = TGraph(self.p, self.names)            # new EdgeListGraph(this.graph)
        for e in g.get_edges():
            fges_g.add_edge(e)
        self.cpdag_marks = fges_g.marks()
        for b in range(self.p):
            adj = fges_g.get_adjacent_nodes(b)
            if len(adj) < 2:
                continue
            for i, j in combinations(range(len(adj)), 2):
                a, c = adj[i], adj[j]
                if g.is_adjacent(a, c) and fges_g.is_adjacent(a, c):
                    if self.minp_sepset(fges_g, a, c) is not None:
                        self.remove_edge_nodes(g, a, c)
                        for x1, y1 in self.st.similar_pairs(a, c):
                            self.remove_edge_nodes(g, x1, y1)
        self.modified_r0(g, fges_g)
        self.final_orientation(g)
        return g

    def _orient_similar_arrow(self, g, x, y):
        for x1, y1 in self.st.similar_pairs(x, y, only_if_x_later=True):
            if g.is_adjacent(x1, y1) and g.get_endpoint(x1, y1) == C:
                g.set_endpoint(x1, y1, "ARROW")

    def modified_r0(self, g, fges_g):
        g.reorient_all_with(C)
        for a, b in self.kn.forbidden_edges(self.names):      # SvarGfci's own fciOrientbk
            if g.get_edge(a, b) is None:
                continue
            g.set_endpoint(b, a, "ARROW")
            g.set_endpoint(a, b, C)
        for b in range(self.p):
            adj = g.get_adjacent_nodes(b)
            if len(adj) < 2:
                continue
            for i, j in combinations(range(len(adj)), 2):
                a, c = adj[i], adj[j]
                if fges_g.is_def_collider(a, b, c):
                    g.set_endpoint(a, b, "ARROW")
                    g.set_endpoint(c, b, "ARROW")
                    self._orient_similar_arrow(g, a, b)
                    self._orient_similar_arrow(g, c, b)
                elif fges_g.is_adjacent(a, c) and not g.is_adjacent(a, c):
                    S = self.minp_sepset(fges_g, a, c)
                    if S is not None and b not in S:
                        g.set_endpoint(a, b, "ARROW")
                        g.set_endpoint(c, b, "ARROW")
                        self._orient_similar_arrow(g, a, b)
                        self._orient_similar_arrow(g, c, b)


def svargfci(ci, score, alpha=0.05, tetrad_compat=False, knowledge=None, tie_seed=None):
    """SVAR-GFCI on ETIA's lagged table with ``knowledge``."""
    if knowledge is None or knowledge.is_empty():
        raise ValueError("svargfci needs the tier knowledge of the lagged table")
    if not tetrad_compat:                     # default: GFCI with the tiers (no similar-pair copies)
        from .gfci import gfci
        return gfci(ci, score, alpha=alpha, knowledge=knowledge)
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    t = _TetradSvarGfci(cache, p, -1, knowledge)
    t.tie_seed = tie_seed
    g = t.search_svargfci(score)
    return GfciResult(g.marks(), t.cpdag_marks, cache, score)
