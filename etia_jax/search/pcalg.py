"""PC and CPC (Spirtes, Glymour and Scheines 2000; Ramsey, Zhang and Spirtes 2006).

The semantics follow Tetrad 7.6.5 as ETIA calls it (``Pc(test)`` and
``Cpc(test)`` with ``setStable(true)`` and ``setGuaranteeCpdag(true)``):

* FAS: the pairs (i, j), i < j, in node order. At depth d, x and y are tested
  with each subset of size d of adj(x) - {y}, in lexicographic order, then
  of adj(y) - {x}. The first independence removes the edge and records the
  separating set. In Tetrad 7.6.5 the "stable" FAS copies the adjacencies at
  the start of a depth but then changes that copy, so later pairs see the
  removals of earlier pairs (the classic, order-dependent PC). With
  ``tetrad_compat=True`` (or ``stable=False``) the FAS does the same. The
  default is the PC-stable of Colombo and Maathuis (2014), which ETIA asks
  for: the adjacencies of a depth are fixed during the depth.
* Colliders (PC): for each node b in node order and each pair (a, c) of its
  adjacents, a and c not adjacent and b not in sepset(a, c): orient
  a -> b <- c, unless both arrowheads exist (PRIORITIZE_EXISTING). A
  collider is skipped when b is already an ancestor of a or c (Tetrad
  7.6.5 tests only c, by a typo: ``tetrad_compat=True``).
* Colliders (CPC): all the separating sets of (a, c) among the subsets of
  adj(a) and of adj(c). Collider if no set has b, non-collider if all have
  b, else ambiguous. Tetrad 7.6.5 uses a side only if it has at least 2
  adjacents (``tetrad_compat=True``): then a triple a - b - c where a and c
  have only b as adjacent is always ambiguous, also a true collider.
* Meek (Tetrad MeekRules, meekPreventCycles = true): first, p -> y becomes
  undirected if p is adjacent to all other parents of y. Then R1, R2, R3 on
  each undirected edge until nothing changes. An orientation that would make
  a directed cycle is made in the other direction.

The CI queries of a depth (FAS) or of all unshielded triples (CPC) go to the
test in one batched call when the test has ``test_batch``.
"""
from itertools import combinations

import numpy as np

from ..graphs.encoding import NONE, ARROW, TAIL
from .cache import CICache
from .knowledge import NO_KNOWLEDGE


# ---------------------------------------------------------------- the skeleton
class Skeleton:
    """Result of FAS: adjacency lists (ordered) and separating sets."""

    def __init__(self, p):
        self.p = p
        self.adj = {i: [j for j in range(p) if j != i] for i in range(p)}
        self.sepset = {}

    def adjacent(self, a, b):
        return b in self.adj[a]

    def remove(self, a, b, Z):
        self.adj[a].remove(b)
        self.adj[b].remove(a)
        self.sepset[(min(a, b), max(a, b))] = tuple(Z)

    def get_sepset(self, a, b):
        return self.sepset.get((min(a, b), max(a, b)))


def _free_degree(adj):
    return max((len(v) - 1 for v in adj.values() if v), default=0)


def fas(cache, p, depth=1000, mode="stable", knowledge=None):
    """Fast adjacency search. ``mode``: "stable" or "sequential" (see the module doc).
    With ``knowledge``, the conditioning sets of side x use only the possible
    parents of x (Tetrad's Fas: no z with z -> x forbidden)."""
    kn = knowledge or NO_KNOWLEDGE
    pp = (lambda a, b, adj: [v for v in adj if v != b and kn.possible_parent(v, a)])
    sk = Skeleton(p)
    edges = [(i, j) for i in range(p) for j in range(i + 1, p)]
    # depth 0 first, on all pairs (Tetrad tests them before the depth loop)
    cache.prefetch([(i, j, ()) for i, j in edges])
    for i, j in edges:
        if cache.independent(i, j, ()):
            sk.remove(i, j, ())
    for d in range(depth + 1):
        # all queries that the depth may ask, from the adjacencies at its start
        queries = []
        for x, y in edges:
            if not sk.adjacent(x, y):
                continue
            for a, b in ((x, y), (y, x)):
                ppa = pp(a, b, sk.adj[a])
                if len(ppa) >= d:
                    queries.extend((a, b, Z) for Z in combinations(ppa, d))
        cache.prefetch(queries)
        if mode == "stable":
            snapshot = {k: list(v) for k, v in sk.adj.items()}
            removals = []
            for x, y in edges:
                if not sk.adjacent(x, y):
                    continue
                found = None
                for a, b in ((x, y), (y, x)):
                    ppa = pp(a, b, snapshot[a])
                    if len(ppa) < d:
                        continue
                    for Z in combinations(ppa, d):
                        if cache.independent(a, b, Z):
                            found = Z
                            break
                    if found is not None:
                        break
                if found is not None:
                    removals.append((x, y, found))
            for x, y, Z in removals:
                sk.remove(x, y, Z)
        else:  # sequential: Tetrad's FAS
            for x, y in edges:
                if not _check_side(cache, sk, d, x, y, pp):
                    _check_side(cache, sk, d, y, x, pp)
        if _free_degree(sk.adj) <= d:
            break
    return sk


def _check_side(cache, sk, d, x, y, pp):
    if not sk.adjacent(x, y):
        return False
    ppx = pp(x, y, sk.adj[x])
    if len(ppx) < d:
        return False
    for Z in combinations(ppx, d):
        if cache.independent(x, y, Z):
            sk.remove(x, y, Z)
            return True
    return False


# ----------------------------------------------------------- the mark matrix
def _skeleton_marks(sk):
    G = np.zeros((sk.p, sk.p), dtype=np.int8)
    for a in range(sk.p):
        for b in sk.adj[a]:
            G[a, b] = TAIL
    return G


def _adjacent(G, a, b):
    return G[a, b] != NONE


def _adj_list(G, a):
    return [int(b) for b in np.flatnonzero(G[a] != NONE)]


# Tetrad's graph returns adjacent nodes from a java.util.HashSet<Node>, so the
# order is the bucket order of the Java hash of the node names ("X1", "X2",
# ... in ETIA). In tetrad_compat mode the orientation steps use this order.
def java_string_hash(s):
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h


def _java_bucket(h, cap):
    return (h ^ (h >> 16)) & (cap - 1)


def java_hashset_order(nodes, names):
    """Iteration order of a HashSet<Node> that holds ``nodes`` (bucket index;
    ties broken by node index, an approximation of the insertion order)."""
    cap = 16
    while len(nodes) > 0.75 * cap:
        cap *= 2
    return sorted(nodes, key=lambda v: (_java_bucket(java_string_hash(names[v]), cap), v))


class _Order:
    """Order of adjacents for the orientation steps: node order, or the
    Java HashSet order of Tetrad (names X1, X2, ...). Ties in a bucket follow
    the iteration order of the node's own edge set (a HashSet<Edge>: bucket
    of h(a) + h(b), then insertion order), when an edge tracker is set."""

    def __init__(self, p, tetrad_compat, names=None):
        self.names = (list(names) if names else [f"X{i + 1}" for i in range(p)]) if tetrad_compat else None
        self.eo = None

    def adj(self, G, a):
        lst = _adj_list(G, a)
        if not self.names:
            return lst
        if self.eo is None:
            return java_hashset_order(lst, self.names)
        return self.eo.adjacency_order(a, lst)


def _table_size_for(c):
    n = 1
    while n < c:
        n *= 2
    return n


class TetradEdgeOrder:
    """The iteration order of Tetrad's EdgeListGraph.getEdges(): a copy
    ``new HashSet<>(edgesSet)``, so (bucket of h(node1) + h(node2) for the
    copy's capacity, insertion position in edgesSet). Removing and adding an
    edge again moves it to the end. Also keeps node1 of each edge (Meek tries
    R1(node1, node2) before R1(node2, node1))."""

    def __init__(self, names):
        self.h = [java_string_hash(n) for n in names]
        self.ins, self.node1, self.t = {}, {}, 0

    def add(self, a, b):
        k = (min(a, b), max(a, b))
        self.t += 1
        self.ins[k], self.node1[k] = self.t, a

    def remove(self, a, b):
        k = (min(a, b), max(a, b))
        self.ins.pop(k, None)
        self.node1.pop(k, None)

    def adjacency_order(self, a, nbrs):
        """Order of Tetrad's getAdjacentNodes(a) (a HashSet<Node> filled in the
        iteration order of the node's edge set)."""
        cap_e = _table_size_for(max(int(len(nbrs) / 0.75) + 1, 16))
        def edge_key(b):
            k = (min(a, b), max(a, b))
            return (_java_bucket((self.h[a] + self.h[b]) & 0xFFFFFFFF, cap_e), self.ins.get(k, 0))
        filled = sorted(nbrs, key=edge_key)              # insertion order into the HashSet<Node>
        cap = 16
        while len(nbrs) > 0.75 * cap:
            cap *= 2
        pos = {b: i for i, b in enumerate(filled)}
        return sorted(nbrs, key=lambda b: (_java_bucket(self.h[b], cap), pos[b]))

    def snapshot(self):
        n = len(self.ins)
        cap = _table_size_for(max(int(n / 0.75) + 1, 16))
        def key(k):
            return (_java_bucket((self.h[k[0]] + self.h[k[1]]) & 0xFFFFFFFF, cap), self.ins[k])
        return [(self.node1[k], k[1] if self.node1[k] == k[0] else k[0]) for k in sorted(self.ins, key=key)]

    def rebuild(self):
        """GraphUtils.replaceNodes: a new graph with the edges in getEdges() order."""
        order = self.snapshot()
        self.ins, self.node1, self.t = {}, {}, 0
        for a, b in order:
            self.add(a, b)


def collider_allowed(G, x, y, z, tetrad_compat, knowledge=None):
    """Tetrad's colliderAllowed: no collider x -> y <- z when the knowledge
    forbids x -> y or z -> y, or when y is already an ancestor of z (Tetrad
    7.6.5 tests z twice: compat), or of x or z."""
    if knowledge and (knowledge.is_forbidden(x, y) or knowledge.is_forbidden(z, y)):
        return False
    if _directed_path_exists(G, y, z):
        return False
    return tetrad_compat or not _directed_path_exists(G, y, x)


def orient_collider(G, x, y, z, eo=None):
    """PRIORITIZE_EXISTING: x -> y <- z unless both arrowheads already exist."""
    if not (G[x, y] == ARROW and G[z, y] == ARROW):
        G[x, y], G[y, x] = ARROW, TAIL
        G[z, y], G[y, z] = ARROW, TAIL
        if eo is not None:
            eo.remove(x, y); eo.remove(z, y)
            eo.add(x, y); eo.add(z, y)


# --------------------------------------------------------------- Meek rules
def _undirected(G, a, b):
    return G[a, b] == TAIL and G[b, a] == TAIL


def _directed(G, a, b):
    return G[a, b] == ARROW and G[b, a] == TAIL


def _parents(G, b):
    return [int(a) for a in np.flatnonzero((G[:, b] == ARROW) & (G[b, :] == TAIL))]


def _directed_path_exists(G, s, t):
    """A directed path s -> ... -> t."""
    seen, stack = {s}, [s]
    while stack:
        v = stack.pop()
        for w in np.flatnonzero((G[v] == ARROW) & (G[:, v] == TAIL)):
            w = int(w)
            if w == t:
                return True
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return False


_EO = None   # the edge-order tracker of the running Meek (compat mode)
_KN = None   # the knowledge of the running Meek


def _direct(G, a, c, prevent_cycles):
    if _KN and _KN.is_forbidden(a, c):          # isArrowheadAllowed(a, c)
        return False
    if not _undirected(G, a, c):
        return False
    G[a, c] = G[c, a] = NONE
    if prevent_cycles and _directed_path_exists(G, c, a):
        G[c, a], G[a, c] = ARROW, TAIL
        tail = c
    else:
        G[a, c], G[c, a] = ARROW, TAIL
        tail = a
    if _EO is not None:
        _EO.remove(a, c)
        _EO.add(tail, c if tail == a else a)
    return True


def _common(G, a, b):
    return [c for c in _adj_list(G, a) if _adjacent(G, b, c)]


def _r1(G, b, c, pc):
    for a in _parents(G, b):
        if _adjacent(G, c, a):
            continue
        if _direct(G, b, c, pc):
            return True
    return False


def _r2(G, a, c, pc):
    oriented = False
    for b in _common(G, a, c):
        if _directed(G, a, b) and _directed(G, b, c):
            if _direct(G, a, c, pc):
                oriented = True
        if _directed(G, c, b) and _directed(G, b, a):
            if _direct(G, c, a, pc):
                oriented = True
    return oriented


def _r3(G, d, a, pc):
    adj = _common(G, a, d)
    if len(adj) < 2:
        return False
    oriented = False
    for i in range(len(adj)):
        for j in range(i + 1, len(adj)):
            b, c = adj[i], adj[j]
            if _adjacent(G, b, c):
                continue
            if (_undirected(G, d, a) and _undirected(G, d, b) and _undirected(G, d, c)
                    and _directed(G, b, a) and _directed(G, c, a)):
                if _direct(G, d, a, pc):
                    oriented = True
    return oriented


def meek_tetrad(G, prevent_cycles=True, revert=True, edge_order=None, knowledge=None):
    """Tetrad 7.6.5 MeekRules.orientImplied (without knowledge: no R4).
    With ``edge_order`` (a TetradEdgeOrder), each pass visits a snapshot of
    the edges in Tetrad's getEdges() order; else in index order."""
    global _EO, _KN
    p = len(G)
    _EO, _KN = edge_order, (knowledge or None)
    try:
        if revert:
            for y in range(p):
                parents = _parents(G, y)
                if edge_order is not None:
                    parents = edge_order.adjacency_order(y, parents)
                for q in parents:
                    if _KN and _KN.is_forbidden(y, q):     # the edge may not be y -> q
                        continue
                    if all(q == r or _adjacent(G, q, r) for r in parents):
                        G[q, y] = G[y, q] = TAIL
                        if edge_order is not None:
                            edge_order.remove(q, y); edge_order.add(q, y)
        oriented = True
        while oriented:
            oriented = False
            if edge_order is not None:
                snap = [(x, y) for x, y in edge_order.snapshot() if _undirected(G, x, y)]
            else:
                snap = [(x, y) for x in range(p) for y in range(x + 1, p) if _undirected(G, x, y)]
            for x, y in snap:
                if (_r1(G, x, y, prevent_cycles) or _r1(G, y, x, prevent_cycles)
                        or _r2(G, x, y, prevent_cycles) or _r2(G, y, x, prevent_cycles)
                        or _r3(G, x, y, prevent_cycles) or _r3(G, y, x, prevent_cycles)):
                    oriented = True
    finally:
        _EO, _KN = None, None
    return G


# ------------------------------------------------------------------ PC / CPC
class SearchResult:
    def __init__(self, G, skeleton, cache, **extra):
        self.graph = G
        self.skeleton = skeleton
        self.n_tests = cache.n_tests
        self.n_computed = cache.n_computed
        for k, v in extra.items():
            setattr(self, k, v)


def _edge_order(G, order):
    """A TetradEdgeOrder with the FAS edges added in (i, j), i < j, order."""
    if order.names is None:
        return None
    eo = TetradEdgeOrder(order.names)
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            if G[i, j] != NONE:
                eo.add(i, j)
    return eo


def _fas_mode(stable, tetrad_compat):
    return "stable" if (stable and not tetrad_compat) else "sequential"


def pc_orient_bk(G, knowledge, eo=None, names=None):
    """GraphSearchUtils.pcOrientbk: each forbidden edge a -> b that is in the
    graph becomes b -> a (in Tetrad's order of the forbidden edges)."""
    if not knowledge:
        return
    for a, b in knowledge.forbidden_edges(names):
        if G[a, b] == NONE:
            continue
        G[b, a], G[a, b] = ARROW, TAIL
        if eo is not None:
            eo.remove(a, b)
            eo.add(b, a)


def _names(p, knowledge):
    return knowledge.names if knowledge is not None and knowledge.names else None


def pc(ci, alpha=0.05, depth=1000, stable=True, tetrad_compat=False, guarantee_cpdag=True, knowledge=None):
    """PC. Returns a SearchResult with ``graph`` (mark matrix; undirected edges
    are tail-tail) and the FAS skeleton with its separating sets."""
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    sk = fas(cache, p, depth, _fas_mode(stable, tetrad_compat), knowledge)
    G = _skeleton_marks(sk)
    names = _names(p, knowledge)
    order = _Order(p, tetrad_compat, names)
    eo = _edge_order(G, order)
    order.eo = eo
    pc_orient_bk(G, knowledge, eo, order.names)
    for b in range(p):
        for a, c in combinations(order.adj(G, b), 2):
            if _adjacent(G, a, c):
                continue
            S = sk.get_sepset(a, c)
            if S is not None and b not in S and collider_allowed(G, a, b, c, tetrad_compat, knowledge):
                orient_collider(G, a, b, c, eo)
    if eo is not None:
        eo.rebuild()
    meek_tetrad(G, prevent_cycles=guarantee_cpdag, edge_order=eo, knowledge=knowledge)
    return SearchResult(G, sk, cache)


def _cpc_sepsets(cache, G, i, k, min_side):
    """All separating sets of (i, k) among the subsets of adj(i) and adj(k).
    A side is used when it has at least ``min_side`` adjacents (Tetrad: 2)."""
    adji, adjk = _adj_list(G, i), _adj_list(G, k)
    out = set()
    for d in range(max(len(adji), len(adjk)) + 1):
        for side in (adji, adjk):
            if len(side) >= min_side and d <= len(side):
                for Z in combinations(side, d):
                    if cache.independent(i, k, Z):
                        out.add(frozenset(Z))
    return out


def _cpc_queries(G, i, k, min_side):
    adji, adjk = _adj_list(G, i), _adj_list(G, k)
    for d in range(max(len(adji), len(adjk)) + 1):
        for side in (adji, adjk):
            if len(side) >= min_side and d <= len(side):
                for Z in combinations(side, d):
                    yield (i, k, Z)


def cpc(ci, alpha=0.05, depth=1000, stable=True, tetrad_compat=False, guarantee_cpdag=True, knowledge=None):
    """CPC. Returns a SearchResult with ``graph``, ``colliders``,
    ``noncolliders`` and ``ambiguous`` (lists of triples (x, y, z))."""
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    sk = fas(cache, p, depth, _fas_mode(stable, tetrad_compat), knowledge)
    min_side = 2 if tetrad_compat else 1
    G = _skeleton_marks(sk)
    order = _Order(p, tetrad_compat, _names(p, knowledge))
    eo = _edge_order(G, order)
    order.eo = eo
    pc_orient_bk(G, knowledge, eo, order.names)
    triples = [(x, y, z) for y in range(p) for x, z in combinations(order.adj(G, y), 2)
               if not _adjacent(G, x, z)]
    cache.prefetch([q for x, y, z in triples for q in _cpc_queries(G, x, z, min_side)])
    col, non, amb = [], [], []
    # Tetrad reads getAdjacentNodes(y) for each y when it comes to y: the
    # colliders of earlier nodes change the edge sets and thus this order.
    triples = (t for y in range(p) for t in
               [(x, y, z) for x, z in combinations(order.adj(G, y), 2) if not _adjacent(G, x, z)])
    for x, y, z in triples:
        S = _cpc_sepsets(cache, G, x, z, min_side)
        if S and all(y not in s for s in S):
            if collider_allowed(G, x, y, z, tetrad_compat, knowledge):
                orient_collider(G, x, y, z, eo)
                col.append((x, y, z))
        elif S and all(y in s for s in S):
            non.append((x, y, z))
        else:
            amb.append((x, y, z))
    if eo is not None:
        eo.rebuild()
    meek_tetrad(G, prevent_cycles=guarantee_cpdag, edge_order=eo, knowledge=knowledge)
    return SearchResult(G, sk, cache, colliders=col, noncolliders=non, ambiguous=amb)
