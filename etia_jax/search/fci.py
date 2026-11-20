"""FCI (Spirtes, Meek and Richardson 1999; Zhang 2008).

Default (``tetrad_compat=False``): the textbook algorithm.
  1. FAS (PC-stable) gives the skeleton and the separating sets.
  2. R0: a *-* b *-* c unshielded, b not in sepset(a, c): a *-> b <-* c.
  3. Possible-D-SEP: an edge x - y goes when x _||_ y | S for a subset S of
     Possible-D-SEP(x, y) or of Possible-D-SEP(y, x) (Spirtes et al. 2000).
  4. All edges back to o-o, R0 again, then Zhang's rules R1-R4, R8-R10
     (no selection bias). R4 uses sepset(d, c) of the path end points.
  With the d-separation oracle the result is the PAG of the true MAG.

``tetrad_compat=True``: Tetrad 7.6.5 ``Fci(test)`` as ETIA calls it (see
``_tetrad_fci``): R0 from a new greedy search for separating sets, Tetrad's
possible-m-sep step, the rule schedule of FciOrient (R4 only when R1-R3
changed something, R5-R7, R10 never fires), the test-based R4 with the
path-blocking search for separating sets, and the iteration orders of
Tetrad's hash sets (``tetrad_graph``). In Tetrad, R4 runs its tasks in
parallel over a hash set without hash codes, so with two or more
discriminating paths in one round its order is not deterministic; ETIA-JAX
uses the order in which the paths are found.
"""
from collections import deque
from itertools import combinations

from math import comb
import numpy as np

from ..graphs.encoding import NONE, CIRCLE, ARROW, TAIL
from ..graphs.orientation import fci_orient, unshielded_triples
from .cache import CICache
from .pcalg import fas, SearchResult
from .tetrad_graph import TGraph, JHashSet, JNode, ARROW as A, TAIL as T, CIRCLE as C


def _sublists(n, depth=-1):
    """Tetrad SublistGenerator(n, depth): subsets by size, each size in
    lexicographic order."""
    top = n if depth < 0 or depth > n else depth
    for k in range(top + 1):
        yield from combinations(range(n), k)


# =========================================================== textbook FCI
def _possible_dsep(G, x, y):
    """Possible-D-SEP(x, y): nodes a != x, y with a path x ... a on which each
    inner node is a collider or the two neighbours on the path are adjacent."""
    p = len(G)
    adj = lambda a, b: G[a, b] != NONE
    out, seen = set(), set()
    stack = [(x, b) for b in range(p) if adj(x, b)]
    while stack:
        a, b = stack.pop()
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if b not in (x, y):
            out.add(b)
        for c in range(p):
            if c in (a, b) or not adj(b, c):
                continue
            collider = G[a, b] == ARROW and G[c, b] == ARROW
            if collider or adj(a, c):
                stack.append((b, c))
    out.discard(x)
    out.discard(y)
    return sorted(out)


def _blocks_of_edges(G):
    """For each edge (x, y), x < y, the node set of its biconnected component
    (block) in the skeleton. A simple path between x and y stays in it."""
    import networkx as nx
    p = len(G)
    U = nx.Graph()
    U.add_nodes_from(range(p))
    U.add_edges_from((i, j) for i in range(p) for j in range(i + 1, p) if G[i, j] != NONE)
    out = {}
    for comp in nx.biconnected_component_edges(U):
        nodes = {v for e in comp for v in e}
        for a, b in comp:
            out[(min(a, b), max(a, b))] = nodes
    return out


_COMBOS = {}


def _combos(n, k):
    """All k-subsets of range(n) as an (m, k) index array, in the order of
    itertools.combinations."""
    if k == 0:
        return np.zeros((1, 0), np.int32)
    if (n, k) not in _COMBOS:
        c = np.fromiter((i for t in combinations(range(n), k) for i in t), np.int32, count=comb(n, k) * k)
        _COMBOS[(n, k)] = c.reshape(-1, k)
    return _COMBOS[(n, k)]


def _pds_step(cache, G):
    """The Possible-D-SEP step of FCI, batched by the size of the set: all
    edges test their sets of size 0, then of size 1, and so on; an edge stops
    at its first separating set. Possible-D-SEP is restricted to the block of
    the edge (Colombo et al. 2012, "biCC"). Returns [(x, y, Z)] to remove."""
    p = len(G)
    blocks = _blocks_of_edges(G)
    pending = {}
    for x in range(p):
        for y in range(x + 1, p):
            if G[x, y] != NONE:
                blk = blocks.get((x, y), set())
                pending[(x, y)] = ([v for v in _possible_dsep(G, x, y) if v in blk],
                                   [v for v in _possible_dsep(G, y, x) if v in blk])
    removed = []
    for (x, y), (a, b) in pending.items():          # an edge stops at its first separating set
        a, b = np.asarray(a, np.int32), np.asarray(b, np.int32)
        for k in range(max(len(a), len(b)) + 1):
            Zs = [side[_combos(len(side), k)] for side in (a, b) if len(side) >= k]
            i = cache.first_independent(x, y, np.concatenate(Zs))
            if i >= 0:
                removed.append((x, y, tuple(int(z) for z in np.concatenate(Zs)[i])))
                break
    return removed


def _r0(G, sepset):
    for a, b, c in unshielded_triples(G):
        S = sepset.get((min(a, c), max(a, c)))
        if S is not None and b not in S:
            G[a, b] = G[c, b] = ARROW


def _textbook_fci(cache, p, depth, complete=True):
    sk = fas(cache, p, depth, "stable")
    G = np.zeros((p, p), dtype=np.int8)
    for a in range(p):
        for b in sk.adj[a]:
            G[a, b] = CIRCLE
    sepset = dict(sk.sepset)
    _r0(G, sepset)
    removed = _pds_step(cache, G)
    for x, y, Z in removed:
        G[x, y] = G[y, x] = NONE
        sepset[(x, y)] = tuple(Z)
    G[G != NONE] = CIRCLE
    _r0(G, sepset)

    def is_noncollider(d, a, b, c):
        S = sepset.get((min(d, c), max(d, c)))
        return S is not None and b in S

    fci_orient(G, is_noncollider, complete=complete)
    return G, sk, sepset


# ============================================================ Tetrad 7.6.5
class _JavaPQ:
    """java.util.PriorityQueue with a comparator on an int key (binary heap,
    the same siftUp / siftDown as Java, so the same order for equal keys)."""

    def __init__(self):
        self.q = []

    def add(self, key, item):
        q = self.q
        q.append((key, item))
        k = len(q) - 1
        x = q[k]
        while k > 0:
            parent = (k - 1) >> 1
            if x[0] >= q[parent][0]:
                break
            q[k] = q[parent]
            k = parent
        q[k] = x

    def poll(self):
        q = self.q
        result = q[0]
        last = q.pop()
        n = len(q)
        if n:
            k, x, half = 0, last, n >> 1
            while k < half:
                c = 2 * k + 1
                r = c + 1
                if r < n and q[c][0] > q[r][0]:
                    c = r
                if x[0] <= q[c][0]:
                    break
                q[k] = q[c]
                k = c
            q[k] = x
        return result

    def __bool__(self):
        return bool(self.q)


class _TetradFci:
    """A port of Tetrad 7.6.5 Fci + FciOrient + R0R4StrategyTestBased +
    SepsetFinder + Paths (the parts that Fci uses) on a TGraph."""

    def __init__(self, cache, p, depth=-1):
        self.cache, self.p, self.depth = cache, p, depth
        self.change = False

    # ------------------------------------------------------------- tests
    def indep(self, x, y, Z):
        return self.cache.independent(x, y, tuple(Z))

    # ------------------------------------------------ helpers of Tetrad
    def arrowhead_allowed(self, g, x, y):
        """FciOrient.isArrowheadAllowed without knowledge."""
        if not g.is_adjacent(x, y):
            return False
        ep = g.get_endpoint(x, y)
        if ep == A:
            return True
        if ep == T:
            return False
        return ep == C

    def traverse_semi_directed(self, node, e):
        if node == e.n1:
            if e.e1 in (T, C):
                return e.n2
        elif node == e.n2:
            if e.e2 in (T, C):
                return e.n1
        return None

    def exists_semi_directed_path(self, g, frm, to):
        Q, V = deque(), set()
        for u in g.get_adjacent_nodes(frm):
            c = self.traverse_semi_directed(frm, g.get_edge(frm, u))
            if c is not None and c not in V:
                V.add(c)
                Q.append(c)
        while Q:
            t = Q.popleft()
            if t == to:
                return True
            for u in g.get_adjacent_nodes(t):
                c = self.traverse_semi_directed(t, g.get_edge(t, u))
                if c is not None and c not in V:
                    V.add(c)
                    Q.append(c)
        return False

    def is_ancestor(self, g, b, z):
        """Paths.isAncestor(b, z): b is an ancestor of some node of z."""
        if b in z:
            return True
        Q, V = deque(z), set(z)
        while Q:
            t = Q.popleft()
            if t == b:
                return True
            for c in g.get_parents(t):
                if c not in V:
                    Q.append(c)
                    V.add(c)
        return False

    # ------------------------------------------------- possible-m-sep
    def possible_msep(self, g, x, y):
        msep = set()
        Q = deque()
        V = set()
        previous = {x: set()}

        def add_to_set(b, c):
            previous.setdefault(c, set()).add(b)

        adj = JHashSet()
        for n in g.get_adjacent_nodes(x):
            adj.add(JNode(n, g.h[n]))
        for bn in adj:
            b = bn.i
            if b == y:
                continue
            Q.append((x, b))
            V.add((x, b))
            add_to_set(b, x)
            msep.add(b)
        while Q:
            a, b = Q.popleft()
            if self._exist_one_path_with_possible_parents(g, previous, b, x, b):
                msep.add(b)
            for c in g.get_adjacent_nodes(b):
                if c in (a, x, y):
                    continue
                add_to_set(b, c)
                if g.is_def_collider(a, b, c) or g.is_adjacent(a, c):
                    u = (a, c)
                    if u in V:
                        continue
                    V.add(u)
                    Q.append(u)
        msep.discard(x)
        msep.discard(y)
        return sorted(msep, reverse=True)        # Collections.sort, then reverse

    def _exist_one_path_with_possible_parents(self, g, previous, w, x, b):
        if w == x:
            return True
        p = previous.get(w)
        if p is None:
            return False
        for r in p:
            if r in (b, x):
                continue
            if self.exists_semi_directed_path(g, r, x) or self.exists_semi_directed_path(g, r, b):
                return True
        return False

    def remove_by_possible_msep(self, g):
        for e in g.get_edges():
            a, b = e.n1, e.n2
            for first, second in ((a, b), (b, a)):
                if first == b and e not in g.edges_set:
                    break
                pm = self.possible_msep(g, first, second)
                adj_a, adj_b = set(g.get_adjacent_nodes(a)), set(g.get_adjacent_nodes(b))
                removed = False
                for ch in _sublists(len(pm), len(pm)):
                    if len(ch) < 2:
                        continue
                    S = [pm[i] for i in ch]
                    if adj_a.issuperset(S) or adj_b.issuperset(S):
                        continue
                    if self.indep(a, b, S):
                        g.remove_edge(e)
                        removed = True
                        break
                if removed:
                    break

    # ------------------------------------------------------------ R0
    def sepset_greedy(self, g, x, y):
        """SepsetFinder.getSepsetContainingGreedy(graph, x, y, {}, test, -1)."""
        adjx = [n for n in g.get_adjacent_nodes(x) if n != y]
        adjy = [n for n in g.get_adjacent_nodes(y) if n != x]
        for adj in (adjx, adjy):
            for ch in _sublists(len(adj), self.depth):
                S = [adj[i] for i in ch]
                if self.indep(x, y, S):
                    return set(S)
        return None

    def rule_r0(self, g):
        g.reorient_all_with(C)
        for b in g.nodes:
            adj = g.get_adjacent_nodes(b)
            if len(adj) < 2:
                continue
            for i, j in combinations(range(len(adj)), 2):
                a, c = adj[i], adj[j]
                if g.is_adjacent(a, c) or g.is_def_collider(a, b, c):
                    continue
                S = self.sepset_greedy(g, a, c)
                if S is not None and b not in S:
                    if not self.arrowhead_allowed(g, a, b) or not self.arrowhead_allowed(g, c, b):
                        continue
                    g.set_endpoint(a, b, A)
                    g.set_endpoint(c, b, A)
                    self.change = True

    # ---------------------------------------------------------- R1-R3
    def r1(self, g, a, b, c):
        if g.is_adjacent(a, c):
            return
        if g.get_endpoint(a, b) == A and g.get_endpoint(c, b) == C:
            if not self.arrowhead_allowed(g, b, c):
                return
            g.set_endpoint(c, b, T)
            g.set_endpoint(b, c, A)
            self.change = True

    def r2(self, g, a, b, c):
        if g.is_adjacent(a, c) and g.get_endpoint(a, c) == C:
            e = g.get_endpoint
            if ((e(a, b) == A and e(b, c) == A and e(b, a) == T)
                    or (e(a, b) == A and e(b, c) == A and e(c, b) == T)):
                if not self.arrowhead_allowed(g, a, c):
                    return
                g.set_endpoint(a, c, A)
                self.change = True

    def r1r2_cycle(self, g):
        for b in g.nodes:
            adj = g.get_adjacent_nodes(b)
            if len(adj) < 2:
                continue
            for i, j in combinations(range(len(adj)), 2):
                a, c = adj[i], adj[j]
                self.r1(g, a, b, c)
                self.r1(g, c, b, a)
                self.r2(g, a, b, c)
                self.r2(g, c, b, a)

    def r3(self, g):
        for b in g.nodes:
            adj = g.get_adjacent_nodes(b)
            for i, j, k in combinations(range(len(adj)), 3):
                a, c, d = adj[i], adj[j], adj[k]
                if (not g.is_adjacent(a, c) and g.is_adjacent(a, d) and g.is_adjacent(c, d)
                        and g.is_def_collider(a, b, c) and g.get_endpoint(a, d) == C
                        and g.get_endpoint(c, d) == C and g.get_endpoint(d, b) == C):
                    if not self.arrowhead_allowed(g, d, b):
                        continue
                    g.set_endpoint(d, b, A)
                    self.change = True

    # -------------------------------------------------------------- R4
    def _path_exists_and_unoriented(self, g, e, a, b, c, collider_path):
        if g.get_endpoint(c, b) != C or g.is_adjacent(e, c):
            return False
        if g.get_endpoint(b, c) != A or g.get_endpoint(b, a) != A:
            return False
        if a not in collider_path:
            return False
        p = [e] + list(collider_path) + [b]
        for i in range(1, len(p) - 2):
            n1, n2, n3 = p[i - 1], p[i], p[i + 1]
            if not g.is_def_collider(n1, n2, n3):
                return False
            if not g.is_parent_of(n2, c):
                return False
        return True

    def list_discriminating_paths(self, g):
        paths = []
        for b in g.nodes:
            poss_a = g.get_nodes_out_to(b, A)
            poss_c = g.get_nodes_in_to(b, C)
            for a in poss_a:
                for c in poss_c:
                    if a == c or not g.is_parent_of(a, c):
                        continue
                    if g.get_endpoint(c, b) != C or g.get_endpoint(b, c) != A:
                        continue
                    self._discriminating_path_bfs(g, a, b, c, paths)
        return paths

    def _discriminating_path_bfs(self, g, a, b, c, paths):
        Q, V = deque([a]), {a, b}
        previous = {a: None, b: a}
        while Q:
            t = Q.popleft()
            for e in g.get_nodes_in_to(t, A):
                if e in V:
                    continue
                previous[e] = t
                collider_path = []
                d = e
                while True:
                    d = previous.get(d)
                    if d is None:
                        break
                    if d != e:
                        collider_path.insert(0, d)
                key = (e, a, b, c, tuple(collider_path))
                if self._path_exists_and_unoriented(g, e, a, b, c, collider_path):
                    if key not in paths:
                        paths.append(key)
                if e not in V:
                    Q.append(e)
                    V.add(e)

    def _is_m_connecting_path(self, g, path, Z):
        if len(path) - 1 == 1:
            return True
        e2 = g.get_edge(path[0], path[1])
        e2v = (e2.n1, e2.n2, e2.e1, e2.e2)
        for i in range(len(path) - 2):
            e1v = e2v
            e = g.get_edge(path[i + 1], path[i + 2])
            e2v = (e.n1, e.n2, e.e1, e.e2)
            b = path[i + 1]
            prox1 = e1v[2] if b == e1v[0] else e1v[3]
            if prox1 == A:
                # allowSelectionBias = True: a nondirected (o-o) edge becomes b -o> other
                if e2v[2] == C and e2v[3] == C:
                    other = e2v[1] if e2v[0] == b else e2v[0]
                    e2v = (b, other, C, A)
            if not self._reachable(g, e1v, e2v, path[i], Z):
                return False
        return True

    def _reachable(self, g, e1v, e2v, a, Z):
        b = e1v[1] if e1v[0] == a else e1v[0]
        prox1 = e1v[2] if b == e1v[0] else e1v[3]
        prox2 = e2v[2] if b == e2v[0] else e2v[3]
        collider = prox1 == A and prox2 == A
        if not collider and b not in Z:                # no underlined triples in FCI
            return True
        return collider and self.is_ancestor(g, b, Z)

    def _block_path(self, g, path, cond, could, blacklist, x, y):
        for n in range(1, len(path) - 1):
            z1, z2, z3 = path[n - 1], path[n], path[n + 1]
            if z1 == x and z3 == y and g.is_def_collider(z1, z2, z3):
                blacklist.add(z2)
                break
            if not g.is_def_collider(z1, z2, z3):
                if z2 in cond:
                    cond.difference_update(blacklist)
                    if g.is_adjacent(z1, z3):
                        could.add(z2)
                cond.add(z2)
                cond.difference_update(blacklist)
                if g.is_adjacent(z1, z3):
                    could.add(z2)
                break

    def _bfs_all_paths_out_of_x(self, g, cond, could, blacklist, max_len, x, y):
        queue = deque([[x]])
        while queue:
            path = queue.popleft()
            if max_len != -1 and len(path) > max_len:
                continue
            node = path[-1]
            for want_collider in (False, True):
                for z3 in g.get_adjacent_nodes(node):
                    if z3 in path:
                        continue
                    new = path + [z3]
                    if len(new) - 1 == 1:
                        if not want_collider:
                            queue.append(new)
                        else:
                            queue.append(new)
                        continue
                    z1, z2 = new[-3], new[-2]
                    if g.is_def_collider(z1, z2, z3) == want_collider:
                        self._block_path(g, new, cond, could, blacklist, x, y)
                        if self._is_m_connecting_path(g, new, cond):
                            queue.append(new)

    def sepset_path_blocking(self, g, x, y):
        """SepsetFinder.getSepsetPathBlockingOutOfX(g, x, y, test, -1, -1, true, {})."""
        max_len = g.p - 1
        cond, could, blacklist = _NodeSet(g), _NodeSet(g), _NodeSet(g)
        self._bfs_all_paths_out_of_x(g, cond, could, blacklist, max_len, x, y)
        could_list = list(could)
        cond.difference_update(could)
        for ch in _sublists(len(could_list), -1):
            sep = _NodeSet(g)
            for k in ch:
                sep.add(could_list[k])
            for n in cond:
                sep.add(n)
            sep.discard(y)
            if self.indep(x, y, list(sep)):
                z = sep
                removed = True
                while removed:
                    removed = False
                    for w in list(_NodeSet.copy_of(g, z)):
                        zz = _NodeSet.copy_of(g, z)
                        zz.discard(w)
                        if self.indep(x, y, list(zz)):
                            removed = True
                            z = zz
                return set(z)
        return None

    def do_discriminating_path(self, g, path):
        e, a, b, c, cp = path
        if not self._path_exists_and_unoriented(g, e, a, b, c, list(cp)):
            return False
        S = self.sepset_path_blocking(g, e, c)
        if S is None:
            return False
        if b not in S:                               # collider rule
            if g.get_endpoint(c, b) != C:
                return False
            if not self.arrowhead_allowed(g, a, b) or not self.arrowhead_allowed(g, c, b):
                return False
            g.set_endpoint(a, b, A)
            g.set_endpoint(c, b, A)
            return True
        if g.get_endpoint(c, b) != C:                # tail rule
            return False
        g.set_endpoint(c, b, T)
        return True

    def r4(self, g):
        results = []
        while True:
            paths = self.list_discriminating_paths(g)
            if not paths:
                break
            res = [self.do_discriminating_path(g, pth) for pth in paths]
            results.extend(res)
            if not any(res):
                break
        if any(results):
            self.change = True

    # --------------------------------------------------------- R5-R10
    def _dijkstra_path(self, g, x, y):
        """R5R9Dijkstra.distances (potentially directed) + getPath."""
        def neighbors(node):
            out = []
            for e in g.get_edges_of(node):
                o = self.traverse_semi_directed(node, e)
                if o is not None:
                    out.append(o)
            return out
        INF = 2 ** 31 - 1
        dist = {n: INF for n in g.nodes}
        pred = {n: None for n in g.nodes}
        dist[x] = 0
        pq = _JavaPQ()
        pq.add(0, x)
        visited = set()
        while pq:
            _, cur = pq.poll()
            if cur in visited:
                continue
            visited.add(cur)
            for to in neighbors(cur):
                pr = pred.get(cur)
                if (to == y and cur == x) or (to == x and cur == y):
                    continue
                if (to == y and pr == x) or (to == x and pr == y):
                    continue
                if pr is not None and pr in neighbors(to):
                    continue
                nd = dist[cur] + 1
                if nd < dist.get(to, INF):
                    dist[to] = nd
                    pred[to] = cur
                    pq.add(nd, to)
                    if to == y:
                        return self._get_path(pred, x, y)
        return self._get_path(pred, x, y)

    def _get_path(self, pred, start, end):
        path, at = [], end
        while at is not None:
            path.append(at)
            at = pred.get(at)
        path.reverse()
        return path if path[0] == start else None

    def r5(self, g):
        for e in g.get_edges():
            if e.e1 == C and e.e2 == C:
                x, y = e.n1, e.n2
                path = self._dijkstra_path(g, x, y)
                if path is None:
                    continue
                g.set_endpoint(x, y, T)
                g.set_endpoint(y, x, T)
                for i in range(len(path) - 1):
                    w, z = path[i], path[i + 1]
                    g.set_endpoint(w, z, T)
                    g.set_endpoint(z, w, T)
                self.change = True

    def r6(self, g):
        for e in g.get_edges():
            if not (e.e1 == T and e.e2 == T):
                continue
            for a, b in ((e.n1, e.n2), (e.n2, e.n1)):
                for c in g.get_adjacent_nodes(b):
                    if c != a and g.get_endpoint(c, b) == C:
                        g.set_endpoint(c, b, T)
                        self.change = True

    def r7(self, g):
        for e in g.get_edges():
            for a, b in ((e.n1, e.n2), (e.n2, e.n1)):
                if g.get_endpoint(a, b) == C and g.get_endpoint(b, a) == T:
                    for c in g.get_adjacent_nodes(b):
                        if c != a and not g.is_adjacent(a, c) and g.get_endpoint(c, b) == C:
                            g.set_endpoint(c, b, T)
                            self.change = True

    def _is_po(self, g, a, c):
        """The edge is a o-> c."""
        return g.get_endpoint(a, c) == A and g.get_endpoint(c, a) == C

    def r8(self, g, a, c):
        if g.get_edge(a, c) is None or not self._is_po(g, a, c):
            return False
        adj_c = set(g.get_adjacent_nodes(c))
        for b in [n for n in g.get_adjacent_nodes(a) if n in adj_c]:
            ep = g.get_endpoint
            if ((ep(b, a) == T and ep(a, b) == A and ep(c, b) == T and ep(b, c) == A)
                    or (ep(b, a) == T and ep(a, b) == C and ep(c, b) == T and ep(b, c) == A)):
                g.set_endpoint(c, a, T)
                self.change = True
                return True
        return False

    def r9(self, g, a, c):
        e = g.get_edge(a, c)
        if e is None or not self._is_po(g, a, c):
            return False
        path = self._dijkstra_path(g, e.n1, e.n2)
        if path is None:
            return False
        g.set_endpoint(c, a, T)
        self.change = True
        return True

    def r8r9r10(self, g):
        for c in g.nodes:
            for a in g.get_nodes_in_to(c, A):
                if g.get_endpoint(c, a) != C:
                    continue
                if not self.r8(g, a, c):
                    self.r9(g, a, c)          # R10 never fires in Tetrad 7.6.5

    def final_orientation(self, g):
        self.change = True
        while self.change:
            self.change = False
            self.r1r2_cycle(g)
            self.r3(g)
            if self.change:
                self.r4(g)
        self.r5(g)
        self.change = True
        while self.change:
            self.change = False
            self.r6(g)
            self.r7(g)
        self.change = True
        while self.change:
            self.change = False
            self.r8r9r10(g)

    # ------------------------------------------------------------ search
    def search(self, skeleton):
        g = TGraph(self.p)
        for i in range(self.p):
            for j in range(i + 1, self.p):
                if skeleton.adjacent(i, j):
                    g.add_undirected_edge(i, j)
        g.reorient_all_with(C)
        self.rule_r0(g)
        self.remove_by_possible_msep(g)
        g.reorient_all_with(C)
        self.rule_r0(g)
        self.final_orientation(g)
        return g


class _NodeSet:
    """A java.util.HashSet<Node> for the sepset search (Java order)."""

    def __init__(self, g):
        self.g, self.s = g, JHashSet()

    @classmethod
    def copy_of(cls, g, other):
        n = cls(g)
        n.s = JHashSet.copy(other.s)
        return n

    def add(self, i):
        self.s.add(JNode(i, self.g.h[i]))

    def discard(self, i):
        self.s.remove(JNode(i, self.g.h[i]))

    def difference_update(self, other):
        for i in list(other):
            self.discard(i)

    def __contains__(self, i):
        return JNode(i, self.g.h[i]) in self.s

    def __iter__(self):
        return (n.i for n in self.s)

    def __len__(self):
        return len(self.s)


def fci(ci, alpha=0.05, depth=-1, tetrad_compat=False, complete=True):
    """FCI. Returns a SearchResult with the PAG (mark matrix) in ``graph``."""
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    d = 1000 if depth < 0 else depth
    if tetrad_compat:
        sk = fas(cache, p, d, "sequential")
        g = _TetradFci(cache, p, depth).search(sk)
        return SearchResult(g.marks(), sk, cache)
    G, sk, sepset = _textbook_fci(cache, p, d, complete)
    return SearchResult(G, sk, cache, sepset=sepset)


# ================================================== FCI-Max, RFCI and CFCI
class _TetradFciFamily(_TetradFci):
    """Tetrad 7.6.5 FciMax, Rfci and Cfci (on the same machinery as Fci)."""

    def query(self, x, y, Z):
        return self.cache.query(x, y, tuple(Z))

    # ------------------------------------------------------------ FCI-Max
    def add_colliders_maxp(self, g, alpha):
        scores = []                                    # (score, a, b, c) in discovery order
        for b in g.nodes:
            adj = g.get_adjacent_nodes(b)
            if len(adj) < 2:
                continue
            for i, j in combinations(range(len(adj)), 2):
                a, c = adj[i], adj[j]
                if g.is_adjacent(a, c):
                    continue
                best, S = float("inf"), None
                for first, second, nb in ((a, c, g.get_adjacent_nodes(a)), (c, a, g.get_adjacent_nodes(c))):
                    for ch in _sublists(len(nb), -1):
                        s = [nb[k] for k in ch]
                        _, _, pval = self.query(first, second, s)
                        score = alpha - pval
                        if score < best:                 # NaN never wins
                            best, S = score, set(s)
                if S is not None and b not in S:
                    scores.append((best, a, b, c))
        # Tetrad: List.sort (stable) by score, descending
        for score, a, b, c in sorted(scores, key=lambda t: -t[0]):
            g.set_endpoint(a, b, A)
            g.set_endpoint(c, b, A)

    def search_fcimax(self, skeleton, alpha):
        g = self._graph_of(skeleton)
        g.reorient_all_with(C)
        self.remove_by_possible_msep(g)
        g.reorient_all_with(C)
        self.add_colliders_maxp(g, alpha)
        self.final_orientation(g)
        return g

    # --------------------------------------------------------------- RFCI
    def _set_min_sepset(self, sepsets, S, x, y):
        S = sorted(S)
        if self.indep(x, y, []):
            sepsets[frozenset((x, y))] = set()
            return
        for k in range(1, len(S) + 1):
            for ch in combinations(range(len(S)), k):
                cond = [S[i] for i in ch]
                if self.indep(x, y, cond):
                    sepsets[frozenset((x, y))] = set(cond)
                    return

    def r0_rfci(self, g, sepsets):
        r = []
        for j in g.nodes:
            adj = g.get_adjacent_nodes(j)
            if len(adj) < 2:
                continue
            for a, b in combinations(range(len(adj)), 2):
                i, k = adj[a], adj[b]
                if not g.is_adjacent(i, k):
                    r.append((i, j, k))
        l = []
        while r:
            i, j, k = r.pop(0)
            nodes1 = sepsets.get(frozenset((i, k)))
            if nodes1 is None:
                continue
            S = set(nodes1)
            S.discard(j)
            ind1 = self.indep(i, j, sorted(S))
            ind2 = self.indep(j, k, sorted(S))
            if not ind1 and not ind2:
                l.append((i, j, k))
                continue
            if ind1:
                self._set_min_sepset(sepsets, S, i, j)
                e = g.get_edge(i, j)
                if e is not None:
                    g.remove_edge(e)
            if ind2:
                self._set_min_sepset(sepsets, S, j, k)
                e = g.get_edge(j, k)
                if e is not None:
                    g.remove_edge(e)
            for t in g.nodes:
                adj_t = g.get_adjacent_nodes(t)
                if ind1 and i in adj_t and j in adj_t:
                    r.append((i, t, j))
                if ind2 and j in adj_t and k in adj_t:
                    r.append((j, t, k))

            def drop(tp):
                return ((ind1 and tp[1] == i and (tp[0] == j or tp[2] == j))
                        or (ind2 and tp[1] == k and (tp[0] == j or tp[2] == j))
                        or (ind1 and tp[1] == j and (tp[0] == i or tp[2] == i))
                        or (ind2 and tp[1] == j and (tp[0] == k or tp[2] == k)))
            r = [tp for tp in r if not drop(tp)]
            l = [tp for tp in l if not drop(tp)]
        for i, j, k in l:
            sep = sepsets.get(frozenset((i, k)))
            if sep is None:
                continue
            if j not in sep and g.is_adjacent(i, j) and g.is_adjacent(j, k):
                if not self.arrowhead_allowed(g, i, j) or not self.arrowhead_allowed(g, k, j):
                    continue
                g.set_endpoint(i, j, A)
                g.set_endpoint(k, j, A)

    def search_rfci(self, skeleton):
        g = self._graph_of(skeleton)
        g.reorient_all_with(C)
        sepsets = {frozenset(k): set(v) for k, v in skeleton.sepset.items()}
        self.r0_rfci(g, sepsets)
        self.final_orientation(g)
        return g

    # --------------------------------------------------------------- CFCI
    def _triple_type(self, g, x, y, z, sepsets):
        cont = notcont = False
        for a, b in ((x, z), (z, x)):
            nodes = _NodeSet(g)
            for n in g.get_adjacent_nodes(a):
                nodes.add(n)
            nodes.discard(b)
            lst = list(nodes)
            for d in range(len(lst) + 1):
                for ch in combinations(range(len(lst)), d):
                    S = [lst[i] for i in ch]
                    if self.indep(x, z, S):
                        if y in S:
                            cont = True
                        else:
                            notcont = True
        S = sepsets.get(frozenset((x, z)))
        if S is not None:
            if y in S:
                cont = True
            else:
                notcont = True
        if cont == notcont:
            return "ambiguous"
        return "noncollider" if not notcont else "collider"

    def r0_cfci(self, g, sepsets):
        for y in g.nodes:
            adj = g.get_adjacent_nodes(y)
            if len(adj) < 2:
                continue
            for a, b in combinations(range(len(adj)), 2):
                x, z = adj[a], adj[b]
                if g.is_adjacent(x, z):
                    continue
                t = self._triple_type(g, x, y, z, sepsets)
                S = sepsets.get(frozenset((x, z)))
                if t == "collider" or (S is not None and y not in S):
                    if self.arrowhead_allowed(g, x, y) and self.arrowhead_allowed(g, z, y):
                        g.set_endpoint(x, y, A)
                        g.set_endpoint(z, y, A)

    def possible_msep_fci(self, g):
        """PossibleMsepFci.search: x's side only, all subsets from size 0."""
        found = {}
        for e in list(g.get_edges()):
            x, y = e.n1, e.n2
            pm = self.possible_msep(g, x, y)
            cond = None
            for d in range(len(pm) + 1):
                for ch in combinations(range(len(pm)), d):
                    S = [pm[i] for i in ch]
                    if self.indep(x, y, S):
                        cond = set(S)
                        break
                if cond is not None:
                    break
            if cond is not None:
                ee = g.get_edge(x, y)
                if ee is not None:
                    g.remove_edge(ee)
                found[frozenset((x, y))] = cond
        return found

    def search_cfci(self, skeleton):
        g = self._graph_of(skeleton)
        g.reorient_all_with(C)
        sepsets = {}                                   # Cfci keeps its own, empty map
        self.r0_cfci(g, sepsets)
        sepsets.update(self.possible_msep_fci(g))
        g.reorient_all_with(C)
        self.r0_cfci(g, sepsets)
        self.final_orientation(g)
        return g

    def _graph_of(self, skeleton):
        g = TGraph(self.p)
        for i in range(self.p):
            for j in range(i + 1, self.p):
                if skeleton.adjacent(i, j):
                    g.add_undirected_edge(i, j)
        return g


# ---------------------------------------------------- textbook defaults
def _maxp_sepsets(cache, G, alpha):
    """Max-P: for each unshielded triple the separating set (subsets of adj(a)
    and adj(c)) with the largest p-value. Returns {(a, c): (p, set)}."""
    p = len(G)
    adj = lambda v: [int(u) for u in np.flatnonzero(G[v] != NONE)]
    out = {}
    qs = []
    for b in range(p):
        for a, c in combinations(adj(b), 2):
            if G[a, c] == NONE:
                for first, second, nb in ((a, c, adj(a)), (c, a, adj(c))):
                    qs.extend((first, second, Z) for k in range(len(nb) + 1) for Z in combinations(nb, k))
    cache.prefetch(qs)
    for b in range(p):
        for a, c in combinations(adj(b), 2):
            if G[a, c] != NONE or (a, c) in out:
                continue
            best, S = -1.0, None
            for first, second, nb in ((a, c, adj(a)), (c, a, adj(c))):
                for k in range(len(nb) + 1):
                    for Z in combinations(nb, k):
                        pv = cache.query(first, second, Z)[2]
                        if pv == pv and pv > best:
                            best, S = pv, set(Z)
            out[(a, c)] = (best, S)
    return out


def _textbook_variant(cache, p, depth, alpha, variant):
    """FCI-Max or CFCI: textbook FCI with the Max-P or the conservative R0."""
    sk = fas(cache, p, depth, "stable")
    G = np.zeros((p, p), dtype=np.int8)
    for a in range(p):
        for b in sk.adj[a]:
            G[a, b] = CIRCLE
    sepset = dict(sk.sepset)
    _r0(G, sepset)
    removed = _pds_step(cache, G)
    for x, y, Z in removed:
        G[x, y] = G[y, x] = NONE
        sepset[(x, y)] = tuple(Z)
    G[G != NONE] = CIRCLE
    if variant == "fcimax":
        mp = _maxp_sepsets(cache, G, alpha)
        for (a, c), (pv, S) in sorted(mp.items(), key=lambda kv: kv[1][0]):   # lowest p first
            sepset[(a, c)] = tuple(sorted(S)) if S is not None else sepset.get((a, c))
        for a, b, c in unshielded_triples(G):
            S = sepset.get((min(a, c), max(a, c)))
            if S is not None and b not in S:
                G[a, b] = G[c, b] = ARROW
    else:  # cfci: collider only if no separating set (among adjacents) contains b
        amb = set()
        trip = list(unshielded_triples(G))
        cache.prefetch([(a, c, Z) for a, b, c in trip
                        for side in ([v for v in np.flatnonzero(G[a] != NONE) if v != c],
                                     [v for v in np.flatnonzero(G[c] != NONE) if v != a])
                        for k in range(len(side) + 1) for Z in combinations(side, k)])
        for a, b, c in trip:
            adj_a = [v for v in np.flatnonzero(G[a] != NONE) if v != c]
            adj_c = [v for v in np.flatnonzero(G[c] != NONE) if v != a]
            seps = [set(Z) for side in (adj_a, adj_c) for k in range(len(side) + 1)
                    for Z in combinations(side, k) if cache.independent(a, c, Z)]
            S0 = sepset.get((min(a, c), max(a, c)))
            if S0 is not None:
                seps.append(set(S0))
            if seps and all(b not in s for s in seps):
                G[a, b] = G[c, b] = ARROW
            elif not (seps and all(b in s for s in seps)):
                amb.add((a, b, c))

    def is_noncollider(d, a, b, c):
        S = sepset.get((min(d, c), max(d, c)))
        return S is not None and b in S

    fci_orient(G, is_noncollider, complete=True)
    return G, sk, sepset


def _rfci_textbook(cache, p, depth):
    """RFCI (Colombo et al. 2012): FAS, the R0 of their Algorithm 4.4, then
    Zhang's rules with the separating sets. Sound, not complete."""
    sk = fas(cache, p, depth, "stable")
    fam = _TetradFciFamily(cache, p)
    g = fam._graph_of(sk)
    g.reorient_all_with(C)
    sepsets = {frozenset(k): set(v) for k, v in sk.sepset.items()}
    fam.r0_rfci(g, sepsets)
    G = g.marks()
    sep = {tuple(sorted(k)): tuple(sorted(v)) for k, v in sepsets.items()}

    def is_noncollider(d, a, b, c):
        S = sep.get((min(d, c), max(d, c)))
        return S is not None and b in S

    fci_orient(G, is_noncollider, complete=True)
    return G, sk, sep


def fcimax(ci, alpha=0.05, depth=-1, tetrad_compat=False):
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    d = 1000 if depth < 0 else depth
    if tetrad_compat:
        sk = fas(cache, p, d, "sequential")
        g = _TetradFciFamily(cache, p, depth).search_fcimax(sk, alpha)
        return SearchResult(g.marks(), sk, cache)
    G, sk, sep = _textbook_variant(cache, p, d, alpha, "fcimax")
    return SearchResult(G, sk, cache, sepset=sep)


def rfci(ci, alpha=0.05, depth=-1, tetrad_compat=False):
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    d = 1000 if depth < 0 else depth
    if tetrad_compat:
        sk = fas(cache, p, d, "sequential")
        g = _TetradFciFamily(cache, p, depth).search_rfci(sk)
        return SearchResult(g.marks(), sk, cache)
    G, sk, sep = _rfci_textbook(cache, p, d)
    return SearchResult(G, sk, cache, sepset=sep)


def cfci(ci, alpha=0.05, depth=-1, tetrad_compat=False):
    p = ci.p if hasattr(ci, "p") else len(ci.observed)
    cache = CICache(ci, alpha)
    d = 1000 if depth < 0 else depth
    if tetrad_compat:
        sk = fas(cache, p, d, "sequential")
        g = _TetradFciFamily(cache, p, depth).search_cfci(sk)
        return SearchResult(g.marks(), sk, cache)
    G, sk, sep = _textbook_variant(cache, p, d, alpha, "cfci")
    return SearchResult(G, sk, cache, sepset=sep)
