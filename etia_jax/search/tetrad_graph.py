"""An emulation of the parts of Tetrad 7.6.5 that decide its iteration orders.

Many steps of Tetrad's searches iterate java.util.HashSet / HashMap objects,
and their results change with that order. To reproduce ETIA exactly
(``tetrad_compat=True``), the searches run on ``TGraph``, a port of
``EdgeListGraph`` whose sets are ``JHashSet``: a Java 8+ HashMap emulation
(bucket index ``(h ^ (h >>> 16)) & (cap - 1)``, insertion order in a bucket,
resize by splitting buckets, the copy constructor's capacity).

Not emulated: treeified bins (more than 8 entries in one bucket of a table of
64 or more). ``JHashSet.treeify_risk`` counts when that would happen.
"""
from .._java import java_string_hash, table_size_for

ARROW, TAIL, CIRCLE = "ARROW", "TAIL", "CIRCLE"
_MARK = {CIRCLE: 1, ARROW: 2, TAIL: 3}


def _spread(h):
    h &= 0xFFFFFFFF
    return h ^ (h >> 16)


class JHashSet:
    """java.util.HashSet<E> (HashMap) with Java's iteration order.
    Elements must have ``jhash`` (int) and ``__eq__``."""
    treeify_risk = 0

    def __init__(self, initial_capacity=None):
        self.cap = table_size_for(initial_capacity) if initial_capacity else 16
        self.table = None              # allocated at the first add (as Java)
        self.size = 0

    @classmethod
    def copy(cls, src):
        """new HashSet<>(src): capacity from (int)(n / .75f) + 1, at least 16."""
        s = cls(max(int(len(src) / 0.75) + 1, 16))
        for x in src:
            s.add(x)
        return s

    def _alloc(self):
        if self.table is None:
            self.table = [[] for _ in range(self.cap)]

    def _bucket(self, x):
        return _spread(x.jhash) & (self.cap - 1)

    def add(self, x):
        self._alloc()
        b = self.table[self._bucket(x)]
        for e in b:
            if e == x:
                return False
        b.append(x)
        self.size += 1
        if len(b) > 8 and self.cap >= 64:
            JHashSet.treeify_risk += 1
        if self.size > 0.75 * self.cap:
            self._resize()
        return True

    def _resize(self):
        old, oldcap = self.table, self.cap
        self.cap *= 2
        self.table = [[] for _ in range(self.cap)]
        for i, b in enumerate(old):
            for e in b:                                 # split preserves the order
                hi = _spread(e.jhash) & oldcap
                self.table[i + oldcap if hi else i].append(e)

    def remove(self, x):
        if self.table is None:
            return False
        b = self.table[self._bucket(x)]
        for k, e in enumerate(b):
            if e == x:
                del b[k]
                self.size -= 1
                return True
        return False

    def __contains__(self, x):
        if self.table is None:
            return False
        return any(e == x for e in self.table[self._bucket(x)])

    def __iter__(self):
        if self.table is None:
            return iter(())
        return (e for b in self.table for e in b)

    def __len__(self):
        return self.size


class JNode:
    """A node of a JHashSet<Node>: equality by index, hash of its name."""
    __slots__ = ("i", "jhash")

    def __init__(self, i, jhash):
        self.i, self.jhash = i, jhash

    def __eq__(self, o):
        return self.i == o.i

    def __hash__(self):
        return self.i


class TEdge:
    """Tetrad's Edge. An edge pointing left (ARROW at node1 and TAIL or
    CIRCLE at node2) is flipped by the constructor."""
    __slots__ = ("n1", "n2", "e1", "e2", "jhash")

    def __init__(self, n1, n2, e1, e2, hashes):
        if e1 == ARROW and e2 in (TAIL, CIRCLE):
            n1, n2, e1, e2 = n2, n1, e2, e1
        self.n1, self.n2, self.e1, self.e2 = n1, n2, e1, e2
        self.jhash = (hashes[n1] + hashes[n2]) & 0xFFFFFFFF

    def __eq__(self, o):
        return ((self.n1 == o.n1 and self.n2 == o.n2 and self.e1 == o.e1 and self.e2 == o.e2)
                or (self.n1 == o.n2 and self.n2 == o.n1 and self.e1 == o.e2 and self.e2 == o.e1))

    def distal(self, node):
        return self.n2 if node == self.n1 else self.n1

    def proximal_endpoint(self, node):
        return self.e1 if node == self.n1 else self.e2

    def distal_endpoint(self, node):
        return self.e2 if node == self.n1 else self.e1


class TGraph:
    """EdgeListGraph (Tetrad 7.6.5) over nodes 0..p-1 named X1..Xp."""

    def __init__(self, p, names=None):
        self.p = p
        self.names = names or [f"X{i + 1}" for i in range(p)]
        self.h = [java_string_hash(n) for n in self.names]
        self.nodes = list(range(p))
        self.edge_lists = {i: JHashSet() for i in range(p)}
        self.edges_set = JHashSet()
        self.parents_hash = {}

    # -------------------------------------------------------------- edges
    def edge(self, a, b, ea, eb):
        return TEdge(a, b, ea, eb, self.h)

    def add_edge(self, e):
        s1 = JHashSet.copy(self.edge_lists[e.n1])
        s2 = JHashSet.copy(self.edge_lists[e.n2])
        s1.add(e)
        s2.add(e)
        self.edge_lists[e.n1], self.edge_lists[e.n2] = s1, s2
        self.edges_set.add(e)
        self.parents_hash.pop(e.n1, None)
        self.parents_hash.pop(e.n2, None)
        return True

    def remove_edge(self, e):
        if e not in self.edges_set:
            return False
        s1 = JHashSet.copy(self.edge_lists[e.n1])
        s2 = JHashSet.copy(self.edge_lists[e.n2])
        self.edges_set.remove(e)
        s1.remove(e)
        s2.remove(e)
        self.edge_lists[e.n1], self.edge_lists[e.n2] = s1, s2
        self.parents_hash.pop(e.n1, None)
        self.parents_hash.pop(e.n2, None)
        return True

    def add_undirected_edge(self, a, b):
        return self.add_edge(self.edge(a, b, TAIL, TAIL))

    def add_directed_edge(self, a, b):
        return self.add_edge(self.edge(a, b, TAIL, ARROW))

    def get_edge(self, a, b):
        for e in self.edge_lists[a]:
            if (e.n1 == a and e.n2 == b) or (e.n1 == b and e.n2 == a):
                return e
        return None

    def get_edges(self):
        """getEdges(): a copy, new HashSet<>(edgesSet)."""
        return list(JHashSet.copy(self.edges_set))

    def get_edges_of(self, node):
        return list(JHashSet.copy(self.edge_lists[node]))

    def get_endpoint(self, a, b):
        """The mark at b on the edge a - b (None if not adjacent)."""
        e = self.get_edge(a, b)
        return None if e is None else e.proximal_endpoint(b)

    def set_endpoint(self, frm, to, ep):
        e = self.get_edge(frm, to)
        if e is None:
            raise ValueError("not adjacent")
        self.remove_edge(e)
        self.add_edge(self.edge(frm, to, e.proximal_endpoint(frm), ep))
        return True

    def reorient_all_with(self, ep):
        for e in self.get_edges():
            self.remove_edge(e)
            self.add_edge(self.edge(e.n1, e.n2, ep, ep))

    # -------------------------------------------------------------- nodes
    def is_adjacent(self, a, b):
        return any(e.distal(a) == b for e in self.edge_lists[a])

    def get_adjacent_nodes(self, a):
        adj = JHashSet()
        for e in self.edge_lists[a]:
            adj.add(JNode(e.distal(a), self.h[e.distal(a)]))
        return [n.i for n in adj]

    def get_nodes_in_to(self, node, ep):
        return [e.distal(node) for e in self.get_edges_of(node) if e.proximal_endpoint(node) == ep]

    def get_nodes_out_to(self, node, ep):
        return [e.distal(node) for e in self.get_edges_of(node) if e.distal_endpoint(node) == ep]

    def get_parents(self, node):
        if node not in self.parents_hash:
            self.parents_hash[node] = [e.distal(node) for e in self.edge_lists[node]
                                       if e.distal_endpoint(node) == TAIL and e.proximal_endpoint(node) == ARROW]
        return self.parents_hash[node]

    def is_parent_of(self, a, b):
        """a -> b."""
        e = self.get_edge(a, b)
        return e is not None and e.proximal_endpoint(a) == TAIL and e.proximal_endpoint(b) == ARROW

    def is_def_collider(self, a, b, c):
        e1, e2 = self.get_edge(a, b), self.get_edge(b, c)
        if e1 is None or e2 is None:
            return False
        return e1.proximal_endpoint(b) == ARROW and e2.proximal_endpoint(b) == ARROW

    # ------------------------------------------------------- conversions
    def marks(self):
        """The ETIA mark matrix: M[i, j] = mark at j on the edge i - j."""
        import numpy as np
        M = np.zeros((self.p, self.p), dtype=np.int8)
        for e in self.edges_set:
            M[e.n1, e.n2] = _MARK[e.e2]
            M[e.n2, e.n1] = _MARK[e.e1]
        return M
