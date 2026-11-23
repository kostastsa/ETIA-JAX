"""Background knowledge of temporal tiers (Tetrad 7.6.5 ``Knowledge`` with
tiers only, as ETIA's ``time_knowledge`` makes it).

Tier 0 is the oldest. An edge from a node of a later tier into a node of an
earlier tier is forbidden: ``is_forbidden(a, b)`` is true iff
tier(a) > tier(b). Nodes without a tier have no constraints. No edge is
required.
"""


from .._java import java_string_hash
from .tetrad_graph import JHashSet


class _JStr:
    """A String in a java.util.HashSet."""
    __slots__ = ("i", "jhash")

    def __init__(self, i, name):
        self.i, self.jhash = i, java_string_hash(name)

    def __eq__(self, o):
        return self.i == o.i


class _JKEdge:
    """A KnowledgeEdge: hash 37 (31 + h(from)) + h(to)."""
    __slots__ = ("a", "b", "jhash")

    def __init__(self, a, b, ha, hb):
        self.a, self.b = a, b
        self.jhash = (37 * ((31 + ha) & 0xFFFFFFFF) + hb) & 0xFFFFFFFF

    def __eq__(self, o):
        return self.a == o.a and self.b == o.b


class Knowledge:
    def __init__(self, tiers=(), names=None):
        self.names = list(names) if names is not None else None
        self.tiers = [list(t) for t in tiers]
        self.tier_of = {}
        for k, t in enumerate(self.tiers):
            for v in t:
                self.tier_of[int(v)] = k

    @classmethod
    def time_lagged(cls, p, n_lags):
        """ETIA's time_knowledge for the lagged table of p variables: the
        lag-l block (columns l p .. l p + p - 1) is tier n_lags - l."""
        names = [f"X{i + 1}" if l == 0 else f"X{i + 1}:{l}" for l in range(n_lags + 1) for i in range(p)]
        return cls([list(range((n_lags - t) * p, (n_lags - t + 1) * p)) for t in range(n_lags + 1)], names)

    def forbidden_edges(self, names=None):
        """Knowledge.getListOfForbiddenEdges() for tiers: (a, b) with a in a
        later tier than b. With names: the Java order (a HashSet of
        KnowledgeEdge filled tier by tier, each tier a HashSet<String>);
        without: index order."""
        if names is None:
            return sorted((a, b) for i in range(len(self.tiers)) for j in range(i)
                          for a in self.tiers[i] for b in self.tiers[j])
        h = [java_string_hash(n) for n in names]

        def tier_order(t):
            st = JHashSet()
            for v in t:
                st.add(_JStr(v, names[v]))
            return [x.i for x in st]
        edges = JHashSet()
        for i in range(len(self.tiers) - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                for a in tier_order(self.tiers[i]):
                    for b in tier_order(self.tiers[j]):
                        edges.add(_JKEdge(a, b, h[a], h[b]))
        return [(e.a, e.b) for e in edges]

    def is_empty(self):
        return not self.tiers

    def __bool__(self):
        return not self.is_empty()

    def is_forbidden(self, a, b):
        """The edge a -> b is forbidden."""
        ta, tb = self.tier_of.get(int(a)), self.tier_of.get(int(b))
        return ta is not None and tb is not None and ta > tb

    def is_required(self, a, b):
        return False

    def no_edge_required(self, a, b):
        return True

    def possible_parent(self, z, x):
        """Fas.possibleParentOf: z may be a parent of x."""
        return not self.is_forbidden(z, x) and not self.is_required(x, z)

    def tier(self, v):
        """Knowledge.isInWhichTier (-1 if none)."""
        return self.tier_of.get(int(v), -1)


NO_KNOWLEDGE = Knowledge()
