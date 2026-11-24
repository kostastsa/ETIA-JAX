"""A port of the parts of dagitty 0.3.4 (libdagitty) that ETIA's
``find_adjset`` reaches: ``adjustmentSets(type="canonical")`` and the first
set of ``adjustmentSets(type="minimal")`` (increment 10).

The port keeps dagitty's data structures, because the order of the minimal
sets depends on them:

* a ``Hash`` is a JavaScript object: integer-like keys in increasing order,
  then the other keys in insertion order;
* a vertex has ``incoming`` and ``outgoing`` edge lists, and its kinship
  lists (children, parents, neighbours, ...) follow these lists;
* the graph that the JavaScript code sees comes from the text that R sends:
  ``pcalg2dagitty`` builds a text, dagitty parses it and writes it again
  (vertex statements sorted, edge statements sorted), and R sets the
  exposures and outcomes the same way. ``from_etia`` makes this graph.

Sources: https://github.com/jtextor/libdagitty (src/Graph.js,
GraphAnalyzer.js, GraphTransformer.js, GraphSerializer.js) and the R
package (R/dagitty.R, inst/js/RUtil.js); pcalg 2.7-12 ``pcalg2dagitty``.
"""
import re

UNDIRECTED, DIRECTED, BIDIRECTED, UNSPECIFIED, PARTDIRECTED, PARTUNDIRECTED = 0, 1, 2, 3, 4, 5
SYMMETRIC = {0: True, 1: False, 2: True, 3: True, 4: False, 5: False}
SYMBOL = {0: "--", 1: "->", 2: "<->", 3: "@-@", 4: "@->", 5: "@--"}
_INDEX_KEY = re.compile(r"^(0|[1-9][0-9]*)$")


class Hash:
    """A JavaScript object used as a map (Object.keys order)."""

    def __init__(self):
        self.kv = {}

    def contains(self, k):
        return k in self.kv

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v):
        self.kv[k] = v

    def unset(self, k):
        self.kv.pop(k, None)

    def keys(self):
        ints = sorted((k for k in self.kv if _INDEX_KEY.match(k) and int(k) < 2 ** 32 - 1), key=int)
        si = set(ints)
        return ints + [k for k in self.kv if k not in si]

    def values(self):
        return [self.kv[k] for k in self.keys()]


# ------------------------------------------------------------ underscore
def _uniq(a):
    out = []
    for x in a:
        if not any(x is y for y in out):
            out.append(x)
    return out


def _union(*arrs):
    return _uniq([x for a in arrs for x in a])


def _contains(a, x):
    return any(x is y for y in a)


def _difference(a, *others):
    rest = [x for o in others for x in o]
    return [x for x in a if not _contains(rest, x)]


def _intersection(a, *others):
    out = []
    for x in _uniq(a):
        if all(_contains(o, x) for o in others):
            out.append(x)
    return out


# ------------------------------------------------------------ graph
class Vertex:
    def __init__(self, vid):
        self.id = vid
        self.incoming = []
        self.outgoing = []
        self.visited = False

    def kin(self, etype, outward=True):
        r = []
        if outward or SYMMETRIC[etype]:
            r += [e.v2 if e.v1 is self else e.v1 for e in self.outgoing if e.directed == etype]
        if not outward or SYMMETRIC[etype]:
            r += [e.v2 if e.v1 is self else e.v1 for e in self.incoming if e.directed == etype]
        return r

    def neighbours(self):
        return self.kin(UNDIRECTED)

    def spouses(self):
        return self.kin(BIDIRECTED)

    def children(self):
        return self.kin(DIRECTED)

    def parents(self):
        return self.kin(DIRECTED, False)

    def possible_neighbours(self):
        return self.kin(UNDIRECTED) + self.kin(UNSPECIFIED)

    def possible_parents(self):
        return self.kin(DIRECTED, False) + self.kin(PARTDIRECTED, False)

    def children_and_neighbours(self):
        return self.neighbours() + self.children()

    def possible_parents_and_neighbours(self):
        return self.possible_neighbours() + self.possible_parents()

    def adjacent(self):
        return self.children_and_neighbours() + self.parents() + self.spouses()


class Edge:
    def __init__(self, v1, v2, directed):
        self.v1, self.v2, self.directed = v1, v2, directed


PROPS = ("source", "target", "adjustedNode", "latentNode", "selectedNode")


class Graph:
    def __init__(self, gtype="digraph"):
        self.vertices = Hash()
        self.edges = []
        self.type = gtype
        self.props = {p: Hash() for p in PROPS}

    # vertices
    def get(self, v):
        if isinstance(v, str):
            return self.vertices.get(v)
        if isinstance(v, Vertex):
            return self.vertices.get(v.id)
        if isinstance(v, list):
            return [self.get(x) for x in v]
        raise TypeError(v)

    def add_vertex(self, vid):
        v = Vertex(vid)
        self.vertices.set(vid, v)
        return v

    def vertex_list(self):
        return self.vertices.values()

    def add_prop(self, v, p):
        vv = self.get(v)
        if vv is not None:
            self.props[p].set(vv.id, vv)

    def with_prop(self, p):
        return self.props[p].values()

    def has_prop(self, v, p):
        vv = self.get(v)
        return vv is not None and self.props[p].contains(vv.id)

    sources = lambda self: self.with_prop("source")
    targets = lambda self: self.with_prop("target")
    adjusted = lambda self: self.with_prop("adjustedNode")
    latent = lambda self: self.with_prop("latentNode")
    selected = lambda self: self.with_prop("selectedNode")

    def copy_props_to(self, g2):
        for p in PROPS:
            for v in self.with_prop(p):
                g2.add_prop(v, p)
        g2.type = self.type

    def clone(self):
        g2 = Graph()
        for v in self.vertex_list():
            g2.add_vertex(v.id)
        for e in self.edges:
            g2.add_edge(e.v1.id, e.v2.id, e.directed)
        self.copy_props_to(g2)
        return g2

    # edges
    def get_edge(self, v1, v2, etype=DIRECTED):
        v1, v2 = self.get(v1), self.get(v2)
        if v1 is None or v2 is None:
            return None
        for e in v1.outgoing:
            if e.v2.id == v2.id and e.directed == etype:
                return e
        if SYMMETRIC[etype]:
            for e in v1.incoming:
                if e.v1.id == v2.id and e.directed == etype:
                    return e
        return None

    def add_edge(self, v1, v2, etype=DIRECTED):
        v1, v2 = self.get(v1), self.get(v2)
        e = self.get_edge(v1, v2, etype)
        if e is None and SYMMETRIC[etype]:
            e = self.get_edge(v2, v1, etype)
        if e is None:
            e = Edge(v1, v2, etype)
            v1.outgoing.append(e)
            v2.incoming.append(e)
            self.edges.append(e)
        return e

    def delete_edge(self, v1, v2, etype):
        e = self.get_edge(v1, v2, etype)
        if e is None:
            return False
        e.v1.outgoing = [x for x in e.v1.outgoing if x is not e]
        e.v2.incoming = [x for x in e.v2.incoming if x is not e]
        self.edges = [x for x in self.edges if x is not e]
        return True

    def change_edge(self, e, etype):
        v1n = e.v1
        v2n = e.v2
        if self.get_edge(v1n, v2n, etype) is not None:
            return False
        e.directed = etype
        return True

    # closures
    def clear_visited(self):
        for v in self.vertex_list():
            v.visited = False

    def closure(self, vs, kin):
        self.clear_visited()
        q = [v for v in vs if not v.visited]
        for v in q:
            v.visited = True
        r = []
        while q:
            v = q.pop()
            for w in [w for w in getattr(v, kin)() if not w.visited]:
                w.visited = True
                q.append(w)
            r.append(v)
        self.clear_visited()
        return r

    def ancestors_of(self, vs):
        return self.closure(vs, "parents")

    def anteriors_of(self, vs):
        return self.closure(vs, "possible_parents_and_neighbours")

    def posteriors_of(self, vs):
        return self.closure(vs, "children_and_neighbours")

    def district_of(self, vs):
        return self.closure(vs, "spouses")

    def parents_of(self, vs):
        r = []
        for v in vs:
            r = _uniq(r + v.parents())
        return _uniq(r)

    def spouses_of(self, vs):
        vh = Hash()
        for v in vs:
            vh.set(v.id, v)
        rh = Hash()
        for v in vs:
            for w in v.spouses():
                if not vh.get(w.id):
                    rh.set(w.id, w)
        return rh.values()

    def neighbours_of(self, vs):
        vh = Hash()
        for v in vs:
            vh.set(v.id, v)
        rh = Hash()
        for v in vs:
            for w in v.neighbours():
                if not vh.get(w.id):
                    rh.set(w.id, w)
        return rh.values()


# ------------------------------------------------------------ text (dot)
_DOT_BARE = re.compile(r"^[0-9a-zA-Z_.]+$")


def _quote(vid):
    return vid if _DOT_BARE.match(vid) else '"' + vid.replace('"', '\\"') + '"'


def to_dot(g):
    """GraphSerializer.toDot without layout: (sorted vertex lines, sorted edge lines)."""
    vl = []
    for v in g.vertex_list():
        p = []
        if g.has_prop(v, "source"):
            p.append("exposure")
        if g.has_prop(v, "target"):
            p.append("outcome")
        if g.has_prop(v, "adjustedNode"):
            p.append("adjusted")
        if g.has_prop(v, "latentNode"):
            p.append("latent")
        if g.has_prop(v, "selectedNode"):
            p.append("selected")
        if p:
            vl.append((_quote(v.id) + " [" + ",".join(p) + "]\n", v.id, p))
        elif not v.adjacent():
            vl.append((_quote(v.id) + "\n", v.id, p))
    vl.sort(key=lambda t: t[0])
    el = []
    for e in g.edges:
        a, b = _quote(e.v1.id), _quote(e.v2.id)
        ida, idb = e.v1.id, e.v2.id
        if SYMMETRIC[e.directed] and a > b:
            a, b, ida, idb = b, a, idb, ida
        el.append((a + " " + SYMBOL[e.directed] + " " + b, ida, idb, e.directed))
    el.sort(key=lambda t: t[0])
    return [(vid, p) for _, vid, p in vl], [(a, b, t) for _, a, b, t in el]


_PROP_OF = {"exposure": "source", "outcome": "target", "adjusted": "adjustedNode", "latent": "latentNode",
            "selected": "selectedNode"}


def reparse(g):
    """The graph that JavaScript parses from ``g.toString()``."""
    verts, edges = to_dot(g)
    h = Graph(g.type)
    for vid, p in verts:
        if h.get(vid) is None:
            h.add_vertex(vid)
        for x in p:
            h.add_prop(vid, _PROP_OF[x])
    for a, b, t in edges:
        for vid in (a, b):
            if h.get(vid) is None:
                h.add_vertex(vid)
        h.add_edge(a, b, t)
    return h


def pcalg2dagitty(amat, labels, gtype):
    """pcalg::pcalg2dagitty: the type and the statements of its text."""
    n = len(labels)
    head = {"cpdag": "pdag", "pdag": "pdag", "dag": "dag", "mag": "mag", "pag": "pag"}[gtype]
    g = Graph(head)

    def vert(name):
        if g.get(name) is None:
            g.add_vertex(name)

    def lone(i):
        if (amat[:, i] + amat[i, :]).sum() == 0:
            vert(labels[i])

    for i in range(n - 1):
        lone(i)
        for j in range(i + 1, n):
            a, b = amat[i, j], amat[j, i]
            if head in ("pdag", "dag"):
                st = ((labels[i], "--", labels[j]) if a and b else (labels[i], "<-", labels[j]) if a and not b
                      else (labels[i], "->", labels[j]) if b and not a else None)
            else:
                st = {(1, 1): (labels[i], "@-@", labels[j]), (2, 3): (labels[i], "->", labels[j]),
                      (3, 2): (labels[i], "<-", labels[j]), (3, 3): (labels[i], "--", labels[j]),
                      (2, 2): (labels[i], "<->", labels[j]), (2, 1): (labels[i], "@->", labels[j]),
                      (1, 2): (labels[i], "<-@", labels[j])}.get((int(a), int(b)))
            if st is None:
                continue
            x, op, y = st
            vert(x)
            vert(y)
            if op == "<-":
                g.add_edge(y, x, DIRECTED)
            elif op == "<-@":
                g.add_edge(y, x, PARTDIRECTED)
            else:
                g.add_edge(x, y, {"--": UNDIRECTED, "->": DIRECTED, "<->": BIDIRECTED, "@-@": UNSPECIFIED,
                                  "@->": PARTDIRECTED}[op])
    lone(n - 1)
    return g


# ------------------------------------------------------------ transforms
def induced_subgraph(g, vs):
    gn = Graph()
    for v in vs:
        gn.add_vertex(v.id)
    ids = [v.id for v in vs]
    for e in g.edges:
        if e.v1.id in ids and e.v2.id in ids:
            gn.add_edge(e.v1.id, e.v2.id, e.directed)
    g.copy_props_to(gn)
    return gn


def ancestor_graph(g):
    V = _union(g.sources(), g.targets(), g.adjusted(), g.selected())
    return induced_subgraph(g, g.anteriors_of(V))


def pag_to_pdag(g):
    if g.type != "pag":
        return None
    g = g.clone()
    for e in list(g.edges):
        if e.directed == UNSPECIFIED:
            g.change_edge(e, UNDIRECTED)
        if e.directed == PARTDIRECTED:
            g.change_edge(e, DIRECTED)
        if e.directed == PARTUNDIRECTED:
            g.change_edge(e, UNDIRECTED)
    g.type = "pdag"
    return g


def is_edge_visible(g, e):
    t = g.type
    if t in ("dag", "pdag", "digraph"):
        return True
    if e.directed != DIRECTED:
        return None
    if _difference(_union(e.v1.parents(), e.v1.spouses()), e.v2.adjacent()):
        return True
    vpar = induced_subgraph(g, _union([e.v2], e.v2.parents()))
    vpar = g.get([v.id for v in vpar.district_of([vpar.get(e.v1.id)])])
    if _difference(_union(g.parents_of(vpar), g.spouses_of(vpar)), e.v2.adjacent()):
        return True
    return False


def proper_possible_causal_paths(g, X, Y):
    in_X = {v.id for v in X}
    visited, r = set(), []

    def visit(v):
        if v.id not in visited:
            visited.add(v.id)
            r.append(v)
            if v.id not in in_X:
                for w in _union(v.parents(), v.neighbours()):
                    visit(w)
    for y in Y:
        visit(y)
    return _intersection(r, g.posteriors_of(X))


def dpcp(g, X=None, Y=None):
    X = g.sources() if X is None else X
    Y = g.targets() if Y is None else Y
    if g.type == "pag":
        gn = pag_to_pdag(g)
        gn.type = "pag"
    else:
        gn = g
    X = gn.get([v.id for v in X])
    Y = gn.get([v.id for v in Y])
    return g.get([v.id for v in gn.posteriors_of(_difference(proper_possible_causal_paths(gn, X, Y), X))])


def back_door_graph(g):
    gtarget = g.clone()
    if g.type == "pag":
        gback = pag_to_pdag(g)
        gback.type = "pag"
    else:
        gback = gtarget
    X, Y = g.sources(), g.targets()
    if not X or not Y:
        return gback
    X = gback.get([v.id for v in X])
    Y = gback.get([v.id for v in Y])
    pcp = proper_possible_causal_paths(gback, X, Y)
    for s in X:
        for c in _intersection(pcp, s.children()):
            if is_edge_visible(gback, gback.get_edge(s.id, c.id)):
                gtarget.delete_edge(gtarget.get(s.id), gtarget.get(c.id), DIRECTED)
    return gtarget


def connected_components(g, kin="neighbours"):
    visited = set()

    def comp(v):
        q, r = [v], [v]
        while q:
            v2 = q.pop()
            visited.add(v2.id)
            for w in getattr(v2, kin)():
                if w.id not in visited:
                    q.append(w)
                    r.append(w)
                    visited.add(w.id)
        return r
    return [comp(v) for v in g.vertex_list() if v.id not in visited]


def moral_graph(g):
    if g.type == "pag":
        g = pag_to_pdag(g)
    mg = Graph()
    for v in g.vertex_list():
        mg.add_vertex(v.id)
    for c in connected_components(g, "spouses"):
        cp = g.parents_of(c) + c
        for i in range(len(cp)):
            for j in range(i + 1, len(cp)):
                mg.add_edge(cp[i].id, cp[j].id, UNDIRECTED)
    for e in g.edges:
        if e.directed == UNDIRECTED:
            mg.add_edge(e.v1.id, e.v2.id, e.directed)
    g.copy_props_to(mg)
    mg.type = "graph"
    return mg


def cg_to_rcg(g):
    gn = Graph(g.type)
    for v in g.vertex_list():
        gn.add_vertex(v.id)
    fail = [False]

    def connected(v, w):
        return any(x.id == w.id for x in v.adjacent())

    def process(a, b):
        if fail[0]:
            return
        for c in b.neighbours():
            if a.id != c.id and not connected(a, c):
                for d in gn.get(c.id).parents():
                    if a.id != d.id and b.id != d.id and not connected(b, d):
                        fail[0] = True
                if fail[0]:
                    return
                if gn.get_edge(b.id, c.id, DIRECTED) is None:
                    gn.add_edge(b.id, c.id, DIRECTED)
                    process(b, c)
    for e in g.edges:
        if e.directed == DIRECTED:
            gn.add_edge(e.v1.id, e.v2.id, DIRECTED)
    for e in list(g.edges):
        if e.directed == DIRECTED:
            process(e.v1, e.v2)
    if fail[0]:
        return None
    for e in g.edges:
        if e.directed != DIRECTED and not connected(gn.get(e.v1.id), e.v2):
            gn.add_edge(e.v1.id, e.v2.id, e.directed)
    g.copy_props_to(gn)
    return gn


# ------------------------------------------------------------ analysis
def d_connected(g, X, Y, Z):
    if g.type == "pag":
        g = pag_to_pdag(g)
    X = [g.get(v) for v in X]
    Y = [g.get(v) for v in Y]
    Z = [g.get(v) for v in Z]
    AnZ = g.ancestors_of(Z)
    fq, bq = [], list(X)
    fv, bv = set(), set()
    Yi, Zi, Ai = {v.id for v in Y}, {v.id for v in Z}, {v.id for v in AnZ}
    while fq or bq:
        if fq:
            v = fq.pop()
            fv.add(v.id)
            if v.id in Yi:
                return True
            if v.id in Ai:
                for w in v.parents():
                    if w.id not in bv:
                        bq.append(w)
                for w in v.spouses():
                    if w.id not in fv:
                        fq.append(w)
            if v.id not in Zi:
                for w in _union(v.children(), v.neighbours()):
                    if w.id not in fv:
                        fq.append(w)
        if bq:
            v = bq.pop()
            bv.add(v.id)
            if v.id in Yi:
                return True
            if v.id in Zi:
                continue
            for w in _union(v.children(), v.spouses()):
                if w.id not in fv:
                    fq.append(w)
            for w in _union(v.parents(), v.neighbours()):
                if w.id not in bv:
                    bq.append(w)
    return False


def is_adjustment_set(g, Z):
    if g.type not in ("dag", "pdag", "mag", "pag"):
        raise ValueError(g.type)
    if not g.sources() or not g.targets():
        return False
    Zg = [g.get(v) for v in Z]
    if _intersection(dpcp(g), Zg):
        return False
    gbd = back_door_graph(g)
    Zb = [gbd.get(v) for v in Zg]
    return not d_connected(gbd, gbd.sources(), gbd.targets(), Zb)


def canonical_adjustment_set(g):
    Z = _difference(g.anteriors_of(_union(g.sources(), g.targets())),
                    _union(g.latent(), g.sources(), g.targets(), dpcp(g)))
    return [Z] if is_adjustment_set(g, Z) else []


def connected_component_avoiding(g, V, U):
    visited = {u.id for u in U}
    r, q = [], []
    for v in V:
        visited.add(v.id)
        r.append(v)
        q.append(v)
    while q:
        for w in q.pop().neighbours():
            if w.id not in visited:
                visited.add(w.id)
                q.append(w)
                r.append(w)
    return r


def list_minimal_separators(g, must=(), must_not=(), max_nr=float("inf")):
    if not g.sources() or not g.targets():
        return []
    Uh = Hash()
    for v in g.targets():
        Uh.set(v.id, v)
    for v in g.neighbours_of(g.targets()):
        Uh.set(v.id, v)
    U = Uh.values()
    R = []
    must = [g.get(v.id) for v in must]
    must_not = [g.get(v.id) for v in must_not]

    def real_n(V):
        g.clear_visited()
        for v in V + must:
            v.visited = True
        r, q = [], list(V)
        while q:
            for w in q.pop().neighbours():
                if not w.visited:
                    w.visited = True
                    if _contains(must_not, w):
                        q.append(w)
                    else:
                        r.append(w)
        return r

    def near_separator(A, b):
        NA = real_n(A)
        C = connected_component_avoiding(g, b, NA + must)
        return real_n(C)

    def list_min_sep(A, U):
        if len(R) >= max_nr:
            return
        SA = near_separator(A, g.targets())
        Astar = connected_component_avoiding(g, g.sources(), SA + must)
        if not _intersection(Astar, U):
            NA = _difference(real_n(Astar), U)
            if NA:
                v = NA[0]
                Astar.append(v)
                list_min_sep(Astar, U)
                Astar.pop()
                U.append(v)
                list_min_sep(Astar, U)
                U.pop()
            else:
                R.append(SA + must)
    list_min_sep(g.sources(), U)
    return R


def list_msas_total_effect(g):
    if g.type not in ("dag", "pdag", "mag", "pag"):
        raise ValueError(g.type)
    if not g.sources() or not g.targets():
        return []
    gg = g
    if g.type == "pdag":
        gg = cg_to_rcg(g)
        if gg is None:                 # dagitty: an error in JavaScript (null graph)
            raise ValueError("dagitty: the PDAG has no consistent extension (cgToRcg failed)")
    if any(gg.has_prop(v, "adjustedNode") for v in dpcp(gg)):
        return []
    gbd = back_door_graph(gg)
    gam = moral_graph(ancestor_graph(gbd))
    adjusted = [gam.get(v.id) for v in gg.adjusted()]
    latent = [x for x in (gam.get(v.id) for v in gg.latent() + dpcp(gg)) if x is not None]
    r = list_minimal_separators(gam, adjusted, latent)
    return [[g.get(v.id) for v in s] for s in r]


def _sorted_ids(sets):
    """DagittyR.adj2r: the ids of each set, sorted as JavaScript strings."""
    return [sorted(v.id for v in s) for s in sets]


# ------------------------------------------------------------ ETIA's call
def from_etia(M, names, graph_type, exposures, outcomes):
    """The dagitty graph of ETIA's find_adjset (after R's text round trips)."""
    import numpy as np
    M = np.asarray(M)
    if graph_type in ("dag", "cpdag"):
        pc = np.zeros(M.shape, dtype=int)
        pc[M == 1] = 1
        pc[M == 2] = 1
        amat = pc.T
    else:
        amat = M.astype(int)
    labels = [_r_name(n) for n in names]
    g = reparse(pcalg2dagitty(amat, labels, graph_type))
    for x in exposures:
        g.add_prop(x, "source")
    g = reparse(g)
    for y in outcomes:
        g.add_prop(y, "target")
    return reparse(g)


def _r_name(n):
    """R's make.names (read.table with header=TRUE changes ':' and the like to '.')."""
    s = re.sub(r"[^0-9A-Za-z._]", ".", str(n))
    if not re.match(r"^([A-Za-z]|\.(?![0-9]))", s):
        s = "X" + s
    return s


def etia_find_adjset(M, names, graph_type, target_name, exposure_names):
    """ETIA's find_adjset, as it runs when R runs the script: returns
    (canonical set or None, first minimal set or None), with variable names."""
    exposures = [_r_name(x.replace(":", ".")) for x in exposure_names]
    outcomes = [_r_name(y) for y in target_name]
    g = from_etia(M, names, graph_type, exposures, outcomes)
    back = {_r_name(n): str(n) for n in names}
    can = _sorted_ids(canonical_adjustment_set(g))
    mini = _sorted_ids(list_msas_total_effect(g))
    fix = lambda s: [back.get(v, v).replace(".", ":") if ":" in back.get(v, v) else back.get(v, v) for v in s]
    return (fix(can[0]) if can else None), (fix(mini[0]) if mini else None)
