"""Markov boundaries, and one DAG or MAG of a CPDAG or PAG (increment 9).

The OCT takes the Markov boundary of each node from the graph of each
configuration. The Markov boundary is the same for all the graphs of one
Markov equivalence class (Biza et al. 2020, p. 6), so the default mode takes
it from one fixed member of the class:

* ``dag_from_cpdag``: the extension of Dor and Tarsi (1992): remove, one by
  one, a node that has no children and whose undirected neighbors are
  adjacent to all its other neighbors; orient its undirected edges into it.
* ``mag_from_pag``: Zhang (2008), Theorem 2: ``o->`` becomes ``->``, and the
  circle component becomes a DAG without new unshielded colliders.
* ``markov_boundary``: the nodes that a collider path connects to the target
  (parents, children, the parents of the children; in a MAG also the
  districts of the target and of its children, and their parents).

``etia_markov_boundary`` is ETIA's function
(``CRV/causal_graph_utils/markov_boundary.py``) with its behavior: a circle
mark raises ``ValueError``, and an undirected edge (tail at both ends) is
not a parent and not a child, so ETIA drops this neighbor.
"""
import numpy as np

from .encoding import NONE, CIRCLE, ARROW, TAIL


def _is_parent(G, a, b):
    return G[a, b] == ARROW and G[b, a] == TAIL


def dag_from_cpdag(G):
    """A DAG (mark matrix) of the CPDAG ``G``. If ``G`` has no consistent
    extension (an estimated graph can have cycles), the remaining
    undirected edges go from the lower to the higher index."""
    G = np.array(G, dtype=np.int8)
    p = len(G)
    left = set(range(p))
    und = lambda a, b: G[a, b] == TAIL and G[b, a] == TAIL
    while left:
        pick = None
        for x in sorted(left):
            if any(_is_parent(G, x, y) for y in left if y != x):
                continue
            nb = [y for y in left if y != x and G[x, y] != NONE]
            un = [y for y in nb if und(x, y)]
            if all(G[y, z] != NONE for y in un for z in nb if z != y):
                pick = x
                break
        if pick is None:
            break
        for y in left:
            if y != pick and und(pick, y):
                G[y, pick], G[pick, y] = ARROW, TAIL
        left.discard(pick)
    for a in range(p):
        for b in range(a + 1, p):
            if und(a, b):
                G[a, b], G[b, a] = ARROW, TAIL
    return G


def mag_from_pag(G):
    """A MAG of the PAG ``G`` (Zhang 2008, Theorem 2)."""
    G = np.array(G, dtype=np.int8)
    p = len(G)
    cc = np.zeros_like(G)
    for a in range(p):
        for b in range(p):
            if G[a, b] == CIRCLE and G[b, a] == CIRCLE:
                cc[a, b] = TAIL
    D = dag_from_cpdag(cc)
    for a in range(p):
        for b in range(p):
            if D[a, b] != NONE:
                G[a, b] = D[a, b]
    for a in range(p):
        for b in range(p):
            if G[a, b] == ARROW and G[b, a] == CIRCLE:      # a o-> b
                G[b, a] = TAIL
            elif G[a, b] == CIRCLE and G[b, a] == TAIL:     # a --o b
                G[a, b] = ARROW
    return G


def member(G):
    """One DAG or MAG of the class of ``G``: a PAG (it has circles) goes
    through ``mag_from_pag``, a graph with undirected edges through
    ``dag_from_cpdag``."""
    G = np.asarray(G)
    if (G == CIRCLE).any():
        return mag_from_pag(G)
    if ((G == TAIL) & (G.T == TAIL)).any():
        return dag_from_cpdag(G)
    return np.array(G, dtype=np.int8)


def markov_boundary(G, t):
    """The Markov boundary of ``t`` in the DAG or MAG ``G``: the nodes that a
    collider path connects to ``t`` (every inner node of the path is a
    collider). A CPDAG or PAG is first replaced by ``member(G)``."""
    G = member(G)
    p = len(G)
    out = set()
    # state: (node, came with an arrowhead into node)
    seen = set()
    stack = []
    for v in range(p):
        if G[t, v] != NONE:
            out.add(v)
            if G[t, v] == ARROW:
                stack.append(v)
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        # b is an inner node: it must be a collider, so the next edge has an arrowhead at b
        for c in range(p):
            if c == t or G[b, c] == NONE or G[c, b] != ARROW:
                continue
            out.add(c)
            if G[b, c] == ARROW:            # b <-> c: c can be the next inner collider
                stack.append(c)
    out.discard(t)
    return sorted(out)


# ---------------------------------------------------------------- ETIA
def _etia_bidirected_path(i, M):
    nb = np.where(np.logical_and(M[i, :] == 2, M[:, i] == 2))[0].tolist()
    if len(nb) == 0:
        return None
    out = list(nb)
    M[i, :] = 0
    M[:, i] = 0
    for j in nb:
        nxt = _etia_bidirected_path(j, M)
        if nxt:
            out = out + nxt
    return out


def etia_markov_boundary(t, M):
    """ETIA's ``markov_boundary(target, matrix)``. Returns the nodes in the
    order of ETIA's ``list(set(...))`` (sorted for small integers)."""
    M = np.asarray(M)
    if np.where(M == 1)[0].size > 0:
        raise ValueError("cannot find MB due to an undirected edge (need DAG or MAG)")
    par = lambda v: np.where(np.logical_and(M[v, :] == 3, M[:, v] == 2))[0].tolist()
    parents = par(t)
    children = np.where(np.logical_and(M[t, :] == 2, M[:, t] == 3))[0].tolist()
    poc = []
    for ch in children:
        poc += par(ch)
    dist_i = _etia_bidirected_path(t, np.copy(M))
    dist_ch = []
    for ch in children:
        d = _etia_bidirected_path(ch, np.copy(M))
        if d:
            dist_ch += d
    pod_i = []
    if dist_i:
        for di in dist_i:
            pod_i += par(di)
    else:
        dist_i = []
    pod_ch = []
    for dc in dist_ch:
        pod_ch += par(dc)
    mb = list(set(parents + children + poc + dist_i + dist_ch + pod_i + pod_ch))
    if t in mb:
        mb.remove(t)
    return mb
