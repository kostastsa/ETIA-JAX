"""Path queries on a graph of the class (increment 10, ETIA ``CRV/queries``).

The mark matrix is ETIA's: ``M[i, j]`` is the mark at ``j`` on the edge
``i - j`` (1 circle, 2 arrow, 3 tail).

``etia_compat=True`` copies ETIA's recursive functions: a depth-first search
that visits the neighbours in index order and returns the first path that
it finds. It returns ``[]`` when the start node has no neighbour of the
kind and ``None`` when no path exists. The search keeps only the current
path, so it can visit a node many times (exponential time in dense
graphs). ``one_potentially_directed_path`` of ETIA follows ``o->``, ``->``
and ``o-o`` edges; it does not follow ``--o`` or an undirected CPDAG edge
``---``.

Default: a breadth-first search (a shortest path, each node visited once)
that returns the path or ``None``. A potentially directed path (Zhang 2008)
has no edge into its earlier node and no edge out of its later node, so it
also follows ``--o``; an undirected CPDAG edge counts as a possible
direction.
"""
from collections import deque

import numpy as np


def _index(names, v):
    if names is None or isinstance(v, (int, np.integer)):
        return int(v)
    return list(names).index(v)


# --------------------------------------------------------------- ETIA
def _etia_search(M, start, end, step, path_=()):
    path_ = list(path_) + [start]
    if start == end:
        return path_
    nb = np.where(step(M, start))[0].tolist()
    if len(nb) == 0:
        return []
    if end in nb:
        return _etia_search(M, end, end, step, path_)
    for node in nb:
        if node not in path_:
            p = _etia_search(M, node, end, step, path_)
            if p:
                return p
    return None


_ETIA_STEP = {
    "directed": lambda M, s: np.logical_and(M[s, :] == 2, M[:, s] == 3),
    "potentially_directed": lambda M, s: np.logical_or(np.logical_or(
        np.logical_and(M[s, :] == 2, M[:, s] == 1), np.logical_and(M[s, :] == 2, M[:, s] == 3)),
        np.logical_and(M[s, :] == 1, M[:, s] == 1)),
    "any": lambda M, s: np.logical_and(M[s, :] != 0, M[:, s] != 0),
    "bidirected": lambda M, s: np.logical_and(M[s, :] == 2, M[:, s] == 2),
}


# --------------------------------------------------------------- default
def _step(kind, M, s):
    to, back = M[s, :], M[:, s]            # mark at the next node, mark at s
    if kind == "directed":
        return (to == 2) & (back == 3)
    if kind == "potentially_directed":
        und = (to == 3) & (back == 3)
        return (to != 0) & (back != 2) & ((to != 3) | und)
    if kind == "any":
        return (to != 0) & (back != 0)
    if kind == "bidirected":
        return (to == 2) & (back == 2)
    raise ValueError(kind)


def _bfs(M, start, end, kind):
    prev = {start: None}
    q = deque([start])
    while q:
        v = q.popleft()
        if v == end:
            out = []
            while v is not None:
                out.append(v)
                v = prev[v]
            return out[::-1]
        for w in np.flatnonzero(_step(kind, M, v)).tolist():
            if w not in prev:
                prev[w] = v
                q.append(w)
    return None


def _query(kind, M, start, end, names, etia_compat):
    M = np.asarray(M)
    s, t = _index(names, start), _index(names, end)
    p = _etia_search(M, s, t, _ETIA_STEP[kind]) if etia_compat else _bfs(M, s, t, kind)
    if p and names is not None and not isinstance(start, (int, np.integer)):
        return [list(names)[i] for i in p]
    return p


def one_directed_path(M, start, end, names=None, etia_compat=False):
    """A directed path start -> ... -> end."""
    return _query("directed", M, start, end, names, etia_compat)


def one_potentially_directed_path(M, start, end, names=None, etia_compat=False):
    """A potentially directed path from start to end."""
    return _query("potentially_directed", M, start, end, names, etia_compat)


def one_path_anytype(M, start, end, names=None, etia_compat=False):
    """A path of any edges between start and end."""
    return _query("any", M, start, end, names, etia_compat)


def one_bidirected_path_from_to(M, start, end, names=None, etia_compat=False):
    """A path of bidirected edges between start and end."""
    return _query("bidirected", M, start, end, names, etia_compat)
