"""Edge-mark encoding shared with ETIA and Tetrad.

``G[i, j]`` is the mark at ``j`` on the edge between ``i`` and ``j``:

* ``NONE = 0``   no edge
* ``CIRCLE = 1`` ``i *-o j``
* ``ARROW = 2``  ``i *-> j``
* ``TAIL = 3``   ``i *-- j``

So ``i -> j`` is ``G[i, j] = ARROW, G[j, i] = TAIL``; ``i <-> j`` is ARROW/ARROW;
``i --- j`` (undirected CPDAG edge) is TAIL/TAIL; ``i o-> j`` is ARROW at j and
CIRCLE at i.
"""
import numpy as np

NONE, CIRCLE, ARROW, TAIL = 0, 1, 2, 3


def adj_to_marks(A):
    """0/1 DAG adjacency (``A[i, j] = 1`` iff ``i -> j``) to a mark matrix."""
    A = np.asarray(A) != 0
    G = np.zeros(A.shape, dtype=np.int8)
    G[A] = ARROW
    G[A.T] = TAIL
    return G


def marks_to_adj(G):
    """Mark matrix of a DAG to its 0/1 adjacency (``i -> j``)."""
    G = np.asarray(G)
    return ((G == ARROW) & (G.T == TAIL)).astype(np.int8)


def skeleton(G):
    G = np.asarray(G)
    return (G != NONE) | (G.T != NONE)


def is_directed(G, i, j):
    return G[i, j] == ARROW and G[j, i] == TAIL


def is_parent(G, i, j):
    """``i -> j``."""
    return G[i, j] == ARROW and G[j, i] == TAIL


def check_marks(G):
    """Raise if ``G`` is not a valid symmetric mark matrix."""
    G = np.asarray(G)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("graph matrix must be square")
    if not np.isin(G, [NONE, CIRCLE, ARROW, TAIL]).all():
        raise ValueError("unknown edge mark")
    if np.any(np.diag(G) != 0):
        raise ValueError("self loops are not allowed")
    if np.any((G == 0) != (G.T == 0)):
        raise ValueError("edge marks must be present at both ends")
