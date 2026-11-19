"""Basic DAG utilities on 0/1 adjacency matrices (``A[i, j] = 1`` iff ``i -> j``)."""
import numpy as np


def random_dag(n_nodes, expected_degree=2.0, rng=None, max_parents=None):
    """Erdos-Renyi DAG in a random causal order. Returns an int8 adjacency."""
    rng = np.random.default_rng(rng)
    p = min(1.0, expected_degree / max(n_nodes - 1, 1))
    A = np.triu(rng.random((n_nodes, n_nodes)) < p, k=1)
    if max_parents is not None:
        for j in range(n_nodes):
            pa = np.flatnonzero(A[:, j])
            if len(pa) > max_parents:
                A[rng.choice(pa, len(pa) - max_parents, replace=False), j] = False
    perm = rng.permutation(n_nodes)
    return A[np.ix_(perm, perm)].astype(np.int8)


def topological_order(A):
    A = np.asarray(A) != 0
    indeg = A.sum(0).astype(int)
    order, stack = [], [i for i in range(len(A)) if indeg[i] == 0]
    while stack:
        i = stack.pop()
        order.append(i)
        for j in np.flatnonzero(A[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    if len(order) != len(A):
        raise ValueError("graph has a directed cycle")
    return order


def is_dag(A):
    try:
        topological_order(A)
        return True
    except ValueError:
        return False


def ancestors_matrix(A):
    """``anc[i, j]`` is True iff ``i`` is an ancestor of ``j`` (i == j counts)."""
    A = np.asarray(A) != 0
    R = A | np.eye(len(A), dtype=bool)
    while True:
        R2 = R | ((R.astype(np.int32) @ R.astype(np.int32)) > 0)
        if (R2 == R).all():
            return R
        R = R2


def ancestors_of(A, nodes):
    """Set of ancestors of ``nodes`` (inclusive) as a boolean mask."""
    A = np.asarray(A) != 0
    mask = np.zeros(len(A), dtype=bool)
    stack = list(nodes)
    while stack:
        v = stack.pop()
        if mask[v]:
            continue
        mask[v] = True
        stack.extend(np.flatnonzero(A[:, v]))
    return mask
