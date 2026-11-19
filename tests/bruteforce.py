"""Exact (exponential) references for small graphs."""
import itertools
import numpy as np
from etia_jax.graphs.encoding import NONE, CIRCLE, ARROW, TAIL
from etia_jax.graphs.dag import is_dag, ancestors_matrix
from etia_jax.graphs.separation import d_separated


def mag_as_dag(M):
    """DAG with one latent parent per bidirected edge (same m-separations)."""
    p = len(M)
    bi = [(i, j) for i in range(p) for j in range(i + 1, p) if M[i, j] == ARROW and M[j, i] == ARROW]
    A = np.zeros((p + len(bi), p + len(bi)), dtype=np.int8)
    A[:p, :p] = (M == ARROW) & (M.T == TAIL)
    for k, (i, j) in enumerate(bi):
        A[p + k, i] = A[p + k, j] = 1
    return A


def is_ancestral(M):
    D = (M == ARROW) & (M.T == TAIL)
    if not is_dag(D):
        return False
    anc = ancestors_matrix(D)
    bi = (M == ARROW) & (M.T == ARROW)
    return not (bi & (anc | anc.T)).any()


def _queries(p):
    for x, y in itertools.combinations(range(p), 2):
        rest = [k for k in range(p) if k not in (x, y)]
        for r in range(len(rest) + 1):
            for Z in itertools.combinations(rest, r):
                yield x, y, list(Z)


def msep_signature(M):
    A = mag_as_dag(M)
    return [d_separated(A, x, y, Z) for x, y, Z in _queries(len(M))]


def same_mseps(M, queries, target):
    A = mag_as_dag(M)
    return all(d_separated(A, x, y, Z) == t for (x, y, Z), t in zip(queries, target))


def exact_pag(M, pag_guess):
    """Marks shared by all MAGs equivalent to ``M`` that agree with the
    non-circle marks of ``pag_guess`` (which must be sound)."""
    queries = list(_queries(len(M)))
    target = msep_signature(M)
    circles = [tuple(ij) for ij in np.argwhere(pag_guess == CIRCLE)]
    members = []
    for marks in itertools.product((ARROW, TAIL), repeat=len(circles)):
        C = pag_guess.copy()
        for (i, j), m in zip(circles, marks):
            C[i, j] = m
        if ((C == TAIL) & (C.T == TAIL)).any():  # no undirected edges
            continue
        if is_ancestral(C) and same_mseps(C, queries, target):
            members.append(C)
    P = members[0].copy()
    for C in members[1:]:
        P[C != P] = CIRCLE
    return P, len(members)


def find_circle_witness(M, P, a, b, max_circle_edges=16):
    """Search a MAG equivalent to ``M``, consistent with ``P``, in which ``a``
    is not an ancestor of ``b`` (so the circle at ``a`` on ``a o-> b`` is
    correct). Tries each set of ``o->`` edges out of ``a`` or into ``b`` to
    make bidirected, and every
    orientation of the circle component. Returns the MAG or None."""
    queries = list(_queries(len(M)))
    target = msep_signature(M)
    oa = (P == ARROW) & (P.T == CIRCLE)         # x o-> y as (x, y): P[x, y] = ARROW
    others = [(int(x), b) for x in np.flatnonzero(oa[:, b]) if x != a]
    others += [(a, int(y)) for y in np.flatnonzero(oa[a]) if y != b]
    for r in range(len(others) + 1):
        for sub in itertools.combinations(others, r):
            G = P.copy()
            for x, y in ((a, b), *sub):
                G[y, x] = ARROW
            G[((G == ARROW) & (G.T == CIRCLE)).T] = TAIL
            U = np.argwhere(np.triu((G == CIRCLE) & (G.T == CIRCLE), 1))
            if len(U) > max_circle_edges:
                raise ValueError("circle component too large")
            for bits in itertools.product((0, 1), repeat=len(U)):
                W = G.copy()
                for (i, j), t in zip(U, bits):
                    x, y = (i, j) if t else (j, i)
                    W[x, y], W[y, x] = ARROW, TAIL
                D = (W == ARROW) & (W.T == TAIL)
                if not is_ancestral(W) or ancestors_matrix(D)[a, b]:
                    continue
                if same_mseps(W, queries, target):
                    return W
    return None
