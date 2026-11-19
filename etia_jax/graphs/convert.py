"""Conversions between DAG, CPDAG, MAG and PAG (mark matrices)."""
import numpy as np
from .encoding import NONE, CIRCLE, ARROW, TAIL, adj_to_marks
from .dag import ancestors_matrix
from .orientation import meek_rules, fci_orient, unshielded_triples
from .separation import d_separated


def dag_to_cpdag(A):
    """CPDAG of DAG ``A`` (0/1 adjacency). Undirected edges are TAIL/TAIL."""
    A = np.asarray(A) != 0
    S = A | A.T
    G = np.where(S, TAIL, NONE).astype(np.int8)
    for a, b, c in unshielded_triples(G):
        if A[a, b] and A[c, b]:
            G[a, b] = G[c, b] = ARROW
            G[b, a] = G[b, c] = TAIL
    meek_rules(G)
    return G


def has_inducing_path(A, x, y, is_latent, anc=None):
    """True iff DAG ``A`` has an inducing path between x and y relative to latents.

    Every non-endpoint is latent or a collider, and every collider is an
    ancestor of x or y (Richardson & Spirtes 2002). Search over (prev, node).
    """
    A = np.asarray(A) != 0
    n = len(A)
    anc = ancestors_matrix(A) if anc is None else anc
    S = A | A.T
    seen = set()
    stack = [(x, v) for v in np.flatnonzero(S[x])]
    while stack:
        u, v = stack.pop()
        if v == y:
            return True
        if (u, v) in seen:
            continue
        seen.add((u, v))
        for w in np.flatnonzero(S[v]):
            if w == u or w == x:
                continue
            collider = A[u, v] and A[w, v]
            if collider:
                ok = anc[v, x] or anc[v, y]
            else:
                ok = is_latent[v]
            if ok:
                stack.append((v, int(w)))
    return False


def dag_to_mag(A, is_latent=None):
    """MAG over the observed nodes of DAG ``A`` (latents marginalized).

    Returns the mark matrix over observed nodes, in their original order.
    """
    A = np.asarray(A) != 0
    n = len(A)
    is_latent = np.zeros(n, dtype=bool) if is_latent is None else np.asarray(is_latent, dtype=bool)
    anc = ancestors_matrix(A)
    obs = np.flatnonzero(~is_latent)
    M = np.zeros((n, n), dtype=np.int8)
    for i, x in enumerate(obs):
        for y in obs[i + 1:]:
            if not (A[x, y] or A[y, x] or has_inducing_path(A, x, y, is_latent, anc)):
                continue
            if anc[x, y]:
                M[x, y], M[y, x] = ARROW, TAIL
            elif anc[y, x]:
                M[y, x], M[x, y] = ARROW, TAIL
            else:
                M[x, y] = M[y, x] = ARROW
    return M[np.ix_(obs, obs)]


def mag_to_pag(M, complete=True):
    """PAG (Markov equivalence class) of MAG ``M``."""
    M = np.asarray(M)
    G = np.where((M != NONE) | (M.T != NONE), CIRCLE, NONE).astype(np.int8)
    for a, b, c in unshielded_triples(G):
        if M[a, b] == ARROW and M[c, b] == ARROW:
            G[a, b] = G[c, b] = ARROW

    def is_noncollider(d, a, b, c):
        return not (M[a, b] == ARROW and M[c, b] == ARROW)

    fci_orient(G, is_noncollider, complete=complete)
    return G


def dag_to_pag(A, is_latent=None, complete=True):
    """PAG of the MAG of DAG ``A`` with latents ``is_latent``."""
    return mag_to_pag(dag_to_mag(A, is_latent), complete=complete)


def mag_adjacency_by_dsep(A, is_latent):
    """Reference check: x, y adjacent in the MAG iff not d-separated given
    every subset; it is enough to test Z = (An({x, y}) & observed) - {x, y}."""
    A = np.asarray(A) != 0
    is_latent = np.asarray(is_latent, dtype=bool)
    anc = ancestors_matrix(A)
    obs = np.flatnonzero(~is_latent)
    k = len(obs)
    S = np.zeros((k, k), dtype=bool)
    for i in range(k):
        for j in range(i + 1, k):
            x, y = obs[i], obs[j]
            Z = [int(z) for z in obs if z not in (x, y) and (anc[z, x] or anc[z, y])]
            S[i, j] = S[j, i] = not d_separated(A, int(x), int(y), Z)
    return S
