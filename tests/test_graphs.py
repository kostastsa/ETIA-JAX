import numpy as np
import pytest

from etia_jax.graphs.encoding import adj_to_marks, marks_to_adj, check_marks, ARROW, TAIL
from etia_jax.graphs.dag import random_dag, is_dag, ancestors_matrix, topological_order
from etia_jax.graphs.separation import d_separated
from etia_jax.graphs.convert import (dag_to_cpdag, dag_to_mag, mag_to_pag, dag_to_pag,
                                     mag_adjacency_by_dsep)
from etia_jax.graphs.metrics import shd, adjacency_scores


def test_random_dag_is_acyclic():
    for s in range(20):
        A = random_dag(15, 3.0, s)
        assert is_dag(A)
        order = topological_order(A)
        pos = {v: i for i, v in enumerate(order)}
        for i, j in zip(*np.nonzero(A)):
            assert pos[i] < pos[j]


def test_marks_roundtrip():
    A = random_dag(10, 2.0, 1)
    G = adj_to_marks(A)
    check_marks(G)
    assert (marks_to_adj(G) == A).all()


def test_chain_fork_collider():
    # 0 -> 1 -> 2, 1 -> 3 <- 4
    A = np.zeros((5, 5), int); A[0, 1] = A[1, 2] = A[1, 3] = A[4, 3] = 1
    assert not d_separated(A, 0, 2)
    assert d_separated(A, 0, 2, [1])
    assert d_separated(A, 0, 4)
    assert not d_separated(A, 0, 4, [3])
    assert not d_separated(A, 2, 4, [3])
    assert d_separated(A, 2, 4, [3, 1])


def test_dsep_matches_tetrad(graph_fixtures):
    n = 0
    for fx in graph_fixtures:
        for row in fx["dsep"]:
            x, y, sep = int(row[0]), int(row[1]), bool(row[2])
            Z = list(np.flatnonzero(row[3:]))
            assert d_separated(fx["A"], x, y, Z) == sep
            n += 1
    assert n > 1000


def test_cpdag_matches_tetrad(graph_fixtures):
    for g, fx in enumerate(graph_fixtures):
        G = dag_to_cpdag(fx["A"])
        check_marks(G)
        assert (G == fx["cpdag"]).all(), f"graph {g}: SHD {shd(fx['cpdag'], G)}"


def test_mag_matches_tetrad(graph_fixtures):
    for g, fx in enumerate(graph_fixtures):
        M = dag_to_mag(fx["A"], fx["latent"])
        check_marks(M)
        assert (M == fx["mag"]).all(), f"graph {g}: SHD {shd(fx['mag'], M)}"


def test_mag_adjacency_by_dsep(graph_fixtures):
    for fx in graph_fixtures:
        M = dag_to_mag(fx["A"], fx["latent"])
        assert ((M != 0) == mag_adjacency_by_dsep(fx["A"], fx["latent"])).all()



def test_pag_without_latents_agrees_with_cpdag(graph_fixtures):
    # Without latents every arrowhead/tail in the PAG is invariant, so a
    # directed PAG edge must be directed the same way in the CPDAG.
    for fx in graph_fixtures:
        P = dag_to_pag(fx["A"])
        C = dag_to_cpdag(fx["A"])
        d = (P == ARROW) & (P.T == TAIL)
        assert ((C == ARROW) & (C.T == TAIL))[d].all()


def test_metrics():
    A = random_dag(8, 2.0, 3)
    G = adj_to_marks(A)
    assert shd(G, G) == 0
    assert adjacency_scores(G, G)["f1"] == 1.0
    G2 = G.copy(); i, j = map(int, np.argwhere(A)[0]); G2[i, j] = G2[j, i] = 0
    assert shd(G, G2) == 1


# Fixture graphs where Tetrad's DagToPag is not exact. The slow test
# test_pag_exact_where_tetrad_differs shows that each difference is a Tetrad error.
TETRAD_INEXACT = {3, 11, 17, 24, 25, 29, 33, 45, 48, 54, 55, 58, 59}


def _check_exact(fx, g):
    from bruteforce import exact_pag
    M = dag_to_mag(fx["A"], fx["latent"])
    P = mag_to_pag(M)
    E, n_members = exact_pag(M, P)
    assert n_members >= 1
    assert (E == P).all(), f"graph {g}: SHD {shd(E, P)}"


def test_pag_exact_bruteforce_small(graph_fixtures):
    """Our PAG equals the exact PAG on the small fixture graphs."""
    n = 0
    for g, fx in enumerate(graph_fixtures):
        P = dag_to_pag(fx["A"], fx["latent"])
        if len(P) <= 7 and (P == 1).sum() <= 10:
            _check_exact(fx, g)
            n += 1
    assert n >= 10


@pytest.mark.slow
def test_pag_exact_where_tetrad_differs(graph_fixtures):
    """Each difference from Tetrad is a Tetrad error:

    * our circle, Tetrad tail: a witness MAG in the class has no tail there;
    * our arrowhead, Tetrad circle: an exact search (with that mark free)
      finds an arrowhead in every MAG of the class.
    """
    from bruteforce import exact_pag, find_circle_witness
    n_tails = n_arrows = 0
    for g in sorted(TETRAD_INEXACT):
        fx = graph_fixtures[g]
        M = dag_to_mag(fx["A"], fx["latent"])
        P, T = mag_to_pag(M), fx["pag"]
        assert ((P != T) <= (((P == 1) & (T == 3)) | ((P == 2) & (T == 1)))).all()
        for i, j in np.argwhere((P == 1) & (T == 3)):
            assert find_circle_witness(M, P, int(j), int(i)) is not None, (g, i, j)
            n_tails += 1
        for i, j in np.argwhere((P == 2) & (T == 1)):
            guess = P.copy(); guess[i, j] = 1
            E, _ = exact_pag(M, guess)
            assert E[i, j] == 2 and (E == P).all(), (g, i, j)
            n_arrows += 1
    assert (n_tails, n_arrows) == (29, 3)


def test_pag_matches_tetrad_elsewhere(graph_fixtures):
    for g, fx in enumerate(graph_fixtures):
        P = dag_to_pag(fx["A"], fx["latent"])
        same = (P == fx["pag"]).all()
        assert same == (g not in TETRAD_INEXACT), f"graph {g}"
