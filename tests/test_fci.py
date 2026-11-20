"""Tests of FCI, FCI-Max, RFCI and CFCI (increment 4)."""
from pathlib import Path

import numpy as np
import pytest

from etia_jax.ci import CI_TESTS
from etia_jax.graphs.dag import random_dag
from etia_jax.graphs.separation import DSepOracle
from etia_jax.graphs.convert import dag_to_pag
from etia_jax.graphs.metrics import shd
from etia_jax.search.fci import fci, fcimax, rfci, cfci
from etia_jax.search.tetrad_graph import TGraph

FIX = Path(__file__).parent.parent / "reference" / "fixtures"
ALGOS = {"fci": fci, "fcimax": fcimax, "rfci": rfci, "cfci": cfci}
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}


def _oracle_cases(n=120):
    for s in range(n):
        rng = np.random.default_rng(s)
        p = int(rng.integers(5, 13))
        A = random_dag(p, 2.5, s)
        L = rng.random(p) < 0.2
        obs = np.flatnonzero(~L)
        if len(obs) >= 3:
            yield s, A, L, obs


# ------------------------------------------------------------------ oracle
@pytest.mark.parametrize("name", ["fci", "fcimax", "cfci"])
def test_oracle_gives_the_pag(name):
    for s, A, L, obs in _oracle_cases():
        P = dag_to_pag(A, L)
        G = ALGOS[name](DSepOracle(A, obs), alpha=0.5).graph
        assert (G == P).all(), (s, shd(P, G))


def test_oracle_rfci_is_sound():
    """RFCI is sound: its skeleton contains the true one, and its marks on
    the true edges agree with the PAG where both are not circles."""
    for s, A, L, obs in _oracle_cases():
        P = dag_to_pag(A, L)
        G = rfci(DSepOracle(A, obs), alpha=0.5).graph
        assert ((P != 0) <= (G != 0)).all(), s
        both = (P != 0) & (G != 0)
        assert ((G == P) | (P == 1) | (G == 1))[both].all(), s


# ------------------------------------------------- the Java order emulation
def test_tgraph_orders_match_tetrad_rules():
    g = TGraph(12)
    for i in range(12):
        for j in range(i + 1, 12):
            if (i + j) % 3 == 0:
                g.add_undirected_edge(i, j)
    before = [(e.n1, e.n2) for e in g.get_edges()]
    g.set_endpoint(0, 3, "ARROW")
    after = [(e.n1, e.n2) for e in g.get_edges()]
    assert sorted(before) == sorted(after)               # same edges
    assert g.get_endpoint(0, 3) == "ARROW" and g.get_endpoint(3, 0) == "TAIL"
    assert set(g.get_adjacent_nodes(0)) == {j for j in range(1, 12) if j % 3 == 0}


# ----------------------------------------------------- Tetrad agreement
def _cases(max_p, groups=None):
    d = np.load(FIX / "search_fci_data.npz")
    r = np.load(FIX / "search_fci_ref.npz")
    s = np.load(FIX / "search_fci_ref_serial.npz")
    for case in d["cases"]:
        case = str(case)
        X = d[f"{case}|X"]
        if X.shape[1] > max_p or (groups and not case.startswith(groups)):
            continue
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        for t in d[f"{case}|tests"]:
            for a in d["alphas"]:
                for algo in ALGOS:
                    key = f"{case}|{t}|{float(a)}|{algo}"
                    if key + "|graphs" in s.files:            # race-free Tetrad (serialized tests)
                        refs = list(s[key + "|graphs"])
                    else:                                       # Tetrad as ETIA runs it
                        refs = [r[key + "|graph"]]
                    yield case, X, kinds, str(t), float(a), algo, refs


def _agreement(max_p, groups=None):
    n, bad, cis = 0, [], {}
    for case, X, kinds, t, a, algo, refs in _cases(max_p, groups):
        ci = cis.setdefault((case, t), CI_TESTS[t](X, kinds, **COMPAT.get(t, {})))
        G = ALGOS[algo](ci, alpha=a, tetrad_compat=True).graph
        n += 1
        if not any((G == h).all() for h in refs):
            bad.append((case, t, a, algo, min(shd(h, G) for h in refs)))
    return n, bad


def test_same_pags_as_tetrad_small():
    n, bad = _agreement(12, ("cont10", "mixed10_s0", "disc10_s0"))
    assert n == 80 and not bad, bad


@pytest.mark.slow
def test_same_pags_as_tetrad_all():
    """All 216 searches. Discrete data: one of the race-free Tetrad outputs."""
    n, bad = _agreement(1000)
    assert n == 216 and not bad, bad


def test_tgraph_orders_equal_real_tetrad():
    """Random graph operations, applied in Tetrad 7.6.5 (JPype) and in TGraph:
    the iteration orders of getEdges, getAdjacentNodes, getParents and
    getNodesInTo are the same (10 to 100 nodes)."""
    import json
    f = json.load(open(FIX / "java_orders.json"))
    for case, ref in zip(f["cases"], f["tetrad"]):
        g = TGraph(case["p"])
        for a, b in case["edges"]:
            g.add_undirected_edge(a, b)
        for op in case["ops"]:
            if op[0] == "set":
                g.set_endpoint(op[1], op[2], op[3])
            elif op[0] == "reorient":
                g.reorient_all_with("CIRCLE")
            else:
                e = g.get_edge(op[1], op[2])
                g.remove_edge(e)
                g.add_edge(e)
        assert [[e.n1, e.n2, e.e1, e.e2] for e in g.get_edges()] == ref["edges"]
        assert [g.get_adjacent_nodes(n) for n in range(g.p)] == ref["adj"]
        assert [g.get_parents(n) for n in range(g.p)] == ref["parents"]
        assert [g.get_nodes_in_to(n, "ARROW") for n in range(g.p)] == ref["into"]
