"""Tests of increment 10: the CRV (path queries, adjustment sets, bootstrap
confidence, graph drawings).

References: dagitty 0.3.4 in R with ETIA's conversion (adjset_ref.json) and
ETIA's CRV functions with Tetrad's FCI on bootstrap rows (crv_ref.npz).
Oracles: exact covariances of linear models with latent variables."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from etia_jax.crv import queries as Q
from etia_jax.crv import dagitty as D
from etia_jax.crv.adjustment import find_adjset, adjustment_sets, graph_type_of
from etia_jax.crv.confidence import (compute_edge_weights, paths_metrics, bootstrap_graphs, is_consistent_edge,
                                     consistent_marks, bootstrap_rows, calculate_confidence)
from etia_jax.crv.visualization import matrix_to_edges, Visualization
from etia_jax.graphs.convert import dag_to_cpdag, dag_to_mag, mag_to_pag
from etia_jax.graphs.dag import random_dag

FIX = Path(__file__).parent.parent / "reference" / "fixtures"
FN = dict(directed=Q.one_directed_path, potentially_directed=Q.one_potentially_directed_path,
          any=Q.one_path_anytype, bidirected=Q.one_bidirected_path_from_to)


def _adj_cases():
    return json.loads((FIX / "adjset_ref.json").read_text())


def _crv():
    return np.load(FIX / "crv_ref.npz")


def _norm(x):
    return None if x is None else ([x] if isinstance(x, str) else sorted(x))


# ------------------------------------------------------------ dagitty
def test_adjustment_sets_match_dagitty():
    """400 graphs (DAG, CPDAG, MAG, PAG): the canonical set, the first
    minimal set and the number of minimal sets of dagitty 0.3.4, with ETIA's
    conversion of the matrix."""
    for c in _adj_cases():
        r = c["dagitty"]
        g = D.from_etia(np.array(c["M"]), c["names"], c["type"], c["exposures"], c["outcomes"])
        can = D._sorted_ids(D.canonical_adjustment_set(g))
        mini = D._sorted_ids(D.list_msas_total_effect(g))
        assert (can[0] if can else None) == _norm(r["canonical"])
        assert (mini[0] if mini else None) == _norm(r["minimal"])
        assert len(mini) == r["n_minimal"]


def test_find_adjset_compat_is_etia():
    for c in _adj_cases()[:60]:
        can, mini = find_adjset(np.array(c["M"]), c["names"], c["type"], c["outcomes"][0], c["exposures"],
                                etia_compat=True)
        assert can == _norm(c["dagitty"]["canonical"]) and mini == _norm(c["dagitty"]["minimal"])


def test_js_object_key_order():
    h = D.Hash()
    for k in ("b", "10", "a", "2"):
        h.set(k, k)
    assert h.keys() == ["2", "10", "b", "a"]


# ------------------------------------------------------------ adjustment oracle
def _sem(A, rng):
    W = np.where(A != 0, rng.uniform(0.5, 1.5, A.shape) * rng.choice([-1, 1], A.shape), 0.0)
    B = np.linalg.inv(np.eye(len(A)) - W.T)               # X = B e
    return B @ np.diag(rng.uniform(0.5, 1.5, len(A))) @ B.T, B


def _coef(S, x, y, Z):
    idx = [x] + list(Z)
    return np.linalg.solve(S[np.ix_(idx, idx)], S[idx, y])[0]


@pytest.mark.parametrize("gtype", ["dag", "cpdag", "mag", "pag"])
def test_default_adjustment_sets_give_the_total_effect(gtype):
    """Default mode on 60 random graphs: each canonical and minimal set
    gives the true total effect (exact covariances). For a DAG, when no set
    is found, no subset of the other variables gives the effect."""
    rng = np.random.default_rng({"dag": 0, "cpdag": 1, "mag": 2, "pag": 3}[gtype])
    found = 0
    for _ in range(60):
        p = 8
        A = random_dag(p, float(rng.uniform(1.5, 3.0)), rng)
        S, B = _sem(A, rng)
        lat = np.zeros(p, bool)
        if gtype in ("mag", "pag"):
            lat[rng.choice(p, 2, replace=False)] = True
        obs = np.flatnonzero(~lat)
        if gtype == "dag":
            M = np.where(A != 0, 2, 0) + np.where(A.T != 0, 3, 0)
        elif gtype == "cpdag":
            M = dag_to_cpdag(A)
        else:
            M = dag_to_mag(A, lat)
            if gtype == "pag":
                M = mag_to_pag(M)
        names = [f"V{int(i)}" for i in obs]
        x, y = rng.choice(len(obs), 2, replace=False)
        can, mins = adjustment_sets(M, names, gtype, names[y], [names[x]], all_minimal=True)
        X, Y = obs[x], obs[y]
        effect = B[Y, X]
        for Zs in ([can] if can is not None else []) + mins:
            Z = [obs[names.index(v)] for v in Zs]
            assert np.isclose(_coef(S, X, Y, Z), effect, atol=1e-8)
            found += 1
        if gtype == "dag" and can is None:
            others = [i for i in range(len(obs)) if i not in (x, y)]
            for k in range(len(others) + 1):
                for Zs in itertools.combinations(others, k):
                    assert not np.isclose(_coef(S, X, Y, [obs[i] for i in Zs]), effect, atol=1e-8)
    assert found > 40


def test_etia_cpdag_conversion_loses_undirected_edges():
    """X0 --- X1 -> X2 with X0 -> X2 is not a valid CPDAG; use a chain
    X0 --- X1 --- X2 and Y = X2: ETIA's conversion drops both edges, so
    dagitty sees no path and gives the empty set as canonical set."""
    M = np.zeros((3, 3), int)
    M[0, 1] = M[1, 0] = M[1, 2] = M[2, 1] = 3
    names = ["X0", "X1", "X2"]
    assert find_adjset(M, names, "cpdag", "X2", ["X0"], etia_compat=True) == ([], [])
    assert find_adjset(M, names, "cpdag", "X2", ["X0"]) == (None, None)       # not amenable


def test_graph_type_of():
    M = np.zeros((2, 2), int)
    assert graph_type_of(M) == "dag"
    M[0, 1] = M[1, 0] = 3
    assert graph_type_of(M) == "cpdag"
    M[0, 1] = M[1, 0] = 2
    assert graph_type_of(M) == "mag"
    M[0, 1] = 1
    assert graph_type_of(M) == "pag"


# ------------------------------------------------------------ queries
def test_queries_match_etia():
    r = _crv()
    cases = _adj_cases()
    for q in json.loads(str(r["queries"])):
        M = np.array(cases[q["case"]]["M"])
        for k, f in FN.items():
            assert f(M, q["start"], q["end"], etia_compat=True) == q[k], (q, k)


def _reach(M, s, t, ok):
    seen, st = {s}, [s]
    while st:
        v = st.pop()
        for w in range(len(M)):
            if w not in seen and ok(M, v, w):
                seen.add(w)
                st.append(w)
    return t in seen


_EDGE_OK = {
    "directed": lambda M, a, b: M[a, b] == 2 and M[b, a] == 3,
    "potentially_directed": lambda M, a, b: M[a, b] != 0 and M[b, a] != 2 and (M[a, b] != 3 or M[b, a] == 3),
    "any": lambda M, a, b: M[a, b] != 0,
    "bidirected": lambda M, a, b: M[a, b] == 2 and M[b, a] == 2,
}


def test_default_queries_are_shortest_valid_paths():
    """Default: a path exists exactly when the end can be reached through
    edges of the kind, each step is such an edge, and no shorter path exists."""
    r = _crv()
    cases = _adj_cases()
    missed = 0
    for q in json.loads(str(r["queries"])):
        M = np.array(cases[q["case"]]["M"])
        for k, f in FN.items():
            p = f(M, q["start"], q["end"])
            assert (p is not None) == _reach(M, q["start"], q["end"], _EDGE_OK[k])
            if p:
                assert p[0] == q["start"] and p[-1] == q["end"]
                assert all(_EDGE_OK[k](M, a, b) for a, b in zip(p[:-1], p[1:]))
                e = q[k]
                if e:
                    assert len(p) <= len(e)
            if k == "potentially_directed" and p and not q[k]:
                missed += 1
    assert missed > 0          # ETIA misses paths through --o and --- edges


def test_queries_with_names():
    M = np.zeros((3, 3), int)
    M[0, 1], M[1, 0], M[1, 2], M[2, 1] = 2, 3, 2, 3
    assert Q.one_directed_path(M, "a", "c", names=["a", "b", "c"]) == ["a", "b", "c"]
    assert Q.one_directed_path(M, 2, 0) is None
    assert Q.one_directed_path(M, 2, 0, etia_compat=True) == []


# ------------------------------------------------------------ confidence
@pytest.mark.parametrize("case", ["cont10_s0", "cont10_s1", "cont20_s0"])
def test_bootstrap_graphs_and_weights_match_etia(case):
    r = _crv()
    d = np.load(FIX / "search_data.npz")
    X = d[f"{case}|X"][:1000]
    names = [f"X{i}" for i in range(X.shape[1])]
    rows = bootstrap_rows(len(X), 50, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(rows, r[f"{case}|rows"]))
    cfg = {"name": "fci", "ci_test": "FisherZ", "significance_level": 0.05}
    boots = bootstrap_graphs(X, cfg, rows=rows, etia_compat=True)
    assert all(np.array_equal(a, b) for a, b in zip(boots, r[f"{case}|boots"]))
    best = r[f"{case}|best"]
    wa, _ = compute_edge_weights(best, boots, True, names=names, etia_compat=True)
    we, _ = compute_edge_weights(best, boots, False, names=names, etia_compat=True)
    assert wa.values.tolist() == json.loads(str(r[f"{case}|w_all"]))
    assert we.values.tolist() == json.loads(str(r[f"{case}|w_edges"]))
    pc, pdisc = paths_metrics(best, boots, {"p": json.loads(str(r[f"{case}|paths"]))}, names=names, etia_compat=True)
    assert np.array_equal(pc["p"], r[f"{case}|path_consistency"])
    assert np.array_equal(pdisc["p"], r[f"{case}|path_discovery"])


def test_consistency_rules():
    """ETIA's table is stricter than the default rule (never looser) in 29
    of the 100 mark pairs: a tail against a circle, the undirected CPDAG
    edge, and o-> against <-o (both can be <->)."""
    marks = [(0, 0)] + [(a, b) for a in (1, 2, 3) for b in (1, 2, 3)]
    diff = []
    for e1 in marks:
        for e2 in marks:
            if is_consistent_edge(*e1, *e2) != consistent_marks(*e1, *e2):
                diff.append((e1, e2))
    assert len(diff) == 29
    for e1, e2 in diff:
        assert not is_consistent_edge(*e1, *e2) and consistent_marks(*e1, *e2)
        assert 3 in e1 + e2 or {e1, e2} == {(1, 2), (2, 1)}, (e1, e2)
    assert is_consistent_edge(2, 3, 1, 3) is False and consistent_marks(2, 3, 1, 3) is True
    assert is_consistent_edge(3, 3, 2, 3) is False and consistent_marks(3, 3, 2, 3) is True
    assert consistent_marks(2, 3, 2, 2) is False


def test_calculate_confidence_runs():
    d = np.load(FIX / "search_data.npz")
    X = d["cont10_s0|X"][:500]
    w, boots = calculate_confidence(X, {"name": "fci", "ci_test": "FisherZ", "significance_level": 0.05},
                                    n_bootstraps=5)
    assert len(boots) == 5 and len(w) > 0
    assert ((w.edge_consistency >= w.edge_discovery) & (w.edge_consistency <= 1)).all()


# ------------------------------------------------------------ drawings
def test_matrix_to_edges_matches_etia():
    r = _crv()
    for k, c in enumerate(_adj_cases()):
        assert matrix_to_edges(np.array(c["M"]), c["names"]).values.tolist() == json.loads(str(r[f"cyto|{k}"]))


def test_visualization_svg():
    M = np.zeros((3, 3), int)
    M[0, 1], M[1, 0] = 2, 3            # X0 -> X1
    M[1, 2], M[2, 1] = 1, 2            # X1 <-o X2: mark at X2 arrow? (M[1,2]=1: circle at X2)
    v = Visualization(M)
    v.set_node_color(["X0"], "#ADD8E6")
    svg = v.plot_cytoscape()
    assert svg.startswith("<svg") and svg.count("<line") == 2 and "#ADD8E6" in svg
    assert 'marker-end="url(#arr)"' in svg and "url(#cir)" in svg
    w, _ = compute_edge_weights(M, [M, np.zeros((3, 3), int)], all_edges=False)
    v.plot_edge_confidence(w)
    v.hide_edges(0.6)
    assert v.to_svg().count("<line") == 0
    v.hide_nodes(["X2"])
    assert "X2</text>" not in v.to_svg()


def test_etia_path_consistency_skips_later_edges():
    """Path X0 - X1 - X2. The bootstrap graph has X0 o-> X1 (consistent with
    X0 -> X1, not the same) and no edge X1 - X2. ETIA stops at the first
    edge and counts the path as consistent; the default mode does not."""
    M = np.zeros((3, 3), int)
    M[0, 1], M[1, 0], M[1, 2], M[2, 1] = 2, 3, 2, 3
    B = np.zeros((3, 3), int)
    B[0, 1], B[1, 0] = 2, 1
    path = {"p": [["X0", "X1", "X2"]]}
    pc, pdisc = paths_metrics(M, [B], path, etia_compat=True)
    assert pc["p"][0] == 1.0 and pdisc["p"][0] == 0.0
    pc, pdisc = paths_metrics(M, [B], path)
    assert pc["p"][0] == 0.0 and pdisc["p"][0] == 0.0
