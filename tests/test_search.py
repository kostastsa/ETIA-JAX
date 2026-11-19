"""Tests of PC and CPC (increment 3)."""
from pathlib import Path

import numpy as np
import pytest

from etia_jax.ci import CI_TESTS, FisherZ
from etia_jax.ci.base import CITest
from etia_jax.graphs.dag import random_dag
from etia_jax.graphs.separation import DSepOracle
from etia_jax.graphs.convert import dag_to_cpdag
from etia_jax.graphs.encoding import ARROW, TAIL
from etia_jax.graphs.metrics import shd
from etia_jax.search import pc, cpc

FIX = Path(__file__).parent.parent / "reference" / "fixtures"
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}


# ----------------------------------------------------------------- oracle
@pytest.mark.parametrize("algo,kw", [(pc, {}), (pc, {"tetrad_compat": True}),
                                     (pc, {"stable": False}), (cpc, {})])
def test_oracle_gives_the_cpdag(algo, kw):
    for s in range(120):
        A = random_dag(int(np.random.default_rng(s).integers(4, 16)), 2.5, s)
        G = algo(DSepOracle(A), alpha=0.5, **kw).graph
        assert (G == dag_to_cpdag(A)).all(), (s, shd(dag_to_cpdag(A), G))


def test_cpc_tetrad_defect_on_the_smallest_collider():
    """x -> y <- z with x and z of degree 1: Tetrad 7.6.5 uses a side of the
    triple only with two or more adjacents, so the collider is ambiguous."""
    A = np.zeros((3, 3), int)
    A[0, 1] = A[2, 1] = 1
    o = DSepOracle(A)
    r = cpc(o, alpha=0.5)
    assert r.colliders == [(0, 1, 2)] and r.graph[0, 1] == ARROW and r.graph[2, 1] == ARROW
    rc = cpc(o, alpha=0.5, tetrad_compat=True)
    assert rc.ambiguous == [(0, 1, 2)] and rc.graph[0, 1] == TAIL and rc.graph[1, 0] == TAIL


# --------------------------------------------------------- Tetrad fixtures
def _fixture_cases(max_p):
    d, r = np.load(FIX / "search_data.npz"), np.load(FIX / "search_ref.npz")
    for case in d["cases"]:
        case = str(case)
        X = d[f"{case}|X"]
        if X.shape[1] > max_p:
            continue
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        for t in d[f"{case}|tests"]:
            t = str(t)
            for a in d["alphas"]:
                for algo in ("pc", "cpc"):
                    yield case, X, kinds, t, float(a), algo, r[f"{case}|{t}|{float(a)}|{algo}|graph"]


def _agreement(max_p):
    bad, n, cis = [], 0, {}
    for case, X, kinds, t, a, algo, T in _fixture_cases(max_p):
        ci = cis.setdefault((case, t), CI_TESTS[t](X, kinds, **COMPAT.get(t, {})))
        G = (pc if algo == "pc" else cpc)(ci, alpha=a, tetrad_compat=True).graph
        n += 1
        if not (G == T).all():
            bad.append((case, t, a, algo, shd(T, G)))
    return n, bad


def test_same_graphs_as_tetrad_up_to_30_variables():
    n, bad = _agreement(30)
    assert n == 100 and not bad, bad


@pytest.mark.slow
def test_same_graphs_as_tetrad_all():
    n, bad = _agreement(1000)
    assert n == 116 and not bad, bad


# -------------------------------------------------- batched replay = sequential
class _NoBatch(CITest):
    """The same test without its batched path: one query at a time."""

    def __init__(self, inner):
        self.inner, self.p, self.symmetric = inner, inner.p, inner.symmetric

    def test(self, x, y, Z=()):
        return self.inner.test(x, y, Z)

    def decide(self, stat, pval, alpha):
        return self.inner.decide(stat, pval, alpha)


@pytest.mark.parametrize("compat", [False, True])
def test_batched_replay_equals_one_query_at_a_time(compat):
    d = np.load(FIX / "search_data.npz")
    for case in ("cont10_s0", "cont20_s1", "disc10_s0"):
        X = d[f"{case}|X"]
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        ci = CI_TESTS["FisherZ" if case.startswith("cont") else "gsquare"](X, kinds)
        for algo in (pc, cpc):
            r1 = algo(ci, alpha=0.05, tetrad_compat=compat)
            r2 = algo(_NoBatch(ci), alpha=0.05, tetrad_compat=compat)
            assert (r1.graph == r2.graph).all() and r1.n_tests == r2.n_tests
            assert r1.n_computed >= r2.n_computed      # prefetch computes extra queries


# ----------------------------------------------------- PC-stable: order free
def test_pc_stable_skeleton_does_not_depend_on_the_order():
    d = np.load(FIX / "search_data.npz")
    for case in ("cont10_s1", "cont20_s0", "cont30_s1"):
        X = d[f"{case}|X"]
        perm = np.random.default_rng(0).permutation(X.shape[1])
        S1 = pc(FisherZ(X), 0.05).graph != 0
        S2 = pc(FisherZ(X[:, perm]), 0.05).graph != 0
        inv = np.argsort(perm)
        assert (S1 == S2[np.ix_(inv, inv)]).all()
