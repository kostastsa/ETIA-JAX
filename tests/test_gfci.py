"""Tests of GFCI (increment 5)."""
from pathlib import Path

import numpy as np
import pytest

from etia_jax.ci import CI_TESTS
from etia_jax.graphs.dag import random_dag
from etia_jax.graphs.separation import DSepOracle
from etia_jax.graphs.convert import dag_to_pag
from etia_jax.graphs.metrics import shd
from etia_jax.scores import SCORES, ETIA_SETTINGS
from etia_jax.search.gfci import gfci
from oracle_scores import DSepScore

FIX = Path(__file__).parent.parent / "reference" / "fixtures"
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}


def test_oracle_gives_the_pag():
    """With d-separation tests and the d-separation score, the default GFCI
    gives the PAG of the true DAG with latent variables."""
    n = 0
    for s in range(120):
        rng = np.random.default_rng(s)
        p = int(rng.integers(5, 11))
        A = random_dag(p, 2.5, s)
        L = rng.random(p) < 0.2
        obs = np.flatnonzero(~L)
        if len(obs) < 3:
            continue
        P = dag_to_pag(A, L)
        G = gfci(DSepOracle(A, obs), DSepScore(A, obs), alpha=0.5).graph
        assert (G == P).all(), (s, shd(P, G))
        n += 1
    assert n > 100


RACY = ("chisquare", "gsquare")       # not thread-safe in Tetrad: only race-free outputs count


def _cases(max_p):
    d = np.load(FIX / "search_fci_data.npz")
    r = np.load(FIX / "search_gfci_ref.npz")
    more = np.load(FIX / "search_gfci_ref_more.npz")
    steps = np.load(FIX / "search_gfci_steps.npz")
    # The last step (FciOrient) depends only on the graph after R0, the test
    # and alpha. Searches with the same dataset, test, alpha and R0 graph
    # (different scores) have the same possible outputs: pool them.
    pool = {}
    for k in more.files:
        case, test, alpha, score, _, _ = k.split("|")
        base = k[:-len("|graphs")]
        outs = list(more[k])
        if test not in RACY and base + "|graphs" in r.files:
            outs += list(r[base + "|graphs"])
        pool.setdefault((case, test, alpha, steps[base + "|r0"].tobytes()), []).extend(outs)
    for k in sorted(more.files):
        base = k[:-len("|graphs")]
        case, test, alpha, score, _ = base.split("|")
        X = d[f"{case}|X"]
        if X.shape[1] > max_p:
            continue
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        refs = pool[(case, test, alpha, steps[base + "|r0"].tobytes())]
        yield case, test, float(alpha), score, X, kinds, refs, steps[base + "|r0"]


def _agreement(max_p):
    """The steps before R4 (FGES, extra-edge removal, GFCI-R0) are
    deterministic in Tetrad: the graph after R0 must be the same. The final
    orientation runs R4 tasks in parallel in Tetrad: the PAG must be one of
    the recorded Tetrad outputs."""
    from etia_jax.search.cache import CICache
    from etia_jax.search.gfci import _TetradGfci
    n, bad, cis = 0, [], {}
    for case, test, alpha, score, X, kinds, refs, r0 in _cases(max_p):
        ci = cis.setdefault((case, test), CI_TESTS[test](X, kinds, **COMPAT.get(test, {})))
        sc = SCORES[score](X, kinds, tetrad_compat=True, **ETIA_SETTINGS[score])
        t = _TetradGfci(CICache(ci, alpha), X.shape[1])
        g, cpdag = t.fges_step(sc)
        t.removal_step(g, cpdag)
        t.r0_step(g, cpdag)
        same_r0 = bool((g.marks() == r0).all())
        t.final_orientation(g)
        G = g.marks()
        n += 1
        if not same_r0 or not any((G == h).all() for h in refs):
            bad.append((case, test, alpha, score, same_r0, min(shd(h, G) for h in refs)))
    return n, bad


def test_same_pags_as_tetrad_small():
    n, bad = _agreement(10)
    assert n > 50 and not bad, bad


@pytest.mark.slow
def test_same_pags_as_tetrad_all():
    n, bad = _agreement(1000)
    assert n > 150 and not bad, bad


def test_public_function_is_the_steps():
    d = np.load(FIX / "search_fci_data.npz")
    X = d["cont10_s0|X"]
    kinds = ["continuous"] * X.shape[1]
    ci = CI_TESTS["FisherZ"](X, kinds)
    a = gfci(ci, SCORES["sem_bic_score"](X, kinds, tetrad_compat=True), alpha=0.05, tetrad_compat=True).graph
    from etia_jax.search.cache import CICache
    from etia_jax.search.gfci import _TetradGfci
    b = _TetradGfci(CICache(ci, 0.05), X.shape[1]).search(SCORES["sem_bic_score"](X, kinds, tetrad_compat=True))
    assert (a == b.marks()).all()
