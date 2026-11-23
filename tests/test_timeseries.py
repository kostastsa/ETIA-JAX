"""Tests of increment 6 (time series): the lagged table of ETIA, the tier
knowledge, and the Tetrad algorithms with time_lagged=True."""
from pathlib import Path

import numpy as np
import pytest
import jax

from etia_jax.ci import CI_TESTS
from etia_jax.data.svar import random_svar, simulate, lagged_table, lagged_names, tiers
from etia_jax.graphs.metrics import shd
from etia_jax.scores import SCORES, ETIA_SETTINGS
from etia_jax.search import pcalg, fci as F
from etia_jax.search.fges import fges
from etia_jax.search.gfci import gfci
from etia_jax.search.knowledge import Knowledge
from etia_jax.search.svar import svarfci, svargfci

FIX = Path(__file__).parent.parent / "reference" / "fixtures"
COMPAT = {"dg_lrt": {"tetrad_compat": True}}
TEST_ALGOS = {"pc": pcalg.pc, "cpc": pcalg.cpc, "fci": F.fci, "fcimax": F.fcimax, "rfci": F.rfci,
              "cfci": F.cfci, "svarfci": svarfci}


def test_lagged_table_layout():
    X = np.arange(12.0).reshape(6, 2)            # x(t) = (2t, 2t + 1)
    T = lagged_table(X, 2)
    assert T.shape == (4, 6)
    assert (T[0] == [4, 5, 2, 3, 0, 1]).all()    # x(2), x(1), x(0)
    assert lagged_names(2, 2) == ["X1", "X2", "X1:1", "X2:1", "X1:2", "X2:2"]
    assert tiers(2, 2) == [[4, 5], [2, 3], [0, 1]]


def test_knowledge_forbids_edges_into_the_past():
    kn = Knowledge.time_lagged(2, 1)             # X1, X2 (tier 1), X1:1, X2:1 (tier 0)
    assert kn.is_forbidden(0, 2) and not kn.is_forbidden(2, 0) and not kn.is_forbidden(0, 1)
    fe = kn.forbidden_edges(kn.names)
    assert sorted(fe) == [(0, 2), (0, 3), (1, 2), (1, 3)]


def test_svar_simulation_kinds():
    spec = random_svar(4, L=1, kinds="mixed", seed=3)
    X = simulate(spec, 300, jax.random.key(0))
    assert X.shape == (300, 4)
    for j, k in enumerate(spec.kinds):
        assert (len(np.unique(X[:, j])) <= spec.arity) == (k == "discrete")


def _ts_cases(algos, max_p=1000):
    d = np.load(FIX / "ts_data.npz")
    r = np.load(FIX / "ts_tetrad_ref.npz")
    for k in sorted(r.files):
        if not k.endswith("|graphs"):
            continue
        case, t, a, sc, algo = k.split("|")[:5]
        if algo not in algos:
            continue
        L = int(d[f"{case}|L"])
        p = len(d[f"{case}|discrete"])
        if p * (L + 1) > max_p:
            continue
        disc = np.tile(d[f"{case}|discrete"], L + 1)
        kinds = ["discrete" if b else "continuous" for b in disc]
        yield case, t, a, sc, algo, d[f"{case}|lagged"], kinds, Knowledge.time_lagged(p, L), list(r[k])


def _run(algo, X, kinds, t, a, sc, kn, tie_seed=None):
    if algo == "fges":
        return fges(SCORES[sc](X, kinds, tetrad_compat=True, **ETIA_SETTINGS[sc]),
                    tetrad_compat=True, knowledge=kn).graph
    ci = CI_TESTS[t](X, kinds, **COMPAT.get(t, {}))
    if algo == "gfci":
        return gfci(ci, SCORES[sc](X, kinds, tetrad_compat=True, **ETIA_SETTINGS[sc]), alpha=float(a),
                    tetrad_compat=True, knowledge=kn).graph
    if algo == "svargfci":
        return svargfci(ci, SCORES[sc](X, kinds, tetrad_compat=True, **ETIA_SETTINGS[sc]), alpha=float(a),
                        tetrad_compat=True, knowledge=kn, tie_seed=tie_seed).graph
    return TEST_ALGOS[algo](ci, alpha=float(a), tetrad_compat=True, knowledge=kn).graph


def _agreement(algos, max_p=1000, tie_seeds=0):
    """Each search must give one of the recorded Tetrad outputs (several for
    the searches where Tetrad runs R4 in parallel). SVAR-GFCI: Tetrad orders
    score changes that are equal except for rounding by its own rounding;
    the test also accepts the order of one of ``tie_seeds`` seeds."""
    n, bad = 0, []
    for case, t, a, sc, algo, X, kinds, kn, refs in _ts_cases(algos, max_p):
        ok = False
        for seed in [None] + list(range(tie_seeds)):
            G = _run(algo, X, kinds, t, a, sc, kn, seed)
            if any((G == h).all() for h in refs):
                ok = True
                break
            if algo != "svargfci":
                break
        n += 1
        if not ok:
            bad.append((case, t, a, sc, algo, min(shd(h, G) for h in refs)))
    return n, bad


def test_pc_cpc_with_tiers_equal_tetrad():
    n, bad = _agreement(("pc", "cpc"), max_p=10)
    assert n > 30 and not bad, bad


def test_fci_family_with_tiers_small():
    n, bad = _agreement(("fci", "rfci", "cfci", "fcimax", "svarfci"), max_p=6)
    assert n == 30 and not bad, bad


@pytest.mark.slow
def test_all_tetrad_algorithms_with_tiers():
    """All 448 searches: PC, CPC, FCI, FCI-Max, RFCI, CFCI, SVAR-FCI, FGES,
    GFCI and SVAR-GFCI on the lagged tables with ETIA's tier knowledge."""
    n, bad = _agreement(tuple(TEST_ALGOS) + ("fges", "gfci", "svargfci"), tie_seeds=20)
    assert n == 448 and not bad, bad


# ------------------------------------------------- default mode with tiers
def _oracle_checks(seeds):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from oracle_scores import DSepScore
    from etia_jax.data.svar import window_dag
    from etia_jax.graphs.separation import DSepOracle
    from etia_jax.graphs.convert import dag_to_pag, dag_to_mag, dag_to_cpdag
    bad = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        p, L = int(rng.integers(2, 4)), int(rng.integers(1, 3))
        spec = random_svar(p, L=L, seed=seed, contemp_degree=0.8, lag_degree=0.8)
        kn = Knowledge.time_lagged(p, L)
        A, lat = window_dag(spec)
        obs = np.flatnonzero(~lat)
        P, M = dag_to_pag(A, lat), dag_to_mag(A, lat)
        ci = DSepOracle(A, obs)
        outs = {n: f(ci, alpha=0.5, knowledge=kn).graph
                for n, f in [("fci", F.fci), ("fcimax", F.fcimax), ("cfci", F.cfci), ("rfci", F.rfci),
                             ("svarfci", svarfci)]}
        outs["gfci"] = gfci(ci, DSepScore(A, obs), alpha=0.5, knowledge=kn).graph
        outs["svargfci"] = svargfci(ci, DSepScore(A, obs), alpha=0.5, knowledge=kn).graph
        for n, G in outs.items():
            sk = ((P != 0) <= (G != 0)).all() if n == "rfci" else ((P != 0) == (G != 0)).all()
            both = (G != 0) & (M != 0)
            sound = ((G == M) | (G == 1))[both].all()
            info = ((P == 1) | (G == P))[(P != 0) & (G != 0)].all()
            if not (sk and sound and info):
                bad.append((seed, n, sk, sound, info))
        A0, _ = window_dag(spec, extra=0)
        C = dag_to_cpdag(A0)
        D = np.where(A0 != 0, 2, 0) + np.where(A0.T != 0, 3, 0)
        ci0 = DSepOracle(A0)
        for n, G in [("pc", pcalg.pc(ci0, alpha=0.5, knowledge=kn).graph),
                     ("cpc", pcalg.cpc(ci0, alpha=0.5, knowledge=kn).graph),
                     ("fges", fges(DSepScore(A0), knowledge=kn).graph)]:
            sk = ((C != 0) == (G != 0)).all()
            sound = ((G == D) | (G == 3) & (G.T == 3))[G != 0].all()
            info = ((C == 3) & (C.T == 3) | (G == C))[C != 0].all()
            if not (sk and sound and info):
                bad.append((seed, n, sk, sound, info))
    return bad


def test_default_mode_with_tiers_oracle():
    """With d-separation oracles on time-series DAGs (latent earlier time
    steps for the FCI family): the skeleton is right, each oriented mark is
    true, and the output is at least as informative as the PAG / CPDAG."""
    bad = _oracle_checks(range(20))
    assert not bad, bad


# ------------------------------------------------------ tigramite (ETIA)
RANDOM_NULL = ("GPDC", "CMIsymb")     # p-values from random draws (null samples, shuffles)


def _tig_agreement(algos, tests):
    from etia_jax.search.pcmci import pcmci, pcmciplus
    from etia_jax.search.lpcmci import lpcmci
    fn = {"PCMCI": pcmci, "PCMCI+": pcmciplus, "LPCMCI": lpcmci}
    d = np.load(FIX / "ts_data.npz")
    r = np.load(FIX / "ts_tigramite_ref.npz")
    n, bad = 0, []
    for k in sorted(r.files):
        if not k.endswith("|graph"):
            continue
        case, test, a, algo = k.split("|")[:4]
        if algo not in algos or test not in tests:
            continue
        X, L = d[f"{case}|X"], int(d[f"{case}|L"])
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        out = fn[algo](X, test, L, float(a), kinds=kinds)
        n += 1
        if not ((out.graph == r[k]).all() and (out.etia_matrix == r[k[:-6] + "|etia_matrix"]).all()):
            bad.append((case, test, a, algo))
    return n, bad


def test_pcmci_family_equals_tigramite():
    n, bad = _tig_agreement(("PCMCI", "PCMCI+", "LPCMCI"), ("ParCor", "RobustParCor", "Gsquared", "RegressionCI"))
    assert n == 156 and not bad, bad


@pytest.mark.slow
def test_pcmci_family_slow_tests():
    """ParCorrWLS and CMIknn: exact. GPDC and CMIsymb have random p-values in
    tigramite: at most a few searches may differ (decisions near alpha)."""
    n, bad = _tig_agreement(("PCMCI", "PCMCI+", "LPCMCI"), ("ParCorrWLS", "CMIknn"))
    assert n > 70 and not bad, bad
    n, bad = _tig_agreement(("PCMCI", "PCMCI+", "LPCMCI"), RANDOM_NULL)
    assert len(bad) <= 0.4 * n, bad
