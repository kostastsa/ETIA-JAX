"""Tests of FGES (increment 5)."""
from pathlib import Path

import numpy as np
import pytest

from etia_jax.graphs.dag import random_dag
from etia_jax.graphs.convert import dag_to_cpdag
from etia_jax.graphs.metrics import shd
from etia_jax.scores import SCORES, ETIA_SETTINGS
from etia_jax.search.fges import fges
from oracle_scores import DSepScore

FIX = Path(__file__).parent.parent / "reference" / "fixtures"


@pytest.mark.parametrize("compat", [False, True])
def test_oracle_gives_the_cpdag(compat):
    """With the d-separation score, FGES gives the CPDAG of the true DAG."""
    for seed in range(60):
        rng = np.random.default_rng(seed)
        A = random_dag(int(rng.integers(4, 12)), 2.5, seed)
        G = fges(DSepScore(A), tetrad_compat=compat).graph
        assert (G == dag_to_cpdag(A)).all(), seed


def _cases(max_p, skip=()):
    d = np.load(FIX / "search_data.npz")
    r = np.load(FIX / "search_fges_ref.npz")
    for k in sorted(r.files):
        if not k.endswith("|graph"):
            continue
        case, name, _, _ = k.split("|")
        X = d[f"{case}|X"]
        if X.shape[1] > max_p or name in skip:
            continue
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        yield case, name, X, kinds, r[k]


def _agreement(max_p, skip=()):
    n, bad = 0, []
    for case, name, X, kinds, ref in _cases(max_p, skip):
        sc = SCORES[name](X, kinds, tetrad_compat=True, **ETIA_SETTINGS[name])
        G = fges(sc, tetrad_compat=True).graph
        n += 1
        if not (G == ref).all():
            bad.append((case, name, shd(ref, G)))
    return n, bad


def test_same_cpdags_as_tetrad_small():
    n, bad = _agreement(20, skip=("cg_bic",))
    assert n == 32 and not bad, bad


@pytest.mark.slow
def test_same_cpdags_as_tetrad_all():
    """All the FGES searches of the fixtures (to 100 variables)."""
    n, bad = _agreement(1000)
    assert n == 69 and not bad, bad


def test_default_mode_on_data():
    """The default mode (index orders, no degree limit) is as accurate as
    Tetrad on the small continuous datasets."""
    d = np.load(FIX / "search_data.npz")
    r = np.load(FIX / "search_fges_ref.npz")
    for case in ("cont10_s0", "cont10_s1", "cont20_s0"):
        C = dag_to_cpdag(d[f"{case}|A"])
        G = fges(SCORES["sem_bic_score"](d[f"{case}|X"])).graph
        assert shd(C, G) <= shd(C, r[f"{case}|sem_bic_score|fges|graph"]), case
