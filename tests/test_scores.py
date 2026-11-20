"""Tests of the five scores (increment 5)."""
from pathlib import Path

import numpy as np
import pytest

from etia_jax.scores import SCORES, ETIA_SETTINGS

FIX = Path(__file__).parent.parent / "reference" / "fixtures"


def _fixture_keys():
    r = np.load(FIX / "scores.npz")
    return sorted(k for k in r.files if k.endswith("|local"))


@pytest.mark.parametrize("key", _fixture_keys())
def test_local_scores_equal_tetrad(key):
    """Tetrad 7.6.5 localScore for random (node, parents) queries, with
    ETIA's settings: relative error below 1e-9, the same NaNs, and the
    same getMaxDegree."""
    d = np.load(FIX / "search_data.npz")
    r = np.load(FIX / "scores.npz")
    case, name, _ = key.split("|")
    kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
    sc = SCORES[name](d[f"{case}|X"], kinds, tetrad_compat=True, **ETIA_SETTINGS[name])
    Q = r[f"{case}|queries"]
    mine = np.array([sc.local_score(int(q[0]), [int(v) for v in q[1:] if v >= 0]) for q in Q])
    ref = r[key]
    assert (np.isnan(mine) == np.isnan(ref)).all()
    ok = ~np.isnan(ref)
    assert np.allclose(mine[ok], ref[ok], rtol=1e-9, atol=1e-9)
    assert sc.max_degree() == int(r[f"{case}|{name}|max_degree"])


def test_batched_equals_single():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    X[:, 1] += X[:, 0]
    a, b = SCORES["sem_bic_score"](X), SCORES["sem_bic_score"](X)
    sets = [(), (0,), (0, 2), (2, 3, 4)]
    batch = a.local_scores(1, sets)
    single = [b.local_score(1, s) for s in sets]
    assert np.allclose(batch, single)


def test_score_equivalence_sem_bic():
    """Two Markov equivalent DAGs (x -> y, y -> x) have the same score."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 2))
    X[:, 1] += 0.5 * X[:, 0]
    s = SCORES["sem_bic_score"](X)
    assert np.isclose(s.local_score_diff(0, 1), s.local_score_diff(1, 0))


def test_default_fixes_tetrad_priors():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 3, size=(400, 5)).astype(float)
    kinds = ["discrete"] * 5
    t = SCORES["discrete_bic"](X, kinds, tetrad_compat=True, structure_prior=1.0)
    d = SCORES["discrete_bic"](X, kinds, structure_prior=1.0)
    # Tetrad's sign: the prior term grows with the number of parents
    assert (t.local_score(0, (1, 2)) - t.local_score(0, ())) > (d.local_score(0, (1, 2)) - d.local_score(0, ()))
