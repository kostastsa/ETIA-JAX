"""Tests of increment 8: FBED and SES (exact MXM 1.5.8 match), the
predictive models and the AFS pipeline of ETIA."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etia_jax.afs.fs import fbed, ses, RfastError, NoSelColumnError
from etia_jax.afs.afs import kfold, predictive_configs, run_afs, afs_target
from etia_jax.afs.models import LinearRegression, MLP, r2_score, roc_auc
from etia_jax.graphs.dag import random_dag

FIX = Path(__file__).parent.parent / "reference" / "fixtures"


def test_kfold_is_sklearn():
    """The first two test folds of KFold(5, shuffle=True, random_state=42)
    for 23 rows (from sklearn 1.x in the ETIA venv)."""
    f = kfold(23)
    assert f[0][1].tolist() == [0, 8, 9, 15, 17]
    assert f[1][1].tolist() == [1, 2, 5, 12, 13]
    assert sorted(np.concatenate([te for _, te in f]).tolist()) == list(range(23))


def test_configs_are_etias_grid():
    c = predictive_configs()
    assert len(c) == 24
    assert c[0] == {"model": "random_forest", "n_estimators": 50, "min_samples_leaf": 0.1, "max_features": "sqrt",
                    "fs_name": "fbed", "alpha": 0.05, "k": 3, "ind_test_name": "testIndFisher"}
    assert c[-1] == {"model": "linear_regression", "fs_name": "ses", "alpha": 0.01, "k": 5,
                     "ind_test_name": "testIndFisher"}


def test_linear_regression_and_scores():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ [1.0, -2.0, 0.5] + 3 + rng.normal(size=200)
    m = LinearRegression().fit(X, y)
    A = np.c_[np.ones(200), X]
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    np.testing.assert_allclose(m.predict(X), A @ b, atol=1e-10)
    assert r2_score(y, y) == 1.0
    assert abs(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-12


def test_mlp_fits_a_nonlinear_function():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (500, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2
    m = MLP(hidden=(64,), steps=1500).fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.9


def _mb_and_pc(A, t):
    pa = set(np.flatnonzero(A[:, t]))
    ch = set(np.flatnonzero(A[t]))
    sp = {int(s) for c in ch for s in np.flatnonzero(A[:, c])} - {t}
    return pa | ch | sp, pa | ch


def _exact_cov_data(A, rng, n=200_000):
    """Rows whose sample covariance is exactly the covariance of a linear
    SEM on A (an oracle: a true independence has partial correlation 0)."""
    p = len(A)
    B = A * rng.choice([-1, 1], A.shape) * rng.uniform(0.5, 1.5, A.shape)
    D = np.diag(rng.uniform(0.5, 1.5, p))
    IB = np.linalg.inv(np.eye(p) - B)
    S = IB.T @ D @ IB
    Z = rng.normal(size=(n, p))
    Z -= Z.mean(0)
    Z = Z @ np.linalg.inv(np.linalg.cholesky(np.cov(Z, rowvar=False)).T)   # sample cov = I
    return Z @ np.linalg.cholesky(S).T


def _weak_edge(X, t, pc, lam=0.01):
    """True when an edge of t has a partial correlation below lam for some
    conditioning set (a near-violation of faithfulness)."""
    from itertools import combinations
    C = np.corrcoef(X, rowvar=False)
    rest = [j for j in range(len(C)) if j != t]
    for v in pc:
        for k in range(len(rest)):
            for z in combinations([j for j in rest if j != v], k):
                idx = [t, v, *z]
                P = np.linalg.inv(C[np.ix_(idx, idx)])
                if abs(P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])) < lam:
                    return True
    return False


def test_fbed_and_ses_oracle():
    """With exact-covariance data, FBED (K = p) finds the Markov blanket of
    the target, and SES (max_k = p) all its parents and children, on 50 DAGs.
    A DAG with a near-violation of faithfulness at the target is skipped."""
    skipped = 0
    for seed in range(50):
        rng = np.random.default_rng(seed)
        A = random_dag(8, 2.0, rng, 3)
        X = _exact_cov_data(A, rng)
        t = int(np.argmax(A.sum(0) + A.sum(1)))
        others = [j for j in range(8) if j != t]
        mb, pc = _mb_and_pc(A, t)
        if _weak_edge(X, t, pc):
            skipped += 1
            continue
        assert {others[i] for i in fbed(X[:, t], X[:, others], 0.01, 7)} == mb, seed
        # SES (MMPC without the symmetry step) conditions only on selected
        # variables: it finds all the parents and children, and can keep a
        # descendant that only a non-selected variable separates.
        assert pc <= {others[i] for i in ses(X[:, t], X[:, others], 0.01, 7)[0]}, seed
    assert skipped <= 10, skipped


# ------------------------------------------------- exact match with MXM

def _fs_cases():
    if not (FIX / "fs_ref.npz").exists():
        return []
    r = np.load(FIX / "fs_ref.npz")
    return sorted({k.rsplit("|", 1)[0] for k in r.files if k.endswith("|sel") or k.endswith("|error")})


@pytest.fixture(scope="module")
def fs_ref():
    return np.load(FIX / "fs_ref.npz"), np.load(FIX / "search_data.npz")


@pytest.mark.parametrize("key", _fs_cases())
def test_fs_matches_mxm(key, fs_ref):
    """The selected columns (and their sequence for FBED) are those of MXM,
    run by ETIA's FeatureSelector."""
    r, d = fs_ref
    case, t, name, alpha, k = key.split("|")
    X = d[f"{case}|X"]
    t, alpha, k = int(t), float(alpha), int(k)
    others = [j for j in range(X.shape[1]) if j != t]
    if key + "|error" in r.files:
        # ETIA's log has the R message; the fixture keeps the RuntimeError.
        assert name == "fbed"
        n_fwd = len(fbed(X[:, t], X[:, others], alpha, 0, backward=False))
        assert n_fwd in (0, 1)                 # 1: Rfast error; 0: no "sel" column
        with pytest.raises(RfastError if n_fwd == 1 else NoSelColumnError):
            fbed(X[:, t], X[:, others], alpha, k, etia_compat=True)
        return
    want = r[key + "|sel"].tolist()
    if name == "fbed":
        got = fbed(X[:, t], X[:, others], alpha, k, etia_compat=True)
    else:
        got = ses(X[:, t], X[:, others], alpha, k)[0]
    assert got == want


# ------------------------------------------------- the AFS pipeline

def _afs_cases(tag):
    if not (FIX / "afs_ref.npz").exists():
        return []
    r = np.load(FIX / "afs_ref.npz")
    return sorted(k.rsplit("|", 1)[0] for k in r.files if k.endswith(f"|{tag}|result"))


@pytest.mark.parametrize("key", _afs_cases("linear"))
def test_afs_linear_matches_etia(key):
    """With ETIA's linear configurations, ETIA-JAX picks the same winner and
    the same columns, and its bootstrap mean is within ETIA's interval."""
    r = np.load(FIX / "afs_ref.npz")
    ref = json.loads(str(r[key + "|result"]))
    case, t, _ = key.split("|")
    X = np.load(FIX / "search_data.npz")[f"{case}|X"]
    cols = [f"X{i}" for i in range(X.shape[1])]
    configs = [c for c in predictive_configs() if c["model"] == "linear_regression"]
    out = run_afs(pd.DataFrame(X, columns=cols), [cols[int(t)]], configs, etia_compat=True)
    assert out["best_config"] == ref["best_config"]
    assert sorted(out["selected_features"][cols[int(t)]]) == ref["selected"]
    lo, hi = ref["ci"]
    assert lo <= out["bbc_score"] <= hi
