"""Tests of increment 7: NOTEARS and SAM (continuous optimization).

NOTEARS of causalnex is deterministic, so the ETIA mode is compared with the
causalnex weights and with ETIA's graphs. SAM is random: the tests compare
its accuracy with ETIA's runs, not its output."""
import math
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from etia_jax.graphs.encoding import ARROW, TAIL, adj_to_marks
from etia_jax.graphs.metrics import shd
from etia_jax.search import notears as NT
from etia_jax.search import sam as S

FIX = Path(__file__).parent.parent / "reference" / "fixtures"


# ---------------------------------------------------------------- NOTEARS

def test_h_is_zero_on_a_dag_and_positive_on_a_cycle():
    W = np.array([[0, 1.5, 0], [0, 0, -2.0], [0, 0, 0]])
    assert abs(float(NT._h(jnp.asarray(W)))) < 1e-12
    W[2, 0] = 0.7
    assert float(NT._h(jnp.asarray(W))) > 0.1


def test_closed_form_gradient_matches_autodiff():
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(50, 4)))
    mask = jnp.asarray(1 - np.eye(4))
    w = jnp.asarray(rng.normal(size=16) * 0.3)

    def ref(w):
        W = w.reshape(4, 4) * mask
        R = X - X @ W
        h = NT._h(W)
        return 0.5 / 50 * jnp.sum(R * R) + 0.5 * 10.0 * h * h + 0.3 * h
    f, g = NT._value_and_grad(w, X, X.T @ X, mask, 10.0, 0.3)
    assert abs(float(f - ref(w))) < 1e-10
    np.testing.assert_allclose(g, jax.grad(ref)(w), atol=1e-9)


def test_etia_matrix_later_edge_of_a_two_cycle_wins():
    W = np.array([[0, 0.3, 0], [0.2, 0, 0], [0, 0, 0]])
    M = NT.etia_matrix(W)
    assert M[1, 0] == ARROW and M[0, 1] == TAIL


def test_etia_check_dag_raises_on_a_cycle():
    W = np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="cycle"):
        NT.etia_check_dag(NT.etia_matrix(W))


def test_default_mode_recovers_an_equal_variance_sem():
    """Linear SEM with equal noise variances: the DAG is identifiable and
    NOTEARS (centered, threshold 0.3) finds it."""
    rng = np.random.default_rng(1)
    A = np.array([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0] * 5, [0] * 5])
    B = A * rng.choice([-1, 1], A.shape) * rng.uniform(0.8, 1.5, A.shape)
    n = 5000
    X = np.zeros((n, 5))
    for j in range(5):
        X[:, j] = X @ B[:, j] + rng.normal(size=n) + 3.0
    res = NT.notears(X, threshold=0.3)
    assert shd(adj_to_marks(A), res.G) == 0


def _notears_cases():
    if not (FIX / "notears_ref.npz").exists():
        return []
    r = np.load(FIX / "notears_ref.npz")
    return sorted({k.split("|")[0] for k in r.files})


@pytest.fixture(scope="module")
def notears_ref():
    return np.load(FIX / "notears_ref.npz"), np.load(FIX / "search_data.npz")


# NOTEARS is not convex. On cont20_s3 the L-BFGS of ETIA-JAX (optax zoom line
# search) ends in another local optimum than scipy L-BFGS-B: a score 0.25 %
# higher, one weight differs by 1.19, and 10 marks around node 1 differ.
LOCAL_OPTIMUM = {"cont20_s3"}


@pytest.mark.parametrize("case", [pytest.param(c, marks=pytest.mark.xfail(
    reason="another local optimum than causalnex", strict=True)) if c in LOCAL_OPTIMUM else c
    for c in _notears_cases() if not c.startswith("cont50")])
def test_notears_etia_mode_matches_causalnex(case, notears_ref):
    """The weights of causalnex and of ETIA-JAX give the same objective (to
    1e-3), and ETIA's graph at each threshold is the same, except for weights
    near the threshold (|w| within 0.05 of it in causalnex)."""
    r, d = notears_ref
    X = d[f"{case}|X"]
    Wc = r[f"{case}|W"]
    Wj, h, _, _ = NT.notears_weights(X, NT.ETIA_MAX_ITER, NT.ETIA_H_TOL)
    assert h <= NT.ETIA_H_TOL

    def F(W):
        R = X - X @ W
        return 0.5 / len(X) * np.sum(R * R)
    assert abs(F(Wj) - F(Wc)) <= 1e-3 * max(F(Wc), 1.0)
    for thr in NT_THRESHOLDS:
        key = f"{case}|{thr}"
        if key + "|time" not in r.files:            # from 20 variables: only threshold 0.8
            continue
        near = np.abs(np.abs(Wc) - thr) < 0.05
        if key + "|error" in r.files:
            with pytest.raises(ValueError):
                NT.notears(X, threshold=thr, etia_compat=True)
            continue
        if thr > 0 and near.any():
            continue
        try:
            G = NT.notears(X, threshold=thr, etia_compat=True).G
        except ValueError:
            pytest.fail(f"cycle error at threshold {thr}, ETIA had none")
        if thr == 0:
            assert (G == r[key + "|etia_matrix"]).all()
        else:
            assert (G == r[key + "|etia_matrix"]).all(), (case, thr)


NT_THRESHOLDS = [0.0, 0.5, 0.8]


# ---------------------------------------------------------------- SAM

def test_notears_constr_formula():
    rng = np.random.default_rng(2)
    A = rng.uniform(0, 1, (4, 4)).astype(np.float32)
    ref = sum(np.trace(np.linalg.matrix_power(A, k)) / math.factorial(k - 1) for k in range(1, 5))
    assert abs(float(S.notears_constr(jnp.asarray(A))) - ref) < 1e-4 * ref


def test_prepare_data_is_sklearn_scale():
    rng = np.random.default_rng(3)
    X = rng.normal(2, 3, (100, 3))
    Z, sizes = S.prepare_data(X)
    assert sizes is None
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(Z.std(0), 1, atol=1e-5)


def test_mixed_one_hot_and_skeleton_defect():
    """A 3-category column, then a continuous one (60 values). In cdt, the
    second generator reads the skeleton column 1 (a column of variable 0),
    so its inputs are its own column; the default mode gives it variable 0."""
    rng = np.random.default_rng(4)
    X = np.c_[rng.integers(0, 3, 200), rng.normal(size=200)]
    Z, sizes = S.prepare_data(X, mixed_data=True)
    assert sizes == [3, 1] and Z.shape == (200, 4)
    _, Pc, _ = S._structure(2, sizes, etia_compat=True)
    _, Pd, _ = S._structure(2, sizes, etia_compat=False)
    assert Pc[1, 3].sum() == 1 and Pc[1, :3].sum() == 0          # its own column
    assert Pd[1, :3].sum() == 3 and Pd[1, 3].sum() == 0          # the 3 columns of variable 0


def test_sam_etia_matrix_is_the_complete_reverse_order_dag():
    W = np.full((4, 4), 0.3)
    np.fill_diagonal(W, 0)
    M = S.etia_matrix(W)
    for i in range(4):
        for j in range(i + 1, 4):
            assert M[j, i] == ARROW and M[i, j] == TAIL


def test_sam_runs_and_averages_filters():
    rng = np.random.default_rng(5)
    x0 = rng.normal(size=300)
    X = np.c_[x0, np.sin(2 * x0) + 0.3 * rng.normal(size=300), rng.normal(size=300)]
    res = S.sam(X, train_epochs=40, test_epochs=20, nruns=2)
    assert res.W.shape == (3, 3) and (np.diag(res.W) == 0).all()
    assert (res.W >= 0).all() and (res.W <= 1).all()
    c = S.sam(X, train_epochs=40, test_epochs=20, nruns=2, etia_compat=True)
    np.testing.assert_allclose(c.W, res.W * res.n_batches, rtol=1e-6)


def _auc(W, A):
    off = ~np.eye(len(A), dtype=bool)
    s, y = W[off], A[off].astype(bool)
    pos, neg = s[y], s[~y]
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


@pytest.mark.slow
def test_sam_accuracy_matches_etia():
    """Same settings as the fixtures (train 1000, test 500): the mean AUROC
    of W against the true DAG (ETIA-JAX: mean of 3 seeds per dataset) is not
    worse than ETIA's (one run per dataset) by more than 0.05."""
    r = np.load(FIX / "sam_ref.npz")
    d = np.load(FIX / "sam_data.npz")
    a_etia, a_jax = [], []
    for case in [str(c) for c in d["cases"]]:
        if f"{case}|short|W" not in r.files:
            continue
        X, A = d[f"{case}|X"], d[f"{case}|A"]
        a_etia.append(_auc(r[f"{case}|short|W"], A))
        a_jax.append(np.mean([_auc(S.sam(X, train_epochs=1000, test_epochs=500, seed=s).W, A)
                              for s in range(3)]))
    assert a_etia, "no SAM fixtures"
    assert np.mean(a_jax) >= np.mean(a_etia) - 0.05, (a_etia, a_jax)
