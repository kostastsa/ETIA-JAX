import numpy as np
import jax
import pytest

from etia_jax.data.scm import random_scm, simulate, sample_params, make_scm
from etia_jax.graphs.dag import random_dag


def test_linear_gaussian_shape_and_regression():
    A = np.zeros((3, 3), int); A[0, 1] = A[1, 2] = 1
    spec = sample_params(A, rng=0)
    X, tr = simulate(spec, 20000, jax.random.key(0))
    assert X.shape == (20000, 3)
    # OLS of x1 on x0 recovers the weight
    w = np.linalg.lstsq(np.c_[X[:, 0], np.ones(len(X))], X[:, 1], rcond=None)[0][0]
    assert abs(w - spec.weights[(0, 1)][0, 0]) < 0.05
    # x0 independent of x2 given x1: partial correlation near 0
    C = np.linalg.inv(np.cov(X.T))
    assert abs(C[0, 2] / np.sqrt(C[0, 0] * C[2, 2])) < 0.03
    assert set(tr.get_choices().keys()) == {"x0", "x1", "x2"}


def test_noise_std():
    A = np.zeros((2, 2), int)
    spec = sample_params(A, rng=1)
    X, _ = simulate(spec, 50000, jax.random.key(1))
    assert np.allclose(X.std(0), spec.noise_std, rtol=0.03)


def test_discrete_and_mixed():
    spec = random_scm(8, 2.0, kinds="mixed", arity=3, rng=2)
    X, _ = simulate(spec, 5000, jax.random.key(2))
    for j, k in enumerate(spec.kinds):
        if k == "discrete":
            v = np.unique(X[:, j])
            assert set(v) <= {0, 1, 2} and len(v) >= 2
    assert np.isfinite(X).all()


def test_nonlinear_link():
    spec = random_scm(6, 2.0, link="tanh", rng=3)
    X, _ = simulate(spec, 1000, jax.random.key(3))
    assert np.isfinite(X).all()


def test_seed_reproducible():
    spec = random_scm(5, 2.0, rng=4)
    X1, _ = simulate(spec, 100, jax.random.key(7))
    X2, _ = simulate(spec, 100, jax.random.key(7))
    X3, _ = simulate(spec, 100, jax.random.key(8))
    assert (X1 == X2).all() and not (X1 == X3).all()
