"""Tests of the JAX CI tests.

* batch == single for the vectorized tests;
* agreement with ETIA (Tetrad 7.6.5 / tigramite 5.2) on the fixture data;
* agreement with causal-learn (independent reference, test-only);
* calibration under the null and power under dependence (GenJAX data).
"""
from pathlib import Path

import numpy as np
import pytest
import jax

from etia_jax.ci import CI_TESTS, FisherZ, ParCorr, RobustParCorr, ParCorrWLS
from etia_jax.ci import ChiSquare, GSquare, Gsquared
from etia_jax.data.scm import sample_params, simulate

FIX = Path(__file__).parent.parent / "reference" / "fixtures"
ALPHA = 0.05

# Exact ports: the p-values must agree to rounding. Tests with a random
# null (CMIsymb, GPDC) or random tie-breaking noise (CMIknn) are compared by
# their decisions and statistics. ``kw`` sets the Tetrad compat modes.
EXACT = {"FisherZ", "ParCor", "RobustParCor", "ParCorrWLS", "chisquare", "gsquare",
         "Gsquared", "RegressionCI", "cg_lrt", "dg_lrt", "cci"}
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}


def _load():
    if not (FIX / "ci_ref.npz").exists():
        pytest.skip("run reference/make_fixtures.py ci in the ETIA venv first")
    return np.load(FIX / "ci_data.npz"), np.load(FIX / "ci_ref.npz")


def _pairs(ref):
    return sorted({tuple(k.split("|")[:2]) for k in ref.files})


def _queries(d, dname):
    return [(int(r[0]), int(r[1]), [int(z) for z in r[4:4 + int(r[2])]]) for r in d[f"{dname}_q"]]


def _run(name, dname, d):
    X = d[f"{dname}_X"]
    kinds = ["discrete" if b else "continuous" for b in d[f"{dname}_discrete"]]
    test = CI_TESTS[name](X, kinds, **COMPAT.get(name, {}))
    out = np.array([test.test(*q) for q in _queries(d, dname)])
    return out[:, 0], out[:, 1]


def compare(name, dname, d, ref):
    """Agreement summary of the JAX test with the ETIA answers."""
    stat, p = _run(name, dname, d)
    rp, rs = ref[f"{name}|{dname}|p"], ref[f"{name}|{dname}|stat"]
    both = ~np.isnan(p) & ~np.isnan(rp)
    # Tetrad computes p = 1 - cdf, which is 0 below about 1e-16
    lp = lambda v: np.log10(np.clip(v, 1e-15, 1))
    res = {
        "nan_same": bool((np.isnan(p) == np.isnan(rp)).all()),
        "max_abs_dp": float(np.max(np.abs(p[both] - rp[both]), initial=0)),
        "max_dlog10p": float(np.max(np.abs(lp(p[both]) - lp(rp[both])), initial=0)),
        "decisions_same": float(np.mean((np.nan_to_num(p, nan=-1) > ALPHA)
                                        == (np.nan_to_num(rp, nan=-1) > ALPHA))),
    }
    if not np.isnan(rs).all():
        res["max_rel_dstat"] = float(np.nanmax(np.abs(stat - rs) / np.maximum(np.abs(rs), 1e-12)))
    return res


@pytest.fixture(scope="module")
def ci_fixtures():
    return _load()


def test_exact_ports_agree_with_etia(ci_fixtures):
    d, ref = ci_fixtures
    bad = {}
    for name, dname in _pairs(ref):
        if name not in EXACT:
            continue
        r = compare(name, dname, d, ref)
        if name == "RegressionCI":
            # sklearn's newton-cholesky stops at tol = 1e-4; our fit is at the optimum
            close = r.get("max_rel_dstat", 0) < 1e-3
        else:
            close = r["max_abs_dp"] < 1e-6 or r["max_dlog10p"] < 1e-3
        if not (r["nan_same"] and r["decisions_same"] == 1.0 and close):
            bad[(name, dname)] = r
    assert not bad, bad


def test_cmiknn_statistic_agrees(ci_fixtures):
    """ETIA runs CMIknn with fixed_thres: the decision uses the CMI value."""
    d, ref = ci_fixtures
    stat, _ = _run("CMIknn", "nonlin", d)
    rs = ref["CMIknn|nonlin|stat"]
    assert np.max(np.abs(stat - rs)) < 5e-3
    assert np.mean((stat > ALPHA) == (rs > ALPHA)) >= 0.95


@pytest.mark.parametrize("name,dname", [("CMIsymb", "disc"), ("GPDC", "nonlin")])
def test_random_null_tests_agree(ci_fixtures, name, dname):
    d, ref = ci_fixtures
    stat, p = _run(name, dname, d)
    rp, rs = ref[f"{name}|{dname}|p"], ref[f"{name}|{dname}|stat"]
    assert np.mean((p > ALPHA) == (rp > ALPHA)) >= 0.9
    assert np.max(np.abs(p - rp)) < 0.15
    tol = 1e-8 if name == "CMIsymb" else 0.05    # GPDC: ML-II optimizers differ
    assert np.max(np.abs(stat - rs)) < tol


# ------------------------------------------------------------- batch == single
@pytest.mark.parametrize("cls,dname", [(FisherZ, "cont"), (ParCorr, "cont"), (RobustParCorr, "cont"),
                                       (ParCorrWLS, "cont"), (ChiSquare, "disc"), (GSquare, "disc"),
                                       (Gsquared, "disc")])
def test_batch_equals_single(cls, dname):
    d = np.load(FIX / "ci_data.npz")
    X = d[f"{dname}_X"]
    kinds = ["discrete" if b else "continuous" for b in d[f"{dname}_discrete"]]
    t = cls(X, kinds)
    qs = [q for q in _queries(d, dname) if len(q[2]) == 2]
    s1 = np.array([t.test(*q) for q in qs])
    s2 = np.column_stack(t.test_batch([q[0] for q in qs], [q[1] for q in qs], [q[2] for q in qs]))
    np.testing.assert_allclose(s1, s2, rtol=1e-12, atol=1e-300)


# ------------------------------------------------------------- causal-learn
@pytest.mark.parametrize("ours,theirs,dname", [(FisherZ, "fisherz", "cont"),
                                               (Gsquared, "gsq", "disc"),
                                               (ChiSquare, "chisq", "disc")])
def test_against_causal_learn(ours, theirs, dname):
    from causallearn.utils.cit import CIT
    d = np.load(FIX / "ci_data.npz")
    X = d[f"{dname}_X"]
    kinds = ["discrete" if b else "continuous" for b in d[f"{dname}_discrete"]]
    a = ours(X, kinds)
    b = CIT(X, theirs)
    qs = _queries(d, dname)
    pa = np.array([a.test(*q)[1] for q in qs])
    pb = np.array([b(q[0], q[1], q[2]) for q in qs])
    agree = np.mean((pa > ALPHA) == (pb > ALPHA))
    if theirs == "fisherz":
        np.testing.assert_allclose(pa, pb, rtol=1e-6, atol=1e-12)
    else:  # causal-learn uses a different dof rule for sparse tables
        assert agree >= 0.9


# ------------------------------------------------- calibration and power
def _fork_data(kinds, n, seed, link_xy=False):
    """Z -> X, Z -> Y (and X -> Y if link_xy), from the GenJAX sampler."""
    A = np.zeros((3, 3), int)
    A[2, 0] = A[2, 1] = 1
    if link_xy:
        A[0, 1] = 1
    spec = sample_params(A, kinds=kinds, rng=seed)
    X, _ = simulate(spec, n, jax.random.key(seed))
    return X, kinds


CALIBRATION = [  # name, kinds, n, replicates
    ("FisherZ", ["continuous"] * 3, 300, 200),
    ("ParCor", ["continuous"] * 3, 300, 200),
    ("RobustParCor", ["continuous"] * 3, 300, 200),
    ("ParCorrWLS", ["continuous"] * 3, 300, 200),
    ("dg_lrt", ["continuous", "discrete", "discrete"], 400, 100),
    ("cg_lrt", ["continuous", "continuous", "discrete"], 400, 100),
    ("RegressionCI", ["discrete", "continuous", "continuous"], 400, 100),
    ("chisquare", ["discrete"] * 3, 600, 100),
    ("gsquare", ["discrete"] * 3, 600, 100),
    ("Gsquared", ["discrete"] * 3, 600, 100),
    ("dg_lrt", ["continuous"] * 3, 300, 100),
    ("cci", ["continuous"] * 3, 200, 60),
]
PARAMS = {"cci": {"bonferroni": True}}


@pytest.mark.slow
@pytest.mark.parametrize("name,kinds,n,reps", CALIBRATION)
def test_calibration_and_power(name, kinds, n, reps):
    """Under X _||_ Y | Z the rejection rate at 0.05 stays near 0.05;
    with X -> Y the test rejects in most replicates."""
    rej0 = rej1 = 0
    for r in range(reps):
        X, k = _fork_data(kinds, n, 1000 + r)
        p0 = CI_TESTS[name](X, k, **PARAMS.get(name, {})).test(0, 1, [2])[1]
        rej0 += (not np.isnan(p0)) and p0 <= ALPHA
        X, k = _fork_data(kinds, n, 5000 + r, link_xy=True)
        p1 = CI_TESTS[name](X, k, **PARAMS.get(name, {})).test(0, 1, [2])[1]
        rej1 += np.isnan(p1) or p1 <= ALPHA
    size, power = rej0 / reps, rej1 / reps
    assert size <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / reps) + 0.02, size
    assert power >= 0.8, power
