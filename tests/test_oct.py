"""Tests of increment 9: the random forest, the Markov boundaries, Tetrad's
random DAG and MAG, the OCT and the CausalLearner.

ETIA's OCT is not reproducible from run to run (Tetrad draws its random
graphs in Java), so the comparisons with ETIA are per step: each step of
ETIA-JAX gets the inputs that ETIA's step had (reference/fixtures/oct_ref.npz)
and must give the same output."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from etia_jax.ci import CI_TESTS
from etia_jax.graphs.convert import dag_to_cpdag, dag_to_mag, mag_to_pag
from etia_jax.graphs.dag import random_dag
from etia_jax.graphs.encoding import NONE, ARROW, TAIL, CIRCLE
from etia_jax.graphs.markov import (dag_from_cpdag, mag_from_pag, markov_boundary, etia_markov_boundary, member)
from etia_jax.learn.forest import RandomForest
from etia_jax.oct.algorithms import search, run_config
from etia_jax.oct.configs import create_causal_configs, parameter_grid, expand, var_types, DEFAULT_CONF
from etia_jax.oct.learner import CausalLearner
from etia_jax.oct.oct import (kfold_indexes, run_oct, oct_scores, oct_select, oct_predictions, boundaries,
                              mi_etia, mi_labels)
from etia_jax.search.tetrad_members import tetrad_mag_from_pag, tetrad_dag_from_cpdag

FIX = Path(__file__).parent.parent / "reference" / "fixtures"
CASES = ["cont10_s0", "cont10_s1", "cont10_s2", "cont20_s0", "mixed10_s0", "disc10_s0"]


def _ref():
    f = FIX / "oct_ref.npz"
    if not f.exists():
        pytest.skip("no oct_ref.npz (reference/make_fixtures.py oct)")
    return np.load(f)


def _data(case):
    d = np.load(FIX / "search_data.npz")
    X, kinds, _ = var_types(d[f"{case}|X"][:1000])
    return X, kinds


def _cases(ref, tag):
    return [c for c in CASES if f"{c}|{tag}|time" in ref.files]


# ------------------------------------------------------------ k-fold, grids
def test_kfold_is_sklearn_without_shuffle():
    from sklearn.model_selection import KFold
    for n, k in ((1000, 10), (23, 5), (101, 10)):
        tr, te = kfold_indexes(n, k)
        for (a, b), x, y in zip(KFold(k).split(np.zeros(n)), tr, te):
            assert a.tolist() == x.tolist() and b.tolist() == y.tolist()


def test_parameter_grid_is_sklearn():
    from sklearn.model_selection import ParameterGrid
    g = {"significance_level": [0.01, 0.05], "ci_test": ["FisherZ", "cg_lrt"], "stable": [True]}
    assert parameter_grid(g) == list(ParameterGrid(g))


def test_configurations_are_etias():
    ref = json.loads((FIX / "oct_configs_ref.json").read_text())
    for key, want in ref.items():
        if key == "default":
            got = create_causal_configs("continuous", exclude_algs=DEFAULT_CONF["exclude_algs"])
        else:
            dt, cs, af, ts, tl = key.split("|")
            got = create_causal_configs(dt, cs == "True", af == "True", ts == "True", tl == "True")
        assert list(got) == list(want), key
        assert got == want, key


def test_default_grid_is_fci_and_fcimax():
    cfg = expand(create_causal_configs("continuous", exclude_algs=DEFAULT_CONF["exclude_algs"]))
    assert [(c["name"], c["significance_level"]) for c in cfg] == [
        ("fci", 0.01), ("fci", 0.05), ("fci", 0.1), ("fcimax", 0.01), ("fcimax", 0.05), ("fcimax", 0.1)]


# ------------------------------------------------------------ random forest
def _friedman(rng, n, d):
    X = rng.normal(size=(n, d))
    y = np.sin(X[:, 0]) * 2 + X[:, 1] ** 2 + 0.5 * X[:, 2] + rng.normal(size=n) * 0.3
    return X, y


def test_forest_regression_like_sklearn():
    """Same settings as ETIA's OCT forest: the test R^2 is within 0.01 of
    scikit-learn's, and the predictions agree (correlation > 0.99)."""
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(0)
    for n, d in ((900, 3), (900, 8), (2000, 5)):
        X, y = _friedman(rng, n + 500, d)
        r2 = lambda p: 1 - ((y[n:] - p) ** 2).mean() / y[n:].var()
        ps = RandomForestRegressor(100, min_samples_leaf=0.01, max_depth=10, random_state=0).fit(X[:n], y[:n])
        ps = ps.predict(X[n:])
        pj = RandomForest(100, 0.01, 10, seed=0).fit(X[:n], y[:n]).predict(X[n:])
        assert abs(r2(ps) - r2(pj)) < 0.01
        assert np.corrcoef(ps, pj)[0, 1] > 0.99


def test_forest_classifier_like_sklearn():
    """AFS settings (min_samples_leaf 0.1, max_features sqrt): the mean
    accuracy over 5 seeds is within 0.02 of scikit-learn's."""
    from sklearn.ensemble import RandomForestClassifier
    acc_s, acc_j = [], []
    for s in range(5):
        rng = np.random.default_rng(10 + s)
        X, y = _friedman(rng, 1400, 6)
        yc = (y > np.median(y)).astype(int) + (y > np.quantile(y, 0.8))
        sk = RandomForestClassifier(100, min_samples_leaf=0.1, max_features="sqrt", random_state=s)
        acc_s.append((sk.fit(X[:900], yc[:900]).predict(X[900:]) == yc[900:]).mean())
        jf = RandomForest(100, 0.1, None, "sqrt", classify=True, seed=s).fit(X[:900], yc[:900])
        acc_j.append((jf.predict(X[900:]) == yc[900:]).mean())
    assert abs(np.mean(acc_s) - np.mean(acc_j)) < 0.02


def test_forest_tree_rules():
    """One tree without bootstrap noise: a step function is learned exactly,
    and no leaf has fewer rows than min_samples_leaf."""
    x = np.linspace(0, 1, 200)[:, None]
    y = (x[:, 0] > 0.5).astype(float)
    f = RandomForest(20, 5, 3, seed=1).fit(x, y)
    assert np.allclose(f.predict(np.array([[0.1], [0.9]])), [0.0, 1.0])
    feat, thr, split, value = (np.asarray(a) for a in f.trees_)
    assert (np.abs(thr[split] - 0.5) < 0.05).any()


# ------------------------------------------------------------ Markov boundaries
def _true_mb(A, t):
    pa = set(np.flatnonzero(A[:, t]))
    ch = set(np.flatnonzero(A[t]))
    sp = {int(s) for c in ch for s in np.flatnonzero(A[:, c])}
    return sorted((pa | ch | sp) - {t})


@pytest.mark.parametrize("seed", range(20))
def test_markov_boundary_of_cpdag_is_true_mb(seed):
    rng = np.random.default_rng(seed)
    A = random_dag(9, 2.5, rng)
    C = dag_to_cpdag(A)
    D = dag_from_cpdag(C)
    assert np.array_equal(dag_to_cpdag((D == ARROW) & (D.T == TAIL)), C)
    for t in range(9):
        assert markov_boundary(C, t) == _true_mb(A, t)


@pytest.mark.parametrize("seed", range(20))
def test_markov_boundary_of_pag_is_mag_mb(seed):
    """A MAG of the PAG (Zhang's Theorem 2) has the PAG, and its Markov
    boundaries are those of the true MAG."""
    rng = np.random.default_rng(100 + seed)
    p = 10
    A = random_dag(p, 2.5, rng)
    lat = np.zeros(p, bool)
    lat[rng.choice(p, 2, replace=False)] = True
    M = dag_to_mag(A, lat)
    P = mag_to_pag(M)
    Mm = mag_from_pag(P)
    assert not (Mm == CIRCLE).any()
    assert np.array_equal(mag_to_pag(Mm), P)
    for t in range(len(M)):
        assert markov_boundary(P, t) == markov_boundary(M, t)


def test_etia_markov_boundary_drops_undirected_neighbors():
    G = np.zeros((3, 3), np.int8)
    G[0, 1] = G[1, 0] = TAIL                     # 0 --- 1
    G[1, 2], G[2, 1] = ARROW, TAIL               # 1 -> 2
    assert etia_markov_boundary(0, G) == []
    assert etia_markov_boundary(2, G) == [1]
    G[0, 1] = CIRCLE
    with pytest.raises(ValueError):
        etia_markov_boundary(0, G)


def test_etia_markov_boundary_from_etia_graphs():
    """ETIA's Markov boundaries of its random MAGs (all cases, all runs)."""
    ref = _ref()
    n = 0
    for case in CASES:
        for tag in ("lin1", "lin8", "rf1", "rf8"):
            key = f"{case}|{tag}"
            if key + "|time" not in ref.files:
                continue
            G = ref[key + "|graph"]
            mbs = json.loads(str(ref[key + "|mbs"]))
            for c in range(G.shape[0]):
                for k in range(G.shape[1]):
                    for t in range(G.shape[2]):
                        assert sorted(etia_markov_boundary(t, G[c, k])) == mbs[c][t][k]
                        n += 1
    assert n > 1000


# ------------------------------------------------------------ Tetrad's random members
def _tv(a, b, n):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)) / n


def _members_ref():
    f = FIX / "members_ref.npz"
    if not f.exists():
        pytest.skip("no members_ref.npz (reference/make_fixtures.py members)")
    return np.load(f)


def _check_distribution(key, draw, n=2000):
    """The total variation distance between Tetrad's draws and ours is not
    larger than the distance between two samples of ours (+ 0.03)."""
    ref = _members_ref()
    G = ref[key + "|graph"]
    java = json.loads(str(ref[key + "|counts"]))
    rng = np.random.default_rng(0)
    ours = Counter(draw(G, rng).tobytes().hex() for _ in range(n))
    ours2 = Counter(draw(G, rng).tobytes().hex() for _ in range(n))
    assert _tv(java, ours, n) <= _tv(ours, ours2, n) + 0.03
    # the frequency of a tail at each endpoint of the class graph that the draw changes
    p = len(G)
    def tails(cnt):
        T = np.zeros((p, p))
        for k, v in cnt.items():
            T += v * (np.frombuffer(bytes.fromhex(k), np.int8).reshape(p, p) == TAIL)
        return T / sum(cnt.values())
    free = (G == CIRCLE) | ((G == TAIL) & (G.T == TAIL))
    a, b = tails(java)[free], tails(ours)[free]
    sd = np.sqrt(np.clip((a + b) / 2 * (1 - (a + b) / 2), 1e-4, None) * 2 / n)
    assert (np.abs(a - b) <= 4.5 * sd + 0.005).all(), np.max(np.abs(a - b) / sd)


def _member_keys(kind):
    f = FIX / "members_ref.npz"
    if not f.exists():
        return []
    return sorted({k.rsplit("|", 1)[0] for k in np.load(f).files if k.startswith(kind)})


@pytest.mark.slow
@pytest.mark.parametrize("key", _member_keys("pag"))
def test_mag_from_pag_distribution_is_tetrads(key):
    _check_distribution(key, tetrad_mag_from_pag)


@pytest.mark.parametrize("key", _member_keys("cpdag"))
def test_dag_from_cpdag_distribution_is_tetrads(key):
    _check_distribution(key, tetrad_dag_from_cpdag)


def test_random_mag_keeps_the_class_marks():
    """Each random MAG keeps the arrowheads and tails of the PAG."""
    ref = _ref()
    P = ref["cont10_s0|lin1|mec"][0, 0]
    rng = np.random.default_rng(3)
    for _ in range(20):
        M = tetrad_mag_from_pag(P, rng)
        assert not (M == CIRCLE).any()
        fixed = (P != CIRCLE) & (P != NONE)
        assert np.array_equal(M[fixed], P[fixed])


# ------------------------------------------------------------ the OCT against ETIA, step by step
@pytest.mark.parametrize("case", CASES)
def test_fold_graphs_match_etia(case):
    """The PAG of each configuration on the training rows of each fold."""
    ref = _ref()
    if f"{case}|lin1|time" not in ref.files:
        pytest.skip("no reference")
    X, kinds = _data(case)
    cfgs = json.loads(str(ref[f"{case}|lin1|configs"]))
    mec = ref[f"{case}|lin1|mec"]
    train, _ = kfold_indexes(len(X), 10)
    for c, params in enumerate(cfgs):
        for k in range(mec.shape[1]):
            got = search(params, X[train[k]], kinds, etia_compat=True)
            assert np.array_equal(got, mec[c, k]), (c, k)


@pytest.mark.parametrize("case", CASES)
def test_predictions_match_etia(case):
    """LinearRegression on ETIA's Markov boundaries: the same pooled test
    predictions (an empty boundary: uniform values in the training range)."""
    ref = _ref()
    if f"{case}|lin1|time" not in ref.files:
        pytest.skip("no reference")
    X, kinds = _data(case)
    key = f"{case}|lin1"
    mbl = json.loads(str(ref[key + "|mbs"]))
    C, p, F = len(mbl), len(mbl[0]), len(mbl[0][0])
    mbs = {(c, t, k): mbl[c][t][k] for c in range(C) for t in range(p) for k in range(F)}
    train, test = kfold_indexes(len(X), 10)
    P, Y = oct_predictions(X, mbs, C, kinds, train, test, F, ("LinearRegression", {}), True)
    want = ref[key + "|pred"]
    assert np.allclose(Y, ref[key + "|y"])
    for c in range(C):
        for t in range(p):
            off = 0
            for k in range(F):
                m = len(test[k])
                if mbs[c, t, k]:
                    assert np.allclose(P[c, t, off:off + m], want[c, t, off:off + m], atol=1e-8)
                else:
                    ytr = X[train[k], t]
                    assert want[c, t, off:off + m].min() >= ytr.min() and want[c, t, off:off + m].max() <= ytr.max()
                off += m


@pytest.mark.parametrize("tag", ["lin1", "lin8", "rf1", "rf8"])
def test_scores_match_etia(tag):
    """ETIA's mutual information of its own predictions."""
    ref = _ref()
    for case in _cases(ref, tag):
        X, kinds = _data(case)
        key = f"{case}|{tag}"
        mu = oct_scores(ref[key + "|pred"], ref[key + "|y"], kinds, True)
        assert np.allclose(mu, ref[key + "|mu"], rtol=1e-10, atol=1e-12, equal_nan=True), case


@pytest.mark.parametrize("tag", ["lin1", "rf1"])
def test_permutation_test_and_choice_match_etia(tag):
    """ETIA's in-process run (n_jobs=1): the same p-values, the same mean
    Markov boundary sizes and the same chosen configuration, from ETIA's
    predictions and permutations."""
    ref = _ref()
    for case in _cases(ref, tag):
        X, kinds = _data(case)
        key = f"{case}|{tag}"
        mbl = json.loads(str(ref[key + "|mbs"]))
        C, p, F = len(mbl), len(mbl[0]), len(mbl[0][0])
        mbs = {(c, t, k): mbl[c][t][k] for c in range(C) for t in range(p) for k in range(F)}
        s = oct_select(ref[key + "|mu"], ref[key + "|pred"], ref[key + "|y"], mbs, kinds, list(ref[key + "|idxs"]),
                       0.01, F, True, lost=False)
        assert np.array_equal(s["p_values"], ref[key + "|p_values"]), case
        assert np.array_equal(s["is_equal"], ref[key + "|is_equal"]), case
        assert np.allclose(s["mean_mb"], ref[key + "|mean_mb"]), case
        assert s["opt_config"] == int(ref[key + "|opt_config"]) and s["chosen"] == int(ref[key + "|chosen"]), case


@pytest.mark.parametrize("tag", ["lin8", "rf8"])
def test_joblib_workers_lose_the_state(tag):
    """ETIA with n_jobs > 1: p-values 1, sizes 0, winner = best score; the
    same rule in etia_compat (default etia_n_jobs)."""
    ref = _ref()
    for case in _cases(ref, tag):
        key = f"{case}|{tag}"
        assert (ref[key + "|p_values"] == 1).all() and (ref[key + "|mean_mb"] == 0).all()
        assert int(ref[key + "|chosen"]) == int(ref[key + "|opt_config"]) == int(np.argmax(np.nanmean(ref[key + "|mu"], 1)))
        X, kinds = _data(case)
        mbl = json.loads(str(ref[key + "|mbs"]))
        C, p, F = len(mbl), len(mbl[0]), len(mbl[0][0])
        mbs = {(c, t, k): mbl[c][t][k] for c in range(C) for t in range(p) for k in range(F)}
        s = oct_select(ref[key + "|mu"], ref[key + "|pred"], ref[key + "|y"], mbs, kinds, list(ref[key + "|idxs"]),
                       0.01, F, True, lost=True)
        assert s["chosen"] == int(ref[key + "|chosen"])


def test_final_graph_uses_the_last_fold_rows():
    ref = _ref()
    train, _ = kfold_indexes(1000, 10)
    for case in _cases(ref, "lin1"):
        assert int(ref[f"{case}|lin1|n_final_rows"]) == len(train[2])


def test_mutual_information_quirks():
    y = np.array([1.0, 2.0, 3.0, 5.0])
    assert np.isclose(mi_etia(y, y)[0], 0.5 * np.log(2 * np.pi * np.e * np.var(y, ddof=1)))
    with pytest.raises(ValueError):
        mi_etia(y, np.ones(4))
    from sklearn.metrics import mutual_info_score
    a, b = np.array([0, 0, 1, 1, 2, 2, 2]), np.array([0.1, 0.1, 0.7, 0.2, 0.9, 0.9, 0.3])
    assert np.isclose(mi_labels(a, b)[0], mutual_info_score(a, b))


# ------------------------------------------------------------ whole runs
def test_run_oct_compat_modes():
    X, kinds = _data("cont10_s0")
    cfgs = expand(create_causal_configs("continuous", exclude_algs=DEFAULT_CONF["exclude_algs"]))
    lost = run_oct(X, cfgs, kinds, regressor=("LinearRegression", {}), etia_compat=True)
    assert (lost["p_values"] == 1).all() and (lost["mean_mb"] == 0).all() and lost["chosen"] == lost["opt_config"]
    one = run_oct(X, cfgs, kinds, regressor=("LinearRegression", {}), etia_compat=True, etia_n_jobs=1)
    assert (one["mean_mb"] > 0).all() and one["is_equal"][one["opt_config"]]
    train, _ = kfold_indexes(1000, 10)
    assert np.array_equal(one["mec_graph"], search(one["config"], X[train[2]], kinds, etia_compat=True))


def test_default_oct_reaches_the_true_markov_boundaries():
    """Default mode on 10-node linear data: the chosen configuration scores
    within 0.02 of the true Markov boundaries (the same regressor), and its
    final graph comes from all the rows."""
    d = np.load(FIX / "search_data.npz")
    for case in ("cont10_s0", "cont10_s1", "cont10_s2"):
        X, kinds = _data(case)
        A = d[f"{case}|A"]
        cfgs = expand(create_causal_configs("continuous", exclude_algs=DEFAULT_CONF["exclude_algs"]))
        r = run_oct(X, cfgs, kinds, regressor=("LinearRegression", {}))
        mbs = {(0, t, k): _true_mb(A, t) for t in range(10) for k in range(3)}
        P, Y = oct_predictions(X, mbs, 1, kinds, r["train"], r["test"], 3, ("LinearRegression", {}), False)
        best = oct_scores(P, Y, kinds, False).mean()
        assert r["mean_mu"][r["chosen"]] > best - 0.02, case
        assert np.array_equal(r["mec_graph"], search(r["config"], X, kinds))


def test_causal_learner():
    X, kinds = _data("cont10_s0")
    cl = CausalLearner(X, regressor=("LinearRegression", {}))
    out = cl.learn_model()
    assert set(out) == {"optimal_conf", "matrix_mec_graph", "matrix_graph", "run_time", "library_results"}
    cfg, mec, g = cl.get_best_model_between_algorithms(["fcimax"])
    assert cfg["name"] == "fcimax"
    with pytest.raises(RuntimeError):
        cl.get_best_model_between_algorithms(["pc"])
    cfg, _, _ = cl.get_best_model_between_family(causal_sufficiency=False)
    assert cfg["name"] in ("fci", "fcimax")
    ce = CausalLearner(X, regressor=("LinearRegression", {}), etia_compat=True)
    ce.learn_model()
    with pytest.raises(ValueError):
        ce.get_best_model_between_algorithms(["fci"])
    with pytest.raises(RuntimeError):
        ce.get_best_model_between_family(causal_sufficiency=False)
    with pytest.raises(KeyError):
        ce.get_best_model_between_family(accepts_missing_values=True)


def test_notears_in_the_kfold_protocol():
    """ETIA's k-fold protocol stops on NOTEARS (two return values); the
    default mode runs NOTEARS on the rows of the fold."""
    X, kinds = _data("cont10_s0")
    params = {"name": "notears", "max_iter": 100, "h_tol": 1e-7, "threshold": 0.5, "causal_sufficiency": True}
    with pytest.raises(ValueError):
        run_config(params, X[:500], kinds, etia_compat=True)
    mec, g = run_config(params, X[:500], kinds)
    assert mec.shape == (10, 10) and not (g == CIRCLE).any()
