"""Out-of-sample causal tuning, OCT (Biza, Tsamardinos and Triantafillou 2020),
as ETIA runs it (``CausalLearning/CDHPO/OCT/OCT.py``, increment 9).

For each configuration and each of the first ``folds_to_run`` folds of an
unshuffled k-fold split, the configuration learns a graph on the training
rows. Each node gets a predictive model on its Markov boundary, fitted on
the training rows and applied to the test rows. The score of a node is the
mutual information of its pooled test predictions with the true values
(Gaussian for a continuous node: -log(1 - r^2) / 2). The score of a
configuration is the mean over the nodes. OCTs: among the configurations
that a permutation test does not find worse than the best one, the
configuration with the smallest mean Markov boundary wins.

``etia_compat=True`` copies ETIA:
  * the Markov boundary comes from a random DAG or MAG of Tetrad
    (``tetrad_members``) and ETIA's ``markov_boundary`` (it drops the
    neighbors on undirected edges);
  * a node with an empty Markov boundary gets uniform random predictions
    between the minimum and the maximum of its training values (a
    categorical node: the most frequent training value);
  * ETIA's Gaussian mutual information: a zero standard deviation raises
    ``ValueError``, and predictions equal to the values give the entropy
    of the values. A categorical node: scikit-learn's ``mutual_info_score``
    of the values and the (regression) predictions;
  * ``etia_n_jobs``: ETIA's CausalLearner runs joblib with
    ``os.cpu_count()`` workers. The workers change their own copies of the
    OCT object, so the Markov boundary sizes stay 0 and the p-values stay
    1: the winner is the configuration with the best score, and the
    permutation test and the sparsity rule have no effect. This is the
    default (``etia_n_jobs=None``). ``etia_n_jobs=1`` gives ETIA's
    in-process run, where they work;
  * the last step runs the winner again on the training rows of the last
    fold (ETIA keeps the ``indexes`` of the fold in the configuration),
    with a new random DAG or MAG.
  The random numbers come from ``np.random.RandomState(seed)`` in ETIA's
  order (the empty Markov boundaries, then the permutations), as
  ``np.random.seed(random_seed)`` of the CausalLearner. Tetrad's random
  graphs and scikit-learn's forest draw other numbers in ETIA, so a whole
  run of ETIA cannot be reproduced.

Default (textbook): the Markov boundary of the class (it is the same in all
its graphs), a constant prediction (the training mean, or the most
frequent value) and score 0 for an empty Markov boundary, a classifier
forest for a categorical node, a working permutation test and sparsity
rule, and the last step on all the rows.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp

from ..graphs.markov import markov_boundary, etia_markov_boundary
from ..learn.forest import fit_predict_many
from .algorithms import run_config

DEFAULT_REGRESSOR = ("RandomForestRegressor", {"n_trees": 100, "min_samples_leaf": 0.01, "max_depth": 10})


def kfold_indexes(n, folds):
    """sklearn KFold(n_splits=folds) without shuffle."""
    sizes = np.full(folds, n // folds)
    sizes[: n % folds] += 1
    stops = np.cumsum(sizes)
    tests = [np.arange(s - z, s) for s, z in zip(stops, sizes)]
    trains = [np.setdiff1d(np.arange(n), t) for t in tests]
    return trains, tests


# ------------------------------------------------------- mutual information
def mi_etia(y, yh):
    """ETIA's mutual_info_continuous (rows of yh: one or more predictions)."""
    yh = np.atleast_2d(yh)
    sy = np.std(y, ddof=1)
    sh = np.std(yh, axis=1, ddof=1)
    if sy == 0 or (sh == 0).any():
        raise ValueError("MutualInfo: zero st_dev")
    yc = y - y.mean()
    hc = yh - yh.mean(1, keepdims=True)
    r = (hc @ yc) / np.sqrt((hc ** 2).sum(1) * (yc ** 2).sum())
    eq = (yh == y).all(1)
    with np.errstate(divide="ignore"):
        out = -0.5 * np.log(1 - np.where(eq, 0.0, r) ** 2)
    out[eq] = 0.5 * math.log(2 * math.pi * math.e * sy ** 2)
    return out


def mi_gauss(y, yh):
    """Gaussian mutual information; 0 when a side is constant."""
    yh = np.atleast_2d(yh)
    yc = y - y.mean()
    hc = yh - yh.mean(1, keepdims=True)
    den = np.sqrt((hc ** 2).sum(1) * (yc ** 2).sum())
    r = np.where(den > 0, (hc @ yc) / np.where(den > 0, den, 1.0), 0.0)
    return -0.5 * np.log(np.clip(1 - r ** 2, 1e-300, None))


def mi_labels(y, yh):
    """scikit-learn mutual_info_score(y, yh) for each row of yh (nats)."""
    yh = np.atleast_2d(yh)
    _, a = np.unique(y, return_inverse=True)
    out = np.empty(len(yh))
    n = len(y)
    for i, row in enumerate(yh):
        _, b = np.unique(row, return_inverse=True)
        C = np.zeros((a.max() + 1, b.max() + 1))
        np.add.at(C, (a, b), 1)
        pi, pj = C.sum(1, keepdims=True), C.sum(0, keepdims=True)
        nz = C > 0
        out[i] = max(0.0, float((C[nz] / n * (np.log(C[nz]) - np.log((pi @ pj)[nz] / n))).sum()))
    return out


# ------------------------------------------------------------ the models
@jax.jit
def _ols_batch(Xtr, ytr, Xte):
    def one(X, y, Xt):
        xm, ym = X.mean(0), y.mean()
        coef = jnp.linalg.lstsq(X - xm, y - ym)[0]
        return Xt @ coef + (ym - xm @ coef)
    return jax.vmap(one)(Xtr, ytr, Xte)


def _predict(problems, X, train, test, is_cat, regressor, etia_compat, seeds):
    """Predictions of the problems (fold, target, Markov boundary) with a
    non-empty boundary. Returns {problem: test predictions}."""
    name, rp = regressor
    out = {}
    by = {}
    for pr in problems:
        k, t, mb = pr
        cls = bool(is_cat[t]) and not etia_compat and name != "LinearRegression"
        by.setdefault((k, cls), []).append(pr)
    for (k, cls), prs in by.items():
        tr, te = train[k], test[k]
        M = max(len(pr[2]) for pr in prs)
        Xtr = np.zeros((len(prs), len(tr), M))
        Xte = np.zeros((len(prs), len(te), M))
        for i, (_, t, mb) in enumerate(prs):
            Xtr[i, :, :len(mb)] = X[np.ix_(tr, mb)]
            Xte[i, :, :len(mb)] = X[np.ix_(te, mb)]
        if name == "LinearRegression":
            ytr = np.stack([X[tr, t] for _, t, _ in prs])
            P = np.asarray(_ols_batch(jnp.asarray(Xtr), jnp.asarray(ytr), jnp.asarray(Xte)))
            for i, pr in enumerate(prs):
                out[pr] = P[i]
                if is_cat[pr[1]] and not etia_compat:
                    codes = np.unique(X[tr, pr[1]])
                    out[pr] = codes[np.abs(P[i][:, None] - codes[None]).argmin(1)]
            continue
        kw = dict(n_estimators=rp.get("n_trees", 100), min_samples_leaf=rp.get("min_samples_leaf", 1),
                  max_depth=rp.get("max_depth", 10) or 10)
        sd = np.array([seeds[pr] for pr in prs], dtype=np.uint32)
        nr = np.full(len(prs), len(tr))
        if not cls:
            Ytr = np.stack([X[tr, t] for _, t, _ in prs])[:, :, None]
            P = np.asarray(fit_predict_many(Xtr, Ytr, Xte, nr, sd, **kw))[:, :, 0]
            for i, pr in enumerate(prs):
                out[pr] = P[i]
        else:
            C = int(max(X[:, t].max() for _, t, _ in prs)) + 1
            Ytr = np.stack([np.eye(C)[X[tr, t].astype(int)] for _, t, _ in prs])
            P = np.asarray(fit_predict_many(Xtr, Ytr, Xte, nr, sd, **kw))
            for i, pr in enumerate(prs):
                out[pr] = P[i].argmax(1).astype(np.float64)
    return out


# ------------------------------------------------------------------ OCT
def learn_graphs(X, configs, kinds, train, folds_to_run, etia_compat, rng, seed=0):
    """The class graphs and the DAGs or MAGs of each configuration and fold."""
    mecs, graphs = [], []
    for params in configs:
        ms, gs = [], []
        for k in range(folds_to_run):
            mec, g = run_config(params, X[train[k]], kinds, etia_compat, rng, seed)
            ms.append(mec)
            gs.append(g)
        mecs.append(ms)
        graphs.append(gs)
    return mecs, graphs


def boundaries(graphs, etia_compat):
    """{(config, node, fold): Markov boundary (sorted list)}."""
    mbs = {}
    for c, gs in enumerate(graphs):
        for k, g in enumerate(gs):
            for t in range(len(g)):
                mbs[c, t, k] = (sorted(int(v) for v in etia_markov_boundary(t, g)) if etia_compat
                                else markov_boundary(g, t))
    return mbs


def oct_predictions(X, mbs, n_configs, kinds, train, test, folds_to_run, regressor, etia_compat, seed=0,
                    rs=None):
    """The pooled test predictions P (configs, nodes, rows) and the pooled
    values Y (nodes, rows). ``rs``: ETIA's np.random (empty boundaries)."""
    n, p = X.shape
    F = folds_to_run
    is_cat = np.array([k == "discrete" for k in kinds])
    rs = rs if rs is not None else np.random.RandomState(seed)
    problems = sorted({(k, t, tuple(mb)) for (c, t, k), mb in mbs.items() if mb})
    if etia_compat and regressor[0] != "LinearRegression":
        seeds = {pr: int(rs.randint(0, 2 ** 31 - 1)) for pr in problems}
    else:
        seeds = {pr: int(np.random.SeedSequence([seed, pr[0], pr[1], *pr[2]]).generate_state(1)[0])
                 for pr in problems}
    preds = _predict(problems, X, train, test, is_cat, regressor, etia_compat, seeds)
    N = sum(len(test[k]) for k in range(F))
    Y = np.stack([np.concatenate([X[test[k], t] for k in range(F)]) for t in range(p)])
    P = np.zeros((n_configs, p, N))
    for c in range(n_configs):
        for t in range(p):
            parts = []
            for k in range(F):
                mb = mbs[c, t, k]
                if mb:
                    parts.append(preds[k, t, tuple(mb)])
                    continue
                ytr = X[train[k], t]
                if is_cat[t]:
                    v, cnt = np.unique(ytr, return_counts=True)
                    parts.append(np.full(len(test[k]), v[np.argmax(cnt)]))
                elif etia_compat:
                    parts.append(rs.uniform(np.min(ytr), np.max(ytr), len(test[k])))
                else:
                    parts.append(np.full(len(test[k]), ytr.mean()))
            P[c, t] = np.concatenate(parts)
    return P, Y


def _scorer(is_cat, etia_compat):
    def score(Yt, rows, t):
        if is_cat[t]:
            return mi_labels(Yt, rows)
        return mi_etia(Yt, rows) if etia_compat else mi_gauss(Yt, rows)
    return score


def oct_scores(P, Y, kinds, etia_compat):
    """mu (configs, nodes): the score of each node."""
    is_cat = np.array([k == "discrete" for k in kinds])
    score = _scorer(is_cat, etia_compat)
    return np.array([[score(Y[t], P[c, t], t)[0] for t in range(P.shape[1])] for c in range(P.shape[0])])


def oct_select(mu, P, Y, mbs, kinds, idxs, alpha, folds_to_run, etia_compat, lost=False):
    """OCTs: p-values, is_equal, mean Markov boundary sizes, the best
    configuration and the chosen configuration. ``lost``: ETIA with joblib
    workers (the sizes stay 0, the p-values stay 1)."""
    C, p, N = P.shape
    is_cat = np.array([k == "discrete" for k in kinds])
    score = _scorer(is_cat, etia_compat)
    mean_mu = np.nanmean(mu, axis=1)
    opt = int(np.argmax(mean_mu))
    B = len(idxs)
    p_values = np.ones(C)
    is_equal = np.ones(C, dtype=bool)
    sizes = np.array([[[len(mbs[c, t, k]) for t in range(p)] for k in range(folds_to_run)] for c in range(C)])
    mean_mb = np.zeros(C) if lost else sizes.mean(axis=2).mean(axis=1)
    if not lost:
        swap = np.zeros((B, N), dtype=bool)
        for i, ix in enumerate(idxs):
            swap[i, ix] = True
        best = np.max(mean_mu)
        for c in range(C):
            if c == opt:
                continue
            sb = np.zeros((B, p))
            sc = np.zeros((B, p))
            for t in range(p):
                b, cur = P[opt, t], P[c, t]
                sb[:, t] = score(Y[t], np.where(swap, cur, b), t)
                sc[:, t] = score(Y[t], np.where(swap, b, cur), t)
            tstat = sb.mean(1) - sc.mean(1)
            p_values[c] = np.count_nonzero(tstat >= best - mean_mu[c]) / B
            is_equal[c] = p_values[c] > alpha
    chosen = opt
    for c in range(C):
        if is_equal[c] and mean_mb[c] < mean_mb[chosen]:
            chosen = c
    return dict(mean_mu=mean_mu, opt_config=opt, p_values=p_values, is_equal=is_equal, mean_mb=mean_mb,
                chosen=chosen)


def run_oct(X, configs, kinds=None, folds=10, folds_to_run=3, regressor=DEFAULT_REGRESSOR, alpha=0.01,
            n_permutations=50, seed=0, etia_compat=False, etia_n_jobs=None):
    """OCT over ``configs`` (a list of parameter dicts with ``name``).
    Returns a dict: ``chosen`` (index), ``config``, ``mec_graph``, ``graph``
    (the winner on the final rows), ``opt_config`` (best score),
    ``mu`` (configs x nodes), ``mean_mu``, ``p_values``, ``is_equal``,
    ``mean_mb``, ``mbs`` ({(c, t, fold): list}), ``graphs``."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    kinds = list(kinds) if kinds is not None else ["continuous"] * p
    train, test = kfold_indexes(n, folds)
    F = folds_to_run
    lost = etia_compat and (etia_n_jobs is None or etia_n_jobs != 1)
    rs = np.random.RandomState(seed)                    # ETIA's np.random after np.random.seed(seed)
    worker_rs = np.random.RandomState([seed, 3]) if lost else rs     # joblib workers: not the seeded state
    grng = np.random.default_rng([seed, 1])             # Tetrad's random graphs

    mecs, graphs = learn_graphs(X, configs, kinds, train, F, etia_compat, grng, seed)
    mbs = boundaries(graphs, etia_compat)
    P, Y = oct_predictions(X, mbs, len(configs), kinds, train, test, F, regressor, etia_compat, seed, worker_rs)
    mu = oct_scores(P, Y, kinds, etia_compat)
    N = P.shape[2]
    idx_rng = rs if etia_compat else np.random.default_rng([seed, 2])
    idxs = [idx_rng.choice(N, N // 2, replace=False) for _ in range(n_permutations)]
    sel = oct_select(mu, P, Y, mbs, kinds, idxs, alpha, F, etia_compat, lost)
    chosen = sel["chosen"]
    rows = train[F - 1] if etia_compat else np.arange(n)
    mec, g = run_config(configs[chosen], X[rows], kinds, etia_compat, grng, seed)
    return dict(config=configs[chosen], mec_graph=mec, graph=g, mu=mu, mbs=mbs, graphs=graphs,
                mec_graphs=mecs, predictions=P, values=Y, idxs=idxs, train=train, test=test, **sel)
