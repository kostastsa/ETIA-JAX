"""AFS (automated feature selection) of ETIA in JAX (increment 8).

ETIA's ``AFS.run_AFS(data, target_features, pred_configs)``, for each target:

1. ``OOS.data_split``: ``KFold(folds, shuffle=True, random_state=42)``.
2. For each configuration and each fold: FBED or SES (``fs.py``) on the
   training rows, then the model on the selected columns, then R^2 (or the
   AUROC) on the test rows. A fold without selected columns is skipped.
3. The configuration with the largest mean score wins (the first on a tie).
4. ``bootstrap_bias_correction``: 1000 bootstrap samples of the pooled test
   predictions of the winner; the mean score and the 95 % percentile interval.
5. The selected columns: the intersection of the selected sets of the
   winner over its folds. With ``depth`` > 1, the search repeats for each
   selected column.
6. A final model with the winner on all rows and the selected columns.

Findings (ETIA), kept only in ``etia_compat=True``:

* ``folds_to_run`` (2 by default) has no effect: all the folds run.
* The "bootstrap bias correction" is a plain bootstrap of the winner's
  pooled predictions. It does not correct the bias of the selection of the
  winner, so it is optimistic. The default mode uses BBC-CV (Tsamardinos,
  Greasidou and Borboudakis 2018): each bootstrap sample selects the best
  configuration on the in-bag rows and scores it on the out-of-bag rows.
* When the main target has no selected column, ETIA returns the selection
  of the last target of the list.
* FBED stops with an R error when its forward phase selects one column
  (a defect of ``Rfast::cor.fbed``) or none (ETIA's R script reads a
  missing column); ETIA then skips the fold.
* The bootstrap of ETIA runs in joblib workers without a seed, so its
  numbers change from run to run.

The FS result of a fold depends only on the FS configuration, so ETIA-JAX
computes it one time for all the models (ETIA runs R again for each model).
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from .fs import fbed, ses, FSError
from .models import make_model, r2_score, roc_auc

PRED_ALGS = {"random_forest": {"n_estimators": [50, 100], "min_samples_leaf": [0.1], "max_features": ["sqrt"]},
             "linear_regression": {}}
FS_ALGS = {"fbed": {"alpha": [0.05, 0.01], "k": [3, 5], "ind_test_name": ["testIndFisher"]},
           "ses": {"alpha": [0.05, 0.01], "k": [3, 5], "ind_test_name": ["testIndFisher"]}}
DEFAULT_OOS = {"name": "KFoldCV", "folds": 5, "folds_to_run": 2}


def _product(d):
    return [dict(zip(d, v)) for v in itertools.product(*d.values())]


def predictive_configs(pred_algs=PRED_ALGS, fs_algs=FS_ALGS):
    """``PredictiveConfigurator.create_predictive_configs`` (preprocess "none")."""
    out = []
    for pred, pinfo in pred_algs.items():
        for v in _product(pinfo):
            for fsn, finfo in fs_algs.items():
                for k in _product(finfo):
                    c = {"model": pred, **v, "fs_name": fsn, **k}
                    out.append(c)
    return out


def kfold(n, n_splits=5, seed=42):
    """sklearn ``KFold(n_splits, shuffle=True, random_state=seed).split``:
    (train, test) index arrays, both ascending."""
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    sizes = np.full(n_splits, n // n_splits)
    sizes[: n % n_splits] += 1
    out, start = [], 0
    for s in sizes:
        test = np.sort(idx[start:start + s])
        mask = np.ones(n, bool)
        mask[test] = False
        out.append((np.flatnonzero(mask), test))
        start += s
    return out


def run_fs(config, y, X, etia_compat=False):
    if config["fs_name"] == "fbed":
        return fbed(y, X, config["alpha"], config["k"], etia_compat=etia_compat)
    if config["fs_name"] == "ses":
        return ses(y, X, config["alpha"], config["k"])[0]
    raise ValueError(f"Unsupported feature selection algorithm: {config['fs_name']}")


@dataclass
class TargetResult:
    selected: list                    # column names
    best_config: dict
    bbc_score: float
    ci: tuple
    scores: list = field(default_factory=list)       # mean score per configuration


def _score(target_type, y, pred):
    if target_type == "categorical":
        return roc_auc(y, pred)
    return r2_score(y, pred)


def _boot_scores(key, y, pred, B, categorical):
    """Scores of B bootstrap samples of (y, pred), in one vmap."""
    n = len(y)
    idx = jax.random.randint(key, (B, n), 0, n)
    y, pred = jnp.asarray(y), jnp.asarray(pred)

    def r2(i):
        yy, pp = y[i], pred[i]
        return 1 - jnp.sum((yy - pp) ** 2) / jnp.sum((yy - yy.mean()) ** 2)

    def auc(i):
        yy, pp = y[i], pred[i]
        pos = yy == yy.max()
        gt = (pp[:, None] > pp[None, :]) + 0.5 * (pp[:, None] == pp[None, :])
        return jnp.sum(gt * (pos[:, None] & ~pos[None, :])) / (pos.sum() * (~pos).sum())
    return np.asarray(jax.vmap(auc if categorical else r2)(idx))


def bbc_cv(key, y, P, B=1000, categorical=False):
    """BBC-CV: P (configs x rows) out-of-sample predictions. For each
    bootstrap sample, the best configuration on the in-bag rows is scored
    on the out-of-bag rows. Returns the B scores."""
    n = len(y)
    y, P = jnp.asarray(y), jnp.asarray(P)

    def one(k):
        i = jax.random.randint(k, (n,), 0, n)
        inbag = jnp.zeros(n, bool).at[i].set(True)
        w_in = jnp.bincount(i, length=n).astype(P.dtype)
        w_out = (~inbag).astype(P.dtype)

        def r2w(p, w):
            m = jnp.sum(w * y) / jnp.sum(w)
            return 1 - jnp.sum(w * (y - p) ** 2) / jnp.sum(w * (y - m) ** 2)
        best = jnp.argmax(jax.vmap(lambda p: r2w(p, w_in))(P))
        return r2w(P[best], w_out)
    if categorical:
        raise NotImplementedError("BBC-CV with the AUROC")
    return np.asarray(jax.vmap(one)(jax.random.split(key, B)))


def afs_target(X, y, names, configs, target_type="continuous", oos=DEFAULT_OOS, B=1000,
               conf_interval=0.95, seed=0, etia_compat=False):
    """One target, depth 1. X: (n, p) features, y: the target, names: the
    feature names. Returns a ``TargetResult``."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    n = len(y)
    if target_type == "categorical":
        raise NotImplementedError("StratifiedKFold (categorical target) is not ported yet")
    folds = kfold(n, oos.get("folds", 5))
    if not etia_compat:
        folds = folds[: oos.get("folds_to_run", len(folds))]
    fs_cache = {}
    results = []
    for cfg in configs:
        fkey = (cfg["fs_name"], cfg["alpha"], cfg["k"])
        scores, preds = [], []
        for f, (tr, te) in enumerate(folds):
            if (fkey, f) not in fs_cache:
                try:
                    fs_cache[(fkey, f)] = run_fs(cfg, y[tr], X[tr], etia_compat)
                except FSError:                      # ETIA: "Feature selection failed", next fold
                    fs_cache[(fkey, f)] = []
            sel = fs_cache[(fkey, f)]
            if not sel:
                continue
            m = make_model(cfg, target_type).fit(X[np.ix_(tr, sel)], y[tr])
            pr = m.predict(X[np.ix_(te, sel)])
            scores.append(_score(target_type, y[te], pr))
            preds.append((f, te, pr, sel))
        results.append((cfg, float(np.mean(scores)) if scores else None, preds))
    valid = [r for r in results if r[1] is not None]
    if not valid:
        return TargetResult([], None, -np.inf, None, [r[1] for r in results])
    best = int(np.argmax([r[1] for r in valid]))
    cfg, _, preds = valid[best]
    key = jax.random.key(seed)
    if etia_compat:
        yy = np.concatenate([y[te] for _, te, _, _ in preds])
        pp = np.concatenate([pr for _, _, pr, _ in preds])
        b = _boot_scores(key, yy, pp, B, target_type == "categorical")
    else:
        rows = np.concatenate([te for _, te, _, _ in preds])
        full = [r for r in valid if {f for f, *_ in r[2]} == {f for f, *_ in preds}]
        P = np.stack([np.concatenate([pr for _, _, pr, _ in r[2]]) for r in full])
        b = bbc_cv(key, y[rows], P, B, target_type == "categorical")
    lo = (1 - conf_interval) / 2
    ci = tuple(np.percentile(b, [100 * lo, 100 * (1 - lo)]))
    common = set.intersection(*(set(sel) for *_, sel in preds))
    return TargetResult(sorted(names[i] for i in common), cfg, float(np.mean(b)), ci,
                        [r[1] for r in results])


def run_afs(data, target_features, pred_configs=None, depth=1, oos=DEFAULT_OOS, seed=0,
            etia_compat=False):
    """``AFS.run_AFS``. data: a pandas DataFrame; target_features: a list of
    names or a dict name -> type. Returns a dict like ETIA's."""
    import pandas as pd
    if depth < 0:
        raise ValueError("Depth must be a non-negative integer.")
    if isinstance(target_features, list):
        target_features = {t: "unknown" for t in target_features}
    configs = predictive_configs() if pred_configs is None else pred_configs
    selected, res_main = {}, None
    main = next(iter(target_features))

    def recurse(target, ttype, d, visited):
        if d == 0:
            return None, []
        visited = visited | {target}
        cols = [c for c in data.columns if c != target and c not in visited]
        if not cols:
            return None, []
        y = data[target].values
        if ttype == "unknown":
            ttype = "continuous" if pd.api.types.is_numeric_dtype(data[target]) else "categorical"
        r = afs_target(data[cols].values, y, cols, configs, ttype, oos, seed=seed, etia_compat=etia_compat)
        out = set()
        for c in r.selected:
            if c not in visited:
                out |= set(recurse(c, "unknown", d - 1, visited)[1])
                out.add(c)
        return r, sorted(out)

    last = None
    for t, ttype in target_features.items():
        r, sel = recurse(t, ttype, depth, set())
        selected[t] = sel
        last = t
        if t == main:
            res_main = r
    keep = sorted(set().union(*selected.values()) | set(target_features))
    reduced = data[keep]
    without = [c for c in keep if c != main]
    out = dict(original_data=data, reduced_data=reduced,
               best_config=None if res_main is None else res_main.best_config,
               bbc_score=-np.inf if res_main is None else res_main.bbc_score,
               ci=None if res_main is None else res_main.ci, trained_model=None,
               selected_features=selected)
    if not without:
        out["selected_features"] = selected[last] if etia_compat else selected[main]
        return out
    m = make_model(res_main.best_config, "continuous").fit(data[without].values, data[main].values)
    out["trained_model"] = m
    return out
