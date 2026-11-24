"""A random forest of CART trees in JAX (increment 9).

ETIA uses scikit-learn forests in two places:

* the OCT: ``RandomForestRegressor(n_estimators=100, min_samples_leaf=0.01,
  max_depth=10, n_jobs=-1)``, without a seed;
* the AFS: ``RandomForestRegressor`` / ``RandomForestClassifier`` with
  ``max_features="sqrt"`` and ``random_state=42``.

This module grows the same kind of forest: bootstrap samples (the counts are
the sample weights), exact CART splits at the midpoints of adjacent values,
the squared error criterion, ``max_depth``, ``min_samples_leaf`` (an
integer, or a fraction of the training rows, rounded up), and
``max_features`` features drawn at random in each node. A classifier is a
regression forest on the one-hot targets: the squared error of a one-hot
vector is the Gini impurity, and the leaf means are the class
probabilities. The random numbers are not the numbers of scikit-learn, so
the trees are not the same trees; the tests compare the forests by their
accuracy.

The trees grow level by level with fixed shapes (a complete binary tree
of depth ``max_depth``), in one ``vmap`` over the trees. In one level, each
feature sorts the rows by (node, value) and finds the best split of each
node with cumulative sums.
"""
from functools import partial
import math

import numpy as np
import jax
import jax.numpy as jnp


def _min_leaf(min_samples_leaf, n):
    if isinstance(min_samples_leaf, float):
        return max(1, int(math.ceil(min_samples_leaf * n)))
    return int(min_samples_leaf)


def _n_features(max_features, d):
    """scikit-learn's rule; 0 means all the features."""
    if max_features is None or max_features == 1.0:
        return 0
    if max_features == "sqrt":
        return max(1, int(math.sqrt(d)))
    if max_features == "log2":
        return max(1, int(math.log2(d)))
    if isinstance(max_features, float):
        return max(1, int(max_features * d))
    return int(max_features)


def _pad_cols(d):
    return max(4, 1 << (d - 1).bit_length())


def _grow_tree(key, X, Y, orders, n_real, msl, k_feat, depth):
    """One tree. X (n, d), Y (n, m); rows from n_real on are padding."""
    n, d = X.shape
    m = Y.shape[1]
    M = 2 ** (depth + 1) - 1
    kb, kf = jax.random.split(key)
    draws = jax.random.randint(kb, (n,), 0, jnp.maximum(n_real, 1))
    w = jnp.zeros(n).at[draws].add(jnp.where(jnp.arange(n) < n_real, 1.0, 0.0))
    feat = jnp.zeros(M, jnp.int32)
    thr = jnp.zeros(M)
    split = jnp.zeros(M, bool)
    value = jnp.zeros((M, m))
    pos = jnp.zeros(n, jnp.int32)
    active = w > 0
    wy = w[:, None] * Y
    wyy = w * (Y ** 2).sum(1)
    idx = jnp.arange(n)
    for lev in range(depth + 1):
        base, ns = 2 ** lev - 1, 2 ** lev
        seg = jnp.where(active, pos - base, ns)
        Sw = jax.ops.segment_sum(w, seg, ns + 1)
        Sy = jax.ops.segment_sum(wy, seg, ns + 1)
        Syy = jax.ops.segment_sum(wyy, seg, ns + 1)
        Sc = jax.ops.segment_sum(active.astype(jnp.int32), seg, ns + 1)
        mean = Sy / jnp.where(Sw > 0, Sw, 1.0)[:, None]
        value = value.at[base:base + ns].set(mean[:ns])
        if lev == depth:
            break
        sse = Syy - (Sy ** 2).sum(1) / jnp.where(Sw > 0, Sw, 1.0)

        def one_feature(f):
            perm = orders[f]
            ix = perm[jnp.argsort(seg[perm], stable=True)]
            xs, ws, ss = X[ix, f], w[ix], seg[ix]
            cw = jnp.cumsum(ws)
            cy = jnp.cumsum(wy[ix], 0)
            cc = jnp.cumsum(active[ix].astype(jnp.int32))
            start = jax.ops.segment_min(idx, ss, ns + 1)
            st = jnp.clip(start, 0, n - 1)
            ew = (cw[st] - ws[st])[ss]
            ey = (cy[st] - wy[ix][st])[ss]
            ec = (cc[st] - active[ix][st])[ss]
            Lw, Ly, Lc = cw - ew, cy - ey, cc - ec
            Rw, Ry, Rc = Sw[ss] - Lw, Sy[ss] - Ly, Sc[ss] - Lc
            nxt = jnp.minimum(idx + 1, n - 1)
            ok = ((ss < ns) & (idx < n - 1) & (ss[nxt] == ss) & (xs < xs[nxt])
                  & (Lc >= msl) & (Rc >= msl))
            score = (Ly ** 2).sum(1) / jnp.where(Lw > 0, Lw, 1.0) + (Ry ** 2).sum(1) / jnp.where(Rw > 0, Rw, 1.0)
            score = jnp.where(ok, score, -jnp.inf)
            best = jax.ops.segment_max(score, ss, ns + 1)
            bp = jax.ops.segment_min(jnp.where(ok & (score == best[ss]), idx, n), ss, ns + 1)
            bp = jnp.clip(bp, 0, n - 2)
            t = (xs[bp] + xs[bp + 1]) / 2
            t = jnp.where(t == xs[bp + 1], xs[bp], t)
            big = jnp.where(ss < ns, xs, -jnp.inf)
            small = jnp.where(ss < ns, xs, jnp.inf)
            nonconst = jax.ops.segment_max(big, ss, ns + 1) > jax.ops.segment_min(small, ss, ns + 1)
            return best, t, nonconst

        best, t, nonconst = jax.vmap(one_feature)(jnp.arange(d))        # (d, ns+1)
        u = jax.random.uniform(jax.random.fold_in(kf, lev), (ns + 1, d))
        u = jnp.where(nonconst.T, u, 2.0)
        rank = jnp.argsort(jnp.argsort(u, 1), 1)
        fmask = jnp.where(k_feat > 0, rank < k_feat, True).T
        gain = jnp.where(fmask, best, -jnp.inf)
        bf = jnp.argmax(gain, 0)
        bs = jnp.max(gain, 0)
        bt = t[bf, jnp.arange(ns + 1)]
        pure = sse <= 1e-12 * jnp.maximum(Sw, 1.0)
        do = (bs > -jnp.inf) & ~pure & (Sc >= 2 * msl)
        feat = feat.at[base:base + ns].set(bf[:ns].astype(jnp.int32))
        thr = thr.at[base:base + ns].set(bt[:ns])
        split = split.at[base:base + ns].set(do[:ns])
        sidx = jnp.minimum(seg, ns)
        go = active & do[sidx]
        left = X[idx, bf[sidx]] <= bt[sidx]
        pos = jnp.where(go, 2 * pos + jnp.where(left, 1, 2), pos)
        active = go
    return feat, thr, split, value


@partial(jax.jit, static_argnames=("depth", "n_trees"))
def _fit(key, X, Y, n_real, msl, k_feat, depth, n_trees):
    orders = jnp.argsort(X.T, axis=1)
    keys = jax.random.split(key, n_trees)
    return jax.vmap(lambda k: _grow_tree(k, X, Y, orders, n_real, msl, k_feat, depth))(keys)


@partial(jax.jit, static_argnames=("depth",))
def _predict(trees, X, depth):
    feat, thr, split, value = trees

    def one(f, t, s, v):
        node = jnp.zeros(X.shape[0], jnp.int32)
        for _ in range(depth):
            go = s[node]
            left = X[jnp.arange(X.shape[0]), f[node]] <= t[node]
            node = jnp.where(go, 2 * node + jnp.where(left, 1, 2), node)
        return v[node]
    return jax.vmap(one)(feat, thr, split, value).mean(0)


class RandomForest:
    """``RandomForestRegressor`` (``classify=False``) or
    ``RandomForestClassifier`` (``classify=True``) of scikit-learn, in JAX.

    ``max_depth=None`` means depth 10 at most here (fixed shapes): a deeper
    tree needs more memory, so give ``max_depth`` for small leaves.
    """

    def __init__(self, n_estimators=100, min_samples_leaf=1, max_depth=None, max_features=1.0,
                 classify=False, seed=0):
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = 10 if max_depth is None else int(max_depth)
        self.max_features = max_features
        self.classify = classify
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X[:, None]
        n, d = X.shape
        if self.classify:
            self.classes_, yi = np.unique(y, return_inverse=True)
            Y = np.eye(len(self.classes_))[yi]
        else:
            Y = y.astype(np.float64).reshape(n, -1)
        msl = _min_leaf(self.min_samples_leaf, n)
        k = _n_features(self.max_features, d)
        # fixed shapes (fewer compilations): rows to a multiple of 128 (weight 0),
        # columns to a power of 2 (constant columns never split)
        self.d_ = d
        dp = _pad_cols(d)
        Xp = np.zeros((-(-n // 128) * 128, dp))
        Xp[:n, :d] = X
        Yp = np.zeros((len(Xp), Y.shape[1]))
        Yp[:n] = Y
        self.trees_ = _fit(jax.random.PRNGKey(self.seed), jnp.asarray(Xp), jnp.asarray(Yp), n, msl, k,
                           self.max_depth, self.n_estimators)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, np.float64).reshape(len(X), -1)
        Xp = np.zeros((-(-len(X) // 128) * 128, _pad_cols(self.d_)))
        Xp[:len(X), :X.shape[1]] = X
        return np.asarray(_predict(self.trees_, jnp.asarray(Xp), self.max_depth))[:len(X)]

    def predict(self, X):
        out = self.predict_proba(X)
        if self.classify:
            return self.classes_[out.argmax(1)]
        return out[:, 0] if out.shape[1] == 1 else out


def fit_predict_many(X_train, Y_train, X_test, n_real, seeds, n_estimators=100, min_samples_leaf=0.01,
                     max_depth=10, max_features=1.0):
    """Many regression forests, one after the other on the device
    (``lax.map``): X_train (B, n, d), Y_train (B, n, m), X_test (B, t, d),
    n_real (B,) real training rows (padding at the end), seeds (B,).
    Padded feature columns must be constant. Returns (B, t, m). The shapes
    are padded (B to a multiple of 8, rows to a multiple of 128, columns to
    a power of 2), so that new calls seldom compile again."""
    X_train, Y_train, X_test = (np.asarray(a, np.float64) for a in (X_train, Y_train, X_test))
    B, n, d = X_train.shape
    t, m = X_test.shape[1], Y_train.shape[2]
    Bp, npad, tp, dp = -(-B // 8) * 8, -(-n // 128) * 128, -(-t // 128) * 128, _pad_cols(d)
    Xtr = np.zeros((Bp, npad, dp)); Xtr[:B, :n, :d] = X_train
    Ytr = np.zeros((Bp, npad, m)); Ytr[:B, :n] = Y_train
    Xte = np.zeros((Bp, tp, dp)); Xte[:B, :t, :d] = X_test
    nr = np.zeros(Bp, np.int64); nr[:B] = n_real
    sd = np.zeros(Bp, np.uint32); sd[:B] = seeds
    msl = np.ones(Bp, np.int64); msl[:B] = [_min_leaf(min_samples_leaf, int(r)) for r in n_real]
    k = _n_features(max_features, d)
    out = _many(jnp.asarray(Xtr), jnp.asarray(Ytr), jnp.asarray(Xte), jnp.asarray(nr), jnp.asarray(msl),
                jnp.asarray(sd), k, int(max_depth), int(n_estimators))
    return np.asarray(out)[:B, :t]


@partial(jax.jit, static_argnames=("depth", "n_trees"))
def _many(Xtr, Ytr, Xte, n_real, msl, seeds, k, depth, n_trees):
    def one(a):
        x, y, xt, r, ms, s = a
        trees = _fit(jax.random.PRNGKey(s), x, y, r, ms, k, depth, n_trees)
        return _predict(trees, xt, depth)
    return jax.lax.map(one, (Xtr, Ytr, Xte, n_real, msl, seeds))
