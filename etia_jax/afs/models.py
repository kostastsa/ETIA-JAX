"""Predictive models of the AFS in JAX (increment 8).

ETIA has two models: ``sklearn.linear_model.LinearRegression`` and a random
forest (``RandomForestRegressor`` / ``Classifier``, ``random_state=42``).

* ``linear_regression``: least squares with an intercept, as sklearn. Its
  predictions agree with sklearn to rounding.
* ``mlp``: a multilayer perceptron with optax (Adam, full batch), in place
  of the random forest. A random forest of sklearn cannot be copied in JAX,
  so the AFS configurations with ``random_forest`` use the MLP; their
  scores are not the scores of ETIA.

Each model has ``fit(X, y)`` and ``predict(X)``; ``fit_predict_folds``
fits one model per fold in one ``vmap`` when the folds have equal sizes.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax


class LinearRegression:
    """Ordinary least squares with an intercept (sklearn LinearRegression)."""

    def fit(self, X, y):
        X = jnp.asarray(X, jnp.float64)
        y = jnp.asarray(y, jnp.float64)
        self.coef_, self.intercept_ = _ols(X, y)
        return self

    def predict(self, X):
        return np.asarray(jnp.asarray(X, jnp.float64) @ self.coef_ + self.intercept_)


@jax.jit
def _ols(X, y):
    xm, ym = X.mean(0), y.mean()
    coef = jnp.linalg.lstsq(X - xm, y - ym)[0]
    return coef, ym - xm @ coef


def _init(key, sizes):
    params = []
    for k, (a, b) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append((jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / a), jnp.zeros(b)))
    return params


def _forward(params, X):
    h = X
    for W, b in params[:-1]:
        h = jax.nn.relu(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[:, 0]


@partial(jax.jit, static_argnames=("hidden", "steps", "classify"))
def _train(key, X, y, hidden, steps, lr, l2, classify):
    xm, xs = X.mean(0), X.std(0) + 1e-12
    ym, ys = (0.0, 1.0) if classify else (y.mean(), y.std() + 1e-12)
    Xs, ysd = (X - xm) / xs, (y - ym) / ys
    params = _init(key, (X.shape[1],) + hidden + (1,))
    opt = optax.adamw(lr, weight_decay=l2)
    state = opt.init(params)

    def loss(p):
        out = _forward(p, Xs)
        if classify:
            return optax.sigmoid_binary_cross_entropy(out, ysd).mean()
        return jnp.mean((out - ysd) ** 2)

    def step(c, _):
        p, s = c
        g = jax.grad(loss)(p)
        u, s = opt.update(g, s, p)
        return (optax.apply_updates(p, u), s), None
    (params, _), _ = jax.lax.scan(step, (params, state), None, length=steps)
    return params, (xm, xs, ym, ys)


class MLP:
    """Feed-forward network (ReLU) with Adam; regression (MSE) or binary
    classification (logistic loss, ``predict`` gives the probability)."""

    def __init__(self, hidden=(64,), steps=500, lr=1e-2, l2=1e-4, classify=False, seed=0):
        self.hidden, self.steps, self.lr, self.l2 = tuple(hidden), int(steps), lr, l2
        self.classify, self.seed = classify, seed

    def fit(self, X, y):
        X = jnp.asarray(X, jnp.float32)
        y = jnp.asarray(y, jnp.float32)
        self.params_, self.scale_ = _train(jax.random.key(self.seed), X, y, self.hidden,
                                           self.steps, self.lr, self.l2, self.classify)
        return self

    def predict(self, X):
        xm, xs, ym, ys = self.scale_
        out = _forward(self.params_, (jnp.asarray(X, jnp.float32) - xm) / xs)
        if self.classify:
            return np.asarray(jax.nn.sigmoid(out), np.float64)
        return np.asarray(out * ys + ym, np.float64)


def make_model(config, target_type="continuous", etia_compat=False):
    """The model of an AFS configuration. ``random_forest`` becomes an MLP
    with ``n_estimators`` / 50 x 32 hidden units (50 -> 32, 100 -> 64)."""
    name = config.get("model")
    if name == "linear_regression":
        return LinearRegression()
    if name in ("random_forest", "mlp"):
        h = int(config.get("hidden", 32 * max(1, int(config.get("n_estimators", 50)) // 50)))
        return MLP(hidden=(h,), classify=target_type == "categorical")
    raise ValueError(f"Unsupported model type: {name}")


def r2_score(y, pred):
    """sklearn ``r2_score`` (one output)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - ss_res / ss_tot


def roc_auc(y, score):
    """Binary AUROC (rank formula with ties), as sklearn ``roc_auc_score``."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("ROC AUC needs two classes")
    pos = y == classes[1]
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
