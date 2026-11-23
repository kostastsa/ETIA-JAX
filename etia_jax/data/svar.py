"""Structural VAR time series as a GenJAX generative function (increment 6).

A model is fixed by an ``SVARSpec``: p variables, a maximum lag L, a
contemporaneous DAG ``B0`` (B0[i, j] != 0 iff x_i(t) -> x_j(t)) and lag
matrices ``B[l]`` (B[l][i, j] != 0 iff x_i(t - l) -> x_j(t), l = 1..L).

* continuous node j: x_j(t) = sum_i B0[i, j] x_i(t) + sum_l sum_i B[l][i, j] x_i(t - l) + eps_j(t)
* discrete node j (k categories): x_j(t) = argmax_c (s_j(t) * w_c + b_c + g_c(t)),
  s_j(t) the linear sum above, g Gumbel noise (a categorical draw).

The random choices are ``"eps"`` (T + burn-in, p) normal noise and ``"gumbel"``
(T + burn-in, p, k) Gumbel noise; a ``jax.lax.scan`` makes the series from them.

``lagged_table`` makes the table of ETIA (time_lagged=True): the columns
X1..Xp at time t, then X1:1..Xp:1 at t - 1, ..., X1:L..Xp:L at t - L.
``window_dag`` gives the true DAG over the window and earlier time steps (as
latent variables), for the oracle tests.
"""
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import genjax
from genjax import gen, normal, Const, const

from ..graphs.dag import random_dag, topological_order


@dataclass
class SVARSpec:
    B0: np.ndarray                 # (p, p) contemporaneous weights (a DAG)
    B: np.ndarray                  # (L, p, p) lag weights, B[l - 1] for lag l
    kinds: list                    # "continuous" or "discrete" per variable
    arity: int = 3
    noise_std: float = 1.0
    cat_w: np.ndarray = None       # (p, arity) slopes of the discrete logits
    cat_b: np.ndarray = None       # (p, arity) intercepts of the discrete logits

    @property
    def p(self):
        return len(self.B0)

    @property
    def L(self):
        return len(self.B)


def random_svar(p, L=1, contemp_degree=1.0, lag_degree=1.0, kinds="continuous", seed=0,
                w_range=(0.3, 0.6), self_lag=0.4, arity=3):
    """A random stable SVAR: a contemporaneous DAG, sparse lag effects and an
    autoregressive term x_j(t - 1) -> x_j(t) for each variable."""
    rng = np.random.default_rng(seed)
    if isinstance(kinds, str):
        kinds = [kinds] * p if kinds != "mixed" else [("discrete" if j % 2 else "continuous") for j in range(p)]
    sign = lambda shape: rng.choice([-1.0, 1.0], size=shape)
    A0 = random_dag(p, contemp_degree, seed) if p > 1 else np.zeros((1, 1), np.int8)
    B0 = A0 * rng.uniform(*w_range, size=(p, p)) * sign((p, p))
    B = np.zeros((L, p, p))
    prob = min(1.0, lag_degree / max(p, 1))
    for l in range(L):
        mask = rng.random((p, p)) < prob / (l + 1)
        B[l] = mask * rng.uniform(*w_range, size=(p, p)) * sign((p, p)) / (l + 1)
    B[0][np.arange(p), np.arange(p)] = self_lag
    cat_w = np.sort(rng.normal(0, 1.5, size=(p, arity)), axis=1)
    cat_b = rng.normal(0, 0.3, size=(p, arity))
    return SVARSpec(B0, B, kinds, arity, 1.0, cat_w, cat_b)


def make_svar(spec: SVARSpec, burn_in=100):
    p, L, k = spec.p, spec.L, spec.arity
    order = topological_order((np.asarray(spec.B0) != 0).astype(np.int8))
    B0, B = jnp.asarray(spec.B0), jnp.asarray(spec.B)
    disc = jnp.asarray([kd == "discrete" for kd in spec.kinds])
    cw, cb = jnp.asarray(spec.cat_w), jnp.asarray(spec.cat_b)

    def step(hist, noise):
        eps, gum = noise                               # (p,), (p, k)
        lagged = jnp.einsum("lip,li->p", B, hist)      # sum_l B[l]^T x(t - l - 1)
        x = jnp.zeros(p)
        for j in order:                                # contemporaneous, in causal order
            s = lagged[j] + x @ B0[:, j]
            cat = jnp.argmax(s * cw[j] + cb[j] + gum[j]).astype(jnp.float64)
            x = x.at[j].set(jnp.where(disc[j], cat, s + spec.noise_std * eps[j]))
        hist = jnp.concatenate([x[None], hist[:-1]], axis=0)
        return hist, x

    @gen
    def svar(T: Const[int]):
        n = T.value + burn_in
        eps = normal(jnp.zeros((n, p)), jnp.ones((n, p))) @ "eps"
        u = genjax.uniform(jnp.full((n, p, k), 1e-12), jnp.ones((n, p, k))) @ "gumbel"
        gum = -jnp.log(-jnp.log(u))
        _, xs = jax.lax.scan(step, jnp.zeros((L, p)), (eps, gum))
        return xs[burn_in:]

    return svar


def simulate(spec: SVARSpec, T, key, burn_in=100):
    """A (T, p) series (float64; discrete columns hold category codes)."""
    tr = genjax.seed(make_svar(spec, burn_in).simulate)(key, const(int(T)))
    return np.asarray(tr.get_retval())


def lagged_table(X, L):
    """The table of ETIA for time_lagged=True: rows t = L..T-1, columns
    [x(t), x(t - 1), ..., x(t - L)] (each block in variable order)."""
    X = np.asarray(X)
    T = len(X)
    return np.concatenate([X[L - l: T - l] for l in range(L + 1)], axis=1)


def lagged_names(p, L):
    """The Tetrad names of ETIA's prepare_data_tetrad: X1..Xp, X1:1.., X1:L.."""
    return [f"X{i + 1}" if l == 0 else f"X{i + 1}:{l}" for l in range(L + 1) for i in range(p)]


def tiers(p, L):
    """ETIA's time_knowledge: the lag-l block is tier L - l (the oldest block is tier 0)."""
    return [list(range((L - t) * p, (L - t + 1) * p)) for t in range(L + 1)]


def window_dag(spec: SVARSpec, extra=None):
    """The true DAG over the window of L + 1 time steps (the columns of
    lagged_table, in the same order) and ``extra`` earlier time steps (latent).
    Returns (A, latent) for dag_to_pag / the oracles."""
    p, L = spec.p, spec.L
    extra = 2 * L if extra is None else extra
    S = L + 1 + extra                             # time slices; slice s holds time t - s
    A = np.zeros((S * p, S * p), np.int8)
    idx = lambda s, i: s * p + i
    for s in range(S):
        for i in range(p):
            for j in range(p):
                if spec.B0[i, j] != 0:
                    A[idx(s, i), idx(s, j)] = 1
                for l in range(1, L + 1):
                    if s + l < S and spec.B[l - 1][i, j] != 0:
                        A[idx(s + l, i), idx(s, j)] = 1
    latent = np.zeros(S * p, bool)
    latent[(L + 1) * p:] = True
    return A, latent
