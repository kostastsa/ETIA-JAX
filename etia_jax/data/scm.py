"""Structural causal models as GenJAX generative functions.

A model is fixed by an ``SCMSpec`` (graph + parameters). ``make_scm(spec)``
returns a GenJAX ``@gen`` function of one static argument ``n`` (sample size).
Each node ``j`` is one random choice ``"x{j}"`` of shape ``(n,)``.

Node kinds:

* ``"continuous"``: x_j = b_j + sum_k W[k, j] f(x_k) + sum_d E_dj[x_d] + eps,
  with eps ~ N(0, sigma_j^2) and f the link (identity, tanh, sin).
* ``"discrete"``: x_j ~ Categorical(logits), logits = b_j + sum_k W_kj x_k
  + sum_d E_dj[x_d] (one row of weights per category).
"""
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
import genjax
from genjax import gen, normal, categorical, Const, const

from ..graphs.dag import random_dag, topological_order

LINKS = {"linear": lambda x: x, "tanh": jnp.tanh, "sin": jnp.sin}


@dataclass
class SCMSpec:
    A: np.ndarray                      # 0/1 adjacency, A[i, j] = 1 iff i -> j
    kinds: list                        # "continuous" or "discrete" per node
    arity: np.ndarray                  # categories per discrete node (0 if continuous)
    weights: dict                      # (k, j) -> array of parameters, see sample_params
    bias: list                         # per node: scalar (cont.) or (arity,) logits
    noise_std: np.ndarray              # per continuous node
    link: str = "linear"
    names: list = field(default=None)

    @property
    def n_nodes(self):
        return len(self.A)

    @property
    def order(self):
        return topological_order(self.A)


def sample_params(A, kinds=None, arity=3, link="linear", w_range=(0.5, 2.0),
                  noise_range=(0.5, 1.5), rng=None):
    """Random parameters for graph ``A``. Weights have random sign and
    |w| in ``w_range`` so that the model is faithful with high probability."""
    rng = np.random.default_rng(rng)
    A = np.asarray(A) != 0
    p = len(A)
    kinds = ["continuous"] * p if kinds is None else list(kinds)
    ar = np.array([arity if k == "discrete" else 0 for k in kinds]) if np.isscalar(arity) \
        else np.asarray(arity)

    def w(shape):
        return rng.choice([-1, 1], size=shape) * rng.uniform(*w_range, size=shape)

    weights, bias = {}, []
    for j in range(p):
        for k in np.flatnonzero(A[:, j]):
            out = 1 if kinds[j] == "continuous" else ar[j]
            inp = 1 if kinds[k] == "continuous" else ar[k]
            W = w((inp, out))
            if kinds[k] == "discrete":
                W = W - W.mean(0, keepdims=True)  # identify category effects
            weights[(int(k), j)] = W
        bias.append(0.0 if kinds[j] == "continuous" else rng.normal(0, 0.5, ar[j]))
    noise = rng.uniform(*noise_range, size=p)
    return SCMSpec(A=A.astype(np.int8), kinds=kinds, arity=ar, weights=weights,
                   bias=bias, noise_std=noise, link=link)


def make_scm(spec: SCMSpec):
    """GenJAX generative function ``scm(n: Const[int]) -> (n, p) array``."""
    order = spec.order
    parents = {j: [int(k) for k in np.flatnonzero(spec.A[:, j])] for j in range(spec.n_nodes)}
    W = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in spec.weights.items()}
    f = LINKS[spec.link]
    p = spec.n_nodes

    @gen
    def scm(n: Const[int]):
        n = n.value
        cols = [None] * p
        for j in order:
            out = 1 if spec.kinds[j] == "continuous" else int(spec.arity[j])
            eta = jnp.zeros((n, out)) + jnp.asarray(spec.bias[j], dtype=jnp.float32)
            for k in parents[j]:
                if spec.kinds[k] == "continuous":
                    eta = eta + f(cols[k])[:, None] * W[(k, j)][0]
                else:
                    eta = eta + W[(k, j)][cols[k].astype(jnp.int32)]
            if spec.kinds[j] == "continuous":
                xj = normal(eta[:, 0], float(spec.noise_std[j])) @ f"x{j}"
            else:
                xj = categorical(eta) @ f"x{j}"
            cols[j] = xj.astype(jnp.float32)
        return jnp.stack(cols, axis=1)

    return scm


def simulate(spec: SCMSpec, n: int, key):
    """Draw ``n`` samples. Returns (data (n, p) numpy array, GenJAX trace)."""
    scm = make_scm(spec)
    tr = genjax.seed(scm.simulate)(key, const(int(n)))
    return np.asarray(tr.get_retval()), tr


def random_scm(n_nodes, expected_degree=2.0, kinds="continuous", arity=3,
               link="linear", rng=None, max_parents=None):
    """Random DAG + random parameters. ``kinds``: "continuous", "discrete",
    "mixed" (half and half, random), or a list."""
    rng = np.random.default_rng(rng)
    A = random_dag(n_nodes, expected_degree, rng, max_parents)
    if kinds == "mixed":
        kinds = list(rng.permutation(["continuous", "discrete"] * (n_nodes // 2)
                                     + ["continuous"] * (n_nodes % 2)))
    elif isinstance(kinds, str):
        kinds = [kinds] * n_nodes
    return sample_params(A, kinds, arity, link, rng=rng)
