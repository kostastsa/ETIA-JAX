"""SAM, the Structural Agnostic Model (Kalainathan et al. 2018), in JAX and
optax (increment 7).

ETIA calls ``cdt.causality.graph.SAM(lr, dlr, mixed_data, lambda1, lambda2,
nh, dnh, train_epochs, test_epochs, batch_size, losstype).predict(data)``
(cdt 0.6.0). The other settings are the cdt defaults: ``dagloss=True``,
``dagstart=0.5``, ``dagpenalization=0``, ``dagpenalization_increase=0.01``,
``functional_complexity='l2_norm'``, ``hlayers=2``, ``dhlayers=2``,
``sampling_type='sigmoidproba'``, ``linear=False``, ``nruns=8``.

One run of ``run_SAM`` (this module copies it step by step):

* data: ``sklearn.preprocessing.scale`` (float32). With ``mixed_data``, a
  column with fewer than 50 values is one-hot (``pd.get_dummies``).
* generators: one network per variable, input = the other columns gated by
  a sampled graph, plus one N(0, 1) noise input. Linear3D (nh) - batch norm -
  tanh - Linear3D (nh) - batch norm - tanh - Linear3D (1 or the categories).
* graph: logits w (start 2); a draw is the straight-through Bernoulli
  ``1[sigmoid(2w + logistic) > 0.5]``; the filters are sigmoid(2w).
* discriminator: for each variable c, the real row with column c replaced by
  the generated column; Linear(dnh) - batch norm - LeakyReLU(0.2), two times,
  then Linear(1).
* each batch: one Adam step of the discriminator on the detached generated
  data, then one Adam step of the generators and the graph logits on
  gen_loss + lambda1/n sum(draw) + lambda2/n sum ||param||_F
  + (epoch - train/2) 0.01 h(filters^2) (h = sum_k tr(A^k)/(k-1)!, k=1..d)
  after the first half of the training epochs.
* test epochs: the filters of each batch are added; the sum is divided by
  ``test`` (not by the number of batches).
* ``predict``: the mean of the 8 runs (runs with NaN are dropped), as a
  ``networkx.DiGraph(W)``.

Findings (ETIA and cdt):

* The output of ETIA is the weight matrix W: no threshold. Its values go up
  to the number of batches per epoch, not to 1.
* ETIA's ``pywhy_graph_to_matrix`` writes each edge i -> j as M[i, j] = 2,
  M[j, i] = 3, in the edge order of the DiGraph. W has all off-diagonal
  entries, so the result is the complete DAG with j -> i for each i < j,
  for all data.
* With ``mixed_data``, ``permutation_matrix`` takes the column of the
  skeleton at the index of the variable, not at the first column of the
  variable: when a categorical variable comes before, a generator gets the
  wrong inputs (its own one-hot columns included).

Modes:

* ``etia_compat=True``: all of the above. ``SamResult.W`` is ETIA's matrix and
  ``SamResult.G`` is ETIA's mark matrix.
* default: the filters are averaged over the test batches (W in [0, 1]),
  the mixed-data skeleton is correct, and ``G`` is the DAG of W > 0.5 (the
  edge with the smallest W on a cycle goes, until no cycle stays).

The random draws are not the draws of torch, so only the accuracy and the
speed are compared with ETIA, not the output.
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax

from ..graphs.encoding import ARROW, TAIL

F32 = jnp.float32
BN_EPS = 1e-5
CATEGORICAL_THRESHOLD = 50

ETIA_DEFAULTS = dict(lr=0.01, dlr=0.001, mixed_data=False, lambda1=10.0, lambda2=0.001,
                     nh=20, dnh=200, train_epochs=3000, test_epochs=1000, batch_size=100,
                     losstype="fgan")
CDT_FIXED = dict(dagloss=True, dagstart=0.5, dagpenalization=0.0, dagpenalization_increase=0.01,
                 hlayers=2, dhlayers=2, nruns=8)


@dataclass
class SamResult:
    W: np.ndarray        # (d, d) mean filters over the runs
    G: np.ndarray        # mark matrix
    runs: np.ndarray     # (nruns, d, d) filters of each run
    n_batches: int


# ---------------------------------------------------------------- data

def prepare_data(X, mixed_data=False):
    """``run_SAM`` data: scaled columns, or one-hot columns with
    ``mixed_data`` (fewer than 50 values). Returns (data float32, cat_sizes)."""
    X = np.asarray(X)
    if not mixed_data:
        Z = (X - X.mean(0)) / _std(X)
        return Z.astype(np.float32), None
    blocks = []
    for j in range(X.shape[1]):
        dum = pd.get_dummies(X[:, j]).values
        if dum.shape[1] < CATEGORICAL_THRESHOLD:
            blocks.append(dum.astype(np.float64))
        else:
            x = X[:, [j]].astype(np.float64)
            blocks.append((x - x.mean(0)) / _std(x))
    sizes = [b.shape[1] for b in blocks]
    return np.concatenate(blocks, 1).astype(np.float32), sizes


def _std(X):
    s = X.std(0)
    return np.where(s == 0, 1.0, s)          # sklearn scale: no division by 0


def _structure(d_vars, cat_sizes, etia_compat):
    """cat_reshape (vars x cols) and the input selection P (vars, cols, max_dim)
    of ``SAM_generators``."""
    if cat_sizes is None:
        C = np.eye(d_vars, dtype=np.float32)
    else:
        C = np.zeros((d_vars, sum(cat_sizes)), np.float32)
        for v, (c, cum) in enumerate(zip(cat_sizes, np.cumsum(cat_sizes))):
            C[v, cum - c:cum] = 1
    cols = C.shape[1]
    S = C.T @ (1 - np.eye(d_vars, dtype=np.float32)) @ C           # cols x cols
    first = np.concatenate([[0], np.cumsum(C.sum(1))[:-1]]).astype(int)
    max_dim = int(S.sum(0).max())
    P = np.zeros((d_vars, cols, max_dim), np.float32)
    for v in range(d_vars):
        col = v if etia_compat else first[v]                       # cdt uses v (defect)
        sel = np.flatnonzero(S[:, col])
        P[v, sel, np.arange(len(sel))] = 1
    return C, P, max_dim


# ---------------------------------------------------------------- networks

def _uniform(key, shape, bound):
    return jax.random.uniform(key, shape, F32, -bound, bound)


def init_params(key, d_vars, cols, max_dim, out_dim, nh, dnh, hlayers=2, dhlayers=2):
    """The torch initial values: Linear3D and Linear U(-1/sqrt(fan_in), ..),
    batch norm 1/0 (generators) and U(0, 1)/0 (discriminator)."""
    ks = iter(jax.random.split(key, 64))
    gen = {"W": [], "b": [], "g": [], "beta": []}
    dims = [max_dim + 1] + [nh] * hlayers
    for l in range(hlayers):
        fin = dims[l]
        gen["W"].append(_uniform(next(ks), (d_vars, fin, nh), 1 / np.sqrt(fin)))
        gen["b"].append(_uniform(next(ks), (d_vars, nh), 1 / np.sqrt(fin)))
        gen["g"].append(jnp.ones((d_vars * nh,), F32))
        gen["beta"].append(jnp.zeros((d_vars * nh,), F32))
    gen["Wo"] = _uniform(next(ks), (d_vars, nh, out_dim), 1 / np.sqrt(nh))
    gen["bo"] = _uniform(next(ks), (d_vars, out_dim), 1 / np.sqrt(nh))
    disc = {"W": [], "b": [], "g": [], "beta": []}
    ddims = [cols] + [dnh] * dhlayers
    for l in range(dhlayers):
        fin = ddims[l]
        disc["W"].append(_uniform(next(ks), (fin, dnh), 1 / np.sqrt(fin)))
        disc["b"].append(_uniform(next(ks), (dnh,), 1 / np.sqrt(fin)))
        disc["g"].append(jax.random.uniform(next(ks), (dnh,), F32))
        disc["beta"].append(jnp.zeros((dnh,), F32))
    disc["Wo"] = _uniform(next(ks), (dnh, 1), 1 / np.sqrt(dnh))
    disc["bo"] = _uniform(next(ks), (1,), 1 / np.sqrt(dnh))
    graph = jnp.full((d_vars, d_vars), 2.0, F32)
    return gen, disc, graph


def _bn(x, g, b, axes):
    m = x.mean(axes, keepdims=True)
    v = ((x - m) ** 2).mean(axes, keepdims=True)
    return (x - m) / jnp.sqrt(v + BN_EPS) * g + b


def generate(gen, x, A, noise, C, P, cat_sizes):
    """x (B, cols), A (vars, vars) drawn graph, noise (B, vars) -> (B, cols)."""
    B = x.shape[0]
    d = A.shape[0]
    adj = C.T @ A                                                  # cols x vars
    z = jnp.einsum("bi,ic,cik->bck", x, adj, P)                    # (B, vars, max_dim)
    h = jnp.concatenate([z, noise[:, :, None]], 2)
    for W, b, g, beta in zip(gen["W"], gen["b"], gen["g"], gen["beta"]):
        h = jnp.einsum("bci,cio->bco", h, W) + b
        nh = h.shape[2]
        h = _bn(h.reshape(B, d * nh), g, beta, 0).reshape(B, d, nh)
        h = jnp.tanh(h)
    out = jnp.einsum("bci,cio->bco", h, gen["Wo"]) + gen["bo"]
    if cat_sizes is None:
        return out[:, :, 0]
    parts = [jax.nn.softmax(out[:, v, :s], axis=1) if s > 1 else out[:, v, :s]
             for v, s in enumerate(cat_sizes)]
    return jnp.concatenate(parts, 1)


def discriminate(disc, inp):
    """inp (B, ..., cols) -> (B, ..., 1); batch norm over the batch axis only."""
    h = inp
    for W, b, g, beta in zip(disc["W"], disc["b"], disc["g"], disc["beta"]):
        h = h @ W + b
        h = _bn(h, g, beta, 0)
        h = jnp.where(h >= 0, h, 0.2 * h)
    return h @ disc["Wo"] + disc["bo"]


def _mix(gen_x, obs, C):
    """(B, vars, cols): the real row with the columns of variable v replaced."""
    return obs[:, None, :] * (1 - C)[None] + gen_x[:, None, :] * C[None]


def _norm(p):
    """Frobenius norm with gradient 0 at 0 (as torch.norm)."""
    s = jnp.sum(p * p)
    safe = jnp.where(s > 0, s, 1.0)
    return jnp.where(s > 0, jnp.sqrt(safe), 0.0)


def _bce(logit, target):
    return jnp.maximum(logit, 0) - logit * target + jnp.log1p(jnp.exp(-jnp.abs(logit)))


def notears_constr(A):
    """cdt ``notears_constr``: sum_{k=1..d} tr(A^k) / (k-1)!."""
    d = A.shape[0]

    def body(k, c):
        M, tot = c
        M = M @ A / k
        return M, tot + jnp.trace(M)
    _, tot = jax.lax.fori_loop(1, d, body, (A, jnp.trace(A)))
    return tot


# ---------------------------------------------------------------- one run

def _run(key, data, C, P, cat_sizes, cfg):
    n, cols = data.shape
    d = C.shape[0]
    B = cfg["batch_size"]
    nb = n // B
    train, test = cfg["train"], cfg["test"]
    n_ep = train + test
    lam1, lam2 = cfg["lambda1"] / n, cfg["lambda2"] / n
    out_dim = 1 if cat_sizes is None else max(cat_sizes)
    mask = 1 - jnp.eye(d, dtype=F32)
    losstype = cfg["losstype"]

    k0, key = jax.random.split(key)
    gen, disc, graph = init_params(k0, d, cols, P.shape[2], out_dim, cfg["nh"], cfg["dnh"])
    g_opt = optax.adam(cfg["lr"])
    d_opt = optax.adam(cfg["dlr"])
    gp = (gen, graph)
    g_state, d_state = g_opt.init(gp), d_opt.init(disc)

    def draw(graph, u):
        logit = 2 * graph + (jnp.log(u) - jnp.log(1 - u))
        soft = jax.nn.sigmoid(logit)
        hard = (soft > 0.5).astype(F32)
        return (hard - jax.lax.stop_gradient(soft) + soft) * mask

    def gen_loss(gp, disc, x, u, noise, epoch):
        gen, graph = gp
        A = draw(graph, u)
        gx = generate(gen, x, A, noise, C, P, cat_sizes)
        if losstype == "mse":
            gl = jnp.mean((gx - x) ** 2)
        else:
            dv = discriminate(disc, _mix(gx, x, C))[..., 0]          # (B, vars)
            if losstype == "fgan":
                gl = -jnp.exp(dv - 1).mean(0).sum()
            else:
                gl = _bce(dv, 1.0).mean(1).sum()
        filt = jax.nn.sigmoid(2 * graph) * mask
        l2 = sum(_norm(p) for p in jax.tree_util.tree_leaves(gen))
        loss = gl + lam1 * A.sum() + lam2 * l2
        if cfg["dagloss"]:
            start = train * cfg["dagstart"]
            coef = cfg["dagpenalization"] + (epoch - start) * cfg["dagpenalization_increase"]
            loss = loss + jnp.where(epoch > start, coef * notears_constr(filt * filt), 0.0)
        return loss, filt

    def disc_loss(disc, gx, x):
        fake = discriminate(disc, _mix(gx, x, C))[..., 0]
        real = discriminate(disc, x)[..., 0]
        if losstype == "fgan":
            return jnp.exp(fake - 1).mean(0).sum() / d - real.mean()
        return _bce(fake, 0.0).mean(1).sum() / d + _bce(real, 1.0).mean()

    def step(carry, inp):
        gp, disc, g_state, d_state, acc = carry
        x, k, epoch = inp
        ku, kn = jax.random.split(k)
        u = jax.random.uniform(ku, (d, d), F32, minval=jnp.finfo(F32).tiny, maxval=1.0)
        noise = jax.random.normal(kn, (x.shape[0], d), F32)
        if losstype != "mse":
            A = draw(gp[1], u)
            gx = jax.lax.stop_gradient(generate(gp[0], x, A, noise, C, P, cat_sizes))
            dg = jax.grad(disc_loss)(disc, gx, x)
            upd, d_state = d_opt.update(dg, d_state, disc)
            disc = optax.apply_updates(disc, upd)
        (_, filt), gg = jax.value_and_grad(gen_loss, has_aux=True)(gp, disc, x, u, noise, epoch)
        last = epoch == n_ep - 1
        upd, g_state_new = g_opt.update(gg, g_state, gp)
        gp_new = optax.apply_updates(gp, upd)
        gp = jax.tree_util.tree_map(lambda a, b: jnp.where(last, a, b), gp, gp_new)
        g_state = jax.tree_util.tree_map(lambda a, b: jnp.where(last, a, b), g_state, g_state_new)
        acc = acc + jnp.where(epoch >= train, filt, 0.0)
        return (gp, disc, g_state, d_state, acc), None

    def epoch_fn(carry, inp):
        epoch, k = inp
        kp, kb = jax.random.split(k)
        perm = jax.random.permutation(kp, n)[: nb * B].reshape(nb, B)
        xs = data[perm]
        ks = jax.random.split(kb, nb)
        carry, _ = jax.lax.scan(step, carry, (xs, ks, jnp.full((nb,), epoch)))
        return carry, None

    carry = (gp, disc, g_state, d_state, jnp.zeros((d, d), F32))
    keys = jax.random.split(key, n_ep)
    carry, _ = jax.lax.scan(epoch_fn, carry, (jnp.arange(n_ep), keys))
    return carry[4]


@partial(jax.jit, static_argnames=("cat_sizes", "cfg_items"))
def _run_many(keys, data, C, P, cat_sizes, cfg_items):
    cfg = dict(cfg_items)
    return jax.vmap(lambda k: _run(k, data, C, P, cat_sizes, cfg))(keys)


def _break_cycles(W, thr):
    from .notears import _reaches
    W = np.where(W > thr, W, 0.0)
    while True:
        A = W != 0
        cyc = [(i, j) for i, j in zip(*np.nonzero(A)) if _reaches(A, j, i)]
        if not cyc:
            return W
        i, j = min(cyc, key=lambda e: W[e])
        W[i, j] = 0.0


def etia_matrix(W):
    """ETIA's ``pywhy_graph_to_matrix`` on ``DiGraph(W)``: edges in row-major
    order, each writes M[i, j] = 2, M[j, i] = 3 (the later edge wins)."""
    d = len(W)
    M = np.zeros((d, d), np.int8)
    for i, j in zip(*np.nonzero(W)):
        M[i, j], M[j, i] = ARROW, TAIL
    return M


def sam(X, lr=0.01, dlr=0.001, mixed_data=False, lambda1=10.0, lambda2=0.001, nh=20,
        dnh=200, train_epochs=3000, test_epochs=1000, batch_size=100, losstype="fgan",
        nruns=8, seed=0, threshold=0.5, etia_compat=False):
    """SAM with ETIA's defaults. Returns a ``SamResult``."""
    if not (0 < lr < 1 and 0 < dlr < 1):
        raise ValueError("Learning rates (lr and dlr) must be between 0 and 1")
    if losstype not in ("fgan", "gan", "mse"):
        raise ValueError("losstype must be one of 'fgan', 'gan', or 'mse'")
    data, cat_sizes = prepare_data(X, mixed_data)
    n = data.shape[0]
    B = n if batch_size in (-1, None) else int(batch_size)
    nb = n // B
    if nb == 0:
        raise ValueError(f"batch_size {B} is larger than the number of rows {n}")
    d = np.asarray(X).shape[1]
    C, P, _ = _structure(d, None if cat_sizes is None else cat_sizes, etia_compat)
    cfg = dict(CDT_FIXED, batch_size=B, train=int(train_epochs), test=int(test_epochs),
               lambda1=float(lambda1), lambda2=float(lambda2), nh=int(nh), dnh=int(dnh),
               lr=float(lr), dlr=float(dlr), losstype=losstype)
    cfg.pop("nruns"); cfg.pop("hlayers"); cfg.pop("dhlayers")
    keys = jax.random.split(jax.random.key(seed), nruns)
    runs = np.asarray(_run_many(keys, jnp.asarray(data), jnp.asarray(C), jnp.asarray(P),
                                None if cat_sizes is None else tuple(cat_sizes),
                                tuple(sorted(cfg.items()))), dtype=np.float64)
    runs = runs / test_epochs if etia_compat else runs / (test_epochs * nb)
    ok = ~np.isnan(runs).any(axis=(1, 2))
    if not ok.any():
        raise AssertionError("All solutions contain NaNs")
    W = runs[ok].mean(0)
    if etia_compat:
        G = etia_matrix(W)
    else:
        Wd = _break_cycles(W.copy(), threshold)
        G = np.zeros((d, d), np.int8)
        G[Wd != 0] = ARROW
        G[(Wd != 0).T] = TAIL
    return SamResult(W=W, G=G, runs=runs, n_batches=nb)
