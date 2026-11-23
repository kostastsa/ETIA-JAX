"""NOTEARS (Zheng et al. 2018) in JAX, as ETIA runs it (increment 7).

ETIA calls ``causalnex.structure.notears.from_pandas(data,
w_threshold=threshold)`` (causalnex 0.12.2). Only ``threshold`` reaches
causalnex: the ``max_iter`` and ``h_tol`` of ETIA's grid are ignored, so
causalnex uses its defaults ``max_iter=100`` and ``h_tol=1e-8``.

The causalnex search (the 2018 code of Zheng et al.):

* score F(W) = 1/(2n) ||X - X W||_F^2 on the raw data (no centering, no L1),
* constraint h(W) = tr(exp(W o W)) - d,
* augmented Lagrangian: minimize F + rho/2 h^2 + alpha h with L-BFGS-B from
  the last solution; while h_new > h/4, rho = 10 rho (up to 1e20); then
  alpha += rho h; stop when h <= h_tol,
* W[|W| <= w_threshold] = 0, and an edge i -> j for each W[i, j] != 0.

ETIA then makes the mark matrix: for each edge (row-major order) M[i, j] = 2,
M[j, i] = 3, so in a 2-cycle the later edge wins. ``matrix_to_pywhy_graph``
adds the edges to a DAG one at a time (row-major) and raises ValueError when
an edge makes a cycle.

Here the inner problem is solved with an L-BFGS solver in JAX (optax, zoom
line search) with the stop rules of scipy's L-BFGS-B (ftol = 2.2e-9,
gtol = 1e-5, at most 15000 steps). The diagonal is fixed at zero by a mask.

Modes:

* ``etia_compat=True``: the causalnex search on the raw data, max_iter=100
  and h_tol=1e-8 whatever the arguments, and ETIA's conversion (ValueError
  on a cycle).
* default: centered data, the given ``max_iter`` and ``h_tol``, and an
  acyclic output: when the thresholded W has a cycle, the edge with the
  smallest |w| on a cycle goes, until no cycle stays.
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax

from ..graphs.encoding import ARROW, TAIL

ETIA_MAX_ITER, ETIA_H_TOL = 100, 1e-8       # causalnex defaults (ETIA's values are ignored)
SCIPY_FTOL = 1e7 * np.finfo(float).eps       # L-BFGS-B factr = 1e7
SCIPY_GTOL = 1e-5
SCIPY_MAXITER = 15000


@dataclass
class NotearsResult:
    W: np.ndarray            # weights after the threshold
    W_raw: np.ndarray        # weights before the threshold
    G: np.ndarray            # mark matrix (ARROW/TAIL)
    h: float                 # h(W_raw)
    n_outer: int             # dual ascent steps
    n_inner: int             # L-BFGS solves (rho increases included)


def _h(W):
    return jnp.trace(jax.scipy.linalg.expm(W * W)) - W.shape[0]


def _value_and_grad(w, X, XtX, mask, rho, alpha):
    """F and its gradient in closed form (the causalnex gradient), with one expm."""
    n, d = X.shape
    W = w.reshape(d, d) * mask
    IW = jnp.eye(d) - W
    loss = 0.5 / n * jnp.sum(IW * (XtX @ IW))
    E = jax.scipy.linalg.expm(W * W)
    h = jnp.trace(E) - d
    f = loss + 0.5 * rho * h * h + alpha * h
    g = -XtX @ IW / n + (rho * h + alpha) * E.T * W * 2
    return f, (g * mask).reshape(-1)


@jax.custom_vjp
def _objective(w, X, XtX, mask, rho, alpha):
    return _value_and_grad(w, X, XtX, mask, rho, alpha)[0]


def _objective_fwd(w, X, XtX, mask, rho, alpha):
    f, g = _value_and_grad(w, X, XtX, mask, rho, alpha)
    return f, g


def _objective_bwd(g, ct):
    # Only w is differentiated (the line search asks for d/dw).
    return ct * g, None, None, None, None, None


_objective.defvjp(_objective_fwd, _objective_bwd)


@partial(jax.jit, static_argnames=("maxiter",))
def _lbfgs(w0, X, mask, rho, alpha, maxiter=SCIPY_MAXITER):
    """Minimize the augmented Lagrangian from w0 with L-BFGS. Stop rules of
    scipy L-BFGS-B: relative decrease <= ftol, or max |grad| <= gtol."""
    XtX = X.T @ X
    fun = partial(_objective, X=X, XtX=XtX, mask=mask, rho=rho, alpha=alpha)
    vg = partial(_value_and_grad, X=X, XtX=XtX, mask=mask, rho=rho, alpha=alpha)
    opt = optax.lbfgs(memory_size=10)

    def cond(c):
        _, _, k, f_old, f, g, _ = c
        rel = (f_old - f) / jnp.maximum(jnp.maximum(jnp.abs(f_old), jnp.abs(f)), 1.0)
        go = (k < maxiter) & (jnp.max(jnp.abs(g)) > SCIPY_GTOL)
        return go & ((k == 0) | (rel > SCIPY_FTOL))

    def body(c):
        w, state, k, _, f, g, _ = c
        updates, state = opt.update(g, state, w, value=f, grad=g, value_fn=fun)
        w_new = optax.apply_updates(w, updates)
        f_new, g_new = vg(w_new)
        return w_new, state, k + 1, f, f_new, g_new, f_new

    f0, g0 = vg(w0)
    c0 = (w0, opt.init(w0), 0, jnp.inf, f0, g0, f0)
    w, _, k, _, f, _, _ = jax.lax.while_loop(cond, body, c0)
    W = w.reshape(mask.shape) * mask
    return w, _h(W), k


def _device(device):
    """NOTEARS is a sequence of small dense steps: the CPU is faster than
    the GPU (kernel launch time). ``device``: "cpu" (default), "gpu", or a
    jax Device."""
    if device is None or device == "cpu":
        return jax.devices("cpu")[0]
    if isinstance(device, str):
        return jax.devices(device)[0]
    return device


def notears_weights(X, max_iter=100, h_tol=1e-8, rho_max=1e20, mask=None, device=None):
    """The augmented Lagrangian loop of causalnex. Returns (W, h, n_outer, n_inner)."""
    with jax.default_device(_device(device)):
        return _notears_weights(X, max_iter, h_tol, rho_max, mask)


def _notears_weights(X, max_iter, h_tol, rho_max, mask):
    X = jnp.asarray(X, dtype=jnp.float64)
    n, d = X.shape
    if X.size == 0:
        raise ValueError("Input data X is empty, cannot learn any structure")
    if not bool(jnp.isfinite(X).all()):
        raise ValueError("Input contains NaN or infinity")
    mask = jnp.asarray(1.0 - np.eye(d) if mask is None else mask, dtype=jnp.float64)
    w_est = jnp.zeros(d * d)
    rho, alpha, h, h_new = 1.0, 0.0, np.inf, np.inf
    n_outer = n_inner = 0
    for n_iter in range(max_iter):
        n_outer += 1
        while rho < rho_max and (h_new > 0.25 * h or h_new == np.inf):
            w_new, h_val, _ = _lbfgs(w_est, X, mask, rho, alpha)
            h_new = float(h_val)
            n_inner += 1
            if h_new > 0.25 * h:
                rho *= 10
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol:
            break
    W = np.asarray(w_est).reshape(d, d) * np.asarray(mask)
    return W, h, n_outer, n_inner


def etia_matrix(W):
    """ETIA's ``structure_model_to_matrix``: edges in row-major order, the
    later edge of a 2-cycle wins."""
    d = len(W)
    M = np.zeros((d, d), dtype=np.int8)
    for i, j in zip(*np.nonzero(W)):
        M[i, j], M[j, i] = ARROW, TAIL
    return M


def etia_check_dag(M):
    """ETIA's ``matrix_to_pywhy_graph`` (DAG): add i -> j in row-major order,
    raise ValueError at the first edge that makes a cycle."""
    d = len(M)
    A = np.zeros((d, d), bool)
    for i in range(d):
        for j in range(d):
            if M[i, j] == ARROW and M[j, i] == TAIL:
                A[i, j] = True
                if _reaches(A, j, i):
                    raise ValueError(f"Adding the edge ({i}, {j}) would create a cycle.")
    return M


def _reaches(A, s, t):
    seen, stack = {s}, [s]
    while stack:
        u = stack.pop()
        if u == t:
            return True
        for v in np.flatnonzero(A[u]):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _break_cycles(W):
    """Remove the smallest |w| edge on a cycle until W is acyclic."""
    W = W.copy()
    while True:
        A = W != 0
        cyc = [(i, j) for i, j in zip(*np.nonzero(A)) if _reaches(A, j, i)]
        if not cyc:
            return W
        i, j = min(cyc, key=lambda e: abs(W[e]))
        W[i, j] = 0.0


def notears(X, threshold=0.8, max_iter=100, h_tol=1e-8, etia_compat=False, device=None):
    """NOTEARS. ``threshold``: |w| <= threshold is set to 0 (ETIA's default 0.8).
    Returns a ``NotearsResult``; ``G`` is the mark matrix of the DAG."""
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold value")
    X = np.asarray(X, dtype=np.float64)
    if etia_compat:
        max_iter, h_tol = ETIA_MAX_ITER, ETIA_H_TOL
    else:
        X = X - X.mean(axis=0)
    W_raw, h, n_outer, n_inner = notears_weights(X, max_iter, h_tol, device=device)
    W = W_raw.copy()
    W[np.abs(W) <= threshold] = 0.0
    if etia_compat:
        G = etia_check_dag(etia_matrix(W))
    else:
        W = _break_cycles(W)
        G = np.zeros(W.shape, dtype=np.int8)
        G[W != 0] = ARROW
        G[(W != 0).T] = TAIL
    return NotearsResult(W=W, W_raw=W_raw, G=G, h=h, n_outer=n_outer, n_inner=n_inner)
