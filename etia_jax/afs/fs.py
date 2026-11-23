"""FBED and SES with the Fisher test, as ETIA runs them (increment 8).

ETIA's AFS writes the training rows to a CSV file and runs
``feature_selectors/fbed_with_idx.R`` or ``ses_with_idx.R`` (MXM 1.5.8):

* FBED: ``fbed.reg(target, x, threshold=alpha, test="testIndFisher", K=k)``.
  With the Fisher test this is ``Rfast::cor.fbed`` (forward phase, K extra
  runs) and then ``MXM::bs.reg`` -> ``cor.bsreg`` (backward phase). The
  output is the list of the selected columns in the forward sequence,
  without the columns that the backward phase removes.
* SES: ``SES(target, x, max_k=k, threshold=alpha, test="testIndFisher")``,
  that is ``InternalSES`` (``IdentifyEquivalence``, ``max_min_assoc``) with
  ``testIndFisher``. The output is ``selectedVars`` (ascending).

This module copies these R functions step by step, with their quirks:

* ``cor.fbed`` compares the Fisher z statistic sqrt(m) |atanh(r)| with a t
  quantile ``qt(1 - alpha/2, m)``; the p-value of the first variable uses
  n - 2 degrees of freedom, the others m.
* ``cor.fbed`` standardizes x and y, and regresses without an intercept.
* ``cor.bsreg`` uses ``lm`` with an intercept on the raw data, and turns its
  t statistics into partial correlations with dof = n - p - 1, then tests
  them with a t distribution with dof - 1 degrees of freedom.
* ``cor.fbed`` stops with an error when its forward phase selects exactly
  one column and K >= 1 (ETIA uses K = 3 or 5): see ``cor_fbed``.
* ETIA's ``fbed_with_idx.R`` fails when FBED selects nothing ("subscript
  out of bounds"): see ``NoSelColumnError``.
* SES: the univariate statistics keep their sign (the conditional ones do
  not); ties of p-values are broken by the larger statistic.

All p-values are natural logarithms, as in MXM.
"""
from itertools import combinations

import numpy as np
from scipy import stats as st

LOG2 = np.log(2.0)


def _logsf_t(stat, df):
    return LOG2 + st.t.logsf(stat, df)


def _fz(r, m):
    """|0.5 log((1 + r) / (1 - r))| sqrt(m)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(m))


def _std(x):
    return (x - x.mean(0)) / x.std(0, ddof=1)


def _cor_cols(a, B):
    """cor(a, B[, j]) for each column j."""
    a = a - a.mean()
    B = B - B.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a @ B) / np.sqrt((a @ a) * (B * B).sum(0))


def _resid(Z, Y):
    """``.lm.fit(Z, Y)$residuals`` (no intercept)."""
    coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return Y - Z @ coef


# ------------------------------------------------------------------ FBED

class FSError(RuntimeError):
    """An error of ETIA's R feature selection (the AFS skips the fold)."""


class RfastError(FSError):
    """The error of ``Rfast::cor.fbed`` (see ``cor_fbed``)."""


class NoSelColumnError(FSError):
    """ETIA's ``fbed_with_idx.R`` reads ``fbed_object$res[, 'sel']``. When
    FBED selects nothing, ``res`` has no column ``sel`` (no backward phase
    renames it): "subscript out of bounds"."""


def cor_fbed(y, x, alpha=0.05, K=0, etia_compat=False):
    """``Rfast::cor.fbed``. Returns (sel, stat, logp): 0-based columns in the
    sequence of selection.

    Defect of Rfast: when the forward phase selects exactly one column, ``z``
    stays a vector, and the K phase (K >= 1) stops with "'x' is not a
    matrix" in ``.lm.fit``. With ``etia_compat`` this raises ``RfastError``
    (ETIA's AFS then skips the fold); the default mode continues."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n, p = x.shape
    quan = 1 - alpha / 2
    x = _std(x)
    y = (y - y.mean()) / y.std(ddof=1)
    yx = x.T @ y / (n - 1)
    sela, sa, pva = [], [], []
    stat = _fz(yx, n - 3)
    s = list(np.flatnonzero(stat > st.t.ppf(quan, n - 3)))
    if not s:
        return sela, sa, pva

    def pick(stat, m):
        s = list(np.flatnonzero(stat > st.t.ppf(quan, m)))
        if s:
            sel = int(np.argmax(stat))
            sela.append(sel)
            sa.append(stat[sel])
            pva.append(_logsf_t(stat[sel], m))
            s.remove(sel)
        return s

    def inner(s):
        while s:
            m = n - 3 - len(sela)
            stat = np.zeros(p)
            Z = x[:, sela]
            er = _resid(Z, np.column_stack([y, x[:, s]]))
            stat[s] = _fz(_cor_cols(er[:, 0], er[:, 1:]), m)
            s = pick(stat, m)

    def phase():
        m = n - 3 - len(sela)
        rest = [j for j in range(p) if j not in sela]
        stat = np.zeros(p)
        if rest:
            er = _resid(x[:, sela], np.column_stack([y, x[:, rest]]))
            stat[rest] = _fz(_cor_cols(er[:, 0], er[:, 1:]), m)
        inner(pick(stat, m))

    sel = int(np.argmax(stat))
    sela.append(sel)
    sa.append(stat[sel])
    pva.append(_logsf_t(stat[sel], n - 2))         # n - 2: as Rfast
    s.remove(sel)
    r = yx[sel]
    if s:
        stat = np.zeros(p)
        xz = _cor_cols(x[:, sel], x[:, s])
        with np.errstate(divide="ignore", invalid="ignore"):
            yxz = np.abs((yx[s] - xz * r) / np.sqrt(1 - xz ** 2) / np.sqrt(1 - r ** 2))
        stat[s] = _fz(yxz, n - 4)
        inner(pick(stat, n - 4))
    card = [len(sela)]
    if etia_compat and K >= 1 and len(sela) == 1:
        raise RfastError("Error in .lm.fit(z, cbind(y, x[, ind[-sela]])) : 'x' is not a matrix")
    if K == 1:
        phase()
    elif K > 1:
        phase()
        card.append(len(sela))
        vim = 1
        while vim < K and card[vim] - card[vim - 1] > 0:
            vim += 1
            phase()
            card.append(len(sela))
    return sela, sa, pva


def cor_drop1(y, X):
    """``MXM:::cor.drop1(y, X, logged=TRUE)``: (stat, logp) per column."""
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    dof = n - rank
    res = y - A @ coef
    s2 = res @ res / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    t = coef[1:] / np.sqrt(np.diag(cov)[1:])
    a = t ** 2
    r = np.sqrt(a / (a + dof))
    stat = _fz(r, dof - 1)
    return stat, _logsf_t(stat, dof - 1)


def cor_bsreg(y, D, alpha):
    """``MXM:::cor.bsreg``: the columns of D (0-based) that stay, in their
    sequence, or None when p > n (no backward phase)."""
    n, p = D.shape
    thr = np.log(alpha)
    if p > n:
        return None
    stat, lp = cor_drop1(y, D)
    rows = list(range(p))                        # mat[, 1]
    mlp = list(lp)
    sel = int(np.argmax(mlp))
    if mlp[sel] < thr:
        return rows
    removed = [rows[sel]]
    last_p = mlp[sel]
    del rows[sel], mlp[sel]
    while last_p > thr and rows:
        stat, lp = cor_drop1(y, D[:, rows])
        if len(rows) == 1:
            if lp[0] > thr:
                return []
            return rows
        mlp = list(lp)
        sel = int(np.argmax(mlp))
        if mlp[sel] < thr:
            return rows
        last_p = mlp[sel]
        removed.append(rows[sel])
        del rows[sel], mlp[sel]
    return rows


def fbed(y, x, alpha=0.05, K=0, backward=True, etia_compat=False):
    """``MXM::fbed.reg(..., test="testIndFisher")``: 0-based selected columns.
    With ``etia_compat``, the Rfast defect raises ``RfastError`` and an empty
    selection raises ``NoSelColumnError`` (the R script of ETIA fails)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    sel, _, _ = cor_fbed(y, x, alpha, K, etia_compat)
    if not sel and etia_compat:
        raise NoSelColumnError('Error in fbed_object$res[, "sel"] : subscript out of bounds')
    if not sel or not backward:
        return sel
    keep = cor_bsreg(y, x[:, sel], alpha)
    if keep is None:
        return sel
    return [sel[i] for i in keep]


# ------------------------------------------------------------------ SES

class _Fisher:
    """``testIndFisher`` with the correlation matrix of [dataset, target]."""

    def __init__(self, y, X):
        self.n, self.p = X.shape
        self.C = np.corrcoef(np.column_stack([X, y]), rowvar=False)
        self.X = X
        a = self.C[:self.p, self.p]
        dof = self.n - 3
        with np.errstate(divide="ignore", invalid="ignore"):
            wa = 0.5 * np.log((1 + a) / (1 - a)) * np.sqrt(dof)
        wa[np.isnan(a)] = 0.0
        self.univ_stat = wa                        # signed, as univregs
        self.univ_p = _logsf_t(np.abs(wa), dof)

    def __call__(self, x, cs):
        cs = list(dict.fromkeys(int(c) for c in cs))
        if x in cs:
            return 0.0, 0.0
        if not cs:
            return float(self.univ_p[x]), float(self.univ_stat[x])
        for c in cs:
            if np.array_equal(self.X[:, x], self.X[:, c]):
                return 0.0, 0.0
        xy = [x, self.p]
        C = self.C
        try:
            R = C[np.ix_(xy, xy)] - C[np.ix_(xy, cs)] @ np.linalg.solve(C[np.ix_(cs, cs)], C[np.ix_(cs, xy)])
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            r = R[0, 1] / np.sqrt(R[0, 0] * R[1, 1])
            z = 0.5 * np.log((1 + r) / (1 - r))
        dof = self.n - len(cs) - 3
        stat = np.sqrt(dof) * abs(z)
        pv = _logsf_t(stat, dof)
        if np.isnan(pv) or np.isnan(stat):
            return 0.0, 0.0
        return float(pv), float(stat)


def _less(p, p2, s, s2):
    """``compare_p_values``: p < p2, or the larger statistic on a tie."""
    if any(np.isnan(v) for v in (p, p2, s, s2)):
        return False
    return s > s2 if p == p2 else p < p2


def _subsets(cs, lastvar, k):
    """The conditioning sets of size k: lastvar with each (k-1)-subset of
    cs without lastvar (``nchoosek`` = ``combn``, columns in sequence)."""
    if k == 1:
        return [[lastvar]]
    temp = [c for c in cs if c != lastvar]
    return [list(c) + [lastvar] for c in combinations(temp, k - 1)]


def ses(y, x, alpha=0.05, max_k=3):
    """``MXM::SES(..., test="testIndFisher")``. Returns (selectedVars,
    queues): 0-based ascending selected columns and, for each, its
    equivalent columns."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n, p = x.shape
    max_k = min(int(np.floor(max_k)), p)
    thr = np.log(alpha)
    test = _Fisher(y, x)
    pv = test.univ_p.copy()
    sv = test.univ_stat.copy()
    if np.nanmin(pv) > thr:
        return [], []
    selected = np.zeros(p, int)
    order = np.zeros(p, int)
    queues = [[i] for i in range(p)]
    v = int(np.nanargmin(pv))
    selected[v] = 1
    order[v] = 1
    remaining = np.ones(p, int)
    remaining[v] = 0
    remaining[pv > thr] = 0

    def lastvar():
        return int(np.flatnonzero(order == order.max())[0])

    def identify_equivalence():
        lv = lastvar()
        for cvar in np.flatnonzero((selected == 1) | (remaining == 1)):
            cvar = int(cvar)
            if cvar == lv:
                continue
            cs = [int(c) for c in np.flatnonzero(selected == 1) if c != cvar]
            k = min(max_k, len(cs))
            done = False
            for kl in range(1, k + 1):
                for z in _subsets(cs, lv, kl):
                    p_, s_ = test(cvar, z)
                    if _less(pv[cvar], p_, sv[cvar], s_):
                        pv[cvar], sv[cvar] = p_, s_
                    if p_ > thr:
                        remaining[cvar] = 0
                        selected[cvar] = 0
                        for w in z:                              # identifyTheEquivalent
                            zp = [c for c in z if c != w] + [cvar]
                            if test(w, zp)[0] > thr:
                                queues[w] = queues[w] + queues[cvar]
                                break
                        done = True
                        break
                if done:
                    break

    def max_min_assoc():
        best, best_p, best_s = -1, 2.0, 0.0
        sel_ids = [int(c) for c in np.flatnonzero(selected == 1)]
        for cvar in np.flatnonzero(remaining == 1):
            cvar = int(cvar)
            ma_p, ma_s = pv[cvar], sv[cvar]
            k = min(max_k, len(sel_ids))
            lv = lastvar()
            for ck in range(1, k + 1):
                for s in _subsets(sel_ids, lv, ck):
                    p_, s_ = test(cvar, s)
                    if not _less(p_, ma_p, s_, ma_s):
                        ma_p, ma_s = p_, s_
                        pv[cvar], sv[cvar] = p_, s_
            if ma_p > thr:
                remaining[cvar] = 0
            if _less(ma_p, best_p, ma_s, best_s):
                best, best_p, best_s = cvar, ma_p, ma_s
        return best, best_p

    while remaining.any():
        identify_equivalence()
        v, vp = max_min_assoc()
        if vp <= thr:
            selected[v] = 1
            order[v] = order.max() + 1
            remaining[v] = 0
    identify_equivalence()
    sel = [int(c) for c in np.flatnonzero(selected == 1)]
    return sel, [list(dict.fromkeys(queues[c])) for c in sel]
