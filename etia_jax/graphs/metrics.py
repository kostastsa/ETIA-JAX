"""Comparison metrics between mark matrices."""
import numpy as np
from .encoding import NONE


def shd(G_true, G_est):
    """Structural Hamming distance: number of node pairs whose edge differs
    (missing, extra, or any end mark different)."""
    G_true, G_est = np.asarray(G_true), np.asarray(G_est)
    iu = np.triu_indices(len(G_true), 1)
    diff = (G_true[iu] != G_est[iu]) | (G_true.T[iu] != G_est.T[iu])
    return int(diff.sum())


def adjacency_scores(G_true, G_est):
    """Adjacency precision, recall and F1."""
    t = np.asarray(G_true) != NONE
    e = np.asarray(G_est) != NONE
    iu = np.triu_indices(len(t), 1)
    t, e = t[iu], e[iu]
    tp = int((t & e).sum())
    prec = tp / max(int(e.sum()), 1)
    rec = tp / max(int(t.sum()), 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}
