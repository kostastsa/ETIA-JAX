"""Bootstrap confidence of edges and paths (increment 10, ETIA ``CRV``).

``bootstrap_graphs`` runs one causal configuration on bootstrap samples of
the rows (``sklearn.utils.resample``: n rows drawn with replacement). Both
modes draw real bootstrap samples. ETIA's ``calculate_confidence`` cannot
run: it resamples the data but runs the search on the original data, gives
a DataFrame where ``model.run`` needs a Dataset, and reads
``opt_conf.data_type_info`` from a dict. ETIA's working functions are
``compute_edge_weights`` and ``paths_metrics``
(``causal_graph_utils/confidence_causal_findings.py``).

* ``edge_consistency``: the fraction of bootstrap graphs whose edge i - j is
  consistent with the edge of the best graph; ``edge_discovery``: the
  fraction with the same edge. A path is consistent (the same) when all its
  edges are.
* ``etia_compat=True``: ETIA's ``is_consistent_edge`` table. A circle is
  consistent with a tail or an arrow only in the listed cases, so
  ``i --> j`` is not consistent with ``i --o j``, and an undirected CPDAG
  edge ``i --- j`` is consistent only with itself.
* ``etia_compat=True``, paths: ETIA stops at the first edge that is not the
  same; when this edge is consistent, the path counts as consistent and the
  later edges are not checked.
* default: two edges are consistent when each end mark agrees or one of the
  two marks is a circle; an undirected CPDAG edge ``---`` counts as
  ``o-o`` (either direction). A path is consistent (the same) when all its
  edges are.
"""
import numpy as np
import pandas as pd

from ..oct.algorithms import search


# ------------------------------------------------------------ ETIA's table
def is_consistent_edge(m1_ij, m1_ji, m2_ij, m2_ji):
    """ETIA's is_consistent_edge."""
    if m1_ij == m2_ij and m1_ji == m2_ji:
        return True
    if m1_ij == 1 and m1_ji == 1 and m2_ij != 0 and m2_ji != 0:
        return True
    if m1_ij == 2 and m1_ji == 1:
        return (m2_ij, m2_ji) in ((2, 2), (2, 3), (1, 1))
    if m1_ij == 1 and m1_ji == 2:
        return (m2_ij, m2_ji) in ((2, 2), (3, 2), (1, 1))
    if m1_ij == 2 and m1_ji == 3:
        return (m2_ij, m2_ji) in ((2, 1), (1, 1))
    if m1_ij == 3 and m1_ji == 2:
        return (m2_ij, m2_ji) in ((1, 2), (1, 1))
    if m1_ij == 2 and m1_ji == 2:
        return (m2_ij, m2_ji) in ((1, 1), (2, 1), (1, 2))
    if m1_ij == 0 and m2_ij != 0:
        return False
    if m1_ij != 0 and m2_ij == 0:
        return False
    if m1_ij == 0 and m2_ij == 0:
        return True
    return False


def consistent_marks(m1_ij, m1_ji, m2_ij, m2_ji):
    """Default: the end marks agree, or one of them is a circle."""
    if (m1_ij == 0) != (m2_ij == 0) or (m1_ji == 0) != (m2_ji == 0):
        return False
    if m1_ij == 0:
        return True
    a = (1, 1) if (m1_ij, m1_ji) == (3, 3) else (m1_ij, m1_ji)
    b = (1, 1) if (m2_ij, m2_ji) == (3, 3) else (m2_ij, m2_ji)
    return all(x == y or x == 1 or y == 1 for x, y in zip(a, b))


def _frame(M, names):
    if isinstance(M, pd.DataFrame):
        return M
    names = names if names is not None else [f"X{i}" for i in range(len(M))]
    return pd.DataFrame(np.asarray(M), index=names, columns=names)


# ------------------------------------------------------------ edges and paths
def _table(cons):
    T = np.zeros((4, 4, 4, 4), dtype=bool)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    T[a, b, c, d] = bool(cons(a, b, c, d))
    return T


_TABLES = {}


def _cons_same(best, boots, names, etia_compat):
    """cons[b, i, j] and same[b, i, j] for all bootstrap graphs and pairs."""
    key = bool(etia_compat)
    if key not in _TABLES:
        _TABLES[key] = _table(is_consistent_edge if etia_compat else consistent_marks)
    B = _frame(best, names)
    M = np.asarray(B).astype(int)
    Bs = np.stack([np.asarray(_frame(b, names)).astype(int) for b in boots])
    cons = _TABLES[key][M[None], M.T[None], Bs, Bs.transpose(0, 2, 1)]
    same = (Bs == M[None]) & (Bs.transpose(0, 2, 1) == M.T[None])
    return B, M, cons, same


def compute_edge_weights(best, boots, all_edges=True, true_graph=None, names=None, etia_compat=False):
    """ETIA's compute_edge_weights: (weights, labels) DataFrames with the
    columns source, target, edge_consistency, edge_discovery."""
    cons_f = is_consistent_edge if etia_compat else consistent_marks
    B, M, cons, same = _cons_same(best, boots, names, etia_compat)
    T = None if true_graph is None else np.asarray(_frame(true_graph, names))
    cols = B.columns.to_list()
    n, nb = len(M), len(cons)
    c_all, d_all = cons.sum(0), same.sum(0)
    w, lab = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if not all_edges and M[i, j] == 0:
                continue
            w.append([cols[i], cols[j], int(c_all[i, j]) / nb, int(d_all[i, j]) / nb])
            if T is not None:
                lab.append([cols[i], cols[j], int(cons_f(M[i, j], M[j, i], T[i, j], T[j, i])),
                            int(T[i, j] == M[i, j] and T[j, i] == M[j, i])])
            else:
                lab.append([None] * 4)
    names4 = ["source", "target", "edge_consistency", "edge_discovery"]
    return pd.DataFrame(w, columns=names4, dtype=object), pd.DataFrame(lab, columns=names4, dtype=object)


def _path_weight(cols, cons, same, path, etia_compat):
    idx = {c: k for k, c in enumerate(cols)}
    i = [idx[u] for u in path[:-1]]
    j = [idx[v] for v in path[1:]]
    C, S = cons[:, i, j], same[:, i, j]
    if etia_compat:
        # ETIA's loop stops at the first edge that is not consistent or not the
        # same; a consistent edge that is not the same ends the loop with the
        # path still "consistent", and the later edges are not checked.
        bad = ~C | ~S
        has = bad.any(1)
        first = bad.argmax(1)
        consistent = np.where(has, C[np.arange(len(C)), first], True)
        similar = ~has
    else:
        consistent, similar = C.all(1), S.all(1)
    return consistent.mean(), similar.mean()


def compute_path_weight(boots, best, path, names=None, etia_compat=False):
    """(path consistency, path discovery) of one path (node names)."""
    B, _, cons, same = _cons_same(best, boots, names, etia_compat)
    return _path_weight(B.columns.to_list(), cons, same, path, etia_compat)


def paths_metrics(best, boots, paths, names=None, etia_compat=False):
    """ETIA's paths_metrics: {key: array} of consistency and of discovery."""
    B, _, cons, same = _cons_same(best, boots, names, etia_compat)
    cols = B.columns.to_list()
    pc, pd_ = {}, {}
    for key, ps in paths.items():
        r = np.array([_path_weight(cols, cons, same, p, etia_compat) for p in ps]).reshape(-1, 2)
        pc[key], pd_[key] = r[:, 0], r[:, 1]
    return pc, pd_


# ------------------------------------------------------------ bootstrap
def bootstrap_rows(n, n_bootstraps, seed=0):
    """The rows of each bootstrap sample (``resample(random_state=...)``:
    ``RandomState(seed).randint(0, n, n)`` for each sample)."""
    rs = np.random.RandomState(seed)
    return [rs.randint(0, n, n) for _ in range(n_bootstraps)]


def bootstrap_graphs(X, config, n_bootstraps=50, kinds=None, seed=0, etia_compat=False, rows=None):
    """The class graph of ``config`` (a configuration of the OCT) on each
    bootstrap sample. Returns the list of mark matrices."""
    X = np.asarray(X, dtype=np.float64)
    rows = rows if rows is not None else bootstrap_rows(len(X), n_bootstraps, seed)
    return [np.asarray(search(config, X[r], kinds, etia_compat=etia_compat)) for r in rows]


def calculate_confidence(X, config, best_graph=None, n_bootstraps=50, names=None, kinds=None, seed=0,
                         etia_compat=False):
    """What ETIA's ``calculate_confidence`` means to do: the edge weights of
    the best graph (default: ``config`` on all the rows) over bootstrap
    graphs. Returns (weights DataFrame, bootstrap graphs)."""
    X = np.asarray(X, dtype=np.float64)
    if best_graph is None:
        best_graph = search(config, X, kinds, etia_compat=etia_compat)
    boots = bootstrap_graphs(X, config, n_bootstraps, kinds, seed, etia_compat)
    w, _ = compute_edge_weights(best_graph, boots, all_edges=False, names=names, etia_compat=etia_compat)
    return w, boots
