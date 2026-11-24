"""Adjustment sets (increment 10, ETIA ``CRV/adjustment``).

``find_adjset(M, names, graph_type, target, exposures)`` returns the
canonical adjustment set and one minimal adjustment set of the total effect
of the exposures on the target (variable names), or ``None`` when no set
exists.

* ``etia_compat=True``: ETIA's ``find_adjset`` with dagitty 0.3.4 (port in
  ``crv.dagitty``). For a DAG or CPDAG, ETIA keeps only the marks 1 and 2,
  so the undirected edges of a CPDAG are lost; pcalg's ``pcalg2dagitty``
  drops ``--o`` edges of a PAG. The minimal set is the first set that
  dagitty lists. (ETIA's own call does not run on Linux or macOS:
  ``subprocess.call`` with a list and ``shell=True`` starts R without the
  script.)
* default: the same dagitty algorithms (the generalized adjustment
  criterion of Perkovic et al. 2018) on the graph with all its edges.
"""
import numpy as np

from . import dagitty as D


def graph_type_of(M):
    M = np.asarray(M)
    if (M == 1).any():
        return "pag"
    if ((M == 2) & (M.T == 2)).any():
        return "mag"
    if ((M == 3) & (M.T == 3)).any():
        return "cpdag"
    return "dag"


def to_dagitty(M, names, graph_type, exposures, outcomes):
    """A dagitty graph with all the edges of the mark matrix."""
    M = np.asarray(M)
    gtype = {"dag": "dag", "cpdag": "pdag", "pdag": "pdag", "mag": "mag", "pag": "pag"}[graph_type]
    g = D.Graph(gtype)
    for n in names:
        g.add_vertex(str(n))
    p = len(names)
    for i in range(p):
        for j in range(i + 1, p):
            a, b = int(M[i, j]), int(M[j, i])          # mark at j, mark at i
            if a == 0 and b == 0:
                continue
            x, y = str(names[i]), str(names[j])
            kind = {(2, 3): (x, y, D.DIRECTED), (3, 2): (y, x, D.DIRECTED), (3, 3): (x, y, D.UNDIRECTED),
                    (2, 2): (x, y, D.BIDIRECTED), (1, 1): (x, y, D.UNSPECIFIED), (2, 1): (x, y, D.PARTDIRECTED),
                    (1, 2): (y, x, D.PARTDIRECTED), (3, 1): (x, y, D.PARTUNDIRECTED),
                    (1, 3): (y, x, D.PARTUNDIRECTED)}[(a, b)]
            g.add_edge(*kind)
    for v in exposures:
        g.add_prop(str(v), "source")
    for v in outcomes:
        g.add_prop(str(v), "target")
    return g


def adjustment_sets(M, names, graph_type, target, exposures, all_minimal=False):
    """Default mode: (canonical set, first minimal set) or with
    ``all_minimal`` (canonical, list of all minimal sets)."""
    t = [target] if isinstance(target, str) else list(target)
    g = to_dagitty(M, names, graph_type, list(exposures), t)
    can = D._sorted_ids(D.canonical_adjustment_set(g))
    try:
        mini = D._sorted_ids(D.list_msas_total_effect(g))
    except ValueError:
        mini = []
    c = can[0] if can else None
    if all_minimal:
        return c, mini
    return c, (mini[0] if mini else None)


def find_adjset(M, names, graph_type, target_name, exposure_names, etia_compat=False):
    """ETIA's ``find_adjset``: (canonical, minimal) variable names or None."""
    names = [str(n) for n in names]
    t = [target_name] if isinstance(target_name, str) else list(target_name)
    if etia_compat:
        return D.etia_find_adjset(M, names, graph_type, t, list(exposure_names))
    return adjustment_sets(M, names, graph_type, t, exposure_names)
