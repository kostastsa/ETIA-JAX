"""PCMCI (Runge et al. 2019) as tigramite 5.2 runs it for ETIA (increment 6).

ETIA calls ``PCMCI(dataframe, cond_ind_test).run_pcmci(tau_max=L,
pc_alpha=alpha, alpha_level=alpha)`` on the raw series. All CI tests of
tigramite use the rows t = 2 tau_max .. T - 1 (``cut_off='2xtau_max'``).
Here one lagged matrix holds these rows, with a column for each variable
and each lag 0 .. 2 tau_max; the CI tests of increment 2 run on it in
batches.

* PC1 (condition selection), for each variable j: the candidate parents
  (i, -tau), tau = 1..tau_max, in the order i, then tau. At condition size d,
  each parent is tested with the first d-subset of the other parents (in the
  current order); a parent goes when it is independent. After each size the
  parents are sorted by their smallest |test value| (largest first).
* MCI: X = (i, -tau), Y = (j, 0), Z = the parents of j (without X) and the
  parents of i shifted by -tau. Lags 0..tau_max.
* The graph: a lagged link is "-->" when p <= alpha. A link at lag 0 is
  "o-o" when both directions have p <= alpha, else "-->" / "<--" when only
  one direction has it. The p-matrix and the value matrix are then made
  symmetric at lag 0 (the larger p-value).
"""
from itertools import combinations

import numpy as np

from ..ci import CI_TESTS

# ETIA's settings of the tigramite tests (TigramiteAlgorithm._ci_test)
ETIA_TIG = {"CMIknn": {"significance": "fixed_thres"}}


class LaggedData:
    """The rows t = 2 tau_max .. T - 1 of the series, with a column for each
    (variable j, lag l), l = 0 .. 2 tau_max: column l N + j holds x_j(t - l)."""

    def __init__(self, X, tau_max, kinds=None):
        X = np.asarray(X, dtype=np.float64)
        self.T, self.N = X.shape
        self.tau_max = tau_max
        self.max_lag = 2 * tau_max
        rows = np.arange(self.max_lag, self.T)
        self.M = np.concatenate([X[rows - l] for l in range(self.max_lag + 1)], axis=1)
        self.kinds = None if kinds is None else [kinds[j] for l in range(self.max_lag + 1) for j in range(self.N)]
        self.constant = self.M.std(axis=0) == 0

    def col(self, node):
        """(j, -l) -> its column."""
        j, lag = node
        return -lag * self.N + j


class _Tester:
    """run_test of tigramite on the lagged matrix: (val, pval, dependent),
    with the removal of constant columns and a cache."""

    def __init__(self, data, test_name, alpha):
        self.d = data
        self.alpha = alpha
        self.ci = CI_TESTS[test_name](data.M, data.kinds, **ETIA_TIG.get(test_name, {}))
        self.fixed = getattr(self.ci, "fixed_threshold", False)
        self.cache = {}
        self.n_tests = 0

    def _key(self, X, Y, Z):
        """construct_array: duplicates in Z and nodes of Z that are X or Y are
        removed (remove_overlaps), then the constant columns."""
        d = self.d
        Zu = [z for z in dict.fromkeys(tuple(z) for z in Z) if z != tuple(X) and z != tuple(Y)]
        return (d.col(X), d.col(Y), tuple(c for c in (d.col(z) for z in Zu) if not d.constant[c]))

    def prefetch(self, queries):
        need = {}
        for X, Y, Z in queries:
            k = self._key(X, Y, Z)
            if k in self.cache or self.d.constant[k[0]] or self.d.constant[k[1]]:
                continue
            need.setdefault(len(k[2]), {})[k] = None
        for size, ks in need.items():
            ks = list(ks)
            xs = np.array([k[0] for k in ks]); ys = np.array([k[1] for k in ks])
            Zs = np.array([k[2] for k in ks], dtype=np.int64).reshape(len(ks), size)
            stats, ps = self.ci.test_batch(xs, ys, Zs)
            for k, s, p in zip(ks, stats, ps):
                self.cache[k] = (float(s), float(p))

    def run(self, X, Y, Z, alpha):
        self.n_tests += 1
        k = self._key(X, Y, Z)
        if self.d.constant[k[0]] or self.d.constant[k[1]]:
            return 0.0, 1.0, False
        if k not in self.cache:
            self.prefetch([(X, Y, Z)])
        val, pval = self.cache[k]
        if self.fixed:
            dep = val >= alpha
            return val, (0.0 if dep else 1.0), dep
        return val, pval, pval <= alpha


def _link_assumptions(N, tau_min, tau_max, contemporaneous):
    """The default link assumptions: {j: [(i, -lag), ...]} in tigramite's order."""
    out = {}
    for j in range(N):
        out[j] = [(i, -lag) for i in range(N) for lag in range(tau_min, tau_max + 1)
                  if not (i == j and lag == 0) and (contemporaneous or lag > 0)]
    return out


def pc1(tester, N, tau_max, pc_alpha, max_conds_dim=None, max_combinations=1):
    """run_pc_stable (one pc_alpha): the parents of each variable, and the
    largest p-value and its test value for each tested link."""
    links = _link_assumptions(N, 1, tau_max, contemporaneous=False)
    max_dim = N * tau_max if max_conds_dim is None else max_conds_dim
    parents = {j: list(links[j]) for j in range(N)}
    val_min = {j: {} for j in range(N)}
    pval_max = {j: {p: None for p in links[j]} for j in range(N)}
    val_dict = {j: {p: None for p in links[j]} for j in range(N)}
    active = set(range(N))
    for d in range(max_dim + 1):
        jobs = []
        for j in sorted(active):
            if len(parents[j]) - 1 < d:
                active.discard(j)
                continue
            for parent in parents[j]:
                others = [q for q in parents[j] if q != parent]
                for ci_, Z in enumerate(combinations(others, d)):
                    if ci_ >= max_combinations:
                        break
                    jobs.append((j, parent, list(Z)))
        if not jobs:
            break
        tester.prefetch([(par, (j, 0), Z) for j, par, Z in jobs])
        nonsig = {j: [] for j in active}
        for j, parent, Z in jobs:
            val, pval, dep = tester.run(parent, (j, 0), Z, pc_alpha)
            val_min[j][parent] = min(abs(val), val_min[j].get(parent, float("inf")))
            if pval_max[j][parent] is None or pval > pval_max[j][parent]:
                pval_max[j][parent] = pval
                val_dict[j][parent] = val
            if not dep:
                nonsig[j].append(parent)
        for j in list(active):
            for parent in nonsig[j]:
                del val_min[j][parent]
            parents[j] = sorted(val_min[j], key=lambda q: abs(val_min[j][q]), reverse=True)
    return parents, pval_max, val_dict


def mci(tester, N, tau_max, parents, alpha_level, max_conds_py=None, max_conds_px=None):
    links = _link_assumptions(N, 0, tau_max, contemporaneous=True)
    mpy = N * (tau_max + 1) if max_conds_py is None else max_conds_py
    mpx = N * (tau_max + 1) if max_conds_px is None else max_conds_px
    jobs = []
    for j in range(N):
        conds_y = parents[j][:mpy]
        for (i, tau) in links[j]:
            conds_x = [(k, tau + kt) for k, kt in parents[i][:mpx]]
            Z = [n for n in conds_y if n != (i, tau)]
            Z += [n for n in conds_x if n not in Z]
            jobs.append((j, i, tau, Z))
    tester.prefetch([((i, tau), (j, 0), Z) for j, i, tau, Z in jobs])
    val = np.zeros((N, N, tau_max + 1))
    p = np.ones((N, N, tau_max + 1))
    for j, i, tau, Z in jobs:
        v, pv, _ = tester.run((i, tau), (j, 0), Z, alpha_level)
        val[i, j, abs(tau)] = v
        p[i, j, abs(tau)] = pv
    if tester.fixed:
        final = val >= alpha_level
    else:
        final = p <= alpha_level
    graph = string_graph(final)
    for i in range(N):                                   # symmetrize_p_and_val_matrix (o?o links)
        for j in range(N):
            if i != j and p[i, j, 0] >= p[j, i, 0]:
                p[j, i, 0] = p[i, j, 0]
                val[j, i, 0] = val[i, j, 0]
    return graph, p, val


def string_graph(final):
    """convert_to_string_graph."""
    g = np.zeros(final.shape, dtype="<U3")
    g[:, :, 1:][final[:, :, 1:]] = "-->"
    f0 = final[:, :, 0]
    g[:, :, 0][f0 & f0.T] = "o-o"
    for i, j in zip(*np.where(f0 & ~f0.T)):
        g[i, j, 0] = "-->"
        g[j, i, 0] = "<--"
    return g


ETIA_EDGE = {"o-o": (1, 1), "-->": (2, 3), "<--": (3, 2), "<->": (2, 2), "o->": (2, 1), "<-o": (1, 2),
             "x-x": (1, 1), "x->": (2, 1), "<-x": (1, 2)}


def etia_matrix(graph):
    """TigramiteAlgorithm.output_to_array: an (N (L + 1))^2 mark matrix;
    node index N t + i is variable i at lag t. A link (i, j, tau) is copied
    to all the time steps t >= tau: i_ = N t + i, j_ = N (t - tau) + j."""
    N, _, T = graph.shape
    M = np.zeros((N * T, N * T), dtype=np.int8)
    for step in range(T):
        for i in range(N):
            for j in range(N):
                e = graph[i, j, step]
                if e == "":
                    continue
                a, b = ETIA_EDGE[e]
                for t in range(step, T):
                    i_, j_ = N * t + i, N * (t - step) + j
                    if e in ("<--", "<-o", "<-x"):
                        M[j_, i_], M[i_, j_] = b, a
                    else:
                        M[i_, j_], M[j_, i_] = a, b
    return M


class TigResult:
    def __init__(self, graph, p_matrix, val_matrix, parents, n_tests):
        self.graph, self.p_matrix, self.val_matrix = graph, p_matrix, val_matrix
        self.parents, self.n_tests = parents, n_tests
        self.etia_matrix = etia_matrix(graph)


def pcmci(X, test="ParCor", tau_max=1, pc_alpha=0.05, alpha_level=None, kinds=None):
    """PCMCI on the raw series X (T x N)."""
    alpha_level = pc_alpha if alpha_level is None else alpha_level
    d = LaggedData(X, tau_max, kinds)
    tester = _Tester(d, test, pc_alpha)
    parents, _, _ = pc1(tester, d.N, tau_max, pc_alpha)
    graph, p, val = mci(tester, d.N, tau_max, parents, alpha_level)
    return TigResult(graph, p, val, parents, tester.n_tests)


# ================================================================ PCMCI+
def _adjt(graph, include_conflicts=True, sort_by=None, contemp=False):
    """_get_adj_time_series(_contemp): adjt[j] = [(i, -tau)] in np.where order
    (i, then tau); with sort_by, sorted by |sort_by[j][.]| (largest first)."""
    N = graph.shape[0]
    adjt = {}
    for j in range(N):
        g = graph[:, j, :]
        m = g != ""
        if not include_conflicts:
            m &= (g != "x-x") & (g != "x?x")
        w = np.where(m)
        adjt[j] = [(int(a), -int(b)) for a, b in zip(*w)]
    if sort_by is not None:
        for j in range(N):
            vals = {k: abs(sort_by[j][k]) for k in sort_by[j] if k in adjt[j]}
            adjt[j] = sorted(vals, key=vals.get, reverse=True)
    if contemp:
        for j in range(N):
            adjt[j] = [a for a in adjt[j] if a[1] == 0]
    return adjt


def _remaining_pairs(graph, adjt, tau_min, tau_max, p):
    N = graph.shape[0]
    return [(i, j, t) for i in range(N) for j in range(N) for t in range(tau_min, tau_max + 1)
            if graph[i, j, t] != "" and len([a for a in adjt[j] if a != (i, -t)]) >= p]


def _pcalg_Z(S, i, abstau, j, lagged_parents):
    conds_y = lagged_parents[j]
    conds_x = lagged_parents[i]
    conds_x_lagged = [(k, -abstau + kt) for k, kt in conds_x]
    Z = list(S)
    Z += [n for n in conds_y if n != (i, -abstau) and n not in Z]
    Z += [n for n in conds_x_lagged if n not in Z]
    return Z


def _pcalg_test(tester, graph, i, abstau, j, S, lagged_parents, alpha):
    Z = _pcalg_Z(S, i, abstau, j, lagged_parents)
    if graph[i, j, abstau] != "" and graph[i, j, abstau][1] == "-":
        return 1.0, 0.0, Z, True
    val, pval, dep = tester.run((i, -abstau), (j, 0), Z, alpha)
    return val, pval, Z, dep


def _skeleton(tester, graph, lagged_parents, pc_alpha, tau_min, tau_max, max_conds_dim):
    N = graph.shape[0]
    graph[range(N), range(N), 0] = ""
    adjt = _adjt(graph, contemp=True)
    val_matrix = np.zeros(graph.shape)
    val_min = {j: {(int(a), -int(b)): np.inf for a, b in zip(*np.where(graph[:, j, :] != ""))} for j in range(N)}
    p_matrix = np.zeros(graph.shape)
    p_matrix[graph == ""] = 1.0
    sepsets = {((i, -t), j): [] for t in range(tau_min, tau_max + 1) for i in range(N) for j in range(N)}
    p = 0
    while _remaining_pairs(graph, adjt, tau_min, tau_max, p) and p <= max_conds_dim:
        pairs = _remaining_pairs(graph, adjt, tau_min, tau_max, p)
        conds = {}
        for (i, j, t) in pairs:
            conds[(i, j, t)] = list(combinations([a for a in adjt[j] if not (a[0] == i and a[1] == -t)], p))
        tester.prefetch([((i, -t), (j, 0), _pcalg_Z(S, i, t, j, lagged_parents))
                         for (i, j, t), cs in conds.items() for S in cs if graph[i, j, t][1:2] != "-"])
        for (i, j, t) in pairs:
            if graph[i, j, t] == "":
                continue
            for S in conds[(i, j, t)]:
                val, pval, Z, dep = _pcalg_test(tester, graph, i, t, j, S, lagged_parents, pc_alpha)
                val_min[j][(i, -t)] = min(abs(val), val_min[j].get((i, -t)))
                if pval >= p_matrix[i, j, t]:
                    p_matrix[i, j, t] = pval
                    val_matrix[i, j, t] = val
                if not dep:
                    if t == 0:
                        graph[i, j, 0] = graph[j, i, 0] = ""
                        sepsets[((i, 0), j)] = sepsets[((j, 0), i)] = list(S)
                        p_matrix[j, i, 0] = p_matrix[i, j, 0]
                    else:
                        graph[i, j, t] = ""
                        sepsets[((i, -t), j)] = list(S)
                    break
        p += 1
        adjt = _adjt(graph, sort_by=val_min, contemp=True)
    return graph, sepsets, p_matrix, val_matrix


def _subsets(s):
    if len(s) == 0:
        return []
    out = []
    for c in range(len(s) + 1):
        out += list(combinations(s, c))
    return [list(x) for x in set(out)]


def _unshielded_triples(graph):
    N = graph.shape[0]
    adjt = _adjt(graph, include_conflicts=False)
    triples = []
    for j in range(N):
        for (k, tk) in adjt[j]:
            if tk == 0 and graph[k, j, 0] == "o-o":
                for (i, ti) in adjt[k]:
                    if ((i, ti) != (j, 0) and graph[i, j, abs(ti)] == ""
                            and graph[i, k, abs(ti)] in ("o-o", "-->")):
                        triples.append(((i, ti), k, j))
    return triples


def _colliders(tester, graph, sepsets, lagged_parents, pc_alpha):
    for old, new in (("o?o", "o-o"), ("-?>", "-->"), ("<?-", "<--")):
        graph[graph == old] = new
    triples = _unshielded_triples(graph)
    adjt = _adjt(graph, contemp=True)
    v_structures, ambiguous = [], []
    plan = []
    for (i, tau), k, j in triples:
        subs = _subsets([(l, tl) for (l, tl) in adjt[j] if not (l == i and tau == tl)])
        if tau == 0:
            subs += _subsets([(l, tl) for (l, tl) in adjt[i] if not (l == j and tl == 0)])
        uniq = []
        for s in subs:
            if s not in uniq:
                uniq.append(s)
        plan.append((((i, tau), k, j), uniq))
    tester.prefetch([((i, tau), (j, 0), _pcalg_Z(S, i, abs(tau), j, lagged_parents))
                     for ((i, tau), k, j), uniq in plan for S in uniq])
    for triple, uniq in plan:
        (i, tau), k, j = triple
        seps = []
        for S in uniq:
            _, _, _, dep = _pcalg_test(tester, graph, i, abs(tau), j, S, lagged_parents, pc_alpha)
            if not dep:
                seps.append(S)
        if not seps:
            ambiguous.append(triple)
            continue
        frac = np.sum([(k, 0) in S for S in seps]) / float(len(seps))
        if frac == 0.5:
            ambiguous.append(triple)
        elif frac < 0.5:
            v_structures.append(triple)
    oriented = []
    for (i, tau), k, j in v_structures:
        if (k, j) not in oriented and (j, k) not in oriented:
            graph[k, j, 0], graph[j, k, 0] = "<--", "-->"
            oriented.append((j, k))
        elif (k, j) in oriented:
            graph[j, k, 0] = graph[k, j, 0] = "x-x"
        if tau == 0:
            if (i, k) not in oriented and (k, i) not in oriented:
                graph[k, i, 0], graph[i, k, 0] = "<--", "-->"
                oriented.append((i, k))
            elif (k, i) in oriented:
                graph[i, k, 0] = graph[k, i, 0] = "x-x"
    return graph, ambiguous


def _rules(graph, ambiguous):
    N = graph.shape[0]

    def triples_r1(g):
        adjt = _adjt(g, include_conflicts=False)
        out = []
        for j in range(N):
            for (k, tk) in adjt[j]:
                if tk == 0 and g[j, k, 0] == "o-o":
                    for (i, ti) in adjt[k]:
                        if (i, ti) != (j, 0) and g[i, j, abs(ti)] == "" and g[i, k, abs(ti)] == "-->":
                            out.append(((i, ti), k, j))
        return out

    def triples_r2(g):
        adjt = _adjt(g, include_conflicts=False, contemp=True)
        out = []
        for j in range(N):
            for (k, _) in adjt[j]:
                if g[k, j, 0] == "-->":
                    for (i, ti) in adjt[k]:
                        if g[i, k, 0] == "-->" and (i, ti) != (j, 0):
                            if g[i, j, 0] == "o-o" and g[j, i, 0] == "o-o":
                                out.append(((i, 0), k, j))
        return out

    def chains_r3(g):
        adjt = _adjt(g, include_conflicts=False, contemp=True)
        out = []
        for j in range(N):
            for (i, _) in adjt[j]:
                if g[j, i, 0] == "o-o":
                    for (k, _) in adjt[j]:
                        for (l, _) in adjt[j]:
                            if (k != l and k != i and l != i and g[k, j, 0] == "-->" and g[l, j, 0] == "-->"
                                    and g[k, i, 0] == "o-o" and g[l, i, 0] == "o-o" and g[k, l, 0] == ""):
                                out.append((((i, 0), k, j), ((i, 0), l, j)))
        return out

    g = np.copy(graph)
    oriented = []
    any1 = any2 = any3 = True
    while any1 or any2 or any3:
        any1 = False
        for t in triples_r1(g):
            if t in ambiguous:
                continue
            any1 = True
            (i, tau), k, j = t
            if (j, k) not in oriented and (k, j) not in oriented:
                g[k, j, 0], g[j, k, 0] = "-->", "<--"
                oriented.append((k, j))
            if (j, k) in oriented:
                g[j, k, 0] = g[k, j, 0] = "x-x"
        any2 = False
        for t in triples_r2(g):
            if t in ambiguous:
                continue
            any2 = True
            (i, tau), k, j = t
            if (j, i) not in oriented and (i, j) not in oriented:
                g[i, j, 0], g[j, i, 0] = "-->", "<--"
                oriented.append((i, j))
            if (j, i) in oriented:
                g[j, i, 0] = g[i, j, 0] = "x-x"
        any3 = False
        for t1, t2 in chains_r3(g):
            if t1 in ambiguous or t2 in ambiguous:
                continue
            any3 = True
            (i, tau), k, j = t1
            if (j, i) not in oriented and (i, j) not in oriented:
                g[i, j, 0], g[j, i, 0] = "-->", "<--"
                oriented.append((i, j))
            if (j, i) in oriented:
                g[j, i, 0] = g[i, j, 0] = "x-x"
    return g


def pcmciplus(X, test="ParCor", tau_max=1, pc_alpha=0.05, kinds=None):
    """PCMCI+ (Runge 2020) as tigramite 5.2 runs it for ETIA
    (contemp_collider_rule='majority', conflict_resolution=True)."""
    d = LaggedData(X, tau_max, kinds)
    N = d.N
    tester = _Tester(d, test, pc_alpha)
    lagged_parents, pval_max, val_dict = pc1(tester, N, tau_max, pc_alpha)
    p1 = np.ones((N, N, tau_max + 1))
    v1 = np.zeros((N, N, tau_max + 1))
    for j in range(N):
        for (k, tau), pv in pval_max[j].items():
            p1[k, j, abs(tau)] = np.nan if pv is None else pv
            v1[k, j, abs(tau)] = np.nan if val_dict[j][(k, tau)] is None else val_dict[j][(k, tau)]
    graph = np.zeros((N, N, tau_max + 1), dtype="<U3")
    links = {}
    for j in range(N):
        links[j] = {par: "-?>" for par in lagged_parents[j]}
        for i in range(N):
            if i != j:
                links[j][(i, 0)] = "o?o"
        for (i, tau), lt in links[j].items():
            graph[i, j, abs(tau)] = lt
    graph, sepsets, ps, vs = _skeleton(tester, graph, lagged_parents, pc_alpha, 0, tau_max, N)
    for i in range(N):                                   # symmetrize (o?o at lag 0)
        for j in range(N):
            if (i, 0) in links[j] and ps[i, j, 0] >= ps[j, i, 0]:
                ps[j, i, 0] = ps[i, j, 0]
                vs[j, i, 0] = vs[i, j, 0]
    p1[:, :, 0], v1[:, :, 0] = ps[:, :, 0], vs[:, :, 0]
    for j in range(N):
        for (i, tau), lt in links[j].items():
            p1[i, j, abs(tau)] = ps[i, j, abs(tau)]
            v1[i, j, abs(tau)] = vs[i, j, abs(tau)]
    graph, ambiguous = _colliders(tester, graph, sepsets, lagged_parents, pc_alpha)
    graph = _rules(graph, ambiguous)
    return TigResult(graph, p1, v1, lagged_parents, tester.n_tests)
