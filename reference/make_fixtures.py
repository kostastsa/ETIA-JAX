"""Make reference fixtures with the original ETIA / Tetrad.

Run with the ETIA venv (Python 3.10, JPype, Java 17):

    JAVA_HOME=... ~/PROJECTS/ETIA/.venv/bin/python reference/make_fixtures.py graphs

Writes ``reference/fixtures/<name>.npz``. Each fixture is a set of random
DAGs and the Tetrad answers for them.
"""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "fixtures"
ETIA_ROOT = Path.home() / "PROJECTS" / "ETIA"


def start_jvm(one_cpu=False, cpus=1):
    """ETIA's JVM. With one_cpu, the JVM sees one processor
    (-XX:ActiveProcessorCount=1): Tetrad's parallel streams and ForkJoin
    pools then run on one thread, so its searches have no races. ``cpus=2``:
    IndTestFisherZ divides by zero with one processor (CovariancesDoubleForkJoin
    uses availableProcessors() / 2 threads), so the OCT runs use two."""
    sys.path.insert(0, str(ETIA_ROOT))
    import jpype
    if one_cpu and not jpype.isJVMStarted():
        import glob, os
        jars = glob.glob(str(ETIA_ROOT / "ETIA" / "CausalLearning" / "algorithms" / "jar_files" / "*.jar"))
        jvm = os.path.join(os.environ["JAVA_HOME"], "lib", "server", "libjvm.so")
        jpype.startJVM(jvm, "-ea", f"-XX:ActiveProcessorCount={cpus}", classpath=":".join(jars), convertStrings=False)
        return jpype
    from ETIA.utils.jvm_manager import start_jvm as sj
    sj("ETIA.CausalLearning.algorithms.jar_files")
    return jpype


def random_dag(p, deg, rng):
    prob = min(1.0, deg / max(p - 1, 1))
    A = np.triu(rng.random((p, p)) < prob, 1)
    perm = rng.permutation(p)
    return A[np.ix_(perm, perm)].astype(np.int8)


MARK = {"CIRCLE": 1, "ARROW": 2, "TAIL": 3}


def to_tetrad(jpype, A, is_latent=None):
    GraphNode = jpype.JClass("edu.cmu.tetrad.graph.GraphNode")
    NodeType = jpype.JClass("edu.cmu.tetrad.graph.NodeType")
    EdgeListGraph = jpype.JClass("edu.cmu.tetrad.graph.EdgeListGraph")
    nodes = []
    for i in range(len(A)):
        nd = GraphNode(f"X{i}")
        if is_latent is not None and is_latent[i]:
            nd.setNodeType(NodeType.LATENT)
        nodes.append(nd)
    g = EdgeListGraph()
    for nd in nodes:
        g.addNode(nd)
    for i, j in zip(*np.nonzero(A)):
        g.addDirectedEdge(nodes[i], nodes[j])
    return g, nodes


def to_marks(graph, names):
    idx = {n: k for k, n in enumerate(names)}
    G = np.zeros((len(names), len(names)), dtype=np.int8)
    for e in graph.getEdges():
        a, b = str(e.getNode1().getName()), str(e.getNode2().getName())
        i, j = idx[a], idx[b]
        G[i, j] = MARK[str(e.getEndpoint2().toString()).upper()]
        G[j, i] = MARK[str(e.getEndpoint1().toString()).upper()]
    return G


def make_graphs(jpype, n_graphs=60, seed=0):
    rng = np.random.default_rng(seed)
    GT = jpype.JClass("edu.cmu.tetrad.graph.GraphTransforms")
    DagToPag = jpype.JClass("edu.cmu.tetrad.search.utils.DagToPag")
    MsepTest = jpype.JClass("edu.cmu.tetrad.search.test.MsepTest")
    HashSet = jpype.JClass("java.util.HashSet")
    out = {}
    for g in range(n_graphs):
        p = int(rng.integers(5, 13))
        A = random_dag(p, rng.uniform(1.5, 3.5), rng)
        is_latent = rng.random(p) < 0.2
        names = [f"X{i}" for i in range(p)]
        obs_names = [n for n, l in zip(names, is_latent) if not l]

        tg, nodes = to_tetrad(jpype, A)
        cpdag = to_marks(GT.dagToCpdag(tg), names)

        tgl, nodes_l = to_tetrad(jpype, A, is_latent)
        mag = to_marks(GT.dagToMag(tgl), obs_names)
        d2p = DagToPag(tgl)
        d2p.setCompleteRuleSetUsed(True)
        pag = to_marks(d2p.convert(), obs_names)

        # d-separation queries on the full DAG
        ms = MsepTest(tg)
        q = []
        for _ in range(40):
            x, y = rng.choice(p, 2, replace=False)
            rest = [k for k in range(p) if k not in (x, y)]
            Z = list(rng.choice(rest, rng.integers(0, min(4, len(rest)) + 1), replace=False))
            s = HashSet()
            for z in Z:
                s.add(nodes[z])
            sep = bool(ms.isMSeparated(nodes[x], nodes[y], s))
            q.append([x, y, sep] + [int(z in Z) for z in range(p)])
        out[f"A_{g}"] = A
        out[f"latent_{g}"] = is_latent
        out[f"cpdag_{g}"] = cpdag
        out[f"mag_{g}"] = mag
        out[f"pag_{g}"] = pag
        out[f"dsep_{g}"] = np.array(q, dtype=np.int8)
    out["n_graphs"] = np.array(n_graphs)
    np.savez_compressed(OUT / "graphs.npz", **out)
    print("wrote", OUT / "graphs.npz")


# ---------------------------------------------------------------------------
# CI tests: ETIA answers and timings on the datasets of make_ci_data.py
# ---------------------------------------------------------------------------
# name: (library, datasets) -- the same constructors and settings as ETIA
CI_TESTS = {
    "FisherZ": ("tetrad", ["cont"]),
    "cci": ("tetrad", ["nonlin"]),
    "cg_lrt": ("tetrad", ["cont", "disc", "mixed"]),
    "dg_lrt": ("tetrad", ["cont", "disc", "mixed"]),
    "chisquare": ("tetrad", ["disc"]),
    "gsquare": ("tetrad", ["disc"]),
    "ParCor": ("tigramite", ["cont"]),
    "RobustParCor": ("tigramite", ["cont", "nonlin"]),
    "ParCorrWLS": ("tigramite", ["cont"]),
    "GPDC": ("tigramite", ["nonlin"]),
    "CMIknn": ("tigramite", ["nonlin"]),
    "Gsquared": ("tigramite", ["disc"]),
    "CMIsymb": ("tigramite", ["disc"]),
    "RegressionCI": ("tigramite", ["cont", "disc", "mixed"]),
}
ALPHA = 0.05


def _tetrad_test(jpype, name, ds):
    T = "edu.cmu.tetrad.search.test."
    if name == "FisherZ":
        return jpype.JClass(T + "IndTestFisherZ")(ds, ALPHA)
    if name == "cci":
        return jpype.JClass(T + "IndTestConditionalCorrelation")(ds, ALPHA)
    if name == "cg_lrt":
        return jpype.JClass(T + "IndTestConditionalGaussianLrt")(ds, ALPHA, True)
    if name == "dg_lrt":
        return jpype.JClass(T + "IndTestDegenerateGaussianLrt")(ds)
    if name == "chisquare":
        return jpype.JClass(T + "IndTestChiSquare")(ds, ALPHA)
    if name == "gsquare":
        return jpype.JClass(T + "IndTestGSquare")(ds, ALPHA)
    raise KeyError(name)


def _tigramite_test(name):
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.independence_tests.robust_parcorr import RobustParCorr
    from tigramite.independence_tests.parcorr_wls import ParCorrWLS
    from tigramite.independence_tests.gpdc import GPDC
    from tigramite.independence_tests.cmiknn import CMIknn
    from tigramite.independence_tests.cmisymb import CMIsymb
    from tigramite.independence_tests.gsquared import Gsquared
    from tigramite.independence_tests.regressionCI import RegressionCI
    return {"ParCor": lambda: ParCorr(),
            "RobustParCor": lambda: RobustParCorr(),
            "GPDC": lambda: GPDC(significance="analytic", gp_params=None),
            "CMIknn": lambda: CMIknn(significance="fixed_thres", model_selection_folds=3),
            "ParCorrWLS": lambda: ParCorrWLS(significance="analytic"),
            "Gsquared": lambda: Gsquared(significance="analytic"),
            "CMIsymb": lambda: CMIsymb(significance="shuffle_test"),
            "RegressionCI": lambda: RegressionCI(significance="analytic")}[name]()


def make_ci(jpype):
    import time
    import pandas as pd
    from tigramite.data_processing import DataFrame
    from ETIA.data.Dataset import Dataset
    from ETIA.CausalLearning.algorithms.utils import prepare_data_tetrad
    HashSet = jpype.JClass("java.util.HashSet")

    f = np.load(OUT / "ci_data.npz")
    out, rows = {}, []
    for name, (lib, dsets) in CI_TESTS.items():
        for dname in dsets:
            X, disc, Q = f[f"{dname}_X"], f[f"{dname}_discrete"], f[f"{dname}_q"]
            pvals = np.full(len(Q), np.nan)
            stats = np.full(len(Q), np.nan)
            times = np.zeros(len(Q))
            t0 = time.perf_counter()
            if lib == "tetrad":
                df = pd.DataFrame(X, columns=[f"V{i}" for i in range(X.shape[1])])
                for j in np.flatnonzero(disc):
                    df[f"V{j}"] = df[f"V{j}"].astype(int)
                ds, _ = prepare_data_tetrad(Dataset(data=df), {})
                test = _tetrad_test(jpype, name, ds)
                nodes = list(ds.getVariables())
            else:
                dtype = np.tile(disc.astype(int), (len(X), 1)) if name == "RegressionCI" else None
                test = _tigramite_test(name)
                test.set_dataframe(DataFrame(X.copy(), data_type=dtype))
            setup = time.perf_counter() - t0
            for i, row in enumerate(Q):
                x, y, k = int(row[0]), int(row[1]), int(row[2])
                Z = [int(z) for z in row[4:4 + k]]
                t1 = time.perf_counter()
                if lib == "tetrad":
                    s = HashSet()
                    for z in Z:
                        s.add(nodes[z])
                    res = test.checkIndependence(nodes[x], nodes[y], s)
                    pvals[i] = float(res.getPValue())
                else:
                    r = test.run_test(X=[(x, 0)], Y=[(y, 0)], Z=[(z, 0) for z in Z], tau_max=0,
                                      **({"alpha_or_thres": ALPHA} if name == "CMIknn" else {}))
                    stats[i] = float(r[0])
                    pvals[i] = np.nan if r[1] is None else float(r[1])
                times[i] = time.perf_counter() - t1
            out[f"{name}|{dname}|p"] = pvals
            out[f"{name}|{dname}|stat"] = stats
            out[f"{name}|{dname}|time"] = times
            out[f"{name}|{dname}|setup"] = np.array(setup)
            print(f"{name:13s} {dname:7s} setup {setup:7.3f}s  queries {times.sum():8.3f}s", flush=True)
    np.savez_compressed(OUT / "ci_ref.npz", **out)
    print("wrote", OUT / "ci_ref.npz")


SCALE_TESTS = {"FisherZ": ("tetrad", "cont"), "gsquare": ("tetrad", "disc"),
               "ParCor": ("tigramite", "cont"), "Gsquared": ("tigramite", "disc")}


def make_ci_scale(jpype):
    """ETIA time for many queries, one by one (as ETIA's PC/PCMCI run them)."""
    import time
    import pandas as pd
    from tigramite.data_processing import DataFrame
    from ETIA.data.Dataset import Dataset
    from ETIA.CausalLearning.algorithms.utils import prepare_data_tetrad
    HashSet = jpype.JClass("java.util.HashSet")
    f = s = np.load(OUT / "ci_scale.npz")
    out = {}
    for name, (lib, dname) in SCALE_TESTS.items():
        X, disc = f[f"{dname}_X"], f[f"{dname}_discrete"]
        if lib == "tetrad":
            df = pd.DataFrame(X, columns=[f"V{i}" for i in range(X.shape[1])])
            for j in np.flatnonzero(disc):
                df[f"V{j}"] = df[f"V{j}"].astype(int)
            ds, _ = prepare_data_tetrad(Dataset(data=df), {})
            nodes = list(ds.getVariables())
        for k in range(4):
            Q = s[f"{dname}_k{k}"]
            if lib == "tetrad":   # a new test object: Tetrad caches facts
                test = _tetrad_test(jpype, name, ds)
            else:
                test = _tigramite_test(name)
                test.set_dataframe(DataFrame(X.copy()))
            p = np.empty(len(Q))
            t0 = time.perf_counter()
            for i, q in enumerate(Q):
                x, y, Z = int(q[0]), int(q[1]), [int(z) for z in q[2:]]
                if lib == "tetrad":
                    hs = HashSet()
                    for z in Z:
                        hs.add(nodes[z])
                    p[i] = float(test.checkIndependence(nodes[x], nodes[y], hs).getPValue())
                else:
                    p[i] = float(test.run_test(X=[(x, 0)], Y=[(y, 0)], Z=[(z, 0) for z in Z],
                                               tau_max=0)[1])
            out[f"{name}|k{k}|time"] = np.array(time.perf_counter() - t0)
            out[f"{name}|k{k}|p"] = p
            print(f"scale {name:9s} k={k} {len(Q)} queries {float(out[f'{name}|k{k}|time']):8.3f}s",
                  flush=True)
    np.savez_compressed(OUT / "ci_scale_ref.npz", **out)
    print("wrote", OUT / "ci_scale_ref.npz")


# ---------------------------------------------------------------------------
# Searches (PC, CPC) as ETIA runs them: Pc(test) / Cpc(test) with
# setStable(true) and setGuaranteeCpdag(true). Times: data preparation (ETIA's
# prepare_data_tetrad) and the search itself.
# ---------------------------------------------------------------------------
TETRAD_CLASS = {"pc": "Pc", "cpc": "Cpc", "fci": "Fci", "fcimax": "FciMax", "rfci": "Rfci", "cfci": "Cfci"}


SCORES_FOR = {"continuous": ("sem_bic_score", "cg_bic", "dg_bic"),
              "discrete": ("bdeu", "discrete_bic", "cg_bic", "dg_bic"),
              "mixed": ("cg_bic", "dg_bic")}


def tetrad_score(jpype, name, ds, penalty=1.0, structure_prior=1.0):
    """The score object as ETIA's TetradAlgorithm._score makes it."""
    S = "edu.cmu.tetrad.search.score."
    if name == "sem_bic_score":
        sc = jpype.JClass(S + "SemBicScore")(ds, True)
        sc.setPenaltyDiscount(penalty)
    elif name == "bdeu":
        sc = jpype.JClass(S + "BdeuScore")(ds)
        sc.setStructurePrior(structure_prior)
    elif name == "discrete_bic":
        sc = jpype.JClass(S + "DiscreteBicScore")(ds)
        sc.setPenaltyDiscount(penalty)
        sc.setStructurePrior(structure_prior)
    elif name == "cg_bic":
        sc = jpype.JClass(S + "ConditionalGaussianScore")(ds, penalty, True)
    elif name == "dg_bic":
        sc = jpype.JClass(S + "DegenerateGaussianScore")(ds, True)
        sc.setPenaltyDiscount(penalty)
    else:
        raise ValueError(name)
    return sc


def data_kind(disc):
    return "discrete" if disc.all() else ("continuous" if not disc.any() else "mixed")


def _etia_dataset(X, disc, n_lags=0):
    """Tetrad's data set as ETIA makes it. With n_lags > 0: ETIA's pre-lagged
    table (time_lagged=True); the Tetrad names are X1.., X1:1.., ..."""
    import pandas as pd
    from ETIA.data.Dataset import Dataset
    from ETIA.CausalLearning.algorithms.utils import prepare_data_tetrad
    p = X.shape[1]
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(p)])
    for j in np.flatnonzero(disc):
        df[f"V{j}"] = df[f"V{j}"].astype(int)
    info = {"n_lags": int(n_lags), "time_lagged": bool(n_lags)}
    ds, _ = prepare_data_tetrad(Dataset(data=df, data_time_info=info), {})
    return ds


def time_knowledge(jpype, ds, n_lags):
    """ETIA's TetradAlgorithm.time_knowledge (the same loop)."""
    knowledge = jpype.JClass("edu.cmu.tetrad.data.Knowledge")()
    var_names = [str(v) for v in ds.getVariableNames()]
    for t, tier in zip(range(n_lags + 1), reversed(range(n_lags + 1))):
        for i, var in enumerate(var_names):
            if (t == 0) and (':' not in var):
                knowledge.addToTier(tier, var_names[i])
            elif (t > 0) and (':' + str(t) in var):
                knowledge.addToTier(tier, var_names[i])
    return knowledge


def make_scores(jpype, n_queries=120, seed=0, max_p=30):
    """localScore(i, parents) of the five scores of ETIA, for random queries
    on the search datasets. Writes scores.npz."""
    f = np.load(OUT / "search_data.npz")
    rng = np.random.default_rng(seed)
    out = {}
    for case in f["cases"]:
        case = str(case)
        X, disc = f[f"{case}|X"], f[f"{case}|discrete"]
        p = X.shape[1]
        if p > max_p:
            continue
        ds = _etia_dataset(X, disc)
        names = [str(v.getName()) for v in ds.getVariables()]
        assert names == [f"V{i}" for i in range(p)] or len(names) == p
        Q = []
        for _ in range(n_queries):
            i = int(rng.integers(p))
            k = int(rng.integers(0, 5))
            par = rng.choice([v for v in range(p) if v != i], size=k, replace=False)
            Q.append([i] + [int(v) for v in par] + [-1] * (4 - k))
        Q = np.array(Q, dtype=np.int32)
        out[f"{case}|queries"] = Q
        for name in SCORES_FOR[data_kind(disc)]:
            sc = tetrad_score(jpype, name, ds)
            vals = []
            for row in Q:
                par = [int(v) for v in row[1:] if v >= 0]
                vals.append(float(sc.localScore(int(row[0]), jpype.JArray(jpype.JInt)(par))))
            out[f"{case}|{name}|local"] = np.array(vals)
            out[f"{case}|{name}|max_degree"] = np.array(int(sc.getMaxDegree()))
            print(case, name, "done", flush=True)
    np.savez_compressed(OUT / "scores.npz", **out)
    print("wrote", OUT / "scores.npz")


def make_fges(jpype, data_file="search_data.npz", out_file="search_fges_ref.npz", reps=1):
    """FGES as ETIA runs it: new Fges(score), search(). One search for each
    dataset and score of its data kind."""
    import time
    f = np.load(OUT / data_file)
    out = {}
    for case in f["cases"]:
        case = str(case)
        X, disc = f[f"{case}|X"], f[f"{case}|discrete"]
        t0 = time.perf_counter()
        ds = _etia_dataset(X, disc)
        out[f"{case}|prep_time"] = np.array(time.perf_counter() - t0)
        names = [str(v.getName()) for v in ds.getVariables()]
        for name in SCORES_FOR[data_kind(disc)]:
            key = f"{case}|{name}|fges"
            graphs, times = [], []
            for _ in range(reps):
                sc = tetrad_score(jpype, name, ds)
                alg = jpype.JClass("edu.cmu.tetrad.search.Fges")(sc)
                alg.setVerbose(False)
                t0 = time.perf_counter()
                g = alg.search()
                times.append(time.perf_counter() - t0)
                graphs.append(to_marks(g, names))
            out[key + "|graph"] = graphs[0]
            out[key + "|time"] = np.array(min(times))
            out[key + "|same_each_run"] = np.array(all((h == graphs[0]).all() for h in graphs))
            print(f"{key:40s} {min(times):8.3f}s  edges {int((graphs[0] != 0).sum() // 2)}", flush=True)
    np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


GFCI_TESTS = ("FisherZ", "cg_lrt", "dg_lrt", "chisquare", "gsquare")     # ETIA causal_configs["gfci"]


def make_gfci(jpype, data_file="search_fci_data.npz", out_file="search_gfci_ref.npz", reps=3):
    """GFCI as ETIA runs it: new GFci(test, score), search()."""
    import time
    f = np.load(OUT / data_file)
    out = {}
    for case in f["cases"]:
        case = str(case)
        X, disc = f[f"{case}|X"], f[f"{case}|discrete"]
        t0 = time.perf_counter()
        ds = _etia_dataset(X, disc)
        out[f"{case}|prep_time"] = np.array(time.perf_counter() - t0)
        names = [str(v.getName()) for v in ds.getVariables()]
        for test_name in [str(t) for t in f[f"{case}|tests"] if str(t) in GFCI_TESTS]:
            for alpha in f["alphas"]:
                for name in SCORES_FOR[data_kind(disc)]:
                    key = f"{case}|{test_name}|{float(alpha)}|{name}|gfci"
                    graphs, times, errors = [], [], []
                    for _ in range(reps):
                        test = _tetrad_test(jpype, test_name, ds)
                        test.setAlpha(float(alpha))
                        sc = tetrad_score(jpype, name, ds)
                        alg = jpype.JClass("edu.cmu.tetrad.search.GFci")(test, sc)
                        alg.setVerbose(False)
                        t0 = time.perf_counter()
                        try:
                            g = alg.search()
                        except Exception as exc:
                            errors.append(f"{exc.__class__.__name__}: {exc}")
                            continue
                        times.append(time.perf_counter() - t0)
                        graphs.append(to_marks(g, names))
                    if errors:
                        out[key + "|error"] = np.array(errors[0])
                        out[key + "|n_errors"] = np.array(len(errors))
                    if graphs:
                        out[key + "|graph"] = graphs[0]
                        out[key + "|graphs"] = np.array(graphs)
                        out[key + "|time"] = np.array(min(times))
                        out[key + "|same_each_run"] = np.array(all((h == graphs[0]).all() for h in graphs))
                    print(f"{key:48s} {min(times) if times else -1:8.3f}s errors {len(errors)} "
                          f"same {all((h == graphs[0]).all() for h in graphs) if graphs else None}", flush=True)
    np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


TS_TEST_ALGOS = ("pc", "cpc", "fci", "fcimax", "rfci", "cfci", "svarfci")
TS_SCORE_ALGOS = ("fges",)
TS_BOTH_ALGOS = ("gfci", "svargfci")
TETRAD_CLASS_TS = {"svarfci": "SvarFci", "svargfci": "SvarGfci", "fges": "Fges", "gfci": "GFci"}


def make_ts(jpype, reps=3, out_file="ts_tetrad_ref.npz", only_errors=False):
    """The Tetrad algorithms of ETIA on the lagged tables, with ETIA's tier
    knowledge (time_lagged=True). Discrete and mixed data: serialized test
    calls (race-free). All distinct outputs of ``reps`` runs are kept."""
    import time
    from make_fixtures_serial import serialized
    f = np.load(OUT / "ts_data.npz")
    S = "edu.cmu.tetrad.search."
    out = {}
    if only_errors:                      # run again only the searches that stopped with an error
        old = dict(np.load(OUT / out_file))
        redo = {k[:-len("|error")] for k in old if k.endswith("|error")}
        out = {k: v for k, v in old.items() if k.rsplit("|", 1)[0] not in redo}
    for case in f["cases"]:
        case = str(case)
        L = int(f[f"{case}|L"])
        disc_raw = f[f"{case}|discrete"]
        X = f[f"{case}|lagged"]
        disc = np.tile(disc_raw, L + 1)
        ds = _etia_dataset(X, disc, L)
        names = [str(v.getName()) for v in ds.getVariables()]
        out[f"{case}|names"] = np.array(names)
        racy = bool(disc.any())

        def run(algo, test_name=None, alpha=None, score_name=None):
            graphs, times, errors = [], [], []
            for _ in range(reps):
                test = None
                if test_name:
                    real = _tetrad_test(jpype, test_name, ds)
                    real.setAlpha(float(alpha))
                    test = serialized(real) if racy else real
                cls = jpype.JClass(S + (TETRAD_CLASS_TS.get(algo) or TETRAD_CLASS[algo]))
                if algo == "fges":
                    alg = cls(tetrad_score(jpype, score_name, ds))
                elif algo in TS_BOTH_ALGOS:
                    alg = cls(test, tetrad_score(jpype, score_name, ds))
                else:
                    alg = cls(test)
                if algo in ("pc", "cpc"):
                    alg.setStable(True)
                    alg.setGuaranteeCpdag(True)
                alg.setVerbose(False)
                alg.setKnowledge(time_knowledge(jpype, ds, L))
                t0 = time.perf_counter()
                try:
                    g = alg.search()
                except Exception as exc:
                    errors.append(f"{exc.__class__.__name__}: {exc}")
                    continue
                times.append(time.perf_counter() - t0)
                G = to_marks(g, names)
                if not any((G == h).all() for h in graphs):
                    graphs.append(G)
            return graphs, times, errors

        jobs = []
        for t in f[f"{case}|tests"]:
            for a in f["alphas"]:
                jobs += [(algo, str(t), float(a), None) for algo in TS_TEST_ALGOS]
                jobs += [(algo, str(t), float(a), str(sc)) for algo in TS_BOTH_ALGOS for sc in f[f"{case}|scores"]]
        jobs += [("fges", None, None, str(sc)) for sc in f[f"{case}|scores"]]
        for algo, t, a, sc in jobs:
            key = f"{case}|{t}|{a}|{sc}|{algo}"
            if only_errors and key not in redo:
                continue
            graphs, times, errors = run(algo, t, a, sc)
            if graphs:
                out[key + "|graphs"] = np.stack(graphs)
                out[key + "|time"] = np.array(min(times))
            if errors:
                out[key + "|error"] = np.array(errors[0])
                out[key + "|n_errors"] = np.array(len(errors))
            print(f"{key:56s} outputs {len(graphs)} errors {len(errors)} "
                  f"{min(times) if times else -1:7.3f}s", flush=True)
    np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


SLOW_TIG_TESTS = ("CMIknn", "GPDC")      # 3-variable data, alpha 0.05 only
VERY_SLOW_TIG_TESTS = ("CMIsymb",)        # a shuffle test: 25-35 min per search; one dataset, alpha 0.05


def make_ts_tigramite(out_file="ts_tigramite_ref.npz"):
    """PCMCI, PCMCI+ and LPCMCI of tigramite 5.2 as ETIA runs them
    (TigramiteAlgorithm: the same CI test settings and calls), on the raw
    series (ETIA allows these algorithms only with time_lagged=False). Also
    ETIA's output_to_array matrix. CMIknn and GPDC: 3-variable data, alpha 0.05."""
    import time
    import pandas as pd
    sys.path.insert(0, str(ETIA_ROOT))
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.lpcmci import LPCMCI
    from ETIA.CausalLearning.algorithms.tigramite_algorithm.TigramiteAlgorithm import TigramiteAlgorithm
    f = np.load(OUT / "ts_data.npz")
    out = dict(np.load(OUT / out_file)) if (OUT / out_file).exists() else {}
    for case in f["cases"]:
        case = str(case)
        X, disc, L = f[f"{case}|X"], f[f"{case}|discrete"], int(f[f"{case}|L"])
        p = X.shape[1]
        for test in [str(t) for t in f[f"{case}|tig_tests"]]:
            for alpha in [float(a) for a in f["alphas"]]:
                if test in SLOW_TIG_TESTS and (p > 3 or alpha != 0.05):
                    continue
                if test in VERY_SLOW_TIG_TESTS and (not case.endswith("_s0") or alpha != 0.05):
                    continue
                for algo in ("PCMCI", "PCMCI+", "LPCMCI"):
                    ta = TigramiteAlgorithm(algo)
                    ta.n_lags = L
                    ta.var_names = [f"V{i}" if t == 0 else f"V{i}:{t}" for t in range(L + 1) for i in range(p)]
                    ci = ta._ci_test({"ci_test": test})
                    names = [f"V{i}" for i in range(p)]
                    if test == "RegressionCI":
                        dtype = np.zeros(X.shape, dtype="int")
                        dtype[:] = disc.astype(int)
                        df = pp.DataFrame(X, data_type=dtype, var_names=names)
                    else:
                        df = pp.DataFrame(X, var_names=names)
                    key = f"{case}|{test}|{alpha}|{algo}"
                    if key + "|graph" in out:              # done in an earlier (stopped) run
                        continue
                    t0 = time.perf_counter()
                    try:
                        o = ta._algo(df, {"significance_level": alpha}, ci)
                    except Exception as exc:
                        out[key + "|error"] = np.array(f"{exc.__class__.__name__}: {exc}")
                        print(f"{key:44s} ERROR {exc}", flush=True)
                        continue
                    dt = time.perf_counter() - t0
                    out[key + "|graph"] = np.asarray(o["graph"]).astype("U3")
                    out[key + "|p_matrix"] = np.asarray(o["p_matrix"], dtype=float)
                    out[key + "|val_matrix"] = np.asarray(o["val_matrix"], dtype=float)
                    out[key + "|etia_matrix"] = np.asarray(ta.output_to_array(o))
                    out[key + "|time"] = np.array(dt)
                    print(f"{key:44s} {dt:8.3f}s", flush=True)
                    np.savez_compressed(OUT / out_file, **out)        # save after each search
    np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


NOTEARS_THRESHOLDS = [0.0, 0.5, 0.8]


def make_notears(out_file="notears_ref.npz"):
    """NOTEARS as ETIA runs it (NoTearsAlgorithm -> causalnex from_pandas), on
    the search datasets with 10 variables, cont20 and cont30_s0. For each case: the raw weights of
    causalnex (w_threshold=0), and for each threshold of ETIA's grid ETIA's
    mark matrix or its error, with the run time."""
    import time, warnings
    import pandas as pd
    import networkx as nx
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ETIA_ROOT))
    from causalnex.structure.notears import from_numpy
    from ETIA.CausalLearning.algorithms.causalnex_algorithm.NoTears import NoTearsAlgorithm
    f = np.load(OUT / "search_data.npz")
    out = dict(np.load(OUT / out_file)) if (OUT / out_file).exists() else {}
    cases = [str(c) for c in f["cases"] if str(c).split("_")[0] in ("cont10", "disc10", "mixed10")]
    cases += [str(c) for c in f["cases"] if str(c).startswith("cont20")] + ["cont30_s0"]
    for case in cases:
        if case + "|W" in out:
            continue
        X = f[f"{case}|X"]
        p = X.shape[1]
        t0 = time.perf_counter()
        sm = from_numpy(X)
        out[case + "|W"] = nx.to_numpy_array(sm, nodelist=range(p), weight="weight")
        out[case + "|W_time"] = np.array(time.perf_counter() - t0)
        df = pd.DataFrame(X, columns=[f"X{i}" for i in range(p)])

        class _D:
            def get_dataset(self):
                return df.copy()
        # causalnex takes minutes from 20 variables: there, only ETIA's default 0.8
        for thr in (NOTEARS_THRESHOLDS if p <= 10 else [0.8]):
            key = f"{case}|{thr}"
            t0 = time.perf_counter()
            try:
                _, res = NoTearsAlgorithm().run(_D(), {"threshold": thr, "max_iter": 100, "h_tol": 1e-7})
                out[key + "|etia_matrix"] = np.asarray(res["matrix_graph"], dtype=np.int8)
            except Exception as exc:
                out[key + "|error"] = np.array(f"{exc.__class__.__name__}: {exc}")
            out[key + "|time"] = np.array(time.perf_counter() - t0)
            print(f"{key:24s} {float(out[key + '|time']):8.3f}s", "error" if key + "|error" in out else "", flush=True)
        np.savez_compressed(OUT / out_file, **out)
    np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


SAM_FIXTURE_PARAMS = {"train_epochs": 1000, "test_epochs": 500}     # the smallest values of ETIA's grid


def make_sam(out_file="sam_ref.npz", full_case="sin10_s0"):
    """SAM as ETIA runs it (SAMAlgorithm -> cdt SAM, GPU), on sam_data.npz with
    train 1000 / test 500 epochs, and once with ETIA's defaults (3000 / 1000)
    on ``full_case``. Stores ETIA's W (the DiGraph weights), ETIA's mark
    matrix and the run time. Run it in a new process (cdt forks workers)."""
    import time, warnings
    import pandas as pd
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ETIA_ROOT))
    from ETIA.CausalLearning.algorithms.cdt_algorithms.SAMAlgorithm import SAMAlgorithm
    from ETIA.CausalLearning.CausalModel.utils import pywhy_graph_to_matrix
    f = np.load(OUT / "sam_data.npz")
    out = dict(np.load(OUT / out_file)) if (OUT / out_file).exists() else {}
    jobs = [(str(c), "short", SAM_FIXTURE_PARAMS) for c in f["cases"]] + [(full_case, "full", {})]
    for case, tag, params in jobs:
        key = f"{case}|{tag}"
        if key + "|W" in out:
            continue
        X = f[f"{case}|X"]
        p = X.shape[1]
        cols = [f"X{i}" for i in range(p)]
        df = pd.DataFrame(X, columns=cols)
        t0 = time.perf_counter()
        g, _ = SAMAlgorithm().run(df, dict(params))
        dt = time.perf_counter() - t0
        W = np.zeros((p, p))
        for u, v, w in g.dag.edges(data="weight"):
            W[cols.index(u), cols.index(v)] = w
        out[key + "|W"] = W
        out[key + "|etia_matrix"] = np.asarray(pywhy_graph_to_matrix(g), dtype=np.int8)
        out[key + "|time"] = np.array(dt)
        print(f"{key:24s} {dt:8.1f}s", flush=True)
        np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


AFS_CASES = ("cont10", "cont20", "cont30", "disc10", "mixed10")
AFS_FS_CONFIGS = [dict(fs_name=f, alpha=a, k=k, ind_test_name="testIndFisher")
                  for f in ("fbed", "ses") for a in (0.05, 0.01) for k in (3, 5)]


def afs_targets(p, case):
    """Three target columns per dataset (fixed by the case name)."""
    rng = np.random.default_rng(sum(map(ord, case)))
    return sorted(int(t) for t in rng.choice(p, 3, replace=False))


def make_fs(out_file="fs_ref.npz"):
    """FBED and SES of MXM 1.5.8 as ETIA runs them (AFS FeatureSelector:
    CSV file, Rscript --vanilla, fbed_with_idx.R / ses_with_idx.R), on the
    search datasets, three targets each, the 8 FS configurations of ETIA's
    grid. MXM must be in ~/R/library (reference/install_mxm.R)."""
    import os, time, warnings
    import pandas as pd
    warnings.filterwarnings("ignore")
    os.environ["R_LIBS"] = str(Path.home() / "R" / "library")
    sys.path.insert(0, str(ETIA_ROOT))
    from ETIA.AFS.feature_selector import FeatureSelector
    f = np.load(OUT / "search_data.npz")
    out = dict(np.load(OUT / out_file)) if (OUT / out_file).exists() else {}
    fs = FeatureSelector(r_path="Rscript")
    for case in [str(c) for c in f["cases"] if str(c).split("_")[0] in AFS_CASES]:
        X = f[f"{case}|X"]
        p = X.shape[1]
        cols = [f"X{i}" for i in range(p)]
        for t in afs_targets(p, case):
            feats = [c for c in cols if c != cols[t]]
            df = pd.DataFrame(X, columns=cols)[feats + [cols[t]]]          # ETIA: features, then the target
            for cfg in AFS_FS_CONFIGS:
                key = f"{case}|{t}|{cfg['fs_name']}|{cfg['alpha']}|{cfg['k']}"
                if key + "|sel" in out or key + "|error" in out:
                    continue
                t0 = time.perf_counter()
                try:
                    sel = fs.feature_selection(dict(cfg), cols[t], df, f"fsref_{case}_{t}")
                    out[key + "|sel"] = np.asarray(sel["sel"].tolist(), dtype=int)   # index in feats
                except Exception as exc:
                    out[key + "|error"] = np.array(f"{exc.__class__.__name__}: {exc}")
                out[key + "|time"] = np.array(time.perf_counter() - t0)
                print(f"{key:36s} {float(out[key + '|time']):6.2f}s", out.get(key + "|sel", "ERROR"), flush=True)
        np.savez_compressed(OUT / out_file, **out)
    np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


def make_afs(out_file="afs_ref.npz"):
    """The whole AFS of ETIA (5 folds, bootstrap) on cont10_s0..s2 and
    cont20_s0, the first target of afs_targets: once with the linear
    configurations only (exact in ETIA-JAX) and once with ETIA's full grid."""
    import os, time, json, warnings
    import pandas as pd
    warnings.filterwarnings("ignore")
    os.environ["R_LIBS"] = str(Path.home() / "R" / "library")
    sys.path.insert(0, str(ETIA_ROOT))
    from ETIA.AFS.AFS import AFS
    from ETIA.AFS.predictive_configurator import PredictiveConfigurator
    allc = PredictiveConfigurator().create_predictive_configs()
    lin = [c for c in allc if c["model"] == "linear_regression"]
    f = np.load(OUT / "search_data.npz")
    out = dict(np.load(OUT / out_file)) if (OUT / out_file).exists() else {}
    for case in ("cont10_s0", "cont10_s1", "cont10_s2", "cont20_s0"):
        X = f[f"{case}|X"]
        p = X.shape[1]
        cols = [f"X{i}" for i in range(p)]
        t = afs_targets(p, case)[0]
        for tag, configs in (("linear", lin), ("full", allc)):
            key = f"{case}|{t}|{tag}"
            if key + "|result" in out:
                continue
            t0 = time.perf_counter()
            res = AFS(random_seed=0).run_AFS(pd.DataFrame(X, columns=cols), [cols[t]],
                                             pred_configs=[dict(c) for c in configs], dataset_name=f"afsref_{case}")
            dt = time.perf_counter() - t0
            sel = res["selected_features"]
            sel = sel.get(cols[t], []) if isinstance(sel, dict) else sel
            out[key + "|result"] = np.array(json.dumps(dict(
                selected=sorted(sel), best_config=res["best_config"], bbc_score=float(res["bbc_score"]),
                ci=[float(v) for v in res["ci"]] if res["ci"] is not None else None)))
            out[key + "|time"] = np.array(dt)
            print(f"{key:28s} {dt:8.1f}s", out[key + "|result"], flush=True)
            np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


def make_search(jpype, algos=("pc", "cpc"), data_file="search_data.npz", out_file="search_ref.npz"):
    import time
    import pandas as pd
    from ETIA.data.Dataset import Dataset
    from ETIA.CausalLearning.algorithms.utils import prepare_data_tetrad
    S = "edu.cmu.tetrad.search."
    f = np.load(OUT / data_file)
    out = {}
    for case in f["cases"]:
        case = str(case)
        X, disc = f[f"{case}|X"], f[f"{case}|discrete"]
        p = X.shape[1]
        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(p)])
        for j in np.flatnonzero(disc):
            df[f"V{j}"] = df[f"V{j}"].astype(int)
        t0 = time.perf_counter()
        ds, _ = prepare_data_tetrad(Dataset(data=df), {})
        prep = time.perf_counter() - t0
        names = [str(v.getName()) for v in ds.getVariables()]
        out[f"{case}|prep_time"] = np.array(prep)
        for test_name in f[f"{case}|tests"]:
            test_name = str(test_name)
            for alpha in f["alphas"]:
                for algo in algos:
                    test = _tetrad_test(jpype, test_name, ds)
                    test.setAlpha(float(alpha)) if hasattr(test, "setAlpha") else None
                    alg = jpype.JClass(S + TETRAD_CLASS[algo])(test)
                    if algo in ("pc", "cpc"):          # ETIA sets these two for PC and CPC only
                        alg.setStable(True)
                        alg.setGuaranteeCpdag(True)
                    t0 = time.perf_counter()
                    key = f"{case}|{test_name}|{float(alpha)}|{algo}"
                    try:
                        g = alg.search()
                    except Exception as exc:                  # a crash inside Tetrad
                        frames = [str(f) for f in exc.getStackTrace()[:6]] if hasattr(exc, "getStackTrace") else []
                        out[key + "|error"] = np.array(f"{exc.__class__.__name__}: {exc} | " + " < ".join(frames))
                        out[key + "|time"] = np.array(time.perf_counter() - t0)
                        print(f"{key:40s} ERROR {exc}", flush=True)
                        continue
                    dt = time.perf_counter() - t0
                    out[key + "|graph"] = to_marks(g, names)
                    out[key + "|time"] = np.array(dt)
                    out[key + "|ntests"] = np.array(int(alg.getNumIndependenceTests())
                                                    if hasattr(alg, "getNumIndependenceTests") else -1)
                    print(f"{key:40s} {dt:8.3f}s", flush=True)
    np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


def _etia_oct_run(X, kinds_unused, nj, reg, case):
    """One ETIA CausalLearner run (random_seed=0). ``reg``: None (ETIA's
    random forest) or "lin" (LinearRegression). Returns (learner, result, s)."""
    import copy, time, warnings, os
    import pandas as pd
    warnings.filterwarnings("ignore")
    from ETIA.data.Dataset import Dataset
    import ETIA.CausalLearning.configurations.configurations as cfgmod
    from ETIA.CausalLearning.configurations.default_parameters import default_conf
    from ETIA.CausalLearning.CausalLearner import CausalLearner
    from ETIA.CausalLearning.configurations.configurations import Configurations
    global _PRISTINE
    if "_PRISTINE" not in globals():
        _PRISTINE = copy.deepcopy(default_conf)
    cfgmod.default_conf = copy.deepcopy(_PRISTINE)       # ETIA changes default_conf in place
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    ds = Dataset(data=df, data_time_info={"n_lags": 0, "time_lagged": False}, time_series=False,
                 dataset_name=f"oct_{case}")
    conf = Configurations(dataset=ds, n_jobs=nj)
    conf.results_folder = str(Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp") + "/"
    if reg == "lin":
        conf.cdhpo_params.set_regressor("LinearRegression", {})
    if case.startswith("disc"):
        # Tetrad's R0 runs the chi-square/G-square test in a parallel stream, and
        # the test is not thread-safe: ETIA stops with Java errors. A lock around
        # each test call (make_fixtures_serial.serialized) gives the serial result.
        from ETIA.CausalLearning.algorithms.tetrad_algorithm.TetradAlgorithm import TetradAlgorithm
        from make_fixtures_serial import serialized
        if not getattr(TetradAlgorithm, "_serial_patch", False):
            real_ci = TetradAlgorithm._ci_test
            TetradAlgorithm._ci_test = lambda self, ds, params: serialized(real_ci(self, ds, params))
            TetradAlgorithm._serial_patch = True
    cl = CausalLearner(configurations=conf, n_jobs=nj, random_seed=0)
    cl.results_folder = conf.results_folder
    cl.cdhpo.results_folder = conf.results_folder
    import contextlib, io
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        res = cl.learn_model()
    return cl, res, time.perf_counter() - t0


def make_oct(jpype, out_file="oct_ref.npz"):
    """ETIA's CausalLearner/OCT: per case, LinearRegression with n_jobs=1
    ("lin1", the in-process path) and n_jobs=8 ("lin8", joblib workers), and
    ETIA's random forest with n_jobs=1 and 8 ("rf1", "rf8"). Dumps the
    graphs, Markov boundaries, pooled predictions, scores, permutations,
    p-values and the chosen configuration."""
    import json
    f = np.load(OUT / "search_data.npz")
    out = dict(np.load(OUT / out_file)) if (OUT / out_file).exists() else {}
    for case in ("cont10_s0", "cont10_s1", "cont10_s2", "cont20_s0", "mixed10_s0", "disc10_s0"):
        X = f[f"{case}|X"][:1000]
        for tag, nj, reg in (("lin1", 1, "lin"), ("lin8", 8, "lin"), ("rf1", 1, None), ("rf8", 8, None)):
            key = f"{case}|{tag}"
            if key + "|time" in out:
                continue
            cl, res, dt = _etia_oct_run(X, None, nj, reg, case)
            o = cl.cdhpo
            C, F, p = len(o.configs_ran), o.oct_params.oos_protocol.protocol.folds_to_run, X.shape[1]
            skip = ("model", "var_type", "indexes", "library")
            cfgs = [{k: (v.item() if hasattr(v, "item") else v) for k, v in c.items() if k not in skip}
                    for c in o.configs_ran]
            out[key + "|configs"] = np.array(json.dumps(cfgs))
            out[key + "|mec"] = np.array([[np.asarray(g) for g in gs] for gs in o.mec_graphs_configs], np.int8)
            out[key + "|graph"] = np.array([[np.asarray(g) for g in gs] for gs in o.graphs_configs], np.int8)
            out[key + "|mbs"] = np.array(json.dumps([[[sorted(int(v) for v in o.mb_configs[c][t][k])
                                                       for k in range(F)] for t in range(p)] for c in range(C)]))
            out[key + "|pred"] = np.array([[np.concatenate(o.pred_configs[c][t]) for t in range(p)]
                                           for c in range(C)])
            out[key + "|y"] = np.array([np.concatenate(o.y_test_nodes[t]) for t in range(p)])
            out[key + "|mu"] = np.asarray(o.mu_configs_np)
            out[key + "|idxs"] = np.array(o.idxs)
            out[key + "|p_values"] = np.asarray(o.p_values)
            out[key + "|is_equal"] = np.asarray(o.is_equal)
            out[key + "|mean_mb"] = np.asarray(o.mean_mb)
            out[key + "|opt_config"] = np.array(o.opt_config)
            chosen = next(i for i, c in enumerate(o.configs_ran) if c is res["optimal_conf"])
            out[key + "|chosen"] = np.array(chosen)
            out[key + "|n_final_rows"] = np.array(len(res["optimal_conf"].get("indexes", X)))
            out[key + "|final_mec"] = np.asarray(res["matrix_mec_graph"], np.int8)
            out[key + "|final_graph"] = np.asarray(res["matrix_graph"], np.int8)
            out[key + "|time"] = np.array(dt)
            print(f"{key:24s} {dt:7.1f}s chosen {chosen} opt {o.opt_config} p {np.round(o.p_values, 2)}",
                  flush=True)
            np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


def make_members(jpype, out_file="members_ref.npz", draws=2000):
    """Tetrad's magFromPag on PAGs of ETIA's OCT (oct_ref.npz) and
    dagFromCpdag on the CPDAGs of true DAGs: the counts of the drawn graphs."""
    import json
    gr = jpype.JPackage("edu.cmu.tetrad.graph")
    EP = {1: gr.Endpoint.CIRCLE, 2: gr.Endpoint.ARROW, 3: gr.Endpoint.TAIL}
    BACK = {"CIRCLE": 1, "ARROW": 2, "TAIL": 3}

    def to_t(M):
        p = len(M)
        nodes = [gr.GraphNode(f"X{i + 1}") for i in range(p)]
        L = jpype.JClass("java.util.ArrayList")()
        for nd in nodes:
            L.add(nd)
        g = gr.EdgeListGraph(L)
        for a in range(p):
            for b in range(a + 1, p):
                if M[a, b]:
                    g.addEdge(gr.Edge(nodes[a], nodes[b], EP[int(M[b, a])], EP[int(M[a, b])]))
        return g, {f"X{i + 1}": i for i in range(p)}

    def marks(g, idx):
        p = len(idx)
        M = np.zeros((p, p), np.int8)
        for e in g.getEdges():
            i, j = idx[str(e.getNode1().getName())], idx[str(e.getNode2().getName())]
            M[i, j] = BACK[str(e.getEndpoint2().name())]
            M[j, i] = BACK[str(e.getEndpoint1().name())]
        return M

    o = np.load(OUT / "oct_ref.npz")
    f = np.load(OUT / "search_data.npz")
    out = {}
    items = []
    for case in ("cont10_s0", "cont10_s1", "cont20_s0", "mixed10_s0"):
        mec = o[f"{case}|lin1|mec"]
        for c, k in ((0, 0), (len(mec) - 1, 2)):
            items.append((f"pag|{case}|{c}|{k}", mec[c, k], "pag"))
    for case in ("cont10_s0", "cont10_s1", "cont20_s0"):
        A = f[f"{case}|A"]
        g, idx = to_t(np.where(A.astype(bool), 2, 0) + np.where(A.T.astype(bool), 3, 0))
        items.append((f"cpdag|{case}", marks(gr.GraphTransforms.dagToCpdag(g), idx), "cpdag"))
    for key, M, kind in items:
        g, idx = to_t(M)
        cnt = {}
        for _ in range(draws):
            h = gr.GraphTransforms.magFromPag(g) if kind == "pag" else gr.GraphTransforms.dagFromCpdag(g)
            b = marks(h, idx).tobytes().hex()
            cnt[b] = cnt.get(b, 0) + 1
        out[key + "|graph"] = M
        out[key + "|counts"] = np.array(json.dumps(cnt))
        print(f"{key:28s} distinct {len(cnt)}", flush=True)
    np.savez_compressed(OUT / out_file, **out)
    print("wrote", OUT / out_file)


def make_oct_configs(out_file="oct_configs_ref.json"):
    """ETIA's create_causal_configs for all the settings (without models)."""
    import json, itertools
    sys.path.insert(0, str(ETIA_ROOT))
    from ETIA.CausalLearning.configurations.class_causal_configurator import CausalDiscoveryConfigurator
    out = {}
    for dt, cs, af, ts, tl in itertools.product(["continuous", "mixed", "categorical"], [True, False],
                                                [True, False], [True, False], [True, False]):
        r = CausalDiscoveryConfigurator().create_causal_configs(dt, cs, af, ts, tl)
        out[f"{dt}|{cs}|{af}|{ts}|{tl}"] = r
    r = CausalDiscoveryConfigurator().create_causal_configs("continuous", False, True, False, False,
                                                             exclude_algs=["gfci", "rfci", "cfci"])
    out["default"] = r
    (OUT / out_file).write_text(json.dumps(out, indent=1))
    print("wrote", OUT / out_file)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["graphs"]
    if "notears" in which:
        make_notears()
    if "sam" in which:
        make_sam()
    if "fs" in which:
        make_fs()
    if "afs" in which:
        make_afs()
    if "oct_configs" in which:
        make_oct_configs()
    if not set(which) - {"notears", "sam", "fs", "afs", "oct_configs"}:
        sys.exit(0)
    # oct: two processors (FisherZ fails with one); oct1: one processor for the
    # discrete case (Tetrad's parallel FCI with chisquare/gsquare has races)
    jp = start_jvm(one_cpu=bool({"fci1", "oct", "oct1"} & set(which)), cpus=2 if "oct" in which else 1)
    if "oct1" in which:
        make_oct(jp)
    if "oct" in which:
        make_oct(jp)
    if "members" in which:
        make_members(jp)
    if "fci1" in which:
        print("processors seen by the JVM:", jp.JClass("java.lang.Runtime").getRuntime().availableProcessors())
    if "graphs" in which:
        make_graphs(jp)
    if "ci" in which:
        make_ci(jp)
    if "scale" in which:
        make_ci_scale(jp)
    if "search" in which:
        make_search(jp)
    if "scores" in which:
        make_scores(jp)
    if "fges" in which:
        make_fges(jp, reps=3)
    if "gfci" in which:
        make_gfci(jp)
    if "ts" in which:
        make_ts(jp)
    if "ts_tigramite" in which:
        make_ts_tigramite()
    if "ts_redo" in which:
        make_ts(jp, only_errors=True)
    if "fci" in which:
        make_search(jp, ("fci", "fcimax", "rfci", "cfci"), "search_fci_data.npz", "search_fci_ref.npz")
    if "fci1" in which:
        make_search(jp, ("fci", "fcimax", "rfci", "cfci"), "search_fci_data.npz", "search_fci_ref_1cpu.npz")
