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


def start_jvm(one_cpu=False):
    """ETIA's JVM. With one_cpu, the JVM sees one processor
    (-XX:ActiveProcessorCount=1): Tetrad's parallel streams and ForkJoin
    pools then run on one thread, so its searches have no races."""
    sys.path.insert(0, str(ETIA_ROOT))
    import jpype
    if one_cpu and not jpype.isJVMStarted():
        import glob, os
        jars = glob.glob(str(ETIA_ROOT / "ETIA" / "CausalLearning" / "algorithms" / "jar_files" / "*.jar"))
        jvm = os.path.join(os.environ["JAVA_HOME"], "lib", "server", "libjvm.so")
        jpype.startJVM(jvm, "-ea", "-XX:ActiveProcessorCount=1", classpath=":".join(jars), convertStrings=False)
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


def _etia_dataset(X, disc):
    import pandas as pd
    from ETIA.data.Dataset import Dataset
    from ETIA.CausalLearning.algorithms.utils import prepare_data_tetrad
    p = X.shape[1]
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(p)])
    for j in np.flatnonzero(disc):
        df[f"V{j}"] = df[f"V{j}"].astype(int)
    ds, _ = prepare_data_tetrad(Dataset(data=df), {})
    return ds


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


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["graphs"]
    jp = start_jvm(one_cpu="fci1" in which)
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
    if "fci" in which:
        make_search(jp, ("fci", "fcimax", "rfci", "cfci"), "search_fci_data.npz", "search_fci_ref.npz")
    if "fci1" in which:
        make_search(jp, ("fci", "fcimax", "rfci", "cfci"), "search_fci_data.npz", "search_fci_ref_1cpu.npz")
