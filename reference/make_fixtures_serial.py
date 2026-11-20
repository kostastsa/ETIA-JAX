"""Race-free Tetrad references for the FCI family (ETIA venv).

Tetrad 7.6.5 runs parts of FCI, FCI-Max, RFCI and CFCI in parallel threads
on one CI test object, and the chi-square / G-square tests are not thread
safe: the p-values can be corrupted, and FCI crashes on discrete data. Here
each test call goes through a lock (a JPype proxy), so the calls are serial.
The order of the parallel R4 tasks can still vary, so each search runs
``REPS`` times and all distinct outputs are kept.

    JAVA_HOME=... ~/PROJECTS/ETIA/.venv/bin/python reference/make_fixtures_serial.py
"""
import threading
import numpy as np
import pandas as pd
import jpype

import sys

from make_fixtures import (OUT, start_jvm, to_marks, _tetrad_test, TETRAD_CLASS, GFCI_TESTS, SCORES_FOR,
                           data_kind, tetrad_score, _etia_dataset)

REPS = 5   # the 4 cases of disc10_s2 that need more were run 40 times (all outputs are merged)


def serialized(real):
    lock = threading.Lock()

    @jpype.JImplements("edu.cmu.tetrad.search.IndependenceTest", deferred=False)
    class Serial:
        @jpype.JOverride
        def checkIndependence(self, x, y, z):
            with lock:
                return real.checkIndependence(x, y, z)

        @jpype.JOverride
        def getVariables(self): return real.getVariables()

        @jpype.JOverride
        def getData(self): return real.getData()

        @jpype.JOverride
        def getAlpha(self): return real.getAlpha()

        @jpype.JOverride
        def setAlpha(self, a): real.setAlpha(a)

        @jpype.JOverride
        def isVerbose(self): return real.isVerbose()

        @jpype.JOverride
        def setVerbose(self, v): real.setVerbose(v)

        @jpype.JOverride
        def indTestSubset(self, v): return real.indTestSubset(v)

        @jpype.JOverride
        def toString(self): return "serial"
    return Serial()


def main():
    jp = start_jvm()
    from ETIA.data.Dataset import Dataset
    from ETIA.CausalLearning.algorithms.utils import prepare_data_tetrad
    f = np.load(OUT / "search_fci_data.npz")
    out = {}
    for case in f["cases"]:
        case = str(case)
        if not case.startswith(("disc", "mixed")):
            continue
        X, disc = f[f"{case}|X"], f[f"{case}|discrete"]
        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(X.shape[1])])
        for j in np.flatnonzero(disc):
            df[f"V{j}"] = df[f"V{j}"].astype(int)
        ds, _ = prepare_data_tetrad(Dataset(data=df), {})
        names = [str(v.getName()) for v in ds.getVariables()]
        for t in f[f"{case}|tests"]:
            for alpha in f["alphas"]:
                for algo in ("fci", "fcimax", "rfci", "cfci"):
                    graphs = []
                    for _ in range(REPS):
                        real = _tetrad_test(jp, str(t), ds)
                        real.setAlpha(float(alpha)) if hasattr(real, "setAlpha") else None
                        g = jp.JClass("edu.cmu.tetrad.search." + TETRAD_CLASS[algo])(serialized(real)).search()
                        G = to_marks(g, names)
                        if not any((G == h).all() for h in graphs):
                            graphs.append(G)
                    key = f"{case}|{t}|{float(alpha)}|{algo}"
                    out[key + "|graphs"] = np.stack(graphs)
                    print(f"{key:40s} distinct outputs: {len(graphs)}", flush=True)
    np.savez_compressed(OUT / "search_fci_ref_serial.npz", **out)
    print("wrote", OUT / "search_fci_ref_serial.npz")


def main_gfci(reps=10):
    """GFCI, ``reps`` runs each, all distinct outputs kept
    (search_gfci_ref_more.npz). Discrete and mixed data: serialized test
    calls. Continuous data: the FisherZ test directly (it is thread-safe,
    and the proxy is slow)."""
    jp = start_jvm()
    f = np.load(OUT / "search_fci_data.npz")
    out = {}
    for case in f["cases"]:
        case = str(case)
        X, disc = f[f"{case}|X"], f[f"{case}|discrete"]
        ds = _etia_dataset(X, disc)
        names = [str(v.getName()) for v in ds.getVariables()]
        for t in [str(t) for t in f[f"{case}|tests"] if str(t) in GFCI_TESTS]:
            for alpha in f["alphas"]:
                for name in SCORES_FOR[data_kind(disc)]:
                    graphs = []
                    for _ in range(reps):
                        real = _tetrad_test(jp, t, ds)
                        real.setAlpha(float(alpha))
                        test = serialized(real) if disc.any() else real     # continuous: FisherZ is thread-safe
                        alg = jp.JClass("edu.cmu.tetrad.search.GFci")(test, tetrad_score(jp, name, ds))
                        G = to_marks(alg.search(), names)
                        if not any((G == h).all() for h in graphs):
                            graphs.append(G)
                    key = f"{case}|{t}|{float(alpha)}|{name}|gfci"
                    out[key + "|graphs"] = np.stack(graphs)
                    print(f"{key:48s} distinct outputs: {len(graphs)}", flush=True)
    np.savez_compressed(OUT / "search_gfci_ref_more.npz", **out)
    print("wrote", OUT / "search_gfci_ref_more.npz")


def main_gfci_steps():
    """The deterministic steps of GFCI (serialized test calls, one run):
    the FGES CPDAG, the graph after the extra-edge removal, and the graph
    after gfciR0 (search_gfci_steps.npz). Only the final orientation
    (parallel R4) can vary between runs."""
    jp = start_jvm()
    S = "edu.cmu.tetrad.search."
    GU = jp.JClass("edu.cmu.tetrad.graph.GraphUtils")
    f = np.load(OUT / "search_fci_data.npz")
    out = {}
    for case in f["cases"]:
        case = str(case)
        X, disc = f[f"{case}|X"], f[f"{case}|discrete"]
        ds = _etia_dataset(X, disc)
        names = [str(v.getName()) for v in ds.getVariables()]
        for t in [str(t) for t in f[f"{case}|tests"] if str(t) in GFCI_TESTS]:
            for alpha in f["alphas"]:
                for name in SCORES_FOR[data_kind(disc)]:
                    real = _tetrad_test(jp, t, ds)
                    real.setAlpha(float(alpha))
                    test = serialized(real) if disc.any() else real
                    fg = jp.JClass(S + "Fges")(tetrad_score(jp, name, ds))
                    fg.setFaithfulnessAssumed(True)
                    fg.setMaxDegree(-1)
                    fg.setVerbose(False)
                    g = fg.search()
                    key = f"{case}|{t}|{float(alpha)}|{name}|gfci"
                    out[key + "|fges"] = to_marks(g, names)
                    cpdag = jp.JClass("edu.cmu.tetrad.graph.EdgeListGraph")(g)
                    seps = jp.JClass(S + "utils.SepsetsMinP")(g, test, -1)
                    GU.gfciExtraEdgeRemovalStep(g, cpdag, jp.JClass("java.util.ArrayList")(test.getVariables()),
                                                seps, -1, False)
                    out[key + "|removal"] = to_marks(g, names)
                    GU.gfciR0(g, cpdag, seps, jp.JClass("edu.cmu.tetrad.data.Knowledge")(), False,
                              jp.JClass("java.util.HashSet")())
                    out[key + "|r0"] = to_marks(g, names)
                    print(f"{key:48s} steps saved", flush=True)
    np.savez_compressed(OUT / "search_gfci_steps.npz", **out)
    print("wrote", OUT / "search_gfci_steps.npz")


if __name__ == "__main__":
    if sys.argv[1:] == ["gfci"]:
        main_gfci()
    elif sys.argv[1:] == ["gfci_steps"]:
        main_gfci_steps()
    else:
        main()
