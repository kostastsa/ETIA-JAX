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

from make_fixtures import OUT, start_jvm, to_marks, _tetrad_test, TETRAD_CLASS

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


if __name__ == "__main__":
    main()
