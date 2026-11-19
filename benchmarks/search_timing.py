"""Time PC and CPC of ETIA-JAX on the search fixtures (increment 3).

    ETIA_JAX_CACHE_DIR= .venv/bin/python benchmarks/search_timing.py
    ETIA_JAX_CACHE_DIR= JAX_PLATFORMS=cpu .venv/bin/python benchmarks/search_timing.py

(ETIA_JAX_CACHE_DIR empty: no disk cache, so "first" includes the compile.)
Writes ``benchmarks/results/search_jax_<device>.csv``: one row per search
with the first-call time, the warm time of the Tetrad mode (the same graph as
ETIA) and of the PC-stable mode, the number of queries, and whether the graph
equals Tetrad's.
"""
import csv
import time
from pathlib import Path

import numpy as np
import jax

from etia_jax.ci import CI_TESTS
from etia_jax.search import pc, cpc

ROOT = Path(__file__).resolve().parents[1]
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}


def main():
    dev = jax.devices()[0].platform
    d = np.load(ROOT / "reference" / "fixtures" / "search_data.npz")
    r = np.load(ROOT / "reference" / "fixtures" / "search_ref.npz")
    rows = []
    for case in d["cases"]:
        case = str(case)
        X = d[f"{case}|X"]
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        for t in d[f"{case}|tests"]:
            t = str(t)
            ci = CI_TESTS[t](X, kinds, **COMPAT.get(t, {}))
            for a in d["alphas"]:
                a = float(a)
                for name, algo in (("pc", pc), ("cpc", cpc)):
                    key = f"{case}|{t}|{a}|{name}"
                    t0 = time.perf_counter()
                    res = algo(ci, alpha=a, tetrad_compat=True)
                    first = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    res = algo(ci, alpha=a, tetrad_compat=True)
                    warm = time.perf_counter() - t0
                    algo(ci, alpha=a)                                 # compile the stable mode
                    t0 = time.perf_counter()
                    stable = algo(ci, alpha=a)
                    warm_stable = time.perf_counter() - t0
                    rows.append(dict(case=case, p=X.shape[1], n=X.shape[0], test=t, alpha=a, algo=name,
                                     etia_search_s=float(r[key + "|time"]),
                                     etia_prep_s=float(r[f"{case}|prep_time"]),
                                     jax_first_s=first, jax_warm_s=warm, jax_stable_warm_s=warm_stable,
                                     n_tests=res.n_tests, n_computed=res.n_computed,
                                     n_tests_stable=stable.n_tests,
                                     same_as_tetrad=bool((res.graph == r[key + "|graph"]).all())))
                    print(f"{dev} {key:36s} etia {rows[-1]['etia_search_s']:7.3f}s  jax first {first:6.2f}s"
                          f" warm {warm:7.4f}s stable {warm_stable:7.4f}s same={rows[-1]['same_as_tetrad']}",
                          flush=True)
    out = ROOT / "benchmarks" / "results" / f"search_jax_{dev}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
