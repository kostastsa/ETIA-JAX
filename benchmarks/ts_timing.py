"""Time the time-series searches of ETIA-JAX (increment 6).

    .venv/bin/python benchmarks/ts_timing.py
    JAX_PLATFORMS=cpu .venv/bin/python benchmarks/ts_timing.py

Writes ``benchmarks/results/ts_jax_<device>.csv``: one row per search with the
ETIA time (Tetrad or tigramite, from the fixtures) and the time of ETIA-JAX
(after a first run that compiles). The Tetrad algorithms run in the Tetrad
mode and in the default mode; the tigramite algorithms have one mode (the
same output as tigramite).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_timeseries import _run, _ts_cases, TEST_ALGOS          # noqa: E402
from etia_jax.ci import CI_TESTS                                  # noqa: E402
from etia_jax.scores import SCORES, ETIA_SETTINGS                 # noqa: E402
from etia_jax.search.fges import fges                             # noqa: E402
from etia_jax.search.gfci import gfci                             # noqa: E402
from etia_jax.search.svar import svargfci                         # noqa: E402
from etia_jax.search.pcmci import pcmci, pcmciplus                # noqa: E402
from etia_jax.search.lpcmci import lpcmci                         # noqa: E402

FIX = ROOT / "reference" / "fixtures"
COMPAT = {"dg_lrt": {"tetrad_compat": True}}


def _run_default(algo, X, kinds, t, a, sc, kn):
    if algo == "fges":
        return fges(SCORES[sc](X, kinds, **ETIA_SETTINGS[sc]), knowledge=kn).graph
    ci = CI_TESTS[t](X, kinds, **COMPAT.get(t, {}))
    if algo in ("gfci", "svargfci"):
        f = gfci if algo == "gfci" else svargfci
        return f(ci, SCORES[sc](X, kinds, **ETIA_SETTINGS[sc]), alpha=float(a), knowledge=kn).graph
    return TEST_ALGOS[algo](ci, alpha=float(a), knowledge=kn).graph


def timed(f):
    f()
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def main():
    dev = jax.devices()[0].platform
    rows = []
    r = np.load(FIX / "ts_tetrad_ref.npz")
    algos = tuple(TEST_ALGOS) + ("fges", "gfci", "svargfci")
    for case, t, a, sc, algo, X, kinds, kn, refs in _ts_cases(algos):
        key = f"{case}|{t}|{a}|{sc}|{algo}"
        tc = timed(lambda: _run(algo, X, kinds, t, a, sc, kn))
        td = timed(lambda: _run_default(algo, X, kinds, t, a, sc, kn))
        rows.append(dict(family="tetrad", algo=algo, case=case, group=case.rsplit("_s", 1)[0], test=t,
                         score="" if sc == "None" else sc, alpha=a, n_vars=X.shape[1],
                         etia_s=float(r[key + "|time"]), jax_compat_s=tc, jax_default_s=td))
        print(f"{dev} {key:52s} etia {rows[-1]['etia_s']:.3f}  compat {tc:.3f}  default {td:.3f}", flush=True)
    d = np.load(FIX / "ts_data.npz")
    tr = np.load(FIX / "ts_tigramite_ref.npz")
    fn = {"PCMCI": pcmci, "PCMCI+": pcmciplus, "LPCMCI": lpcmci}
    for k in sorted(tr.files):
        if not k.endswith("|graph"):
            continue
        case, test, a, algo = k.split("|")[:4]
        X, L = d[f"{case}|X"], int(d[f"{case}|L"])
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        tc = timed(lambda: fn[algo](X, test, L, float(a), kinds=kinds))
        rows.append(dict(family="tigramite", algo=algo, case=case, group=case.rsplit("_s", 1)[0], test=test,
                         score="", alpha=a, n_vars=X.shape[1], etia_s=float(tr[k[:-6] + "|time"]),
                         jax_compat_s=tc, jax_default_s=tc))
        print(f"{dev} {k[:-6]:52s} etia {rows[-1]['etia_s']:.3f}  jax {tc:.3f}", flush=True)
    out = ROOT / "benchmarks" / "results" / f"ts_jax_{dev}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
