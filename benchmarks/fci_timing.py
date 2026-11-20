"""Time FCI, FCI-Max, RFCI and CFCI of ETIA-JAX on the FCI fixtures (increment 4).

    .venv/bin/python benchmarks/fci_timing.py
    JAX_PLATFORMS=cpu .venv/bin/python benchmarks/fci_timing.py

Writes ``benchmarks/results/fci_jax_<device>.csv``: one row per search with
the ETIA time (Tetrad as ETIA runs it; empty when Tetrad crashed), the
compiled time of the Tetrad mode and of the default mode, and the numbers
of CI queries.
"""
import csv
import time
from pathlib import Path

import numpy as np
import jax

from etia_jax.ci import CI_TESTS
from etia_jax.search.fci import fci, fcimax, rfci, cfci

ROOT = Path(__file__).resolve().parents[1]
ALGOS = {"fci": fci, "fcimax": fcimax, "rfci": rfci, "cfci": cfci}
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}


def timed(f, *a, **k):
    f(*a, **k)                                   # compile and warm the CI tests
    t0 = time.perf_counter()
    r = f(*a, **k)
    return r, time.perf_counter() - t0


def main():
    dev = jax.devices()[0].platform
    d = np.load(ROOT / "reference" / "fixtures" / "search_fci_data.npz")
    r = np.load(ROOT / "reference" / "fixtures" / "search_fci_ref.npz")
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
                for name, algo in ALGOS.items():
                    key = f"{case}|{t}|{a}|{name}"
                    crashed = key + "|error" in r.files
                    rc, t_compat = timed(algo, ci, alpha=a, tetrad_compat=True)
                    rd, t_default = timed(algo, ci, alpha=a)
                    rows.append(dict(case=case, group=case.rsplit("_s", 1)[0], p=X.shape[1], n=X.shape[0],
                                     test=t, alpha=a, algo=name,
                                     etia_s=None if crashed else float(r[key + "|time"]),
                                     etia_prep_s=float(r[f"{case}|prep_time"]), tetrad_crashed=crashed,
                                     jax_compat_s=t_compat, jax_default_s=t_default,
                                     n_tests_compat=rc.n_tests, n_tests_default=rd.n_tests))
                    print(f"{dev} {key:38s} etia {rows[-1]['etia_s']}  compat {t_compat:7.3f}s  default {t_default:7.3f}s",
                          flush=True)
    out = ROOT / "benchmarks" / "results" / f"fci_jax_{dev}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
