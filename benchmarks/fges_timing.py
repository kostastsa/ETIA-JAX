"""Time FGES and GFCI of ETIA-JAX on the fixtures (increment 5).

    .venv/bin/python benchmarks/fges_timing.py
    JAX_PLATFORMS=cpu .venv/bin/python benchmarks/fges_timing.py

Writes ``benchmarks/results/fges_jax_<device>.csv``: one row per search
with the ETIA time (Tetrad as ETIA runs it; empty when Tetrad crashed), and
the time of the Tetrad mode and of the default mode (after a first run that
compiles and warms the caches of the CI tests; each run makes a new score
object, so the local scores are computed again).
"""
import csv
import time
from pathlib import Path

import numpy as np
import jax

from etia_jax.ci import CI_TESTS
from etia_jax.scores import SCORES, ETIA_SETTINGS
from etia_jax.search.fges import fges
from etia_jax.search.gfci import gfci

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "reference" / "fixtures"
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}


def timed(make, run):
    run(make())                                   # compile and warm
    obj = make()
    t0 = time.perf_counter()
    r = run(obj)
    return r, time.perf_counter() - t0


def main():
    dev = jax.devices()[0].platform
    rows = []
    # ------------------------------------------------------------- FGES
    d = np.load(FIX / "search_data.npz")
    r = np.load(FIX / "search_fges_ref.npz")
    for k in sorted(r.files):
        if not k.endswith("|graph"):
            continue
        case, name, _, _ = k.split("|")
        X = d[f"{case}|X"]
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        mk_c = lambda: SCORES[name](X, kinds, tetrad_compat=True, **ETIA_SETTINGS[name])
        mk_d = lambda: SCORES[name](X, kinds, **ETIA_SETTINGS[name])
        rc, tc = timed(mk_c, lambda s: fges(s, tetrad_compat=True))
        rd, td = timed(mk_d, lambda s: fges(s))
        rows.append(dict(algo="fges", case=case, group=case.rsplit("_s", 1)[0], p=X.shape[1], n=X.shape[0],
                         test="", alpha="", score=name,
                         etia_s=float(r[f"{case}|{name}|fges|time"]), etia_prep_s=float(r[f"{case}|prep_time"]),
                         tetrad_crashed=False, same_as_etia=bool((rc.graph == r[k]).all()),
                         jax_compat_s=tc, jax_default_s=td, n_scores_compat=rc.n_scores))
        print(f"{dev} fges {case:12s} {name:14s} etia {rows[-1]['etia_s']:.3f}  compat {tc:7.3f}s  "
              f"default {td:7.3f}s  same {rows[-1]['same_as_etia']}", flush=True)
    # ------------------------------------------------------------- GFCI
    d = np.load(FIX / "search_fci_data.npz")
    r = np.load(FIX / "search_gfci_ref.npz")
    cis = {}
    for k in sorted(r.files):
        if not (k.endswith("|graph") or k.endswith("|error")):
            continue
        base = k.rsplit("|", 1)[0]
        if k.endswith("|error") and base + "|graph" in r.files:
            continue                                         # counted with its graph
        case, test, alpha, name, _ = base.split("|")
        alpha = float(alpha)
        X = d[f"{case}|X"]
        kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
        ci = cis.setdefault((case, test), CI_TESTS[test](X, kinds, **COMPAT.get(test, {})))
        mk_c = lambda: SCORES[name](X, kinds, tetrad_compat=True, **ETIA_SETTINGS[name])
        mk_d = lambda: SCORES[name](X, kinds, **ETIA_SETTINGS[name])
        rc, tc = timed(mk_c, lambda s: gfci(ci, s, alpha=alpha, tetrad_compat=True))
        rd, td = timed(mk_d, lambda s: gfci(ci, s, alpha=alpha))
        crashed = base + "|graph" not in r.files
        rows.append(dict(algo="gfci", case=case, group=case.rsplit("_s", 1)[0], p=X.shape[1], n=X.shape[0],
                         test=test, alpha=alpha, score=name,
                         etia_s=None if crashed else float(r[base + "|time"]),
                         etia_prep_s=float(r[f"{case}|prep_time"]), tetrad_crashed=base + "|error" in r.files,
                         same_as_etia=(not crashed) and bool((rc.graph == r[base + "|graph"]).all()),
                         jax_compat_s=tc, jax_default_s=td, n_scores_compat=rc.n_scores))
        print(f"{dev} gfci {base:44s} etia {rows[-1]['etia_s']}  compat {tc:7.3f}s  default {td:7.3f}s",
              flush=True)
    out = ROOT / "benchmarks" / "results" / f"fges_jax_{dev}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
