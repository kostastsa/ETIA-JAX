"""Time the CRV of ETIA-JAX against ETIA (increment 10).

    .venv/bin/python benchmarks/crv_timing.py
    JAX_PLATFORMS=cpu .venv/bin/python benchmarks/crv_timing.py

Writes ``benchmarks/results/crv_jax_<platform>.csv``: one row per kind of
call. ETIA's times come from reference/fixtures (crv_ref.npz, adjset_time.json:
one R call of ETIA's script with dagitty). The ETIA-JAX times are the times
of a second run.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import jax

from etia_jax.crv import queries as Q
from etia_jax.crv.adjustment import find_adjset
from etia_jax.crv.confidence import bootstrap_graphs, paths_metrics

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "reference" / "fixtures"
FIELDS = ["kind", "calls", "etia_time", "jax_time", "speedup", "same"]


def timed(f):
    f()
    t0 = time.perf_counter()
    out = f()
    return time.perf_counter() - t0, out


def main():
    platform = jax.devices()[0].platform
    r = np.load(FIX / "crv_ref.npz")
    cases = json.loads((FIX / "adjset_ref.json").read_text())
    rows = []
    # adjustment sets: ETIA = one R call per query
    rtimes = json.loads((FIX / "adjset_time.json").read_text())
    norm = lambda x: None if x is None else ([x] if isinstance(x, str) else sorted(x))

    def adj():
        return [find_adjset(np.array(c["M"]), c["names"], c["type"], c["outcomes"][0], c["exposures"],
                            etia_compat=True) for c in cases]
    dt, got = timed(adj)
    same = sum(g == (norm(c["dagitty"]["canonical"]), norm(c["dagitty"]["minimal"])) for g, c in zip(got, cases))
    rows.append(dict(kind="adjustment set", calls=len(cases), etia_time=float(np.median(rtimes)),
                     jax_time=dt / len(cases), same=same))
    # queries
    qs = json.loads(str(r["queries"]))
    fn = dict(directed=Q.one_directed_path, potentially_directed=Q.one_potentially_directed_path,
              any=Q.one_path_anytype, bidirected=Q.one_bidirected_path_from_to)

    def run_q():
        return [[f(np.array(cases[q["case"]]["M"]), q["start"], q["end"], etia_compat=True) for f in fn.values()]
                for q in qs]
    dt, got = timed(run_q)
    same = sum(g == [q[k] for k in fn] for g, q in zip(got, qs))
    n = 4 * len(qs)
    rows.append(dict(kind="path query", calls=n, etia_time=float(r["queries_time"]) / n, jax_time=dt / n,
                     same=same * 4))
    # bootstrap (FCI on 50 bootstrap samples) and path metrics
    d = np.load(FIX / "search_data.npz")
    for case in ("cont10_s0", "cont10_s1", "cont20_s0"):
        X = d[f"{case}|X"][:1000]
        names = [f"X{i}" for i in range(X.shape[1])]
        cfg = {"name": "fci", "ci_test": "FisherZ", "significance_level": 0.05}
        dt, boots = timed(lambda: bootstrap_graphs(X, cfg, rows=list(r[f"{case}|rows"]), etia_compat=True))
        same = sum(np.array_equal(a, b) for a, b in zip(boots, r[f"{case}|boots"]))
        rows.append(dict(kind=f"bootstrap 50 FCI {case}", calls=1, etia_time=float(r[f"{case}|boot_time"]),
                         jax_time=dt, same=same))
        paths = {"p": json.loads(str(r[f"{case}|paths"]))}
        dt, (pc, _) = timed(lambda: paths_metrics(r[f"{case}|best"], boots, paths, names=names, etia_compat=True))
        rows.append(dict(kind=f"path metrics {case}", calls=1, etia_time=float(r[f"{case}|paths_time"]),
                         jax_time=dt, same=int(np.array_equal(pc["p"], r[f"{case}|path_consistency"]))))
    for x in rows:
        x["speedup"] = x["etia_time"] / x["jax_time"]
        print(f"{x['kind']:34s} etia {x['etia_time']:.4f}s  jax {x['jax_time']:.5f}s  {x['speedup']:7.1f}x  same {x['same']}")
    out = ROOT / "benchmarks" / "results" / f"crv_jax_{platform}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
