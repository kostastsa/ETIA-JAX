"""Time FBED, SES and the AFS of ETIA-JAX against ETIA (increment 8).

    .venv/bin/python benchmarks/afs_timing.py
    JAX_PLATFORMS=cpu .venv/bin/python benchmarks/afs_timing.py

Writes ``benchmarks/results/afs_jax_<platform>.csv``. One row per FS call
(ETIA: one Rscript run of MXM through ETIA's FeatureSelector, with its CSV
files) and one row per whole AFS run. The ETIA-JAX time is the time of a
second run (the first run compiles).
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax

ROOT = Path(__file__).resolve().parents[1]
from etia_jax.afs.fs import fbed, ses, FSError           # noqa: E402
from etia_jax.afs.afs import run_afs, predictive_configs  # noqa: E402

FIX = ROOT / "reference" / "fixtures"
FIELDS = ["kind", "key", "p", "etia_time", "jax_time", "speedup", "same"]


def timed(f):
    f()
    t0 = time.perf_counter()
    out = f()
    return time.perf_counter() - t0, out


def main():
    platform = jax.devices()[0].platform
    r = np.load(FIX / "fs_ref.npz")
    d = np.load(FIX / "search_data.npz")
    rows = []
    for key in sorted({k.rsplit("|", 1)[0] for k in r.files if k.endswith("|time")}):
        case, t, name, alpha, k = key.split("|")
        X = d[f"{case}|X"]
        t, alpha, k = int(t), float(alpha), int(k)
        others = [j for j in range(X.shape[1]) if j != t]

        def f():
            try:
                if name == "fbed":
                    return fbed(X[:, t], X[:, others], alpha, k, etia_compat=True)
                return ses(X[:, t], X[:, others], alpha, k)[0]
            except FSError:
                return "error"
        dt, got = timed(f)
        want = "error" if key + "|error" in r.files else r[key + "|sel"].tolist()
        et = float(r[key + "|time"])
        rows.append(dict(kind=name, key=key, p=X.shape[1], etia_time=et, jax_time=dt, speedup=et / dt,
                         same=int(got == want)))
    a = np.load(FIX / "afs_ref.npz")
    for key in sorted(k.rsplit("|", 1)[0] for k in a.files if k.endswith("|result")):
        case, t, tag = key.split("|")
        X = d[f"{case}|X"]
        cols = [f"X{i}" for i in range(X.shape[1])]
        configs = predictive_configs()
        if tag == "linear":
            configs = [c for c in configs if c["model"] == "linear_regression"]
        df = pd.DataFrame(X, columns=cols)
        dt, out = timed(lambda: run_afs(df, [cols[int(t)]], configs, etia_compat=True))
        ref = json.loads(str(a[key + "|result"]))
        same = out["best_config"] == ref["best_config"] and \
            sorted(out["selected_features"][cols[int(t)]]) == ref["selected"]
        et = float(a[key + "|time"])
        rows.append(dict(kind=f"afs_{tag}", key=key, p=X.shape[1], etia_time=et, jax_time=dt,
                         speedup=et / dt, same=int(same)))
        print(f"{key:24s} etia {et:7.1f}s  jax {dt:6.2f}s  same {int(same)}", flush=True)
    out = ROOT / "benchmarks" / "results" / f"afs_jax_{platform}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    fs = [x for x in rows if x["kind"] in ("fbed", "ses")]
    print(f"FS calls: {len(fs)}, same {sum(x['same'] for x in fs)}, "
          f"median speed-up {np.median([x['speedup'] for x in fs]):.0f}x")
    print("wrote", out)


if __name__ == "__main__":
    main()
