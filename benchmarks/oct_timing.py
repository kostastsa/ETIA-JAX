"""Time the OCT of ETIA-JAX against ETIA's CausalLearner (increment 9).

    .venv/bin/python benchmarks/oct_timing.py
    JAX_PLATFORMS=cpu .venv/bin/python benchmarks/oct_timing.py

Writes ``benchmarks/results/oct_jax_<platform>.csv``. One row per ETIA run
of reference/fixtures/oct_ref.npz (a case and a tag: ``lin`` LinearRegression
or ``rf`` ETIA's random forest; ``1`` n_jobs=1, ``8`` joblib workers). The
ETIA-JAX run has the same settings in ``etia_compat`` (``etia_n_jobs``), and
its time is the time of a second run (the first run compiles). ``pags_same``
is the fraction of the fold PAGs that are the same as ETIA's; ``default_s``
is the time of the default mode.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import jax

from etia_jax.oct.configs import var_types
from etia_jax.oct.oct import run_oct, DEFAULT_REGRESSOR

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "reference" / "fixtures"
FIELDS = ["case", "tag", "p", "configs", "etia_time", "jax_time", "speedup", "default_s", "pags_same",
          "etia_chosen", "jax_chosen"]


def main():
    platform = jax.devices()[0].platform
    ref = np.load(FIX / "oct_ref.npz")
    d = np.load(FIX / "search_data.npz")
    rows = []
    for key in sorted(k.rsplit("|", 1)[0] for k in ref.files if k.endswith("|time")):
        case, tag = key.split("|")
        X, kinds, _ = var_types(d[f"{case}|X"][:1000])
        cfgs = json.loads(str(ref[key + "|configs"]))
        reg = ("LinearRegression", {}) if tag.startswith("lin") else DEFAULT_REGRESSOR
        nj = 1 if tag.endswith("1") else None
        run = lambda compat: run_oct(X, cfgs, kinds, regressor=reg, etia_compat=compat, etia_n_jobs=nj)
        run(True)
        t0 = time.perf_counter()
        r = run(True)
        dt = time.perf_counter() - t0
        run(False)
        t0 = time.perf_counter()
        run(False)
        dd = time.perf_counter() - t0
        mec = ref[key + "|mec"]
        same = np.mean([np.array_equal(np.array(r["mec_graphs"][c][k]), mec[c, k])
                        for c in range(mec.shape[0]) for k in range(mec.shape[1])])
        et = float(ref[key + "|time"])
        rows.append(dict(case=case, tag=tag, p=X.shape[1], configs=len(cfgs), etia_time=et, jax_time=dt,
                         speedup=et / dt, default_s=dd, pags_same=same, etia_chosen=int(ref[key + "|chosen"]),
                         jax_chosen=r["chosen"]))
        print(f"{key:20s} etia {et:6.1f}s  jax {dt:6.2f}s  default {dd:6.2f}s  pags {same:.2f}", flush=True)
    out = ROOT / "benchmarks" / "results" / f"oct_jax_{platform}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
