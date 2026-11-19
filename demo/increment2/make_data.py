"""Data files of the increment-2 demo page (CI tests and timings).

Run from the repository root, after benchmarks/ci_timing.py (GPU and CPU)
and reference/make_fixtures.py ci scale (ETIA venv):

    .venv/bin/python demo/increment2/make_data.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "tests"))
FIX = ROOT / "reference" / "fixtures"
RES = ROOT / "benchmarks" / "results"

from test_ci import compare, _pairs, _fork_data, ALPHA, PARAMS   # noqa: E402
from etia_jax.ci import CI_TESTS                                 # noqa: E402

ORIGIN = {"FisherZ": "Tetrad", "cci": "Tetrad", "cg_lrt": "Tetrad", "dg_lrt": "Tetrad",
          "chisquare": "Tetrad", "gsquare": "Tetrad"}
d, ref = np.load(FIX / "ci_data.npz"), np.load(FIX / "ci_ref.npz")
gpu, cpu = np.load(RES / "ci_jax_gpu.npz"), np.load(RES / "ci_jax_cpu.npz")
sref = np.load(FIX / "ci_scale_ref.npz")

# ------------------------------------------------------------- agreement
rows = []
for name, dname in _pairs(ref):
    r = compare(name, dname, d, ref)
    if name == "CMIknn":
        stat_j = gpu[f"{name}|{dname}|stat"]
        r["max_abs_dp"] = float("nan")
        r["decisions_same"] = float(np.mean((stat_j > ALPHA) == (ref[f"{name}|{dname}|stat"] > ALPHA)))
    rows.append({"test": name, "dataset": dname, "origin": ORIGIN.get(name, "tigramite"),
                 "decisions_same_pct": 100 * r["decisions_same"],
                 "max_abs_dp": r["max_abs_dp"], "max_rel_dstat": r.get("max_rel_dstat", np.nan),
                 "mode": "tetrad_compat" if name in ("cci", "dg_lrt") else "default"})
agree = pd.DataFrame(rows)
agree.to_csv(OUT / "agreement.csv", index=False)

# --------------------------------------------------------------- timing
rows = []
for name, dname in _pairs(ref):
    k = f"{name}|{dname}"
    etia = float(ref[f"{k}|time"].sum())
    row = {"test": name, "dataset": dname, "origin": ORIGIN.get(name, "tigramite"),
           "etia_s": etia,
           "jax_gpu_s": float(gpu[f"{k}|steady"]), "jax_cpu_s": float(cpu[f"{k}|steady"]),
           "jax_gpu_batched_s": float(gpu[f"{k}|batched"]), "jax_cpu_batched_s": float(cpu[f"{k}|batched"]),
           "jax_gpu_first_s": float(gpu[f"{k}|first"]), "jax_cpu_first_s": float(cpu[f"{k}|first"])}
    best = np.nanmin([row["jax_gpu_s"], row["jax_cpu_s"], row["jax_gpu_batched_s"], row["jax_cpu_batched_s"]])
    row["speedup_best"] = etia / best
    rows.append(row)
timing = pd.DataFrame(rows)
timing.to_csv(OUT / "timing_60.csv", index=False)

# long form for the per-query chart (ms per query, log scale)
long = []
for _, r in timing.iterrows():
    lab = f"{r.test} ({r.dataset})"
    for col, who in [("etia_s", "ETIA"), ("jax_cpu_s", "JAX CPU, one by one"),
                     ("jax_gpu_s", "JAX GPU, one by one"), ("jax_cpu_batched_s", "JAX CPU, batched"),
                     ("jax_gpu_batched_s", "JAX GPU, batched")]:
        if np.isfinite(r[col]):
            long.append({"test": lab, "who": who, "ms_per_query": 1000 * r[col] / 60})
pd.DataFrame(long).to_csv(OUT / "timing_60_long.csv", index=False)

# ---------------------------------------------------------------- scale
rows = []
for name in ("FisherZ", "gsquare", "ParCor", "Gsquared"):
    for k in range(4):
        n = len(sref[f"{name}|k{k}|p"])
        for who, t in [("ETIA", float(sref[f"{name}|k{k}|time"])),
                       ("JAX CPU, one by one", float(cpu[f"scale|{name}|k{k}|single"])),
                       ("JAX GPU, one by one", float(gpu[f"scale|{name}|k{k}|single"])),
                       ("JAX CPU, batched", float(cpu[f"scale|{name}|k{k}|batched"])),
                       ("JAX GPU, batched", float(gpu[f"scale|{name}|k{k}|batched"]))]:
            rows.append({"test": name, "k": k, "who": who, "seconds": t,
                         "queries_per_s": n / t, "series": f"{name} · {who}"})
        # the JAX and ETIA p-values on the same 2000 queries
        pj, pe = gpu[f"scale|{name}|k{k}|p"], sref[f"{name}|k{k}|p"]
        ok = ~np.isnan(pj) & ~np.isnan(pe)
        rows[-1]["max_abs_dp"] = float(np.max(np.abs(pj[ok] - pe[ok]), initial=0))
scale = pd.DataFrame(rows)
scale.to_csv(OUT / "scale.csv", index=False)
wide = scale.pivot_table(index=["test", "k"], columns="who", values="seconds").reset_index()
wide["speedup_batched_gpu"] = wide["ETIA"] / wide["JAX GPU, batched"]
wide["speedup_batched_cpu"] = wide["ETIA"] / wide["JAX CPU, batched"]
wide.columns = [c.replace(", ", "_").replace(" ", "_") for c in wide.columns]
wide.to_csv(OUT / "scale_wide.csv", index=False)

# ------------------------------------------------- calibration and power
CAL = [("FisherZ", ["continuous"] * 3, 300), ("ParCor", ["continuous"] * 3, 300),
       ("RobustParCor", ["continuous"] * 3, 300), ("ParCorrWLS", ["continuous"] * 3, 300),
       ("dg_lrt", ["continuous", "discrete", "discrete"], 400),
       ("cg_lrt", ["continuous", "continuous", "discrete"], 400),
       ("RegressionCI", ["discrete", "continuous", "continuous"], 400),
       ("chisquare", ["discrete"] * 3, 600), ("gsquare", ["discrete"] * 3, 600),
       ("Gsquared", ["discrete"] * 3, 600)]
REPS = 100
rows = []
for name, kinds, n in CAL:
    for mode, kw in ([("default", {})] + ([("tetrad_compat", {"tetrad_compat": True})]
                                          if name == "dg_lrt" else [])):
        r0 = r1 = 0
        for r in range(REPS):
            X, k = _fork_data(kinds, n, 1000 + r)
            p = CI_TESTS[name](X, k, **kw).test(0, 1, [2])[1]
            r0 += (not np.isnan(p)) and p <= ALPHA
            X, k = _fork_data(kinds, n, 5000 + r, link_xy=True)
            p = CI_TESTS[name](X, k, **kw).test(0, 1, [2])[1]
            r1 += np.isnan(p) or p <= ALPHA
        rows.append({"test": name, "mode": mode, "n": n, "size_pct": 100 * r0 / REPS,
                     "power_pct": 100 * r1 / REPS})
for mode, kw in [("default", {}), ("bonferroni", {"bonferroni": True}),
                 ("tetrad_compat", {"tetrad_compat": True})]:
    r0 = r1 = 0
    for r in range(60):
        X, k = _fork_data(["continuous"] * 3, 200, 1000 + r)
        r0 += CI_TESTS["cci"](X, k, **kw).test(0, 1, [2])[1] <= ALPHA
        X, k = _fork_data(["continuous"] * 3, 200, 5000 + r, link_xy=True)
        r1 += CI_TESTS["cci"](X, k, **kw).test(0, 1, [2])[1] <= ALPHA
    rows.append({"test": "cci", "mode": mode, "n": 200, "size_pct": 100 * r0 / 60, "power_pct": 100 * r1 / 60})
pd.DataFrame(rows).to_csv(OUT / "calibration.csv", index=False)

# ---------------------------------------------------------------- summary
summary = {
    "n_tests": len(CI_TESTS),
    "n_pairs": len(agree),
    "decisions_all_same": int((agree.decisions_same_pct == 100).sum()),
    "etia_total_s": float(timing.etia_s.sum()),
    "jax_gpu_total_s": float(timing.jax_gpu_s.sum()),
    "jax_cpu_total_s": float(timing.jax_cpu_s.sum()),
    "cmisymb_etia_s": float(timing[timing.test == "CMIsymb"].etia_s.iloc[0]),
    "cmisymb_gpu_s": float(timing[timing.test == "CMIsymb"].jax_gpu_s.iloc[0]),
    "scale_best_speedup_gpu": float(wide.speedup_batched_gpu.max()),
    "scale_best_speedup_cpu": float(wide.speedup_batched_cpu.max()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
