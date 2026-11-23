"""Data files of the increment-6 demo (time series).

Run after benchmarks/ts_timing.py (GPU and CPU):
    .venv/bin/python demo/increment6/make_data.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_timeseries import _ts_cases, _run, _tig_agreement, TEST_ALGOS   # noqa: E402

OUT = Path(__file__).parent
RES = ROOT / "benchmarks" / "results"
grp = lambda case: case.rsplit("_s", 1)[0]

# ------------------------------------------------- Tetrad algorithms (tiers)
rows = []
for case, t, a, sc, algo, X, kinds, kn, refs in _ts_cases(tuple(TEST_ALGOS) + ("fges", "gfci", "svargfci")):
    ok, seed_used = False, None
    for seed in [None] + list(range(20)):
        G = _run(algo, X, kinds, t, a, sc, kn, seed)
        if any((G == h).all() for h in refs):
            ok, seed_used = True, seed
            break
        if algo != "svargfci":
            break
    rows.append(dict(algo=algo, case=case, group=grp(case), agrees=ok, tie_seed=seed_used is not None,
                     tetrad_outputs=len(refs)))
ta = pd.DataFrame(rows)
(ta.groupby(["algo"]).agg(searches=("case", "size"), agree=("agrees", "sum"),
                          nondeterministic_in_tetrad=("tetrad_outputs", lambda v: int((v > 1).sum())),
                          rounding_ties=("tie_seed", "sum")).reset_index()
   .to_csv(OUT / "tetrad_agreement.csv", index=False))

# ------------------------------------------------------ tigramite algorithms
trows = []
for test in ("ParCor", "RobustParCor", "ParCorrWLS", "CMIknn", "GPDC", "Gsquared", "CMIsymb", "RegressionCI"):
    for algo in ("PCMCI", "PCMCI+", "LPCMCI"):
        n, bad = _tig_agreement((algo,), (test,))
        if n:
            trows.append(dict(algo=algo, test=test, searches=n, same_graph=n - len(bad),
                              random_p_values=test in ("GPDC", "CMIsymb")))
tg = pd.DataFrame(trows)
tg.to_csv(OUT / "tigramite_agreement.csv", index=False)

# ------------------------------------------------- default mode: the oracle
from test_timeseries import _oracle_checks                                    # noqa: E402
bad = _oracle_checks(range(40))
names = ["fci", "fcimax", "cfci", "rfci", "svarfci", "gfci", "svargfci", "pc", "cpc", "fges"]
nbad = collections.Counter(b[1] for b in bad)
pd.DataFrame([dict(algo=n, oracle_cases=40, correct=40 - nbad[n]) for n in names]).to_csv(OUT / "oracle.csv", index=False)

# ---------------------------------------------------------------- timing
gpu, cpu = pd.read_csv(RES / "ts_jax_gpu.csv"), pd.read_csv(RES / "ts_jax_cpu.csv")
tm = gpu.rename(columns={"jax_compat_s": "gpu_compat_s", "jax_default_s": "gpu_default_s"})
tm["cpu_compat_s"], tm["cpu_default_s"] = cpu["jax_compat_s"].values, cpu["jax_default_s"].values
tm["best_compat_s"] = tm[["gpu_compat_s", "cpu_compat_s"]].min(axis=1)
tm["speedup_compat"] = tm["etia_s"] / tm["best_compat_s"]
tm["speedup_default"] = tm["etia_s"] / tm[["gpu_default_s", "cpu_default_s"]].min(axis=1)
tm.to_csv(OUT / "timing.csv", index=False)
by = (tm.groupby(["family", "algo"]).agg(searches=("case", "size"), etia_s=("etia_s", "median"),
                                         gpu_compat_s=("gpu_compat_s", "median"), cpu_compat_s=("cpu_compat_s", "median"),
                                         gpu_default_s=("gpu_default_s", "median"), cpu_default_s=("cpu_default_s", "median"),
                                         speedup=("speedup_compat", "median")).reset_index())
by.to_csv(OUT / "by_algo.csv", index=False)
long = []
for _, r in by.iterrows():
    for col, who in [("etia_s", "ETIA"), ("cpu_compat_s", "ETIA-JAX same as ETIA, CPU"),
                     ("gpu_compat_s", "ETIA-JAX same as ETIA, GPU")]:
        long.append(dict(family=r.family, algo=r.algo, who=who, seconds=r[col]))
    if r.family == "tetrad":
        for col, who in [("cpu_default_s", "ETIA-JAX default, CPU"), ("gpu_default_s", "ETIA-JAX default, GPU")]:
            long.append(dict(family=r.family, algo=r.algo, who=who, seconds=r[col]))
pd.DataFrame(long).to_csv(OUT / "by_algo_long.csv", index=False)

summary = dict(
    tetrad_searches=int(len(ta)), tetrad_agree=int(ta.agrees.sum()), rounding_ties=int(ta.tie_seed.sum()),
    tetrad_nondeterministic=int((ta.tetrad_outputs > 1).sum()),
    tig_searches=int(tg.searches.sum()), tig_same=int(tg.same_graph.sum()),
    tig_exact_tests_same=int(tg[~tg.random_p_values].same_graph.sum()),
    tig_exact_tests_searches=int(tg[~tg.random_p_values].searches.sum()),
    oracle_cases=40 * len(names), oracle_correct=int(40 * len(names) - len(bad)),
    median_speedup_tetrad=float(tm[tm.family == "tetrad"].speedup_compat.median()),
    median_speedup_tigramite=float(tm[tm.family == "tigramite"].speedup_compat.median()),
    max_speedup_tigramite=float(tm[tm.family == "tigramite"].speedup_compat.max()),
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
