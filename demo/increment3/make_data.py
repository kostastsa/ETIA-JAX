"""Data files of the increment-3 demo (PC and CPC).

Run after benchmarks/search_timing.py (GPU and CPU):
    .venv/bin/python demo/increment3/make_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from etia_jax.ci import CI_TESTS
from etia_jax.search import pc, cpc
from etia_jax.graphs.convert import dag_to_cpdag
from etia_jax.graphs.metrics import shd, adjacency_scores

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
RES = ROOT / "benchmarks" / "results"
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}

gpu = pd.read_csv(RES / "search_jax_gpu.csv")
cpu = pd.read_csv(RES / "search_jax_cpu.csv")
df = gpu.rename(columns={"jax_first_s": "gpu_first_s", "jax_warm_s": "gpu_warm_s",
                         "jax_stable_warm_s": "gpu_stable_warm_s"})
df["cpu_first_s"] = cpu["jax_first_s"].values
df["cpu_warm_s"] = cpu["jax_warm_s"].values
df["cpu_stable_warm_s"] = cpu["jax_stable_warm_s"].values
df["group"] = df["case"].str.replace(r"_s\d+$", "", regex=True)
df["etia_total_s"] = df["etia_search_s"] + df["etia_prep_s"]
df["best_warm_s"] = df[["gpu_warm_s", "cpu_warm_s"]].min(axis=1)
df["speedup_warm"] = df["etia_search_s"] / df["best_warm_s"]

# ------------------------------------------------------------- agreement
agree = (df.groupby(["group", "test", "algo"])
           .agg(searches=("same_as_tetrad", "size"), same_graph=("same_as_tetrad", "sum"),
                p=("p", "first"), n=("n", "first"))
           .reset_index())
agree["same_pct"] = 100 * agree["same_graph"] / agree["searches"]
agree.to_csv(OUT / "agreement.csv", index=False)

# --------------------------------------------------- run time by graph size
cont = df[df.test == "FisherZ"]
size = (cont.groupby(["p", "algo"])
            .agg(etia_search_s=("etia_search_s", "median"), etia_prep_s=("etia_prep_s", "median"),
                 gpu_warm_s=("gpu_warm_s", "median"), cpu_warm_s=("cpu_warm_s", "median"),
                 gpu_first_s=("gpu_first_s", "median"), cpu_first_s=("cpu_first_s", "median"),
                 gpu_stable_warm_s=("gpu_stable_warm_s", "median"),
                 n_tests=("n_tests", "median"), n_computed=("n_computed", "median"))
            .reset_index())
size.to_csv(OUT / "by_size.csv", index=False)
long = []
for _, r in size.iterrows():
    for col, who in [("etia_search_s", "ETIA (Tetrad search)"), ("gpu_warm_s", "JAX GPU, compiled"),
                     ("cpu_warm_s", "JAX CPU, compiled"), ("gpu_first_s", "JAX GPU, first call"),
                     ("cpu_first_s", "JAX CPU, first call")]:
        long.append({"p": int(r.p), "series": f"{r.algo.upper()} · {who}", "seconds": r[col]})
pd.DataFrame(long).to_csv(OUT / "by_size_long.csv", index=False)

cols = ["case", "test", "alpha", "algo", "p", "n", "etia_search_s", "etia_prep_s", "gpu_first_s", "gpu_warm_s",
        "cpu_first_s", "cpu_warm_s", "gpu_stable_warm_s", "speedup_warm", "n_tests", "n_computed", "same_as_tetrad"]
df[cols].to_csv(OUT / "timing.csv", index=False)

# ------------------------------------------- accuracy: Tetrad mode vs stable
d = np.load(ROOT / "reference" / "fixtures" / "search_data.npz")
rows = []
for case in d["cases"]:
    case = str(case)
    X, A = d[f"{case}|X"], d[f"{case}|A"]
    kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
    C = dag_to_cpdag(A)
    for t in d[f"{case}|tests"]:
        t = str(t)
        ci = CI_TESTS[t](X, kinds, **COMPAT.get(t, {}))
        for name, algo in (("pc", pc), ("cpc", cpc)):
            for mode, kw in (("Tetrad (same as ETIA)", {"tetrad_compat": True}), ("PC-stable (default)", {})):
                G = algo(ci, alpha=0.05, **kw).graph
                sc = adjacency_scores(C, G)
                rows.append({"case": case, "group": case.rsplit("_s", 1)[0], "test": t, "algo": name,
                             "mode": mode, "shd": shd(C, G), "adj_f1": sc["f1"]})
q = pd.DataFrame(rows)
q.to_csv(OUT / "quality.csv", index=False)
(q.groupby(["group", "algo", "mode"]).agg(shd_mean=("shd", "mean"), adj_f1_mean=("adj_f1", "mean"),
                                          runs=("shd", "size")).reset_index()
  .to_csv(OUT / "quality_summary.csv", index=False))

def speedup(p, algo):
    r = size[(size.p == p) & (size.algo == algo)].iloc[0]
    return float(r.etia_search_s / min(r.gpu_warm_s, r.cpu_warm_s))


summary = {
    "n_searches": int(len(df)), "same_graph": int(df.same_as_tetrad.sum()),
    "p_max": int(df.p.max()),
    "speedup_warm_p100_pc": speedup(100, "pc"),
    "speedup_warm_p100_cpc": speedup(100, "cpc"),
    "first_call_median_s": float(df.gpu_first_s.median()),
    "etia_prep_median_s": float(df.etia_prep_s.median()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
