"""Data files of the increment-4 demo (FCI, FCI-Max, RFCI, CFCI).

Run after benchmarks/fci_timing.py (GPU and CPU):
    .venv/bin/python demo/increment4/make_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from etia_jax.ci import CI_TESTS
from etia_jax.search.fci import fci, fcimax, rfci, cfci
from etia_jax.graphs.convert import dag_to_pag
from etia_jax.graphs.metrics import shd, adjacency_scores

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
FIX = ROOT / "reference" / "fixtures"
RES = ROOT / "benchmarks" / "results"
ALGOS = {"fci": fci, "fcimax": fcimax, "rfci": rfci, "cfci": cfci}
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}

d = np.load(FIX / "search_fci_data.npz")
ref = np.load(FIX / "search_fci_ref.npz")
ser = np.load(FIX / "search_fci_ref_serial.npz")

# ------------------------------------------------------------- agreement
rows, quality = [], []
for case in d["cases"]:
    case = str(case)
    X, A, L = d[f"{case}|X"], d[f"{case}|A"], d[f"{case}|latent"]
    kinds = ["discrete" if b else "continuous" for b in d[f"{case}|discrete"]]
    P = dag_to_pag(A, L)
    for t in d[f"{case}|tests"]:
        t = str(t)
        ci = CI_TESTS[t](X, kinds, **COMPAT.get(t, {}))
        for a in d["alphas"]:
            a = float(a)
            for name, algo in ALGOS.items():
                key = f"{case}|{t}|{a}|{name}"
                G = algo(ci, alpha=a, tetrad_compat=True).graph
                crashed = key + "|error" in ref.files
                same_etia = (not crashed) and bool((G == ref[key + "|graph"]).all())
                if key + "|graphs" in ser.files:
                    outs = ser[key + "|graphs"]
                    serial_n = len(outs)
                    same_serial = any((G == h).all() for h in outs)
                else:
                    serial_n, same_serial = 1, same_etia
                rows.append(dict(case=case, group=case.rsplit("_s", 1)[0], test=t, alpha=a, algo=name,
                                 tetrad_crashed=crashed, same_as_etia=same_etia,
                                 tetrad_outputs_race_free=serial_n, among_race_free=same_serial))
                if a == 0.05:
                    Gd = algo(ci, alpha=a).graph
                    for mode, M in (("Tetrad (same as ETIA)", G), ("default (textbook)", Gd)):
                        quality.append(dict(case=case, group=case.rsplit("_s", 1)[0], test=t, algo=name, mode=mode,
                                            shd=shd(P, M), adj_f1=adjacency_scores(P, M)["f1"]))
ag = pd.DataFrame(rows)
summ = (ag.groupby(["algo", "group"])
          .agg(searches=("case", "size"), tetrad_crashes=("tetrad_crashed", "sum"),
               same_as_etia=("same_as_etia", "sum"),
               race_free_deterministic=("tetrad_outputs_race_free", lambda v: int((v == 1).sum())),
               among_race_free=("among_race_free", "sum"))
          .reset_index())
summ.to_csv(OUT / "agreement.csv", index=False)
q = pd.DataFrame(quality)
q.to_csv(OUT / "quality.csv", index=False)
(q.groupby(["algo", "group", "mode"]).agg(shd_mean=("shd", "mean"), adj_f1_mean=("adj_f1", "mean"),
                                          runs=("shd", "size")).reset_index()
  .to_csv(OUT / "quality_summary.csv", index=False))

# ---------------------------------------------------------------- timing
gpu, cpu = pd.read_csv(RES / "fci_jax_gpu.csv"), pd.read_csv(RES / "fci_jax_cpu.csv")
tm = gpu.rename(columns={"jax_compat_s": "gpu_compat_s", "jax_default_s": "gpu_default_s"})
tm["cpu_compat_s"], tm["cpu_default_s"] = cpu["jax_compat_s"].values, cpu["jax_default_s"].values
tm["best_compat_s"] = tm[["gpu_compat_s", "cpu_compat_s"]].min(axis=1)
tm["best_default_s"] = tm[["gpu_default_s", "cpu_default_s"]].min(axis=1)
tm["speedup_compat"] = tm["etia_s"] / tm["best_compat_s"]
tm["speedup_default"] = tm["etia_s"] / tm["best_default_s"]
tm.drop(columns=["n_tests_default"]).to_csv(OUT / "timing.csv", index=False)
grp = (tm.groupby(["algo", "group"])
         .agg(p=("p", "first"), etia_s=("etia_s", "median"), etia_prep_s=("etia_prep_s", "median"),
              gpu_compat_s=("gpu_compat_s", "median"), cpu_compat_s=("cpu_compat_s", "median"),
              gpu_default_s=("gpu_default_s", "median"), cpu_default_s=("cpu_default_s", "median"))
         .reset_index())
grp.to_csv(OUT / "by_group.csv", index=False)
long = []
for _, r in grp.iterrows():
    for col, who in [("etia_s", "ETIA (Tetrad)"), ("cpu_compat_s", "ETIA-JAX Tetrad mode, CPU"),
                     ("gpu_compat_s", "ETIA-JAX Tetrad mode, GPU"), ("cpu_default_s", "ETIA-JAX default, CPU"),
                     ("gpu_default_s", "ETIA-JAX default, GPU")]:
        if pd.notna(r[col]):
            long.append(dict(algo=r.algo, group=r.group, who=who, seconds=r[col]))
pd.DataFrame(long).to_csv(OUT / "by_group_long.csv", index=False)

ok = ag[~ag.tetrad_crashed & (ag.tetrad_outputs_race_free == 1)]
summary = dict(
    searches=int(len(ag)), tetrad_crashes=int(ag.tetrad_crashed.sum()),
    deterministic=int(len(ok)), deterministic_same=int(ok.same_as_etia.sum() + 0),
    race_free_cases=int(len(ser.files)), race_free_among=int(ag[ag.group.str.startswith("disc")].among_race_free.sum()),
    nondeterministic=int((ag.tetrad_outputs_race_free > 1).sum()),
    nondeterministic_among=int(ag[ag.tetrad_outputs_race_free > 1].among_race_free.sum()),
    all_among=int(ag.among_race_free.sum()),
    median_speedup_default=float(tm.speedup_default.median()),
    median_speedup_compat=float(tm.speedup_compat.median()),
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
