"""Data files of the increment-8 demo (FBED, SES, the AFS).

Run after benchmarks/afs_timing.py (GPU) and the test suite:
    .venv/bin/python demo/increment8/make_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from etia_jax.afs.afs import run_afs, predictive_configs     # noqa: E402

OUT = Path(__file__).parent
RES = ROOT / "benchmarks" / "results"
FIX = ROOT / "reference" / "fixtures"

tm = pd.read_csv(RES / "afs_jax_gpu.csv")
tm.to_csv(OUT / "timing.csv", index=False)

# ------------------------------------------------- FBED and SES against MXM
fs = tm[tm.kind.isin(["fbed", "ses"])].copy()
fs["group"] = fs.key.str.split("|").str[0].str.rsplit("_s", n=1).str[0]
r = np.load(FIX / "fs_ref.npz")
fs["r_error"] = [k + "|error" in r.files for k in fs.key]
agree = (fs.groupby(["kind", "group"]).agg(calls=("key", "size"), same=("same", "sum"),
                                           r_errors=("r_error", "sum"),
                                           etia_s=("etia_time", "median"), jax_s=("jax_time", "median"))
         .reset_index())
agree["speedup"] = agree.etia_s / agree.jax_s
agree.to_csv(OUT / "fs_agreement.csv", index=False)

# ------------------------------------------------------ whole AFS runs
a = np.load(FIX / "afs_ref.npz")
d = np.load(FIX / "search_data.npz")
rows = []
for key in sorted(k.rsplit("|", 1)[0] for k in a.files if k.endswith("|result")):
    case, t, tag = key.split("|")
    ref = json.loads(str(a[key + "|result"]))
    X = d[f"{case}|X"]
    cols = [f"X{i}" for i in range(X.shape[1])]
    configs = predictive_configs()
    if tag == "linear":
        configs = [c for c in configs if c["model"] == "linear_regression"]
    df = pd.DataFrame(X, columns=cols)
    c = run_afs(df, [cols[int(t)]], configs, etia_compat=True)
    dflt = run_afs(df, [cols[int(t)]], configs)
    fmt = lambda cfg: "none" if cfg is None else f"{cfg['model']} {cfg['fs_name']} a={cfg['alpha']} k={cfg['k']}"
    rows.append(dict(case=case, target=cols[int(t)], grid=tag,
                     etia_best=fmt(ref["best_config"]), jax_best=fmt(c["best_config"]),
                     same_best=c["best_config"] == ref["best_config"],
                     etia_selected=" ".join(ref["selected"]),
                     jax_selected=" ".join(sorted(c["selected_features"][cols[int(t)]])),
                     same_selected=sorted(c["selected_features"][cols[int(t)]]) == ref["selected"],
                     etia_bootstrap=ref["bbc_score"], jax_bootstrap=c["bbc_score"],
                     default_bbc_cv=dflt["bbc_score"],
                     default_selected=" ".join(sorted(dflt["selected_features"][cols[int(t)]]))))
afs = pd.DataFrame(rows)
afs.to_csv(OUT / "afs_runs.csv", index=False)

# ------------------------------------------------------ time chart (long)
long = []
for kind, g in tm.groupby("kind", sort=False):
    long.append(dict(kind=kind, who="ETIA", seconds=g.etia_time.median()))
    long.append(dict(kind=kind, who="ETIA-JAX", seconds=g.jax_time.median()))
pd.DataFrame(long).to_csv(OUT / "time_long.csv", index=False)

summary = dict(
    fs_calls=int(len(fs)), fs_same=int(fs.same.sum()), r_errors=int(fs.r_error.sum()),
    fs_median_speedup=float(fs.speedup.median()),
    afs_runs=int(len(afs)), afs_same_best=int(afs.same_best.sum()), afs_same_selected=int(afs.same_selected.sum()),
    afs_linear_runs=int((afs.grid == "linear").sum()),
    afs_linear_same=int(((afs.grid == "linear") & afs.same_best & afs.same_selected).sum()),
    afs_median_speedup=float(tm[tm.kind.str.startswith("afs")].speedup.median()),
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
