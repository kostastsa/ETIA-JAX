"""Data files of the increment-7 demo (NOTEARS and SAM).

Run after benchmarks/opt_timing.py (GPU and CPU) and the test suite:
    .venv/bin/python demo/increment7/make_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from etia_jax.search import notears as NT          # noqa: E402

OUT = Path(__file__).parent
RES = ROOT / "benchmarks" / "results"
FIX = ROOT / "reference" / "fixtures"

# ------------------------------------------------ NOTEARS against causalnex
r = np.load(FIX / "notears_ref.npz")
d = np.load(FIX / "search_data.npz")
rows, pts = [], []
for case in sorted({k.split("|")[0] for k in r.files}):
    X = d[f"{case}|X"]
    Wc = r[f"{case}|W"]
    Wj, h, n_outer, n_inner = NT.notears_weights(X, NT.ETIA_MAX_ITER, NT.ETIA_H_TOL)
    F = lambda W: 0.5 / len(X) * np.sum((X - X @ W) ** 2)
    row = dict(case=case, p=X.shape[1], max_abs_diff=float(np.abs(Wj - Wc).max()),
               objective_rel_diff=float(abs(F(Wj) - F(Wc)) / F(Wc)))
    for thr in (0.0, 0.5, 0.8):
        key = f"{case}|{thr}"
        if key + "|time" not in r.files:
            row[f"graph_{thr}"] = ""
            continue
        near = int((np.abs(np.abs(Wc) - thr) < 0.05).sum()) if thr > 0 else 0
        if key + "|error" in r.files:
            row[f"graph_{thr}"] = "both: cycle error"
            continue
        G = NT.notears(X, threshold=thr, etia_compat=True).G
        same = bool((G == r[key + "|etia_matrix"]).all())
        why = "another local optimum" if row["objective_rel_diff"] > 1e-3 else f"{near} weights near {thr}"
        row[f"graph_{thr}"] = "same" if same else f"differs ({why})"
    rows.append(row)
    off = ~np.eye(len(Wc), dtype=bool)
    pts += [dict(case=case, causalnex=float(a), etia_jax=float(b)) for a, b in zip(Wc[off], Wj[off])]
pd.DataFrame(rows).to_csv(OUT / "notears_agreement.csv", index=False)
pd.DataFrame(pts).to_csv(OUT / "notears_weights.csv", index=False)

# ----------------------------------------------------------- timing, accuracy
gpu, cpu = pd.read_csv(RES / "opt_jax_gpu.csv"), pd.read_csv(RES / "opt_jax_cpu.csv")
allrows = pd.concat([gpu, cpu[cpu.algo == "sam"]], ignore_index=True)
allrows.to_csv(OUT / "timing.csv", index=False)
nt = allrows[(allrows.algo == "notears")]
sm = allrows[(allrows.algo == "sam")]

long = []
for case, g in nt[nt["mode"] == "etia"].groupby("case", sort=False):
    long.append(dict(algo="NOTEARS", case=case, who="ETIA (causalnex)", seconds=g.etia_time.iloc[0]))
    for dev in ("cpu", "gpu"):
        s = g[g.device == dev]
        if len(s):
            long.append(dict(algo="NOTEARS", case=case, who=f"ETIA-JAX, {dev.upper()}", seconds=s.jax_time.iloc[0]))
for (case, setting), g in sm[sm["mode"] == "etia"].groupby(["case", "setting"], sort=False):
    lab = case if setting.startswith("train") else case + " (defaults)"
    long.append(dict(algo="SAM", case=lab, who="ETIA (cdt, GPU)", seconds=g.etia_time.iloc[0]))
    for dev in ("cpu", "gpu"):
        s = g[g.device == dev]
        if len(s):
            long.append(dict(algo="SAM", case=lab, who=f"ETIA-JAX, {dev.upper()}", seconds=s.jax_time.iloc[0]))
pd.DataFrame(long).to_csv(OUT / "time_long.csv", index=False)

acc = sm[sm.device == "gpu"][["case", "setting", "mode", "auc_etia", "auc_jax", "shd_etia", "shd_jax"]]
acc.to_csv(OUT / "sam_accuracy.csv", index=False)
nacc = nt[nt.device == "cpu"][["case", "p", "mode", "shd_etia", "shd_jax", "same_graph"]]
nacc.to_csv(OUT / "notears_accuracy.csv", index=False)

# SAM: ETIA-JAX over 10 seeds against the one ETIA run of each dataset
from test_optim import _auc                          # noqa: E402
from etia_jax.search.sam import sam                  # noqa: E402
rs, ds = np.load(FIX / "sam_ref.npz"), np.load(FIX / "sam_data.npz")
seed_rows = []
for c in [str(c) for c in ds["cases"]]:
    X, A = ds[f"{c}|X"], ds[f"{c}|A"]
    a = [_auc(sam(X, train_epochs=1000, test_epochs=500, seed=s).W, A) for s in range(10)]
    e = _auc(rs[f"{c}|short|W"], A)
    seed_rows.append(dict(case=c, auc_etia=e, jax_mean=float(np.mean(a)), jax_min=float(min(a)),
                          jax_max=float(max(a)), etia_in_jax_range=bool(min(a) <= e <= max(a))))
sseeds = pd.DataFrame(seed_rows)
sseeds.to_csv(OUT / "sam_seeds.csv", index=False)

tests = pd.read_csv(OUT / "tests.csv") if (OUT / "tests.csv").exists() else None
nag = pd.DataFrame(rows)
summary = dict(
    notears_cases=int(len(nag)),
    notears_graph_same=int(sum((nag[c] == "same").sum() for c in nag.columns if c.startswith("graph_"))),
    notears_graph_checked=int(sum((nag[c] != "").sum() for c in nag.columns if c.startswith("graph_"))),
    notears_median_speedup_cpu=float((nt[(nt["mode"] == "etia") & (nt.device == "cpu")].speedup).median()),
    sam_median_speedup_gpu=float(sm[(sm["mode"] == "etia") & (sm.device == "gpu")].speedup.median()),
    sam_auc_etia=float(sseeds.auc_etia.mean()),
    sam_auc_jax=float(sseeds.jax_mean.mean()),
    sam_etia_in_range=int(sseeds.etia_in_jax_range.sum()),
    sam_cases=int(len(sseeds)),
    sam_full_etia_s=float(sm[(sm.setting == "ETIA defaults")].etia_time.max()) if (sm.setting == "ETIA defaults").any() else None,
    sam_full_jax_s=float(sm[(sm.setting == "ETIA defaults") & (sm.device == "gpu") & (sm["mode"] == "etia")].jax_time.min())
    if (sm.setting == "ETIA defaults").any() else None,
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
