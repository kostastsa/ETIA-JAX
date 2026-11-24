"""Data files of the increment-9 demo (the OCT and the CausalLearner).

Run after benchmarks/oct_timing.py (GPU) and the test suite:
    .venv/bin/python demo/increment9/make_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from etia_jax.oct.configs import var_types                                   # noqa: E402
from etia_jax.oct.oct import (kfold_indexes, oct_predictions, oct_scores, oct_select,  # noqa: E402
                              run_oct, DEFAULT_REGRESSOR)
from etia_jax.oct.algorithms import search                                    # noqa: E402
from etia_jax.graphs.markov import etia_markov_boundary                       # noqa: E402

OUT = Path(__file__).parent
FIX = ROOT / "reference" / "fixtures"
RES = ROOT / "benchmarks" / "results"
ref = np.load(FIX / "oct_ref.npz")
d = np.load(FIX / "search_data.npz")
CASES = sorted({k.split("|")[0] for k in ref.files if k.endswith("|time")}, key=lambda c: (c[:4], len(c), c))


def mbs_of(key):
    mbl = json.loads(str(ref[key + "|mbs"]))
    C, p, F = len(mbl), len(mbl[0]), len(mbl[0][0])
    return {(c, t, k): mbl[c][t][k] for c in range(C) for t in range(p) for k in range(F)}, C, p, F


# ------------------------------------------------ step-by-step agreement with ETIA
steps = {s: [0, 0] for s in ("fold PAGs", "Markov boundaries", "predictions", "scores", "p-values", "choice")}
for case in CASES:
    X, kinds, _ = var_types(d[f"{case}|X"][:1000])
    train, test = kfold_indexes(1000, 10)
    for tag in ("lin1", "lin8", "rf1", "rf8"):
        key = f"{case}|{tag}"
        if key + "|time" not in ref.files:
            continue
        mbs, C, p, F = mbs_of(key)
        G = ref[key + "|graph"]
        for (c, t, k), mb in mbs.items():
            steps["Markov boundaries"][0] += sorted(etia_markov_boundary(t, G[c, k])) == mb
            steps["Markov boundaries"][1] += 1
        mu = oct_scores(ref[key + "|pred"], ref[key + "|y"], kinds, True)
        same = np.isclose(mu, ref[key + "|mu"], rtol=1e-10, atol=1e-12, equal_nan=True)
        steps["scores"][0] += int(same.sum())
        steps["scores"][1] += same.size
        s = oct_select(ref[key + "|mu"], ref[key + "|pred"], ref[key + "|y"], mbs, kinds, list(ref[key + "|idxs"]),
                       0.01, F, True, lost=tag.endswith("8"))
        steps["p-values"][0] += int((s["p_values"] == ref[key + "|p_values"]).sum())
        steps["p-values"][1] += C
        steps["choice"][0] += s["chosen"] == int(ref[key + "|chosen"])
        steps["choice"][1] += 1
        if tag == "lin1":
            cfgs = json.loads(str(ref[key + "|configs"]))
            mec = ref[key + "|mec"]
            for c, params in enumerate(cfgs):
                for k in range(F):
                    steps["fold PAGs"][0] += np.array_equal(search(params, X[train[k]], kinds, etia_compat=True),
                                                            mec[c, k])
                    steps["fold PAGs"][1] += 1
            P, _ = oct_predictions(X, mbs, C, kinds, train, test, F, ("LinearRegression", {}), True)
            want = ref[key + "|pred"]
            off = np.cumsum([0] + [len(test[k]) for k in range(F)])
            for (c, t, k), mb in mbs.items():
                if mb:
                    steps["predictions"][0] += np.allclose(P[c, t, off[k]:off[k + 1]], want[c, t, off[k]:off[k + 1]],
                                                           atol=1e-8)
                    steps["predictions"][1] += 1
what = {"fold PAGs": "the PAG of each configuration on the training rows of each fold (FCI and FCI-Max ports)",
        "Markov boundaries": "ETIA's markov_boundary on ETIA's random MAGs",
        "predictions": "LinearRegression on ETIA's Markov boundaries (non-empty), pooled test rows",
        "scores": "mutual information of ETIA's predictions, per configuration and node",
        "p-values": "permutation test from ETIA's predictions and permutations",
        "choice": "the chosen configuration (n_jobs=1: OCTs; n_jobs=8: best score)"}
agree = pd.DataFrame([dict(step=k, compared=what[k], same=v[0], total=v[1]) for k, v in steps.items()])
agree.to_csv(OUT / "step_agreement.csv", index=False)

# ------------------------------------------------ run-to-run spread of the scores
rows = []
for case in ("cont10_s0", "cont20_s0"):
    X, kinds, _ = var_types(d[f"{case}|X"][:1000])
    cfgs = json.loads(str(ref[f"{case}|lin1|configs"]))
    label = [f"{c['name']} {c['significance_level']}" for c in cfgs]
    for tag in ("lin1", "lin8"):
        for c, v in enumerate(np.nanmean(ref[f"{case}|{tag}|mu"], 1)):
            rows.append(dict(case=case, config=label[c], who=f"ETIA run {tag[-1]}", score=v, low=v, high=v))
    reps = np.array([np.nanmean(run_oct(X, cfgs, kinds, regressor=("LinearRegression", {}), etia_compat=True,
                                        seed=s)["mu"], 1) for s in range(10)])
    for c in range(len(cfgs)):
        rows.append(dict(case=case, config=label[c], who="ETIA-JAX etia_compat, 10 seeds", score=reps[:, c].mean(),
                         low=reps[:, c].min(), high=reps[:, c].max()))
    dflt = run_oct(X, cfgs, kinds, regressor=("LinearRegression", {}))
    for c, v in enumerate(dflt["mean_mu"]):
        rows.append(dict(case=case, config=label[c], who="ETIA-JAX default", score=v, low=v, high=v))
spread = pd.DataFrame(rows)
spread["err"] = (spread.high - spread.low) / 2
spread.to_csv(OUT / "score_spread.csv", index=False)

# ------------------------------------------------ timing
tm = pd.read_csv(RES / "oct_jax_gpu.csv")
cpu = RES / "oct_jax_cpu.csv"
if cpu.exists():
    c = pd.read_csv(cpu)[["case", "tag", "jax_time"]].rename(columns={"jax_time": "jax_cpu_time"})
    tm = tm.merge(c, on=["case", "tag"], how="left")
tm.to_csv(OUT / "timing.csv", index=False)
long = []
for tag, g in tm.groupby("tag", sort=False):
    long.append(dict(run=tag, who="ETIA", seconds=g.etia_time.median()))
    long.append(dict(run=tag, who="ETIA-JAX GPU", seconds=g.jax_time.median()))
    if "jax_cpu_time" in g:
        long.append(dict(run=tag, who="ETIA-JAX CPU", seconds=g.jax_cpu_time.median()))
pd.DataFrame(long).to_csv(OUT / "time_long.csv", index=False)

# ------------------------------------------------ AFS with the forest
afs = []
a = np.load(FIX / "afs_ref.npz")
from etia_jax.afs.afs import run_afs, predictive_configs                     # noqa: E402
import time                                                                  # noqa: E402
for key in sorted(k.rsplit("|", 1)[0] for k in a.files if k.endswith("|result") and k.endswith("full|result")):
    case, t, _ = key.split("|")
    X = d[f"{case}|X"]
    cols = [f"X{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    run_afs(df, [cols[int(t)]], predictive_configs(), etia_compat=True)
    t0 = time.perf_counter()
    r = run_afs(df, [cols[int(t)]], predictive_configs(), etia_compat=True)
    dt = time.perf_counter() - t0
    want = json.loads(str(a[key + "|result"]))
    fmt = lambda g: f"{g['model']} {g['fs_name']} a={g['alpha']} k={g['k']}" + (
        f" trees={g['n_estimators']}" if g["model"] == "random_forest" else "")
    afs.append(dict(case=case, target=cols[int(t)], etia_best=fmt(want["best_config"]), jax_best=fmt(r["best_config"]),
                    same_best=r["best_config"] == want["best_config"],
                    same_selected=sorted(r["selected_features"][cols[int(t)]]) == want["selected"],
                    etia_s=float(a[key + "|time"]), jax_s=dt))
pd.DataFrame(afs).to_csv(OUT / "afs_forest.csv", index=False)

summary = dict(
    pags_same=steps["fold PAGs"][0], pags_total=steps["fold PAGs"][1],
    steps_same=int(sum(v[0] for k, v in steps.items() if k != "fold PAGs")),
    steps_total=int(sum(v[1] for k, v in steps.items() if k != "fold PAGs")),
    choice_same=steps["choice"][0], choice_total=steps["choice"][1],
    median_speedup=float(tm.speedup.median()),
    afs_same=int(sum(r["same_best"] and r["same_selected"] for r in afs)), afs_runs=len(afs),
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
