"""Data files of the increment-5 demo (scores, FGES, GFCI).

Run after benchmarks/fges_timing.py (GPU and CPU) and the fixtures:
    .venv/bin/python demo/increment5/make_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from etia_jax.ci import CI_TESTS
from etia_jax.scores import SCORES, ETIA_SETTINGS
from etia_jax.search.fges import fges
from etia_jax.search.gfci import gfci
from etia_jax.graphs.convert import dag_to_cpdag, dag_to_pag
from etia_jax.graphs.metrics import shd, adjacency_scores
from test_gfci import _agreement as gfci_agreement_rows, _cases as gfci_cases, RACY

OUT = Path(__file__).parent
FIX = ROOT / "reference" / "fixtures"
RES = ROOT / "benchmarks" / "results"
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}
grp = lambda case: case.rsplit("_s", 1)[0]

# ---------------------------------------------------------------- scores
sd = np.load(FIX / "search_data.npz")
sr = np.load(FIX / "scores.npz")
rows = []
for k in sorted(sr.files):
    if not k.endswith("|local"):
        continue
    case, name, _ = k.split("|")
    kinds = ["discrete" if b else "continuous" for b in sd[f"{case}|discrete"]]
    sc = SCORES[name](sd[f"{case}|X"], kinds, tetrad_compat=True, **ETIA_SETTINGS[name])
    Q = sr[f"{case}|queries"]
    mine = np.array([sc.local_score(int(q[0]), [int(v) for v in q[1:] if v >= 0]) for q in Q])
    ref = sr[k]
    ok = ~np.isnan(ref)
    rel = np.abs(mine[ok] - ref[ok]) / np.maximum(1.0, np.abs(ref[ok]))
    rows.append(dict(score=name, case=case, group=grp(case), queries=len(Q),
                     max_rel_error=float(rel.max()) if ok.any() else 0.0,
                     same_nan=bool((np.isnan(mine) == np.isnan(ref)).all()),
                     same_max_degree=sc.max_degree() == int(sr[f"{case}|{name}|max_degree"])))
scores = pd.DataFrame(rows)
sc_tab = (scores.groupby("score").agg(datasets=("case", "size"), queries=("queries", "sum"),
                                      max_rel_error=("max_rel_error", "max"), same_nan=("same_nan", "all"),
                                      same_max_degree=("same_max_degree", "all")).reset_index())
sc_tab["max_rel_error"] = sc_tab["max_rel_error"].map(lambda v: f"{v:.1e}")     # text: the table has no exponent format
sc_tab.to_csv(OUT / "scores.csv", index=False)

# ------------------------------------------------------------------ FGES
fr = np.load(FIX / "search_fges_ref.npz")
frows, quality = [], []
for k in sorted(fr.files):
    if not k.endswith("|graph"):
        continue
    case, name, _, _ = k.split("|")
    X = sd[f"{case}|X"]
    kinds = ["discrete" if b else "continuous" for b in sd[f"{case}|discrete"]]
    Gc = fges(SCORES[name](X, kinds, tetrad_compat=True, **ETIA_SETTINGS[name]), tetrad_compat=True).graph
    frows.append(dict(score=name, case=case, group=grp(case), same=bool((Gc == fr[k]).all()),
                      tetrad_same_each_run=bool(fr[k[:-6] + "|same_each_run"])))
    if X.shape[1] <= 50:
        C = dag_to_cpdag(sd[f"{case}|A"])
        Gd = fges(SCORES[name](X, kinds, **ETIA_SETTINGS[name])).graph
        for mode, M in (("Tetrad (same as ETIA)", Gc), ("default", Gd)):
            quality.append(dict(algo="fges", score=name, case=case, group=grp(case), mode=mode,
                                shd=shd(C, M), adj_f1=adjacency_scores(C, M)["f1"]))
fa = pd.DataFrame(frows)
(fa.groupby(["score", "group"]).agg(searches=("case", "size"), same_as_etia=("same", "sum"),
                                    tetrad_deterministic=("tetrad_same_each_run", "sum")).reset_index()
   .to_csv(OUT / "fges_agreement.csv", index=False))

# ------------------------------------------------------------------ GFCI
gd = np.load(FIX / "search_fci_data.npz")
graw = np.load(FIX / "search_gfci_ref.npz")
gmore = np.load(FIX / "search_gfci_ref_more.npz")
n_all, bad = gfci_agreement_rows(1000)
bad_keys = {(b[0], b[1], b[2], b[3]) for b in bad}
grows = []
for case, test, alpha, name, X, kinds, refs, r0 in gfci_cases(1000):
    base = f"{case}|{test}|{alpha}|{name}|gfci"
    grows.append(dict(score=name, test=test, case=case, group=grp(case), alpha=alpha,
                      tetrad_crashed=int(graw[base + "|n_errors"]) > 0 if base + "|n_errors" in graw.files else False,
                      race_free_outputs=len(gmore[base + "|graphs"]),
                      agrees=(case, test, alpha, name) not in bad_keys))
    if alpha == 0.05:
        P = dag_to_pag(gd[f"{case}|A"], gd[f"{case}|latent"])
        ci = CI_TESTS[test](X, kinds, **COMPAT.get(test, {}))
        for mode, compat in (("Tetrad (same as ETIA)", True), ("default", False)):
            M = gfci(ci, SCORES[name](X, kinds, tetrad_compat=compat, **ETIA_SETTINGS[name]), alpha=alpha,
                     tetrad_compat=compat).graph
            quality.append(dict(algo="gfci", score=name, case=case, group=grp(case), mode=mode,
                                shd=shd(P, M), adj_f1=adjacency_scores(P, M)["f1"]))
ga = pd.DataFrame(grows)
(ga.groupby(["test", "group"]).agg(searches=("case", "size"), tetrad_crashes=("tetrad_crashed", "sum"),
                                   race_free_deterministic=("race_free_outputs", lambda v: int((v == 1).sum())),
                                   agree=("agrees", "sum")).reset_index()
   .to_csv(OUT / "gfci_agreement.csv", index=False))

q = pd.DataFrame(quality)
q.to_csv(OUT / "quality.csv", index=False)
(q.groupby(["algo", "group", "mode"]).agg(shd_mean=("shd", "mean"), adj_f1_mean=("adj_f1", "mean"),
                                          runs=("shd", "size")).reset_index()
   .to_csv(OUT / "quality_summary.csv", index=False))

# ---------------------------------------------------------------- timing
gpu, cpu = pd.read_csv(RES / "fges_jax_gpu.csv"), pd.read_csv(RES / "fges_jax_cpu.csv")
tm = gpu.rename(columns={"jax_compat_s": "gpu_compat_s", "jax_default_s": "gpu_default_s"})
tm["cpu_compat_s"], tm["cpu_default_s"] = cpu["jax_compat_s"].values, cpu["jax_default_s"].values
tm["best_compat_s"] = tm[["gpu_compat_s", "cpu_compat_s"]].min(axis=1)
tm["best_default_s"] = tm[["gpu_default_s", "cpu_default_s"]].min(axis=1)
tm["speedup_compat"] = tm["etia_s"] / tm["best_compat_s"]
tm["speedup_default"] = tm["etia_s"] / tm["best_default_s"]
tm.to_csv(OUT / "timing.csv", index=False)
by = (tm.groupby(["algo", "group"])
        .agg(p=("p", "first"), etia_s=("etia_s", "median"), etia_prep_s=("etia_prep_s", "median"),
             gpu_compat_s=("gpu_compat_s", "median"), cpu_compat_s=("cpu_compat_s", "median"),
             gpu_default_s=("gpu_default_s", "median"), cpu_default_s=("cpu_default_s", "median"))
        .reset_index())
by.to_csv(OUT / "by_group.csv", index=False)
long = []
for _, r in by.iterrows():
    for col, who in [("etia_s", "ETIA (Tetrad)"), ("cpu_compat_s", "ETIA-JAX Tetrad mode, CPU"),
                     ("gpu_compat_s", "ETIA-JAX Tetrad mode, GPU"), ("cpu_default_s", "ETIA-JAX default, CPU"),
                     ("gpu_default_s", "ETIA-JAX default, GPU")]:
        if pd.notna(r[col]):
            long.append(dict(algo=r.algo, group=r.group, who=who, seconds=r[col]))
pd.DataFrame(long).to_csv(OUT / "by_group_long.csv", index=False)

summary = dict(
    score_queries=int(scores.queries.sum()), score_max_rel_error=float(scores.max_rel_error.max()),
    fges_searches=int(len(fa)), fges_same=int(fa.same.sum()),
    gfci_searches=int(len(ga)), gfci_agree=int(ga.agrees.sum()),
    gfci_tetrad_crashes=int(ga.tetrad_crashed.sum()),
    gfci_nondeterministic=int((ga.race_free_outputs > 1).sum()),
    median_speedup_fges_compat=float(tm[tm.algo == "fges"].speedup_compat.median()),
    median_speedup_gfci_compat=float(tm[tm.algo == "gfci"].speedup_compat.median()),
    median_speedup_fges_default=float(tm[tm.algo == "fges"].speedup_default.median()),
    median_speedup_gfci_default=float(tm[tm.algo == "gfci"].speedup_default.median()),
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
