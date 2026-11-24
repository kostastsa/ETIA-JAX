"""Data files of the increment-10 demo (the CRV).

Run after benchmarks/crv_timing.py (GPU and CPU) and the test suite:
    .venv/bin/python demo/increment10/make_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from etia_jax.crv import queries as Q                                        # noqa: E402
from etia_jax.crv import dagitty as D                                        # noqa: E402
from etia_jax.crv.adjustment import adjustment_sets, find_adjset              # noqa: E402
from etia_jax.crv.confidence import compute_edge_weights, bootstrap_graphs    # noqa: E402
from etia_jax.crv.visualization import matrix_to_edges, Visualization        # noqa: E402
from etia_jax.graphs.convert import dag_to_cpdag, dag_to_mag, mag_to_pag     # noqa: E402
from etia_jax.graphs.dag import random_dag                                   # noqa: E402
from test_crv import _sem, _coef                                             # noqa: E402

OUT = Path(__file__).parent
FIX = ROOT / "reference" / "fixtures"
RES = ROOT / "benchmarks" / "results"
cases = json.loads((FIX / "adjset_ref.json").read_text())
r = np.load(FIX / "crv_ref.npz")
norm = lambda x: None if x is None else ([x] if isinstance(x, str) else sorted(x))

# ------------------------------------------------ agreement with ETIA
rows = []
ok = 0
for c in cases:
    g = D.from_etia(np.array(c["M"]), c["names"], c["type"], c["exposures"], c["outcomes"])
    can = D._sorted_ids(D.canonical_adjustment_set(g))
    mini = D._sorted_ids(D.list_msas_total_effect(g))
    ok += ((can[0] if can else None) == norm(c["dagitty"]["canonical"]) and
           (mini[0] if mini else None) == norm(c["dagitty"]["minimal"]) and len(mini) == c["dagitty"]["n_minimal"])
rows.append(dict(part="adjustment sets", reference="dagitty 0.3.4 (R), ETIA's conversion", same=ok, total=len(cases)))
qs = json.loads(str(r["queries"]))
fn = dict(directed=Q.one_directed_path, potentially_directed=Q.one_potentially_directed_path,
          any=Q.one_path_anytype, bidirected=Q.one_bidirected_path_from_to)
ok = sum(f(np.array(cases[q["case"]]["M"]), q["start"], q["end"], etia_compat=True) == q[k]
         for q in qs for k, f in fn.items())
rows.append(dict(part="path queries", reference="ETIA CRV/queries", same=ok, total=4 * len(qs)))
ok = sum(matrix_to_edges(np.array(c["M"]), c["names"]).values.tolist() == json.loads(str(r[f"cyto|{k}"]))
         for k, c in enumerate(cases))
rows.append(dict(part="edges for drawing", reference="ETIA matrix_to_cyto", same=ok, total=len(cases)))
d = np.load(FIX / "search_data.npz")
bs = ws = 0
boots_all = {}
for case in ("cont10_s0", "cont10_s1", "cont20_s0"):
    X = d[f"{case}|X"][:1000]
    names = [f"X{i}" for i in range(X.shape[1])]
    cfg = {"name": "fci", "ci_test": "FisherZ", "significance_level": 0.05}
    boots = bootstrap_graphs(X, cfg, rows=list(r[f"{case}|rows"]), etia_compat=True)
    boots_all[case] = boots
    bs += sum(np.array_equal(a, b) for a, b in zip(boots, r[f"{case}|boots"]))
    for tag, ae in (("w_all", True), ("w_edges", False)):
        w, _ = compute_edge_weights(r[f"{case}|best"], boots, ae, names=names, etia_compat=True)
        ws += w.values.tolist() == json.loads(str(r[f"{case}|{tag}"]))
rows.append(dict(part="bootstrap graphs", reference="Tetrad FCI in ETIA on the same rows", same=bs, total=150))
rows.append(dict(part="edge weight tables", reference="ETIA compute_edge_weights", same=ws, total=6))
pd.DataFrame(rows).to_csv(OUT / "agreement.csv", index=False)

# ------------------------------------------------ adjustment oracle (both modes)
orc = []
for gtype, seed in (("dag", 0), ("cpdag", 1), ("mag", 2), ("pag", 3)):
    rng = np.random.default_rng(seed)
    st = dict(graph=gtype, graphs=0, default_sets=0, default_correct=0, etia_sets=0, etia_correct=0)
    for _ in range(60):
        p = 8
        A = random_dag(p, float(rng.uniform(1.5, 3.0)), rng)
        S, B = _sem(A, rng)
        lat = np.zeros(p, bool)
        if gtype in ("mag", "pag"):
            lat[rng.choice(p, 2, replace=False)] = True
        obs = np.flatnonzero(~lat)
        M = (np.where(A != 0, 2, 0) + np.where(A.T != 0, 3, 0) if gtype == "dag" else dag_to_cpdag(A)
             if gtype == "cpdag" else dag_to_mag(A, lat))
        if gtype == "pag":
            M = mag_to_pag(M)
        names = [f"V{int(i)}" for i in obs]
        x, y = rng.choice(len(obs), 2, replace=False)
        st["graphs"] += 1
        for mode in ("default", "etia"):
            can, mini = find_adjset(M, names, gtype, names[y], [names[x]], etia_compat=mode == "etia")
            for Zs in (can, mini):
                if Zs is None:
                    continue
                st[f"{mode}_sets"] += 1
                Z = [obs[names.index(v)] for v in Zs]
                st[f"{mode}_correct"] += int(np.isclose(_coef(S, obs[x], obs[y], Z), B[obs[y], obs[x]], atol=1e-8))
    orc.append(st)
pd.DataFrame(orc).to_csv(OUT / "adjust_oracle.csv", index=False)

# ------------------------------------------------ queries: ETIA against the default
qd = []
for k, f in fn.items():
    e_found = d_found = e_len = d_len = both = 0
    for q in qs:
        M = np.array(cases[q["case"]]["M"])
        pe, pdf = q[k], f(M, q["start"], q["end"])
        e_found += bool(pe)
        d_found += bool(pdf)
        if pe and pdf:
            both += 1
            e_len += len(pe) - 1
            d_len += len(pdf) - 1
    qd.append(dict(kind=k, queries=len(qs), etia_found=e_found, default_found=d_found,
                   etia_mean_edges=e_len / max(both, 1), default_mean_edges=d_len / max(both, 1)))
pd.DataFrame(qd).to_csv(OUT / "query_diff.csv", index=False)

# ------------------------------------------------ confidence of cont10_s0 and its drawing
case = "cont10_s0"
names = [f"X{i}" for i in range(10)]
best = r[f"{case}|best"]
we, _ = compute_edge_weights(best, boots_all[case], False, names=names, etia_compat=True)
wd, _ = compute_edge_weights(best, boots_all[case], False, names=names)
conf = we.rename(columns={"edge_consistency": "etia_consistency", "edge_discovery": "discovery"})
conf["default_consistency"] = wd.edge_consistency.values
conf["edge"] = matrix_to_edges(best, names).set_index(["source", "target"]).loc[
    list(zip(conf.source, conf.target))].interaction_type.values
conf[["source", "target", "edge", "etia_consistency", "default_consistency", "discovery"]].to_csv(
    OUT / "edge_confidence.csv", index=False)
v = Visualization(best, names=names)
v.plot_edge_confidence(wd)
(OUT / "graph_cont10_s0.svg").write_text(v.to_svg())

# ------------------------------------------------ timing
tm = pd.read_csv(RES / "crv_jax_gpu.csv")
cpu = pd.read_csv(RES / "crv_jax_cpu.csv")[["kind", "jax_time"]].rename(columns={"jax_time": "jax_cpu_time"})
tm = tm.merge(cpu, on="kind")
tm.to_csv(OUT / "timing.csv", index=False)
long = []
for _, x in tm.iterrows():
    long += [dict(call=x.kind, who="ETIA", seconds=x.etia_time), dict(call=x.kind, who="ETIA-JAX GPU", seconds=x.jax_time),
             dict(call=x.kind, who="ETIA-JAX CPU", seconds=x.jax_cpu_time)]
pd.DataFrame(long).to_csv(OUT / "time_long.csv", index=False)

ag = pd.DataFrame(rows)
o = pd.DataFrame(orc)
summary = dict(same=int(ag.same.sum()), total=int(ag.total.sum()),
               adjust_same=int(ag.same[0]), adjust_total=int(ag.total[0]),
               default_correct=int(o.default_correct.sum()), default_sets=int(o.default_sets.sum()),
               etia_correct=int(o.etia_correct.sum()), etia_sets=int(o.etia_sets.sum()),
               adjust_speedup=float(tm.speedup[tm.kind == "adjustment set"].iloc[0]),
               boot_speedup_cpu=float((tm.etia_time / tm.jax_cpu_time)[tm.kind.str.startswith("bootstrap")].median()))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print(o.to_string()); print(pd.DataFrame(qd).to_string())
