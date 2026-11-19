"""Make the figures and data files of the increment-1 demo page.

Run from the repository root:  .venv/bin/python demo/increment1/make_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
OUT = Path(__file__).parent
(OUT / "fig").mkdir(exist_ok=True)

from etia_jax.graphs.dag import random_dag, topological_order
from etia_jax.graphs.convert import dag_to_cpdag, dag_to_mag, mag_to_pag
from etia_jax.graphs.encoding import adj_to_marks, CIRCLE, ARROW, TAIL
from etia_jax.graphs.separation import d_separated
from etia_jax.graphs.metrics import shd
from etia_jax.data.scm import sample_params, simulate

SEED = 90

# --------------------------------------------------------------------- model
rng = np.random.default_rng(SEED)
A = random_dag(9, 2.5, rng)
L = np.zeros(9, bool)
L[rng.choice(9, 2, replace=False)] = True
obs = np.flatnonzero(~L)
names = [f"L{i}" if L[i] else f"X{i}" for i in range(9)]
spec = sample_params(A, rng=SEED)
X, trace = simulate(spec, 5000, jax.random.key(SEED))

G_dag = adj_to_marks(A)
G_cpdag = dag_to_cpdag(A)
G_mag = dag_to_mag(A, L)
G_pag = mag_to_pag(G_mag)

# ------------------------------------------------------------------ drawing
# Fixed positions for the seed-90 graph (clear of overlaps)
pos = {7: (-2.8, 0.4), 6: (-1.0, 0.4), 4: (0.6, 0.4), 2: (2.8, 0.4),
       5: (-1.8, -1.6), 8: (1.5, -1.6), 3: (2.8, -3.0),
       0: (-0.2, -3.0), 1: (-0.2, -4.3)}

FG, ACC, LAT = "#1f2937", "#2563eb", "#9ca3af"
import os
PREVIEW = os.environ.get("PREVIEW")  # also write small PNGs to this folder


def _save(fig, fname):
    fig.savefig(OUT / "fig" / fname, bbox_inches="tight", facecolor="white")
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / fname.replace(".svg", ".png"), bbox_inches="tight",
                    facecolor="white", dpi=60)
    plt.close(fig)


def draw(ax, G, nodes, title, latent=()):
    """Draw a mark matrix ``G`` over ``nodes`` (DAG indices)."""
    R = 0.23
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            if G[a, b] == 0:
                continue
            p, q = np.array(pos[nodes[a]], float), np.array(pos[nodes[b]], float)
            u = (q - p) / np.linalg.norm(q - p)
            s, e = p + u * R, q - u * R
            col = LAT if nodes[a] in latent or nodes[b] in latent else FG
            ax.plot(*zip(s, e), color=col, lw=1.6, zorder=1)
            for end, other, mark in ((e, s, G[a, b]), (s, e, G[b, a])):
                d = (end - other) / np.linalg.norm(end - other)
                if mark == ARROW:
                    ax.annotate("", xy=end, xytext=end - d * 0.2,
                                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.35",
                                                color=col, lw=1.6), zorder=2)
                elif mark == CIRCLE:
                    c = end - d * 0.07
                    ax.add_patch(plt.Circle(c, 0.065, fc="white", ec=col, lw=1.5, zorder=3))
    for v in nodes:
        lat = v in latent
        ax.add_patch(plt.Circle(pos[v], R, fc="#f3f4f6" if lat else "white",
                                ec=LAT if lat else ACC, lw=2, ls="--" if lat else "-", zorder=4))
        ax.text(*pos[v], names[v], ha="center", va="center", fontsize=11,
                color=LAT if lat else FG, zorder=5)
    xs = [pos[v][0] for v in range(9)]
    ys = [pos[v][1] for v in range(9)]
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, color=FG)


def save(G, nodes, title, fname, latent=()):
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    draw(ax, G, nodes, title, latent)
    _save(fig, fname)


all_nodes = list(range(9))
save(G_dag, all_nodes, "DAG (latents dashed)", "dag.svg", latent=set(np.flatnonzero(L)))
save(G_cpdag, all_nodes, "CPDAG (no latents)", "cpdag.svg")
save(G_mag, list(obs), "MAG over observed", "mag.svg")
save(G_pag, list(obs), "PAG over observed", "pag.svg")

# ------------------------------------------------- data files for the page
pd.DataFrame(X[:400][:, obs], columns=[names[v] for v in obs]).round(4) \
    .to_csv(OUT / "samples.csv", index=False)

# OLS on the full data (latents included) recovers each structural weight
rows = []
for j in range(9):
    pa = np.flatnonzero(A[:, j])
    if len(pa) == 0:
        continue
    Z = np.c_[X[:, pa], np.ones(len(X))]
    beta = np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
    for k, b in zip(pa, beta):
        rows.append({"edge": f"{names[k]} → {names[j]}", "true_weight": spec.weights[(int(k), j)][0, 0],
                     "ols_estimate": b})
w = pd.DataFrame(rows)
w["abs_error"] = (w.ols_estimate - w.true_weight).abs()
w.round(4).to_csv(OUT / "weights.csv", index=False)


# Fisher-z tests on the observed data versus the d-separation truth
def fisher_z(X, x, y, Z):
    C = np.corrcoef(X[:, [x, y, *Z]].T)
    P = np.linalg.inv(C)
    r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    z = np.sqrt(len(X) - len(Z) - 3) * np.arctanh(np.clip(r, -0.999999, 0.999999))
    return r, 2 * stats.norm.sf(abs(z))


q = []
qrng = np.random.default_rng(1)
for _ in range(300):
    x, y = qrng.choice(obs, 2, replace=False)
    rest = [v for v in obs if v not in (x, y)]
    Z = list(qrng.choice(rest, qrng.integers(0, 4), replace=False))
    r, pv = fisher_z(X, x, y, Z)
    sep = d_separated(A, int(x), int(y), [int(z) for z in Z])
    q.append({"query": f"{names[x]} ⫫ {names[y]} | {{{', '.join(names[z] for z in Z)}}}",
              "abs_partial_corr": abs(r), "neg_log10_p": min(-np.log10(max(pv, 1e-300)), 50),
              "truth": "d-separated" if sep else "d-connected"})
qd = pd.DataFrame(q).drop_duplicates("query")
qd.round(4).to_csv(OUT / "ci_queries.csv", index=False)
alpha = 0.01
sep_ = qd.truth == "d-separated"
rej = qd.neg_log10_p > -np.log10(alpha)
ci_summary = {"n": len(qd), "n_sep": int(sep_.sum()),
              "false_reject": float(rej[sep_].mean()), "power": float(rej[~sep_].mean())}

# ----------------------------------------------- comparison with Tetrad
f = np.load(ROOT / "reference" / "fixtures" / "graphs.npz")
rows = []
for g in range(int(f["n_graphs"])):
    Ag, Lg = f[f"A_{g}"], f[f"latent_{g}"]
    C, M = dag_to_cpdag(Ag), dag_to_mag(Ag, Lg)
    P = mag_to_pag(M)
    T = f[f"pag_{g}"]
    n_extra_tail = int(((P == CIRCLE) & (T == TAIL)).sum())
    n_missing_arrow = int(((P == ARROW) & (T == CIRCLE)).sum())
    rows.append({"graph": g, "nodes": len(Ag), "latents": int(Lg.sum()), "edges": int(Ag.sum()),
                 "cpdag": "same" if (C == f[f"cpdag_{g}"]).all() else "differs",
                 "mag": "same" if (M == f[f"mag_{g}"]).all() else "differs",
                 "pag_shd_to_tetrad": shd(T, P),
                 "tetrad_wrong_tails": n_extra_tail, "tetrad_missed_arrows": n_missing_arrow})
tdf = pd.DataFrame(rows)
tdf.to_csv(OUT / "tetrad_comparison.csv", index=False)

# graph 55: our PAG, Tetrad's PAG, and the witness MAG
from bruteforce import find_circle_witness
M55 = f["mag_55"]
P55, T55 = mag_to_pag(M55), f["pag_55"]
W55 = find_circle_witness(M55, P55, 2, 3)
A55 = f["A_55"]; L55 = f["latent_55"]
obs55 = np.flatnonzero(~L55)
pos_save, names_save = dict(pos), list(names)
pos.clear()
ang = np.linspace(0, 2 * np.pi, len(obs55), endpoint=False)
for k in range(len(obs55)):
    pos[k] = (1.4 * np.sin(ang[k]), 1.4 * np.cos(ang[k]))
names[:] = [f"X{k}" for k in range(len(obs55))] + [""] * 9
_orig_xs = None


def save_round(G, title, fname):
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    draw_nodes = list(range(len(G)))
    # draw() sets limits from 9 nodes; use local limits here
    draw(ax, G, draw_nodes, title)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    _save(fig, fname)


pos.update({k: pos[k] for k in range(len(obs55))})
for k in range(len(obs55), 9):
    pos[k] = (0, 0)
save_round(P55, "ETIA-JAX PAG", "g55_ours.svg")
save_round(T55, "Tetrad DagToPag", "g55_tetrad.svg")
save_round(W55, "Witness MAG", "g55_witness.svg")

# -------------------------------------------------------- summary numbers
import json
summary = {
    "n_graphs": int(f["n_graphs"]),
    "cpdag_same": int((tdf.cpdag == "same").sum()),
    "mag_same": int((tdf.mag == "same").sum()),
    "pag_same": int((tdf.pag_shd_to_tetrad == 0).sum()),
    "tetrad_wrong_tails": int(tdf.tetrad_wrong_tails.sum()),
    "tetrad_missed_arrows": int(tdf.tetrad_missed_arrows.sum()),
    "weights_max_abs_error": float(w.abs_error.max()),
    "n_weights": len(w),
    **{f"ci_{k}": v for k, v in ci_summary.items()},
    "latents": [names_save[i] for i in np.flatnonzero(L)],
    "mag_edges": int((G_mag != 0).sum() // 2),
    "genjax_choices": sorted(trace.get_choices().keys()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
