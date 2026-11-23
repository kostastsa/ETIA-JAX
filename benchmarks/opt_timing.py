"""Time and score NOTEARS and SAM of ETIA-JAX against ETIA (increment 7).

    .venv/bin/python benchmarks/opt_timing.py                    # GPU (SAM), CPU and GPU (NOTEARS)
    JAX_PLATFORMS=cpu .venv/bin/python benchmarks/opt_timing.py  # all on the CPU

Writes ``benchmarks/results/opt_jax_<platform>.csv``: one row per run, with
the ETIA time (causalnex or cdt, from the fixtures), the ETIA-JAX time of a
first run (with compilation) and of a second run, and the accuracy against
the true DAG (SHD at the threshold, and the AUROC of the weights for SAM).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_optim import _auc                                     # noqa: E402
from etia_jax.graphs.encoding import adj_to_marks              # noqa: E402
from etia_jax.graphs.metrics import shd                         # noqa: E402
from etia_jax.search import notears as NT                       # noqa: E402
from etia_jax.search import sam as S                            # noqa: E402

FIX = ROOT / "reference" / "fixtures"
FIELDS = ["algo", "case", "p", "setting", "device", "mode", "etia_time", "jax_first", "jax_time",
          "speedup", "shd_etia", "shd_jax", "auc_etia", "auc_jax", "same_graph"]


def timed(f):
    t0 = time.perf_counter()
    f()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = f()
    return first, time.perf_counter() - t0, out


def dag_marks(W, thr):
    """The DAG of |W| > thr after the cycle removal of the default mode."""
    Wt = np.where(np.abs(W) > thr, W, 0.0)
    Wt = NT._break_cycles(Wt)
    G = np.zeros(W.shape, np.int8)
    G[Wt != 0], G[(Wt != 0).T] = 2, 3
    return G


def notears_rows(devices):
    r = np.load(FIX / "notears_ref.npz")
    d = np.load(FIX / "search_data.npz")
    cases = sorted({k.split("|")[0] for k in r.files})
    rows = []
    for case in cases:
        X, A = d[f"{case}|X"], d[f"{case}|A"]
        T = adj_to_marks(A)
        key = f"{case}|0.8"
        etia_t = float(r[key + "|time"])
        G_etia = r[key + "|etia_matrix"] if key + "|etia_matrix" in r.files else None
        for dev in devices:
            for mode in ("etia", "default"):
                f = lambda: NT.notears(X, threshold=0.8, etia_compat=mode == "etia", device=dev)
                first, t, res = timed(f)
                rows.append(dict(algo="notears", case=case, p=X.shape[1], setting="threshold 0.8",
                                 device=dev, mode=mode, etia_time=etia_t, jax_first=first, jax_time=t,
                                 speedup=etia_t / t,
                                 shd_etia="" if G_etia is None else shd(T, G_etia),
                                 shd_jax=shd(T, res.G),
                                 same_graph="" if G_etia is None or mode != "etia" else int((res.G == G_etia).all())))
                print(f"notears {case:10s} {dev} {mode:7s} etia {etia_t:8.2f}s  jax {t:7.3f}s "
                      f"(first {first:6.2f}s)  shd {rows[-1]['shd_etia']}/{rows[-1]['shd_jax']}", flush=True)
    return rows


def sam_rows(platform):
    r = np.load(FIX / "sam_ref.npz")
    d = np.load(FIX / "sam_data.npz")
    rows = []
    jobs = [(str(c), "short", dict(train_epochs=1000, test_epochs=500)) for c in d["cases"]]
    jobs += [(k.split("|")[0], "full", {}) for k in r.files if k.endswith("|full|W")]
    for case, tag, params in jobs:
        if f"{case}|{tag}|W" not in r.files:
            continue
        X, A = d[f"{case}|X"], d[f"{case}|A"]
        T = adj_to_marks(A)
        We = r[f"{case}|{tag}|W"]
        etia_t = float(r[f"{case}|{tag}|time"])
        nb = X.shape[0] // 100
        for mode in ("etia", "default"):
            f = lambda: S.sam(X, etia_compat=mode == "etia", **params)
            first, t, res = timed(f)
            Wj = res.W / nb if mode == "etia" else res.W
            rows.append(dict(algo="sam", case=case, p=X.shape[1],
                             setting="train 1000 / test 500" if tag == "short" else "ETIA defaults",
                             device=platform, mode=mode, etia_time=etia_t, jax_first=first, jax_time=t,
                             speedup=etia_t / t, shd_etia=shd(T, dag_marks(We / nb, 0.5)),
                             shd_jax=shd(T, dag_marks(Wj, 0.5)), auc_etia=_auc(We, A), auc_jax=_auc(Wj, A),
                             same_graph=int((res.G == r[f"{case}|{tag}|etia_matrix"]).all()) if mode == "etia" else ""))
            print(f"sam {case:10s} {tag:5s} {mode:7s} etia {etia_t:8.1f}s  jax {t:7.2f}s (first {first:6.2f}s)"
                  f"  auc {rows[-1]['auc_etia']:.3f}/{rows[-1]['auc_jax']:.3f}", flush=True)
    return rows


def main():
    platform = jax.devices()[0].platform
    devices = ["cpu"] + (["gpu"] if platform == "gpu" else [])
    rows = notears_rows(devices) + sam_rows(platform)
    out = ROOT / "benchmarks" / "results" / f"opt_jax_{platform}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
