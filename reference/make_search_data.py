"""Datasets for the search fixtures (PC, CPC; later FCI, ...). ETIA-JAX venv:

    .venv/bin/python reference/make_search_data.py [fci|sam]

Writes ``reference/fixtures/search_data.npz``: for each case, the data, the
kinds, the true DAG, the CI test name and alpha. Then run, in the ETIA venv,
``reference/make_fixtures.py search``.
"""
from pathlib import Path

import numpy as np
import jax

from etia_jax.data.scm import random_scm, simulate

OUT = Path(__file__).parent / "fixtures"

#        name      p   kinds         n     tests                     seeds
CASES = [("cont10", 10, "continuous", 1000, ["FisherZ"], range(6)),
         ("cont20", 20, "continuous", 1000, ["FisherZ"], range(4)),
         ("cont30", 30, "continuous", 1000, ["FisherZ"], range(3)),
         ("disc10", 10, "discrete", 2000, ["chisquare", "gsquare"], range(3)),
         ("mixed10", 10, "mixed", 1000, ["cg_lrt", "dg_lrt"], range(3)),
         ("cont50", 50, "continuous", 2000, ["FisherZ"], range(2)),
         ("cont100", 100, "continuous", 2000, ["FisherZ"], range(2))]
ALPHAS = [0.05, 0.01]


def main():
    out, names = {}, []
    for name, p, kinds, n, tests, seeds in CASES:
        for s in seeds:
            seed = 100 * len(names) + s
            spec = random_scm(p, 2.5, kinds=kinds, rng=seed, max_parents=4)
            X, _ = simulate(spec, n, jax.random.key(seed))
            key = f"{name}_s{s}"
            out[f"{key}|X"] = X.astype(np.float64)
            out[f"{key}|discrete"] = np.array([k == "discrete" for k in spec.kinds])
            out[f"{key}|A"] = spec.A
            out[f"{key}|tests"] = np.array(tests)
            names.append(key)
    out["cases"] = np.array(names)
    out["alphas"] = np.array(ALPHAS)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "search_data.npz", **out)
    print("wrote", OUT / "search_data.npz", len(names), "datasets")


FCI_CASES = [("cont10", 12, "continuous", 1000, ["FisherZ"], range(6)),
             ("cont20", 24, "continuous", 1000, ["FisherZ"], range(4)),
             ("cont30", 36, "continuous", 1000, ["FisherZ"], range(3)),
             ("disc10", 12, "discrete", 2000, ["chisquare", "gsquare"], range(3)),
             ("mixed10", 12, "mixed", 1000, ["cg_lrt", "dg_lrt"], range(3)),
             ("cont50", 60, "continuous", 2000, ["FisherZ"], range(2))]


def main_fci():
    """Data with latent variables (about 1 in 6 nodes): the latent columns are
    dropped. A is the full DAG and ``latent`` marks the dropped nodes."""
    out, names = {}, []
    for name, p_all, kinds, n, tests, seeds in FCI_CASES:
        for s in seeds:
            seed = 7000 + 100 * len(names) + s
            rng = np.random.default_rng(seed)
            spec = random_scm(p_all, 2.5, kinds=kinds, rng=seed, max_parents=4)
            X, _ = simulate(spec, n, jax.random.key(seed))
            latent = np.zeros(p_all, bool)
            latent[rng.choice(p_all, p_all // 6, replace=False)] = True
            key = f"{name}_s{s}"
            out[f"{key}|X"] = X[:, ~latent].astype(np.float64)
            out[f"{key}|discrete"] = np.array([k == "discrete" for k in spec.kinds])[~latent]
            out[f"{key}|A"] = spec.A
            out[f"{key}|latent"] = latent
            out[f"{key}|tests"] = np.array(tests)
            names.append(key)
    out["cases"] = np.array(names)
    out["alphas"] = np.array(ALPHAS)
    np.savez_compressed(OUT / "search_fci_data.npz", **out)
    print("wrote", OUT / "search_fci_data.npz", len(names), "datasets")


#          name     p   link      n     seeds
SAM_CASES = [("lin5", 5, "linear", 500, range(1)),
             ("sin5", 5, "sin", 500, range(1)),
             ("lin10", 10, "linear", 500, range(1)),
             ("sin10", 10, "sin", 500, range(1)),
             ("lin20", 20, "linear", 500, range(1)),
             ("sin20", 20, "sin", 500, range(1))]


def main_sam():
    """Continuous data for SAM (increment 7): linear and sine links."""
    out, names = {}, []
    for name, p, link, n, seeds in SAM_CASES:
        for s in seeds:
            seed = 9000 + 100 * len(names) + s
            spec = random_scm(p, 2.0, kinds="continuous", link=link, rng=seed, max_parents=3)
            X, _ = simulate(spec, n, jax.random.key(seed))
            key = f"{name}_s{s}"
            out[f"{key}|X"] = X.astype(np.float64)
            out[f"{key}|A"] = spec.A
            names.append(key)
    out["cases"] = np.array(names)
    np.savez_compressed(OUT / "sam_data.npz", **out)
    print("wrote", OUT / "sam_data.npz", len(names), "datasets")


if __name__ == "__main__":
    import sys
    if "fci" in sys.argv[1:]:
        main_fci()
    elif "sam" in sys.argv[1:]:
        main_sam()
    else:
        main()

