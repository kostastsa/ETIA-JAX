"""Datasets for the search fixtures (PC, CPC; later FCI, ...). ETIA-JAX venv:

    .venv/bin/python reference/make_search_data.py

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


if __name__ == "__main__":
    main()
