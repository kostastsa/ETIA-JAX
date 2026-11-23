"""Time-series datasets for the increment-6 fixtures. ETIA-JAX venv:

    .venv/bin/python reference/make_ts_data.py

Writes ``reference/fixtures/ts_data.npz``. For each case: the raw series X
(T x p), the lagged table of ETIA (columns X1..Xp, X1:1.., X1:L..), the
discrete flags (of the raw variables), the SVAR weights, the true window DAG
with its latent mask, the Tetrad CI tests and scores, and the tigramite CI
tests. Then run, in the ETIA venv, ``reference/make_fixtures.py ts`` and
``ts_tigramite``.
"""
from pathlib import Path

import numpy as np
import jax

from etia_jax.data.svar import random_svar, simulate, lagged_table, window_dag

OUT = Path(__file__).parent / "fixtures"

TETRAD_TESTS = {"continuous": ["FisherZ"], "discrete": ["chisquare", "gsquare"], "mixed": ["cg_lrt", "dg_lrt"]}
SCORES = {"continuous": ["sem_bic_score", "dg_bic"], "discrete": ["bdeu", "discrete_bic"], "mixed": ["cg_bic", "dg_bic"]}
TIGRAMITE_TESTS = {"continuous": ["ParCor", "RobustParCor", "ParCorrWLS", "CMIknn", "GPDC"],
                   "discrete": ["Gsquared", "CMIsymb"], "mixed": ["RegressionCI"]}

#        name     p  L  kinds         T     seeds
CASES = [("cont3", 3, 1, "continuous", 1000, range(3)),
         ("cont3", 3, 2, "continuous", 1000, range(2)),
         ("cont5", 5, 1, "continuous", 1000, range(2)),
         ("cont5", 5, 2, "continuous", 1000, range(2)),
         ("cont8", 8, 2, "continuous", 1500, range(2)),
         ("disc4", 4, 1, "discrete", 2000, range(2)),
         ("mixed4", 4, 1, "mixed", 1000, range(2))]
ALPHAS = [0.05, 0.01]


def main():
    out, names = {}, []
    for name, p, L, kinds, T, seeds in CASES:
        for s in seeds:
            seed = 1000 + 100 * len(names) + s
            spec = random_svar(p, L=L, kinds=kinds, seed=seed)
            X = simulate(spec, T, jax.random.key(seed))
            key = f"{name}_L{L}_s{s}"
            A, latent = window_dag(spec)
            kind = kinds if kinds in TETRAD_TESTS else "mixed"
            out[f"{key}|X"] = X
            out[f"{key}|lagged"] = lagged_table(X, L)
            out[f"{key}|L"] = np.array(L)
            out[f"{key}|discrete"] = np.array([k == "discrete" for k in spec.kinds])
            out[f"{key}|B0"], out[f"{key}|B"] = spec.B0, spec.B
            out[f"{key}|A"], out[f"{key}|latent"] = A, latent
            out[f"{key}|tests"] = np.array(TETRAD_TESTS[kind])
            out[f"{key}|scores"] = np.array(SCORES[kind])
            out[f"{key}|tig_tests"] = np.array(TIGRAMITE_TESTS[kind])
            names.append(key)
    out["cases"] = np.array(names)
    out["alphas"] = np.array(ALPHAS)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "ts_data.npz", **out)
    print("wrote", OUT / "ts_data.npz", len(names), "datasets")


if __name__ == "__main__":
    main()
