"""Make the datasets and queries for the CI-test fixtures (ETIA-JAX venv).

    .venv/bin/python reference/make_ci_data.py

Writes ``reference/fixtures/ci_data.npz``. Then run, in the ETIA venv,
``reference/make_fixtures.py ci`` to add the ETIA answers and timings.
"""
from pathlib import Path

import numpy as np
import jax

from etia_jax.data.scm import random_scm, simulate
from etia_jax.graphs.separation import d_separated

OUT = Path(__file__).parent / "fixtures"

DATASETS = {
    #  name     nodes kinds          link      n     seed
    "cont":   (10, "continuous", "linear", 1000, 11),
    "nonlin": (6, "continuous", "tanh", 300, 12),
    "disc":   (8, "discrete", "linear", 2000, 13),
    "mixed":  (8, "mixed", "linear", 1000, 14),
}
N_QUERIES = 60
MAX_K = 3


def queries(A, rng):
    """N_QUERIES random queries, balanced over |Z| = 0..MAX_K."""
    p = len(A)
    out = []
    for i in range(N_QUERIES):
        k = i % (MAX_K + 1)
        x, y = rng.choice(p, 2, replace=False)
        rest = [v for v in range(p) if v not in (x, y)]
        Z = sorted(rng.choice(rest, k, replace=False).tolist())
        sep = d_separated(A, int(x), int(y), Z)
        out.append([x, y, k, int(sep)] + Z + [-1] * (MAX_K - k))
    return np.array(out, dtype=np.int32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    out = {}
    for name, (p, kinds, link, n, seed) in DATASETS.items():
        spec = random_scm(p, 2.0, kinds=kinds, link=link, rng=seed, max_parents=3)
        X, _ = simulate(spec, n, jax.random.key(seed))
        out[f"{name}_X"] = X.astype(np.float64)
        out[f"{name}_discrete"] = np.array([k == "discrete" for k in spec.kinds])
        out[f"{name}_A"] = spec.A
        out[f"{name}_q"] = queries(spec.A, rng)
    np.savez_compressed(OUT / "ci_data.npz", **out)
    print("wrote", OUT / "ci_data.npz")




def make_scale(n_per_k=2000, p=30, seed=1):
    """Unique random queries (unordered x, y; sorted Z) on 30-variable data,
    so that the result caches of Tetrad and tigramite do not help."""
    rng = np.random.default_rng(seed)
    out = {}
    for dname, kinds, n in (("cont", "continuous", 1000), ("disc", "discrete", 2000)):
        spec = random_scm(p, 2.0, kinds=kinds, rng=seed + len(dname), max_parents=3)
        X, _ = simulate(spec, n, jax.random.key(seed))
        out[f"{dname}_X"] = X.astype(np.float64)
        out[f"{dname}_discrete"] = np.array([k == "discrete" for k in spec.kinds])
        for k in range(MAX_K + 1):
            from math import comb
            n_k = min(n_per_k, comb(p, 2) * comb(p - 2, k))   # all queries if fewer
            seen, rows = set(), []
            while len(rows) < n_k:
                v = rng.choice(p, 2 + k, replace=False)
                key = (frozenset(v[:2].tolist()), tuple(sorted(v[2:].tolist())))
                if key not in seen:
                    seen.add(key)
                    rows.append(list(v[:2]) + sorted(v[2:].tolist()))
            out[f"{dname}_k{k}"] = np.array(rows, dtype=np.int32)
    np.savez_compressed(OUT / "ci_scale.npz", **out)
    print("wrote", OUT / "ci_scale.npz")


if __name__ == "__main__":
    import sys
    if "scale" in sys.argv[1:]:
        make_scale()
    else:
        main()
