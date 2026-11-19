"""Time the ETIA-JAX CI tests on the fixture datasets and save their answers.

    .venv/bin/python benchmarks/ci_timing.py            # GPU (default device)
    JAX_PLATFORMS=cpu .venv/bin/python benchmarks/ci_timing.py

Writes ``benchmarks/results/ci_jax_<device>.npz`` with, per (test, dataset):
p-values and statistics, the setup time, the time of the first query
(includes the JIT compile), the steady time of all queries one by one, and
(for tests with a vectorized path) the steady time of the batched queries.
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
from etia_jax.ci import CI_TESTS                         # noqa: E402
from etia_jax.ci.base import CITest                      # noqa: E402

PAIRS = {  # the same pairs as reference/make_fixtures.py
    "FisherZ": ["cont"], "cci": ["nonlin"], "cg_lrt": ["cont", "disc", "mixed"],
    "dg_lrt": ["cont", "disc", "mixed"], "chisquare": ["disc"], "gsquare": ["disc"],
    "ParCor": ["cont"], "RobustParCor": ["cont", "nonlin"], "ParCorrWLS": ["cont"],
    "GPDC": ["nonlin"], "CMIknn": ["nonlin"], "Gsquared": ["disc"], "CMIsymb": ["disc"],
    "RegressionCI": ["cont", "disc", "mixed"],
}


# the same computation as ETIA: Tetrad compat modes
COMPAT = {"dg_lrt": {"tetrad_compat": True}, "cci": {"tetrad_compat": True}}


def block(x):
    jax.block_until_ready(x)
    return x


def run(name, dname, f):
    X, disc, Q = f[f"{dname}_X"], f[f"{dname}_discrete"], f[f"{dname}_q"]
    kinds = ["discrete" if d else "continuous" for d in disc]
    qs = [(int(r[0]), int(r[1]), [int(z) for z in r[4:4 + int(r[2])]]) for r in Q]

    t0 = time.perf_counter()
    test = CI_TESTS[name](X, kinds, **COMPAT.get(name, {}))
    setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    test.test(*qs[0])
    first = time.perf_counter() - t0
    for k in sorted({len(q[2]) for q in qs}):            # compile every |Z| once
        test.test(*next(q for q in qs if len(q[2]) == k))

    stats, pvals = np.zeros(len(qs)), np.zeros(len(qs))
    t0 = time.perf_counter()
    for i, q in enumerate(qs):
        stats[i], pvals[i] = test.test(*q)
    steady = time.perf_counter() - t0

    batched = np.nan
    if type(test).test_batch is not CITest.test_batch:
        groups = {}
        for i, (x, y, Z) in enumerate(qs):
            groups.setdefault(len(Z), []).append(i)
        def all_batches():
            for k, ii in groups.items():
                block(test.test_batch([qs[i][0] for i in ii], [qs[i][1] for i in ii],
                                      [qs[i][2] for i in ii] if k else np.zeros((len(ii), 0), int)))
        all_batches()                                    # compile
        t0 = time.perf_counter()
        all_batches()
        batched = time.perf_counter() - t0
    return dict(stat=stats, p=pvals, setup=setup, first=first, steady=steady, batched=batched)


SCALE = {"FisherZ": "cont", "gsquare": "disc", "ParCor": "cont", "Gsquared": "disc"}


def run_scale(f, s, out, dev):
    """2000 queries per |Z|: one by one, and batched (both after a warm-up)."""
    for name, dname in SCALE.items():
        X, disc = f[f"{dname}_X"], f[f"{dname}_discrete"]
        kinds = ["discrete" if d else "continuous" for d in disc]
        test = CI_TESTS[name](X, kinds)
        for k in range(4):
            Q = s[f"{dname}_k{k}"]
            xs, ys, Zs = Q[:, 0], Q[:, 1], Q[:, 2:]
            test.test(int(xs[0]), int(ys[0]), [int(z) for z in Zs[0]])      # compile
            t0 = time.perf_counter()
            for i in range(len(Q)):
                test.test(int(xs[i]), int(ys[i]), [int(z) for z in Zs[i]])
            single = time.perf_counter() - t0
            block(test.test_batch(xs, ys, Zs))                                # compile
            t0 = time.perf_counter()
            _, p = block(test.test_batch(xs, ys, Zs))
            batched = time.perf_counter() - t0
            out[f"scale|{name}|k{k}|single"] = np.array(single)
            out[f"scale|{name}|k{k}|batched"] = np.array(batched)
            out[f"scale|{name}|k{k}|p"] = np.asarray(p)
            print(f"{dev} scale {name:9s} k={k} single {single:7.3f}s batched {batched:8.5f}s", flush=True)


def main():
    dev = jax.devices()[0].platform
    f = np.load(ROOT / "reference" / "fixtures" / "ci_data.npz")
    out = {}
    for name, dsets in PAIRS.items():
        for dname in dsets:
            r = run(name, dname, f)
            for k, v in r.items():
                out[f"{name}|{dname}|{k}"] = np.asarray(v)
            print(f"{dev} {name:13s} {dname:7s} setup {r['setup']:6.3f}s first {r['first']:6.3f}s "
                  f"steady {r['steady']:7.3f}s batched {r['batched']:7.4f}s", flush=True)
    sc = np.load(ROOT / "reference" / "fixtures" / "ci_scale.npz")
    run_scale(sc, sc, out, dev)
    (ROOT / "benchmarks" / "results").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "benchmarks" / "results" / f"ci_jax_{dev}.npz", **out)


if __name__ == "__main__":
    main()
