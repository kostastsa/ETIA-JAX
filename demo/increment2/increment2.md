---
title: ETIA-JAX — increment 2
subtitle: The 14 CI tests of ETIA in JAX, compared with Tetrad 7.6.5 and tigramite 5.2 for answers and for speed
date: 2026-11-19
tags: [🧪 increment 2 of 10, ✅ tests pass, ⏱ timing ETIA vs ETIA-JAX]
view: page
---

::: {.callout .result title="Result"}
All 14 CI tests of ETIA now run in JAX. On the fixture data, 20 of 21 test/dataset pairs make the same decision as ETIA on all 60 queries; GPDC agrees on 59 of 60 because its null distribution is random. With batched queries, the fast tests are 16× to 590× faster than ETIA (on the faster device). Four tests are still slower than ETIA when they run one query at a time.
:::

::: stats
- 14 / 14 | CI tests ported | 6 from Tetrad 7.6.5, 8 from tigramite 5.2
- 20 / 21 | Pairs with the same decisions | p-values equal to ~1e-10 for the exact ports
- 16–590× | Speed-up, batched | 2000 unique queries, FisherZ, G², ParCorr, Gsquared
- 14 000× | CMIsymb speed-up | 1449 s in ETIA, 0.10 s on the GPU
- 4 | Tetrad defects found | in CCI and DG-LRT; kept only in `tetrad_compat`
- 43 | Tests pass | 30 fast (1.8 min) + 13 slow (9 min)
:::

## What increment 2 contains

`etia_jax.ci.CI_TESTS` has the 14 CI tests of the ETIA configuration, with the same names. Each test answers `test(x, y, Z) -> (statistic, p-value)`. The linear-Gaussian, contingency-table and GPDC tests also have `test_batch(xs, ys, Zs)`, which answers many queries with one `jax.vmap` call.

| Family | Tests | File |
|---|---|---|
| Linear Gaussian | `FisherZ`, `ParCor`, `RobustParCor`, `ParCorrWLS`, `dg_lrt` | `etia_jax/ci/gaussian.py` |
| Contingency tables | `chisquare`, `gsquare`, `Gsquared`, `CMIsymb` | `etia_jax/ci/discrete.py` |
| Mixed data | `cg_lrt`, `RegressionCI` | `etia_jax/ci/mixed.py` |
| Nonparametric | `CMIknn`, `GPDC`, `cci` | `etia_jax/ci/nonparametric.py` |

## Agreement with ETIA

`reference/make_fixtures.py ci` runs each test in ETIA's own way (`prepare_data_tetrad` and the Tetrad classes through JPype, or tigramite's `run_test`) on four GenJAX datasets: `cont` (10 variables, n = 1000), `nonlin` (6, tanh links, n = 300), `disc` (8, 3 categories, n = 2000) and `mixed` (8, n = 1000). Each dataset has 60 queries with |Z| = 0 … 3.

::: {.table src="agreement.csv" labels="decisions_same_pct: Same decision at α = 0.05 (%) | max_abs_dp: max abs Δp | max_rel_dstat: max rel Δ statistic" dec="max_abs_dp:3 max_rel_dstat:3 decisions_same_pct:1" sort="origin:asc" height=560}
**Table 1.** One row per test and dataset. A Δp of the order of 1e-10 or less means the same p-value up to rounding. `CMIknn` has no p-value in ETIA (fixed threshold on the CMI), so the decision uses the statistic. `CMIsymb` and `GPDC` use random null distributions, so their p-values differ by the Monte Carlo error.
:::

### Four defects in Tetrad 7.6.5, found from its source

The formulas come from the source of Tetrad 7.6.5, the version of `tetradlib.jar` in ETIA. ETIA-JAX implements them in JAX and does not copy the Java code. The option `tetrad_compat=True` keeps the defects, so that the results agree with ETIA. The default mode corrects them.

| Test | Defect in Tetrad 7.6.5 | Effect |
|---|---|---|
| `cci` | The kernel distance uses (x, z₁ … z₍ₖ₋₁₎) in place of (z₁ … zₖ). | The regression of x on Z partly uses x itself and omits the last variable of Z. |
| `cci` | The correlation of the standardized basis values divides by n − 1. | The statistic is n / (n − 1) too large. |
| `dg_lrt` | The statistic is −2 Δlik, but `lik` is already 2 × log-likelihood. | The statistic is twice too large: the test rejects too often on all data types. |
| `dg_lrt` | The covariance uses the mean sum / (n − 1) with the divisor n. | A small bias in each covariance entry. |

## Calibration and power

For each test, the data are 100 replicates from a GenJAX SCM with Z → X and Z → Y. Size is the rejection rate at α = 0.05 when X ⫫ Y | Z. Power is the rejection rate when X → Y is added.

::: {.table src="calibration.csv" labels="size_pct: Size (%) | power_pct: Power (%)" dec="size_pct:0 power_pct:0" height=460}
**Table 2.** The size must be near 5 %. The Tetrad mode of `dg_lrt` and the default (uncorrected) `cci` reject too often under independence. `cci` with `bonferroni=True` corrects the p-value for the 100 basis pairs.
:::

## Timing: ETIA against ETIA-JAX

All runs are on the same computer: Intel Core i9-14900K (32 threads), NVIDIA RTX 4090, JAX 0.11.2 in float64. ETIA runs in its own venv (Python 3.10, Java 17 for Tetrad).

- **One by one**: one call per query, as ETIA's PC and PCMCI run them.
- **Batched**: all queries with the same |Z| in one `vmap` call. Increment 3 (PC) uses this path.
- The times exclude the first JIT compile. The compile time is in a separate column of Table 3.

::: {.chart src="timing_60_long.csv" type="scatter" x="test" y="ms_per_query" color="who" log-y=true y-label="ms per query (log scale)" height=420 .wide}
**Figure 1.** Mean time per query on the 60 fixture queries, one point per mode. Lower is better.
:::

::: {.table src="timing_60.csv" labels="etia_s: ETIA (s) | jax_gpu_s: JAX GPU (s) | jax_cpu_s: JAX CPU (s) | jax_gpu_batched_s: JAX GPU batched (s) | jax_cpu_batched_s: JAX CPU batched (s) | jax_gpu_first_s: GPU first call (s) | jax_cpu_first_s: CPU first call (s) | speedup_best: Speed-up (best JAX)" dec="etia_s:3 jax_gpu_s:3 jax_cpu_s:3 jax_gpu_batched_s:4 jax_cpu_batched_s:4 jax_gpu_first_s:2 jax_cpu_first_s:2 speedup_best:1" sort="speedup_best:desc" height=560 .wide}
**Table 3.** Total time of the 60 queries. The first-call column includes the JIT compile of the first |Z| size. The speed-up compares ETIA with the fastest JAX mode.
:::

### Many queries

PC and FCI run thousands of CI queries. This benchmark uses 30-variable data (`cont`: n = 1000; `disc`: n = 2000) and unique queries: all 435 pairs for |Z| = 0, and 2000 random queries for each |Z| = 1, 2, 3. The queries are unique so that the result caches of Tetrad and tigramite do not help.

::: {.chart src="scale.csv" type="line" x="k" y="queries_per_s" color="series" log-y=true x-label="|Z|" y-label="queries per second (log scale)" height=440 .wide}
**Figure 2.** Throughput of the four fastest tests. Higher is better.
:::

::: {.table src="scale_wide.csv" labels="k: abs Z | JAX_GPU_batched: JAX GPU batched (s) | JAX_CPU_batched: JAX CPU batched (s) | JAX_GPU_one_by_one: JAX GPU one by one (s) | JAX_CPU_one_by_one: JAX CPU one by one (s) | ETIA: ETIA (s) | speedup_batched_gpu: Speed-up GPU batched | speedup_batched_cpu: Speed-up CPU batched" dec="ETIA:3 JAX_GPU_batched:4 JAX_CPU_batched:4 JAX_GPU_one_by_one:3 JAX_CPU_one_by_one:3 speedup_batched_gpu:0 speedup_batched_cpu:0" height=460 .wide}
**Table 4.** Total time for all the queries of one |Z|. On all these queries (7435 per test), the p-values of ETIA and ETIA-JAX agree to better than 1e-10.
:::

::: {.callout .note title="What the timings show"}
- **Batched queries are the strength of JAX.** For 2000 unique queries, one `vmap` call takes 0.6–38 ms. ETIA takes 0.02–4.6 s for the same queries. PC and FCI test many queries of the same |Z| at each level, so increment 3 will use this path.
- **One query per call is not always faster.** Each JAX call has a fixed cost (Python, dispatch, the device). On the CPU this cost is about 40 µs, so FisherZ and ParCorr are still faster than Tetrad and tigramite one by one. On the GPU the cost is about 0.4 ms, and Tetrad's Java code (60–125 µs per FisherZ query) is faster.
- **Slower than ETIA today:** `cg_lrt`, `dg_lrt` and `RegressionCI` on mixed data (2–8× slower than ETIA, on the faster device), and the contingency-table tests one by one (`gsquare` about 6× slower). These tests have no batched path yet. The plan is to add batched paths for the tests that PC and FCI use (increment 3) and to keep the compiled shapes fixed.
- **CMIsymb in ETIA** takes 29 s per query, because numba cannot compile tigramite's shuffle loop (`np.all(axis=...)`) and falls back to Python. The JAX version shuffles in a `vmap` and takes 2 ms per query on the GPU.
- **GPDC** is 1.3× faster on the GPU (batched). Its GP fit uses float64 Cholesky factorizations, which are slow on this GPU, and on the CPU it is slower than sklearn.
- **Compile time:** the first call of each |Z| size compiles (0.1–2 s, Table 3). A PC run compiles at most one shape per level.
:::

## Tests

::: {.table src="tests.csv" height=420}
**Table 5.** The test suite after increment 2. Run `pytest -q` for the fast tests and `pytest -m slow` for the calibration and exact-PAG tests.
:::

## Next: increment 3

Increment 3 adds PC and CPC. They run their CI queries level by level, so all the queries of one level go into one batched call.

::: {.toggle title="How to remake this page"}
```bash
cd ~/PROJECTS/ETIA-JAX
.venv/bin/python reference/make_ci_data.py && .venv/bin/python reference/make_ci_data.py scale
JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 ~/PROJECTS/ETIA/.venv/bin/python reference/make_fixtures.py ci scale
.venv/bin/python benchmarks/ci_timing.py
JAX_PLATFORMS=cpu .venv/bin/python benchmarks/ci_timing.py
.venv/bin/python demo/increment2/make_data.py
pageview demo/increment2/increment2.md
```
:::
