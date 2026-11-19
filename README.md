# ETIA-JAX

ETIA-JAX is a port of ETIA (automated causal discovery) to Python and JAX. It does not use Java or R.

- JAX does the numerical work: CI tests, scores, regressors, and predictive models.
- GenJAX generates data from structural causal models (SCMs).
- NumPy does the combinatorial graph work: orientation rules and separation. JAX gives no advantage for this work.

## Setup

The port needs Python 3.11 or later. The venv uses Python 3.12 and JAX with CUDA 12.

```bash
UV=~/PROJECTS/ETIA/.uvboot/bin/uv
$UV venv --python 3.12 .venv
$UV pip install --python .venv/bin/python "jax[cuda12]" -e ".[test]"
.venv/bin/python -m pytest -q
```

## Graph encoding

The port uses the ETIA and Tetrad encoding. `G[i, j]` is the mark at `j` on the edge `i - j`:

| value | mark |
|---|---|
| 0 | no edge |
| 1 | circle |
| 2 | arrowhead |
| 3 | tail |

An undirected CPDAG edge is tail-tail (3, 3), as in Tetrad.

## Tests and references

- **Oracle tests.** A d-separation oracle replaces the CI test. With the oracle, each algorithm must find the exact CPDAG or PAG.
- **Reference fixtures.** `reference/make_fixtures.py` runs the original ETIA and Tetrad in the old ETIA venv. It saves the outputs in `reference/fixtures/`. The tests compare the port with these outputs.
- **Exact PAG reference.** `tests/bruteforce.py` finds the exact PAG of a small MAG. It examines all MAGs that are consistent with the PAG.

### Known differences from Tetrad

The `DagToPag` class of the Tetrad jar in ETIA does not always give the exact PAG. On 13 of the 60 fixture graphs, it differs from ETIA-JAX:

- At 29 marks, Tetrad gives a tail and ETIA-JAX gives a circle. For each of these marks, the test finds a witness MAG in the equivalence class that has no tail at that mark. Thus the tail is not invariant.
- At 3 marks, Tetrad gives a circle and ETIA-JAX gives an arrowhead. An exhaustive search shows that every MAG in the class has an arrowhead at that mark.

The slow test `test_pag_exact_where_tetrad_differs` makes these checks (`pytest -m slow`, about 6 minutes).

## CI tests (increment 2)

`etia_jax.ci.CI_TESTS` has the 14 CI tests of the ETIA configuration, with the same names. Each test answers `test(x, y, Z) -> (statistic, p-value)`. The linear-Gaussian and the contingency-table tests also have `test_batch(xs, ys, Zs)`, which answers many queries with one `jax.vmap` call.

| Name | Origin in ETIA | Agreement with ETIA (60 queries per dataset) |
|---|---|---|
| `FisherZ`, `chisquare`, `gsquare`, `cg_lrt` | Tetrad 7.6.5 | exact (\|Δp\| < 1e-6) |
| `dg_lrt`, `cci` | Tetrad 7.6.5 | exact with `tetrad_compat=True` |
| `ParCor`, `RobustParCor`, `ParCorrWLS`, `Gsquared` | tigramite 5.2 | exact |
| `RegressionCI` | tigramite 5.2 | statistic within 1e-3 (sklearn stops at tol 1e-4) |
| `CMIknn` | tigramite 5.2 | same statistic (ETIA uses a fixed threshold) |
| `CMIsymb`, `GPDC` | tigramite 5.2 | same decisions; the null distributions are random |

### Known defects of Tetrad 7.6.5 (kept only in `tetrad_compat` mode)

The formulas come from the Tetrad 7.6.5 source (the version of `tetradlib.jar` in ETIA). ETIA-JAX does not copy the Java code. It implements the formulas in JAX.

- **CCI.** The kernel distance uses the rows (x, z1, ..., z(k-1)) in place of (z1, ..., zk). Thus the regression of x on Z partly uses x itself and omits the last variable of Z. Also, the correlation of the standardized basis values divides by n - 1 in place of n.
- **DG-LRT.** The covariance uses a mean of sum / (n - 1) with a divisor n. Tetrad also drops the indicator of the category that appears first latest in the row order. ETIA-JAX drops the smallest category in the default mode.

The default mode of ETIA-JAX uses the correct formulas.

## Plan of increments

Each increment has a build step, a test step and a check step.

1. Graph core: encoding, DAG utilities, d-separation, CPDAG, MAG and PAG conversions, metrics. GenJAX SCM sampler.
2. CI tests in JAX: FisherZ, partial correlation (with robust and weighted variants), G-square, chi-square, conditional Gaussian and discrete LRTs, RegressionCI, CMIknn, CMIsymb, GPDC, CCI.
3. Constraint-based search: PC and CPC.
4. FCI family: FCI, FCI-Max, RFCI, CFCI.
5. Scores in JAX (SEM-BIC, BDeu, CG-BIC, DG-BIC) and FGES. Then GFCI.
6. Time series: SVAR-FCI, SVAR-GFCI, PCMCI, PCMCI+ and LPCMCI.
7. NOTEARS and SAM (with optax).
8. Predictive models (MLP and linear, in place of the random forest) and AFS (FBED, SES).
9. OCT tuning, model validation protocols, and the CausalLearner.
10. CRV: queries, adjustment sets, bootstrap confidence, and visualization without Cytoscape.
