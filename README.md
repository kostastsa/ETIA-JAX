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

## PC and CPC (increment 3)

`etia_jax.search.pc` and `etia_jax.search.cpc` take a CI test or the d-separation oracle. They return the graph (a mark matrix), the skeleton with its separating sets, and the number of CI queries.

```python
from etia_jax.ci import FisherZ
from etia_jax.search import pc
res = pc(FisherZ(X), alpha=0.05)                       # PC-stable (Colombo and Maathuis 2014)
res = pc(FisherZ(X), alpha=0.05, tetrad_compat=True)   # the same graph as ETIA (Tetrad 7.6.5)
```

- **Batched levels.** At the start of each depth, the search finds all the queries that are possible at that depth. It sends them to the CI test in one `test_batch` call. Then it replays the sequential logic from a cache. The result is the same as with one query at a time (a test checks this).
- **Agreement with ETIA.** With `tetrad_compat=True`, PC and CPC give the same graph as ETIA on all 116 fixture searches (10 to 100 variables, FisherZ, chi-square, G-square, CG-LRT and DG-LRT, alpha 0.05 and 0.01).
- **Compiled code on disk.** JAX keeps its compiled code in `~/.cache/etia_jax/jax`. Set `ETIA_JAX_CACHE_DIR` to change the folder. An empty value disables the cache.

### What Tetrad 7.6.5 does in PC and CPC (kept only in `tetrad_compat` mode)

- **The "stable" FAS is not stable.** At each depth, the FAS copies the adjacencies, but then it changes that copy. Later pairs thus see the removals of earlier pairs, and the result changes with the sequence of the variables. ETIA asks for `stable=True`. Thus the default of ETIA-JAX is the PC-stable algorithm of Colombo and Maathuis.
- **colliderAllowed.** Tetrad does not orient a collider x -> y <- z when y is an ancestor of z. A typo makes it test z two times, and it does not test x. The default of ETIA-JAX tests x and z.
- **CPC.** Tetrad looks for separating sets in adj(x) or adj(z) only when that set has two or more nodes. Thus a collider x -> y <- z, where x and z have only y as adjacent, stays ambiguous. The default of ETIA-JAX uses each side that has one or more nodes.
- **The sequence of the orientation steps.** Tetrad visits the adjacent nodes and the edges in the sequence of Java hash sets. The collider rules and the Meek rules give different results for a different sequence. Thus ETIA-JAX copies this sequence in `tetrad_compat` mode.
- **The cache of the CI tests.** Some tests are not symmetric (CG-LRT, DG-LRT). For these tests, the cache of Tetrad gives a different answer for each direction of a pair. ETIA-JAX does the same.

## FCI, FCI-Max, RFCI and CFCI (increment 4)

`etia_jax.search.fci` has `fci`, `fcimax`, `rfci` and `cfci`. Each takes a CI test or the d-separation oracle and returns a PAG (a mark matrix).

- **Default mode.** The textbook algorithms: FAS (PC-stable), R0 with the separating sets, the Possible-D-SEP step, and the rules R1 to R4 and R8 to R10 of Zhang (2008). FCI-Max finds the colliders with the Max-P rule. CFCI uses the conservative rule. RFCI uses the R0 of Colombo et al. (2012). With the d-separation oracle, FCI, FCI-Max and CFCI give the correct PAG, and RFCI is sound. The Possible-D-SEP step uses only the nodes of the biconnected component of the edge (Colombo et al. 2012). For each edge, it makes the subsets of each size as one index array. It tests these subsets in chunks that become larger. It stops at the first chunk that contains a separating set. The results do not go into the cache. On `cont50_s1`, this step makes 14.6 million tests in 2.2 s.
- **`tetrad_compat=True`.** A Python version of Tetrad 7.6.5 `Fci`, `FciMax`, `Rfci` and `Cfci` with the defaults that ETIA uses. It runs on `TGraph`, an emulation of the `EdgeListGraph` of Tetrad. `TGraph` copies the iteration sequence of the Java hash sets. A test applies random graph operations in Tetrad and in `TGraph`, and the sequences are the same.

### Agreement with ETIA

- **Continuous and mixed data.** Tetrad gives the same result each time. ETIA-JAX gives the same PAG in all searches.
- **Discrete data.** In Tetrad 7.6.5, FCI, FCI-Max, RFCI and CFCI use one CI test object from parallel threads. The chi-square and G-square tests are not safe for this. Thus FCI stops with an error in 11 of 12 searches, and other searches can get incorrect p-values. `reference/make_fixtures_serial.py` sends the calls of the test through a lock. With this change, Tetrad gives the same result each time in 88 of 96 searches, and ETIA-JAX gives the same PAG in all 88.
- **The other 8 searches.** In Tetrad, R4 starts its tasks in parallel on a hash set without hash codes. Thus the sequence of the tasks can be different in two searches, and these 8 searches have two possible outputs. The output of ETIA-JAX is one of the outputs of Tetrad in all 8.

### What Tetrad 7.6.5 does in the FCI family (kept only in `tetrad_compat` mode)

- **R0 of FCI** finds a new separating set with a greedy search. It does not use the separating sets of FAS.
- **R4** uses a search for separating sets that blocks paths. A defect in `DiscriminatingPath` puts the nodes of the path in the opposite sequence. Thus R4 accepts only the shortest discriminating paths.
- **The rule schedule.** Tetrad uses R4 only when R1, R2 or R3 changed a mark in the same cycle. R5, R6 and R7 (the rules for selection bias) also apply.
- **R10** does not change marks. The code compares a list of edges with one edge, and this is always false.
- **The possible-m-sep step** of FCI starts its search only from x. CFCI uses a different possible-m-sep step that tests only the side of x.
- **CFCI** keeps its own map of separating sets. At the first R0 this map is empty.

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
