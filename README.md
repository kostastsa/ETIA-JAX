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

## Scores, FGES and GFCI (increment 5)

`etia_jax.scores` has the five scores of ETIA: `SemBicScore`, `BdeuScore`, `DiscreteBicScore`, `ConditionalGaussianScore` and `DegenerateGaussianScore`. `ETIA_SETTINGS` gives the settings that ETIA uses. A score gives `local_score(i, parents)`, and a cache keeps each local score. The SEM BIC score uses stacked numpy solves for small batches and one JAX call for large batches. The CG BIC score uses the vmapped kernel of the CG-LRT test.

- **Agreement of the scores.** For random queries on the search datasets, the local scores of ETIA-JAX and of Tetrad have a relative difference below 2e-12. The NaN values and the degree limits are also the same.

`etia_jax.search.fges.fges(score)` returns a CPDAG. `etia_jax.search.gfci.gfci(ci, score, alpha)` returns a PAG.

- **Default mode.** FGES uses the same three rounds of forward steps and backward steps as Tetrad, but with index orders and no degree limit. GFCI follows Ogarrio et al. (2016): FGES, the removal of extra edges, the GFCI collider rule, the Possible-D-SEP step of FCI, and the rules of Zhang. For a pair that FGES did not connect, R4 needs a separating set. Thus the default mode finds one when R4 asks for it. With a d-separation score and d-separation tests, FGES gives the correct CPDAG, and GFCI gives the correct PAG.
- **`tetrad_compat=True`.** A Python version of Tetrad 7.6.5 `Fges`, `Bes`, `MeekRules` and `GFci`, on `TGraph`. FGES gives the same CPDAG as ETIA in all 69 fixture searches.
- **Run time.** FGES of ETIA-JAX is 3 to 6 times slower than Tetrad on continuous data, because the search runs in Python. GFCI of ETIA-JAX is faster than Tetrad on 20 to 50 variables (for example 0.29 s and 2.15 s on 30 variables), because the CI tests run in batches.

### Agreement of GFCI with ETIA

- **The steps before R4.** FGES, the removal of extra edges and the GFCI collider rule are deterministic in Tetrad. ETIA-JAX gives the same graph after these steps in all 162 searches.
- **The last step.** As in FCI, Tetrad starts its R4 tasks in parallel on a hash set without hash codes. Thus 43 searches have more than one possible PAG (up to 7). The PAG of ETIA-JAX is one of the PAGs of Tetrad in all 162 searches. Some searches have the same data, test, alpha and graph before R4. These searches have the same possible PAGs, thus the test uses all their outputs.
- **Discrete data.** The search for separating sets of GFCI uses the CI test from parallel threads. Thus GFCI with the chi-square or G-square test stops with an error or gets incorrect p-values. The fixtures use a changed Tetrad that sends the test calls through a lock (`reference/make_fixtures_serial.py gfci`).

### What Tetrad 7.6.5 does in the scores, FGES and GFCI (kept only in `tetrad_compat` mode)

- **Degree limit of FGES.** FGES takes the degree limit of the score. For most scores, the limit is ceil(ln n). For the discrete BIC score, the limit is 1000. GFCI sets no limit.
- **BDeu structure prior.** The prior uses N - 1, where N is the number of rows. It must use the number of other variables.
- **Discrete BIC structure prior.** The prior term has the wrong sign. Thus more parents give a higher score.
- **Separating sets of GFCI (`SepsetsMinP`).** The search takes the subset of adj(x) with the smallest p-value. It keeps this subset only if its p-value is above alpha. Thus all the subsets must show independence. The second search uses adj(x) again. Thus the search does not use adj(y).
- **No Possible-D-SEP step in GFCI.** The FGES of GFCI also assumes faithfulness, thus it does only two rounds.
- **GFCI collider rule.** The rule also looks at a pair x, z that is adjacent in the CPDAG and in the graph. It searches for a separating set of x and z. If it finds one, it removes the edge.

## Time series (increment 6)

ETIA has two kinds of time-series algorithms. ETIA-JAX has both.

- **Tetrad algorithms with tiers.** With `time_lagged=True`, ETIA gives all its Tetrad algorithms a lagged table and tier knowledge. The table has the columns X1..Xp at time t, then X1:1..Xp:1 at time t - 1, and so on (`etia_jax.data.svar.lagged_table`). The knowledge puts each lag in one tier. An edge from a later tier into an earlier tier is forbidden (`etia_jax.search.knowledge.Knowledge.time_lagged`). PC, CPC, the FCI family, FGES and GFCI take `knowledge=`. SVAR-FCI and SVAR-GFCI are in `etia_jax.search.svar`.
- **Tigramite algorithms.** `etia_jax.search.pcmci` has `pcmci` and `pcmciplus`, and `etia_jax.search.lpcmci` has `lpcmci`. Each takes the raw series (T x N), the name of a CI test of ETIA, `tau_max` and `pc_alpha`. The result has the tigramite graph and the mark matrix of ETIA (`etia_matrix`). One lagged matrix holds the rows of tigramite (t = 2 tau_max .. T - 1), and the CI tests of increment 2 run on it in batches.
- **Data.** `etia_jax.data.svar` makes structural VAR time series with GenJAX (continuous, discrete or mixed). `window_dag` gives the true DAG of the lagged window, with the earlier time steps as latent variables.

### Agreement with ETIA

- **Tetrad algorithms.** All 448 reference searches agree (`reference/fixtures/ts_tetrad_ref.npz`). Some searches have more than one possible output, because Tetrad runs R4 in parallel. For 8 SVAR-GFCI searches, two score changes are equal except for rounding. Tetrad orders them by its own rounding, so the test also accepts the other order (`tie_seed`).
- **Tigramite algorithms.** PCMCI gives the same graph as tigramite in 84 of 85 searches, PCMCI+ in 83 of 85, and LPCMCI in 85 of 85. The 3 other searches use GPDC or CMIsymb. In tigramite, these tests get their p-values from random draws, so a decision near alpha can change.
- **Default mode.** The tests use d-separation oracles on time-series DAGs. The skeleton is correct, and each oriented mark is true. The output is at least as informative as the PAG or the CPDAG.

### What ETIA, Tetrad and tigramite do (kept only in `tetrad_compat` mode where it applies)

- **ETIA** does not make the lagged table. `Dataset.convert_to_time_lag` is empty, so the user must give the lagged columns. ETIA lets the tigramite algorithms run only on raw series.
- **Knowledge in Tetrad.** The Meek rule R4 stays off, because MeekRules decides `useRule4` before it gets the knowledge. FCI-Max, RFCI and CFCI give an empty knowledge to their final orientation. Only their first orientation step uses the tiers.
- **SVAR-FCI and SVAR-GFCI** copy each edge removal and each mark to the "similar pairs" at the other lags. A mark is copied only when the first node is in the later tier. SVAR-GFCI uses its own older FGES class (`SvarFges`), which also copies each edge.
- **Tier order.** `Knowledge.getTier` sorts the names as strings. Thus X10:1 comes before X1:1, and with 10 or more variables the positions of a variable in two tiers can be different.
- **Tigramite CMIsymb** stops with a numba error when the conditioning set is not empty. In ETIA, this stops the search.

`etia_jax/search/lpcmci.py` is derived from tigramite 5.2 (GNU GPL v3.0). The [license](#license) of ETIA-JAX agrees with it.

## NOTEARS and SAM (increment 7)

ETIA has two algorithms that use continuous optimization. ETIA-JAX has both, with optax.

- **NOTEARS** (`etia_jax.search.notears.notears`). ETIA calls `from_pandas` of causalnex 0.12.2. ETIA-JAX does the same augmented Lagrangian loop. The inner problem uses an L-BFGS solver of optax with the stop rules of scipy L-BFGS-B, and a closed-form gradient. NOTEARS runs on the CPU by default (`device="cpu"`), because its small sequential steps are slower on the GPU.
- **SAM** (`etia_jax.search.sam.sam`). ETIA calls `SAM` of cdt 0.6.0 (torch). ETIA-JAX copies `run_SAM` step by step: the generators, the graph sampler, the discriminator, the losses, the sequence of the Adam steps and the DAG penalty. The 8 runs of cdt are one `vmap`, and the epochs are one `lax.scan`.
- **Modes.** `etia_compat=True` copies ETIA, causalnex and cdt (the list below). The default mode is the correct algorithm.

### Agreement with ETIA

- **NOTEARS.** causalnex has no random step, so the tests compare the weights and the graphs (`reference/fixtures/notears_ref.npz`, 17 datasets). On 16 datasets, the score of the two solutions agrees to 1e-3, and ETIA's graph is the same at each threshold of ETIA's grid. A test skips a threshold when a causalnex weight is within 0.05 of it. On `cont20_s3`, ETIA-JAX ends in another local optimum: its score is 0.25 % higher, and 10 marks are different (the test is an expected failure).
- **SAM.** SAM is random, and ETIA-JAX cannot make the random draws of torch. Thus the tests compare the accuracy (the AUROC of the weights against the correct DAG) on 6 datasets (`reference/fixtures/sam_ref.npz`). ETIA has one run per dataset (about 13 minutes each). For 10 seeds of ETIA-JAX, the ETIA value is in the range of ETIA-JAX on 5 datasets. The mean AUROC is 0.877 for ETIA and 0.856 for ETIA-JAX.

### Speed

- **NOTEARS (CPU).** With 10 variables, ETIA-JAX is 3 to 6 times faster than causalnex. With 20 and 30 variables, causalnex takes 2.5 to 9 minutes, and ETIA-JAX takes 1 to 6 seconds (83 to 210 times faster).
- **SAM (GPU).** With 1000 training and 500 test epochs, ETIA-JAX is 76 to 306 times faster than cdt on the GPU. With the defaults of ETIA (3000 and 1000 epochs, 10 variables, 500 rows), cdt takes 21 minutes and ETIA-JAX takes 12 seconds.

### What ETIA, causalnex and cdt do (kept only in `etia_compat` mode)

- **NOTEARS parameters.** ETIA gives only `threshold` to causalnex. The `max_iter` and `h_tol` of its grid have no effect: causalnex uses 100 and 1e-8. Thus the 27 points of the grid give 3 different results.
- **NOTEARS data.** causalnex does not center the data, and its score has no intercept. The default mode centers the data.
- **NOTEARS at threshold 0.** All weights stay. ETIA writes each edge into one mark matrix, and in a 2-cycle the later edge wins. The result is the complete DAG with j -> i for each i < j. The default mode removes the smallest weight on a cycle until no cycle stays.
- **SAM output.** ETIA keeps the averaged filters of cdt as a weighted DiGraph, with no threshold. cdt divides the sum over the test batches by the number of test epochs only. Thus the values go up to the number of batches per epoch. The default mode gives values in [0, 1] and the DAG of the values above 0.5.
- **SAM graph matrix.** `pywhy_graph_to_matrix` of ETIA writes all the edges of the DiGraph. The result is the complete DAG with j -> i for each i < j, for all data.
- **SAM with mixed data.** cdt reads the skeleton column at the index of the variable. It must read the first column of the variable. When a categorical variable comes first, a generator gets the wrong inputs. The default mode gives the correct inputs.
- **SAM processes.** cdt starts its runs in forked processes. When the calling program uses CUDA before the call, the runs stop with an error, and cdt waits with no end.
- **CausalLearner.** The k-fold protocol of ETIA expects three return values from an algorithm. NOTEARS and SAM return two, so the protocol stops with a ValueError. NOTEARS also uses all the rows, not the rows of the fold. Increment 9 handles the protocols.

## Feature selection and the AFS (increment 8)

ETIA's AFS (automated feature selection) runs FBED and SES of the R package MXM, then a predictive model on the selected columns. ETIA-JAX has no R code.

- **FBED and SES** (`etia_jax.afs.fs`). The modules copy the R functions that ETIA reaches with the Fisher test: `Rfast::cor.fbed` and `MXM:::cor.bsreg` for FBED, and `InternalSES` with `testIndFisher` for SES. They use NumPy (float64), because each step is a small regression.
- **Predictive models** (`etia_jax.afs.models`). `LinearRegression` is least squares with an intercept, as sklearn. `MLP` (JAX and optax) replaces the random forest of ETIA.
- **AFS** (`etia_jax.afs.afs.run_afs`). The folds of `KFold(5, shuffle=True, random_state=42)`, the 24 configurations of ETIA, the winner, the bootstrap, and the intersection of the selected sets. One FS result per fold serves all the models.
- **Modes.** `etia_compat=True` copies ETIA, MXM and Rfast (the list below). The default mode honors `folds_to_run` and uses BBC-CV ([Tsamardinos et al. 2018](#ref-tsamardinos2018)).
- **References.** MXM was archived on CRAN on 2026-11-03. `reference/install_mxm.R` installs MXM 1.5.8 into `~/R/library` for the fixtures.

### Agreement with ETIA

- **FBED and SES.** All 456 reference calls (19 datasets, 3 targets each, the 8 FS configurations of ETIA) give the same result as MXM in ETIA. For FBED, the sequence of the columns is also the same. In 58 of these calls, the R code of ETIA stops with one of the two errors below, and ETIA-JAX gives the same error.
- **AFS.** With the linear grid, all 4 reference runs give the same winner and the same columns as ETIA, and the bootstrap mean is in ETIA's interval. With the full grid, the MLP replaces the random forest, so the winner can be different. In the 4 reference runs, linear regression wins in both programs.
- **Oracle.** With exact-covariance data (50 DAGs), FBED finds the Markov blanket of the target. SES finds all the parents and children. A DAG is skipped when an edge of the target has a partial correlation below 0.01.

### Speed

- **One FS call.** ETIA writes a CSV file and starts R for each call (about 2 s). ETIA-JAX takes about 2 ms. The median speed-up is about 1000 times.
- **A complete AFS run.** With the linear grid, ETIA takes 12 s and ETIA-JAX 0.06 to 0.23 s. With the full grid, ETIA takes 20 s and ETIA-JAX 1.2 to 2.4 s (the MLPs). ETIA runs its configurations in parallel on 32 cores, and ETIA-JAX runs them one after the other.

### What ETIA, MXM and Rfast do (kept only in `etia_compat` mode)

- **Rfast `cor.fbed`.** When the forward phase selects exactly one column, `z` stays a vector, and the K phase stops with "'x' is not a matrix". ETIA uses K = 3 or 5. Thus FBED fails, and ETIA skips the fold.
- **ETIA's FBED script.** When FBED selects no column, MXM returns a result with no column `sel`. The script reads `fbed_object$res[, "sel"]` and stops with "subscript out of bounds". ETIA skips the fold, as for an empty selection.
- **Folds.** The protocol of ETIA has `folds_to_run: 2`, but all 5 folds run.
- **Bootstrap.** The "bootstrap bias correction" of ETIA is a plain bootstrap of the pooled test predictions of the winner. It does not correct the bias that comes from the selection of the winner. It runs in joblib workers without a seed, so its numbers change from run to run.
- **No selected column.** When the main target has no selected column, ETIA returns the selection of the last target of the list.

These details are in MXM itself, so both modes keep them:

- `cor.fbed` compares a Fisher z statistic with a t quantile, and the p-value of the first column uses n - 2 degrees of freedom.
- The backward phase of FBED uses `lm` with an intercept on the raw data.
- In SES, the univariate statistics keep their sign, and a tie of p-values goes to the larger statistic.
- SES conditions only on selected columns. Thus it finds all the parents and children of the target, and it can also keep a descendant.

## Tuning: the OCT and the CausalLearner (increment 9)

ETIA's CausalLearner tunes the causal configurations with OCT, the out-of-sample causal tuning of [Biza et al. (2020)](#ref-biza2020). Each configuration learns a graph on the training rows of each fold. Each node gets a predictive model on its Markov boundary, and the mutual information of the test predictions scores the configuration. OCTs then takes the sparsest configuration that a permutation test does not find worse than the best one.

- **Modules.** `etia_jax/oct/configs.py` (ETIA's tables of algorithms, tests and scores, and the configurator), `etia_jax/oct/algorithms.py` (one configuration on the rows of a fold), `etia_jax/oct/oct.py` (`run_oct`, in steps that the tests compare with ETIA one by one) and `etia_jax/oct/learner.py` (`CausalLearner`).
- **Graphs.** `etia_jax/graphs/markov.py` has the Markov boundary of a DAG or MAG (the nodes that a collider path connects to the target), one DAG of a CPDAG ([Dor and Tarsi 1992](#ref-dortarsi1992)), and one MAG of a PAG ([Zhang 2008](#ref-zhang2008), Theorem 2). `etia_jax/search/tetrad_members.py` ports Tetrad's random `dagFromCpdag` and `magFromPag`.
- **Random forest.** `etia_jax/learn/forest.py` grows CART trees level by level, in one `vmap` over the trees: bootstrap weights, exact splits, `max_depth`, `min_samples_leaf` and `max_features`. A classifier is a regression forest on one-hot targets (the squared error of a one-hot vector is the Gini impurity). The forest is the regressor of the OCT. In the AFS, it replaces the MLP of increment 8.
- **Modes.** `etia_compat=True` copies ETIA (the list below). `etia_n_jobs=None` copies ETIA's default with joblib workers, and `etia_n_jobs=1` copies ETIA's run without workers. The default mode takes the Markov boundary of the class: it is the same in all the graphs of the class ([Biza et al. 2020](#ref-biza2020), p. 6).

### Agreement with ETIA

ETIA's OCT gives other scores and winners in each run, also with `random_seed` and one job, because Tetrad draws its random MAGs in Java. Thus the tests compare each step with the inputs that ETIA's step had (24 stored ETIA runs: 6 datasets, linear regression or ETIA's forest, 1 or 8 workers):

- **Fold PAGs.** FCI and FCI-Max on the training rows of each fold give the same PAG as ETIA in all 120 cases (continuous, mixed and discrete data).
- **Markov boundaries, scores, p-values and choices.** The tests give ETIA's random MAGs, predictions and permutations to the steps of ETIA-JAX. All 8813 compared outputs are the same as ETIA's outputs, also the 24 chosen configurations.
- **Predictions.** Linear regression on ETIA's Markov boundaries gives ETIA's test predictions (difference below 1e-8).
- **Random graphs.** The draws of the ported `magFromPag` and `dagFromCpdag` agree with Tetrad's draws. The tests use 11 class graphs and 2000 draws each. The distance to Tetrad's draws is not larger than the distance between two samples of ETIA-JAX, and the frequency of each mark agrees.
- **Forest.** With ETIA's OCT settings, the test R^2 of the forest is less than 0.01 from scikit-learn's, and the predictions correlate above 0.99. With the full AFS grid, all 4 reference runs give the same winner and the same columns as ETIA.
- **Oracle.** In the default mode, the score of the chosen configuration is less than 0.02 below the score of the true Markov boundaries.

### Speed

The times are for one complete OCT run in `etia_compat` mode with the settings of the stored ETIA run (10 folds, 3 folds to run, 50 permutations). The ETIA-JAX time is the time of a second run (the first run compiles).

- **Continuous data, 10 nodes.** ETIA takes 3 to 16 s with linear regression and 44 to 62 s with its forest and one job (7 to 8 s with all cores). ETIA-JAX takes 0.3 to 0.7 s with linear regression and 0.5 to 1 s with the forest on the GPU (6 to 129 times faster).
- **Larger and other data.** With 20 nodes, and with mixed or discrete data, the FCI searches in Python take most of the time (2 to 13 s). Thus ETIA-JAX is 1 to 14 times faster on continuous and mixed data. On discrete data, it is about as fast as ETIA (0.8 to 1 times) with linear regression or with all cores.
- **CPU.** Without the forest, the CPU is faster than the GPU (0.15 to 7 s). The forest needs the GPU: on the CPU, one run takes 17 to 117 s.
- **AFS.** With the forest instead of the MLP, a complete AFS run with the full grid takes 0.25 to 0.7 s after the first compilation. ETIA takes 20 s.

### What ETIA and Tetrad do (kept only in `etia_compat` mode)

- **Random MAG.** The Markov boundaries come from one random DAG or MAG of Tetrad. `magFromPag` can set a tail at both ends of an edge, and ETIA's `markov_boundary` does not see such a neighbor. FGES has `causal_sufficiency: [False]` in ETIA's table. Thus its CPDAG also goes through `magFromPag` and loses its undirected neighbors.
- **joblib workers.** With more than one job (ETIA's default: all cores), the workers change copies of the OCT object. The Markov boundary sizes stay 0 and the p-values stay 1. Thus the permutation test and the sparsity rule have no effect, and the winner is the configuration with the best score.
- **Last graph.** The chosen configuration keeps the `indexes` of the last fold. Thus the final graph comes from the training rows of that fold, not from all the rows.
- **Empty Markov boundary.** The node gets uniform random predictions between the minimum and the maximum of its training values.
- **Categorical nodes.** The regressor predicts the category codes, and `mutual_info_score` takes each predicted value as a label.
- **Two methods.** `get_best_model_between_algorithms` stops with a `ValueError` (three values into two names). `get_best_model_between_family` compares a list with a bool. Thus it finds no algorithm (`RuntimeError`), and `accepts_missing_values` reads a key that the table does not have (`KeyError`).
- **NOTEARS, SAM and tigramite.** Their wrappers return two values, and the k-fold protocol expects three. Thus the OCT stops with a `ValueError`. The default mode runs NOTEARS and SAM on the rows of the fold. The OCT of ETIA-JAX does not have the time-series algorithms.

Other findings, not part of the modes:

- **Global configuration.** `Configurations` changes ETIA's global `default_conf`. A second CausalLearner in the same Python session keeps the algorithms (and the algorithm objects) of the first dataset. ETIA-JAX makes a new table for each learner.
- **Default grid.** ETIA's table gives `pc` and `cpc` `assume_faithfulness: [False]` and `fges` `causal_sufficiency: [False]`. Thus ETIA's default settings run only FCI and FCI-Max (3 alphas each) on continuous data.
- **Tetrad threads.** R0 of Tetrad's FCI runs the chi-square or G-square test in a parallel stream. The test is not thread-safe. ETIA's OCT on `disc10_s0` stopped with Java errors in all 4 tries, also with one processor. With one processor, `IndTestFisherZ` divides by zero (`availableProcessors() / 2` threads). The discrete reference puts a lock around each test.

## The CRV: queries, adjustment sets and confidence (increment 10)

ETIA's CRV works on the graph that the CausalLearner finds. It has path queries, adjustment sets (through dagitty in R), the bootstrap confidence of edges and paths, and drawings in Cytoscape. ETIA-JAX has no R code and no Cytoscape.

- **Modules.** `etia_jax/crv/queries.py` (four path queries), `etia_jax/crv/dagitty.py` (a port of dagitty 0.3.4 and of pcalg's `pcalg2dagitty`), `etia_jax/crv/adjustment.py` (`find_adjset`), `etia_jax/crv/confidence.py` (bootstrap graphs, edge and path weights) and `etia_jax/crv/visualization.py` (SVG drawings, a force-directed layout of [Fruchterman and Reingold (1991)](#ref-fruchterman1991)).
- **Adjustment sets.** dagitty constructs the sets of the generalized adjustment criterion of [Perković et al. (2018)](#ref-perkovic2018). The port keeps dagitty's data structures (the key order of JavaScript objects, the edge lists of the vertices, and the sorted text that R sends). Thus it lists the minimal sets in dagitty's order, and ETIA takes the first one.
- **Modes.** `etia_compat=True` copies ETIA (the list below). The default mode keeps all the edges, follows all the possible edges in a query, and uses the consistency of the end marks.

### Agreement with ETIA

- **Adjustment sets.** The tests use 400 random graphs (100 each of DAG, CPDAG, MAG and PAG). The canonical set, the first minimal set and the number of minimal sets are the same as dagitty 0.3.4 in R with ETIA's steps.
- **Path queries and edges.** All 4800 path queries and all 400 edge tables (`matrix_to_cyto`) are the same as ETIA's.
- **Confidence.** Tetrad's FCI in ETIA and ETIA-JAX give the same graph on all 150 bootstrap samples (3 datasets, the same rows). The edge weights and the path metrics are also the same.
- **Oracle.** In the default mode, all 304 adjustment sets of 240 random linear models give the true total effect (exact covariances). For a DAG, when no set is found, no subset of the other variables gives the effect.

### Speed

- **Adjustment set.** ETIA starts R for each set (about 0.31 s). ETIA-JAX takes about 0.6 ms (about 490 times faster).
- **Bootstrap.** FCI on 50 bootstrap samples takes 2 to 6 s in ETIA, and 0.2 to 1.3 s in ETIA-JAX on the CPU (5 to 10 times faster). The FCI search runs in Python, so the GPU is slower.
- **Path metrics.** ETIA-JAX compares all the edges of all the bootstrap graphs in one table look-up: 18 to 38 times faster.
- **Path queries.** One query takes about 10 µs in the two programs.

### What ETIA does (kept only in `etia_compat` mode)

- **CPDAG conversion.** For a DAG or CPDAG, ETIA keeps only the marks 1 and 2 before it calls dagitty. Thus the undirected edges of a CPDAG are lost. On 60 random CPDAGs, 19 of ETIA's 104 sets give a wrong total effect.
- **`--o` edges.** pcalg's `pcalg2dagitty` has no case for a tail and a circle. Thus it drops these edges of a PAG.
- **Potentially directed path.** The query follows `o->`, `-->` and `o-o`, but not `--o` or `---`. In 1200 queries it misses 87 of 392 paths. The depth-first search keeps only the current path, so it can visit a node many times.
- **Edge consistency.** ETIA's table is stricter than the end marks in 29 of 100 mark pairs: a tail against a circle, the undirected CPDAG edge, and `o->` against `<-o`.
- **Path consistency.** The loop stops at the first edge that is not the same. When this edge is consistent, the path counts as consistent, and the later edges are not checked.

Other findings, not part of the modes:

- **R call.** `subprocess.call([...], shell=True)` on Linux and macOS runs only "R", without the script. Thus `find_adjset` cannot run there. The R script also changes its folder from a guess about the current folder, and it loads pcalg, which ETIA does not install.
- **`calculate_confidence`.** It resamples the rows, but it runs the search on the original data. It gives a DataFrame where `model.run` needs a Dataset, and it reads attributes that a dict does not have. Thus it cannot run. Both modes of ETIA-JAX draw real bootstrap samples.
- **Drawing.** `plot_edge_confidence` and `hide_edges` read the columns `edge_confidence` and `confidence`, which the Cytoscape network does not have.

## Plan of increments

The details and the status of each increment are in [docs/increments.md](docs/increments.md).

Each increment has a build step, a test step and a check step.

1. Graph core: encoding, DAG utilities, d-separation, CPDAG, MAG and PAG conversions, metrics. GenJAX SCM sampler.
2. CI tests in JAX: FisherZ, partial correlation (with robust and weighted variants), G-square, chi-square, conditional Gaussian and discrete LRTs, RegressionCI, CMIknn, CMIsymb, GPDC, CCI.
3. Constraint-based search: PC and CPC.
4. FCI family: FCI, FCI-Max, RFCI, CFCI.
5. Scores in JAX (SEM-BIC, BDeu, CG-BIC, DG-BIC) and FGES. Then GFCI.
6. Time series: SVAR-FCI, SVAR-GFCI, PCMCI, PCMCI+ and LPCMCI.
7. NOTEARS and SAM (with optax).
8. Predictive models (linear, MLP; the random forest of increment 9 replaces the MLP) and AFS (FBED, SES).
9. OCT tuning, model validation protocols, and the CausalLearner.
10. CRV: queries, adjustment sets (a port of dagitty), bootstrap confidence, and SVG drawings without Cytoscape.

## License

ETIA-JAX is free software. You can distribute it and change it under the terms of the GNU General Public License, version 3 or a later version. The full text is in [LICENSE](LICENSE).

ETIA-JAX copies the behavior of these programs. The licenses of the programs are:

| Program | License | Used in |
|---|---|---|
| ETIA | MIT (from its README) | all increments |
| Tetrad 7.6.5 | GNU GPL v3 | increments 1 to 6 and 9 (random DAG and MAG) |
| libdagitty (the code of dagitty 0.3.4) | GNU GPL v3 (the R package: GPL-2) | increment 10 (`crv/dagitty.py` is derived from libdagitty) |
| pcalg 2.7-12 | GNU GPL v2 or later | increment 10 (`pcalg2dagitty` in `crv/dagitty.py` and `reference/adjset_dagitty.R`) |
| tigramite 5.2 | GNU GPL v3 | increment 6 (`lpcmci.py` is derived from it) |
| causalnex 0.12.2 | Apache 2.0 (from the header of `notears.py`) | increment 7 (NOTEARS) |
| cdt 0.6.0 | MIT | increment 7 (SAM) |

## References (outside the corpus)

- <a id="ref-biza2020"></a>Biza, K., Tsamardinos, I. and Triantafillou, S. (2020). Tuning causal discovery algorithms. *Proceedings of the 10th International Conference on Probabilistic Graphical Models*, PMLR 138, 17–28. [proceedings.mlr.press/v138/biza20a](https://proceedings.mlr.press/v138/biza20a.html). Algorithm 3 (OCT) and the OCTs variant are on pp. 6–7. The paper does not define the mutual information score or the permutation test; ETIA-JAX takes them from ETIA's code.
- <a id="ref-dortarsi1992"></a>Dor, D. and Tarsi, M. (1992). A simple algorithm to construct a consistent extension of a partially oriented graph. Technical Report R-185, Cognitive Systems Laboratory, UCLA. The report was not read; the algorithm (remove a potential sink, one at a time) is as later papers describe it.
- <a id="ref-zhang2008"></a>Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. *Artificial Intelligence* 172(16–17), 1873–1896. [doi:10.1016/j.artint.2008.08.001](https://doi.org/10.1016/j.artint.2008.08.001). The number "Theorem 2" comes from Tetrad's `zhangMagFromPag`; the statement was not checked in the paper.
- <a id="ref-fruchterman1991"></a>Fruchterman, T. M. J. and Reingold, E. M. (1991). Graph drawing by force-directed placement. *Software: Practice and Experience* 21(11), 1129–1164. [doi:10.1002/spe.4380211102](https://doi.org/10.1002/spe.4380211102). Only the bibliographic record and summaries were read; the layout of ETIA-JAX uses the forces k²/d and d²/k with a falling temperature.
- <a id="ref-perkovic2018"></a>Perković, E., Textor, J., Kalisch, M. and Maathuis, M. H. (2018). Complete graphical characterization and construction of adjustment sets in Markov equivalence classes of ancestral graphs. *Journal of Machine Learning Research* 18(220), 1–62. [jmlr.org/papers/v18/16-319](https://jmlr.org/papers/v18/16-319.html). The abstract says that the criterion is sound and complete for DAGs, CPDAGs, MAGs and PAGs, and that dagitty has the algorithms; ETIA-JAX ports dagitty's code, not the paper.
- <a id="ref-tsamardinos2018"></a>Tsamardinos, I., Greasidou, E. and Borboudakis, G. (2018). Bootstrapping the out-of-sample predictions for efficient and accurate cross-validation. *Machine Learning* 107(12), 1895–1922. [doi:10.1007/s10994-018-5714-4](https://doi.org/10.1007/s10994-018-5714-4)

