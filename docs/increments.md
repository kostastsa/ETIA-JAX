# ETIA-JAX increments

ETIA-JAX is a port of ETIA to Python and JAX, without Java and R. The work has 10 increments. Each increment has a build step, a test step and a check step. Each increment also gives a demo with a timing report of ETIA against ETIA-JAX.

## Rules for all increments

- **Two modes.** Each algorithm that ETIA takes from Tetrad 7.6.5 has two modes:
  - `tetrad_compat=True`: the same output as ETIA. This mode copies the Tetrad defects and the iteration sequence of the Java hash sets.
  - Default: the correct algorithm of the paper. The d-separation oracle tests this mode.
- **References.** `reference/make_fixtures.py` runs ETIA and Tetrad in the ETIA venv (Python 3.10, Java 17). The tests compare ETIA-JAX with these outputs.
- **Defects.** The README lists each Tetrad defect that ETIA-JAX found. The default mode does not have these defects.
- **Demos.** Each demo has a plate (plateview) or a page (pageview). Each chart and table has a "How to read" text in STE.
- **Computer.** All times come from algo29: Intel i9-14900K, RTX 4090, float64.

## Status

| Increment | Content | Status | Tests | Commit |
|---|---|---|---|---|
| 1 | Graph core and GenJAX data | done | 17 | `bd4402c` |
| 2 | The 14 CI tests of ETIA | done | 43 | `bd4402c` |
| 3 | PC and CPC | done | 53 | `726d050` |
| 4 | FCI, FCI-Max, RFCI and CFCI | done | 61 | `6d6264c` |
| 5 | Five scores, FGES and GFCI | done | 130 | `e219726` |
| 6 | Time series: tiers, SVAR-FCI, SVAR-GFCI, PCMCI family | done | 139 | `f08958a` |
| 7 | NOTEARS and SAM | done | 167 | `ba4da9b` |
| 8 | Predictive models and AFS | done | 632 | this commit |
| 9 | OCT and CausalLearner | next | | |
| 10 | CRV | planned | | |

The same table is in `deck/increments.csv` (the deck reads it). The "Tests" column gives the number of tests in the suite at the end of the increment (slow tests included).

## Increment 1: graph core and GenJAX data

- **Modules.** `etia_jax/graphs` (encoding, DAG utilities, d-separation, CPDAG, MAG and PAG conversions, metrics) and `etia_jax/data/scm.py` (GenJAX SCMs).
- **Agreement with Tetrad.** On 60 random graphs, the CPDAGs and the MAGs are the same as in Tetrad. The PAGs are the same on 47 graphs.
- **Tetrad defect.** On 13 graphs, `DagToPag` of Tetrad is not exact. It gives 29 tails that are not invariant, and it misses 3 arrowheads. A search of all the MAGs of the class shows this.
- **Data.** GenJAX samples the linear SCM. The weights that a regression finds have a maximum error of 0.028.
- **Demo.** A pageview page: `demo/increment1/increment1.md`.

## Increment 2: the 14 CI tests

- **Modules.** `etia_jax/ci`: FisherZ, ParCor, RobustParCor, ParCorrWLS, chi-square, G-square, Gsquared, CMIsymb, CG-LRT, DG-LRT, RegressionCI, CMIknn, GPDC and CCI.
- **Batches.** The linear-Gaussian tests and the contingency-table tests answer many queries with one `jax.vmap` call.
- **Agreement with ETIA.** The tests of Tetrad and the parametric tests of tigramite give the same p-values (difference below 1e-6). The random tests (CMIsymb, GPDC) give the same decisions.
- **Tetrad defects.** CCI uses the wrong rows in its kernel distance. DG-LRT uses a wrong mean and drops a different category.
- **Speed.** The queries of the timing set take 1,459 s in ETIA. In ETIA-JAX, they take 28 s on the GPU and 105 s on the CPU. The largest speed-up is 586 times (GPU).
- **Demo.** Plate `9bafa2416f53`.

## Increment 3: PC and CPC

- **Modules.** `etia_jax/search/pcalg.py`, `cache.py` and `tetrad_graph.py`.
- **Method.** At each depth, the search sends all the possible queries in one batch. Then it replays the sequential logic from a cache.
- **Agreement with ETIA.** PC and CPC give the same graph as ETIA in all 116 fixture searches (10 to 100 variables, five CI tests, two values of alpha).
- **Tetrad defects.**
  - The "stable" FAS is not stable.
  - A typo in `colliderAllowed` tests z two times and does not test x.
  - CPC does not search a side that has only one node.
  - The orientation steps follow the sequence of the Java hash sets.
- **Speed.** On 100 variables, PC is 3.3 times faster and CPC is 6.1 times faster than ETIA.
- **Demo.** Plate `944c7c26d4e2`.

## Increment 4: the FCI family

- **Modules.** `etia_jax/search/fci.py`. `TGraph` copies `EdgeListGraph` of Tetrad, with the same iteration sequence.
- **Agreement with ETIA.** All 216 searches agree with ETIA, or with one of the outputs of a race-free Tetrad.
- **Tetrad defects.**
  - Tetrad uses one CI test object from parallel threads. Thus FCI crashes on 11 of 12 discrete searches.
  - R4 starts its tasks in parallel in a random sequence. Thus 12 searches have more than one possible PAG.
  - R4 accepts only the shortest discriminating paths, because a defect reverses the path.
  - R10 does not change marks.
  - FCI and CFCI use a different possible-m-sep step.
- **Speed.** The median speed-up is 1.6 times. On one dataset with 50 variables, the textbook Possible-D-SEP step makes 14.6 million tests. A chunked search with early stop does these tests in 2.2 s.
- **Demo.** Plate `1d60395c2236`.

## Increment 5: scores, FGES and GFCI

- **Modules.** `etia_jax/scores` (SEM BIC, BDeu, discrete BIC, CG BIC and DG BIC), `etia_jax/search/fges.py` and `etia_jax/search/gfci.py`.
- **Agreement with ETIA.**
  - **Scores.** 6,840 local scores have a relative difference below 1.6e-12.
  - **FGES.** All 69 CPDAGs are the same as in ETIA.
  - **GFCI.** In all 162 searches, the graph before R4 is the same as in Tetrad, and the PAG is one of the Tetrad outputs.
- **Oracle.** With a d-separation score, FGES gives the correct CPDAG, and GFCI gives the correct PAG.
- **Tetrad defects.**
  - FGES uses the degree limit of the score: ceil(ln n).
  - The BDeu structure prior uses the number of rows.
  - The discrete BIC structure prior has the wrong sign.
  - The separating sets of GFCI (`SepsetsMinP`) must show independence for all subsets, and the search does not use adj(y).
  - GFCI has no Possible-D-SEP step.
  - GFCI crashes in 30 discrete searches (parallel threads).
  - R4 gives up to 7 possible PAGs.
- **Speed.** GFCI is up to 15 times faster than ETIA. FGES is 3 to 6 times slower than ETIA on continuous data, because its search runs in Python.
- **Demo.** Plate `9149fa95e827`.

## Increment 6: time series

- **Modules.** `etia_jax/data/svar.py` (GenJAX SVAR data, the lagged table of ETIA), `etia_jax/search/knowledge.py` (tier knowledge), `etia_jax/search/svar.py` (SVAR-FCI, SVAR-GFCI), `etia_jax/search/pcmci.py` (PCMCI, PCMCI+) and `etia_jax/search/lpcmci.py` (LPCMCI).
- **Agreement with ETIA.**
  - **Tetrad algorithms with tiers.** All 448 searches of the ten Tetrad algorithms agree with Tetrad. For 8 SVAR-GFCI searches, Tetrad orders equal score changes by its own rounding, and the test also accepts the other order.
  - **Tigramite algorithms.** 252 of 255 searches give the same graph as tigramite. The 3 other searches use GPDC or CMIsymb, which get their p-values from random draws.
- **Oracle.** In the default mode with tiers, all 400 oracle cases of the ten algorithms are correct.
- **Findings.**
  - ETIA does not make the lagged table, and it runs the tigramite algorithms only on raw series.
  - In Tetrad, Meek R4 stays off with knowledge. FCI-Max, RFCI and CFCI give an empty knowledge to their final orientation.
  - SVAR-GFCI uses an older FGES class. The similar-pair copies of a mark apply in one direction only.
  - Tigramite CMIsymb stops with an error when the conditioning set is not empty.
- **Speed.** The PCMCI family of ETIA-JAX has a median speed-up of 10 times. With CMIsymb, one search takes 25 to 105 minutes in tigramite and seconds in ETIA-JAX. The Tetrad family has a median speed-up of 1.2 times.
- **Demo.** Plate `603633780311`.

## Increment 7: NOTEARS and SAM

- **Modules.** `etia_jax/search/notears.py` (NOTEARS, an L-BFGS solver of optax) and `etia_jax/search/sam.py` (SAM, optax, the 8 runs in one `vmap`).
- **Agreement with ETIA.**
  - **NOTEARS.** causalnex has no random step. On 16 of 17 datasets, the score of the two solutions agrees to 1e-3. At the thresholds of ETIA, 40 of 41 graphs are the same. On one dataset, ETIA-JAX ends in another local optimum.
  - **SAM.** SAM is random, so the check compares the AUROC against the correct DAG. For 10 seeds of ETIA-JAX, the one ETIA run is in the range on 5 of 6 datasets. The mean AUROC is 0.877 for ETIA and 0.856 for ETIA-JAX.
- **Findings.**
  - ETIA gives only `threshold` to causalnex. Its values of `max_iter` and `h_tol` have no effect.
  - At threshold 0, the NOTEARS graph of ETIA is the complete DAG j -> i (i < j). The SAM graph matrix of ETIA is this DAG for all data.
  - cdt divides the SAM filters by the number of test epochs only, so the values go up to the number of batches.
  - With mixed data, cdt gives some SAM generators the wrong inputs.
  - The k-fold protocol of ETIA stops on NOTEARS and SAM, because they return two values, not three.
- **Speed.** NOTEARS on the CPU is 3 to 6 times faster than causalnex with 10 variables. With 20 and 30 variables, it is 83 to 210 times faster. SAM on the GPU is 76 to 306 times faster than cdt on the GPU. With the defaults of ETIA, one SAM search takes 21 minutes in ETIA and 12 seconds in ETIA-JAX.
- **Demo.** Plate `014ea6c8c2af`.

## Increment 8: predictive models and AFS

- **Modules.** `etia_jax/afs/fs.py` (FBED and SES of MXM 1.5.8, with the Fisher test), `etia_jax/afs/models.py` (linear regression, MLP) and `etia_jax/afs/afs.py` (the AFS pipeline). No R code.
- **Agreement with ETIA.**
  - **FBED and SES.** All 456 reference calls give the same result as MXM in ETIA. For FBED, the sequence of the columns is also the same. 58 calls are R errors of ETIA, and ETIA-JAX gives the same errors.
  - **AFS.** All 8 complete runs (4 with the linear grid, 4 with the full grid) give the same winner and the same columns as ETIA.
- **Oracle.** With exact-covariance data, FBED finds the Markov blanket and SES all the parents and children of the target.
- **Findings.**
  - Rfast `cor.fbed` stops with an error when its forward phase selects one column. ETIA's FBED script stops when FBED selects none. ETIA skips the fold in both cases.
  - ETIA ignores `folds_to_run`, and its "bootstrap bias correction" corrects no bias. The default mode uses BBC-CV.
  - MXM was archived on CRAN on 2026-11-03.
- **Speed.** One FS call is about 1000 times faster, because ETIA starts R for each call. A complete AFS run is 9 to 190 times faster.
- **Demo.** Plate `89493558771c`.

## Increment 9: OCT and CausalLearner (next)

- OCT tuning of the causal configurations, the validation protocols, and the CausalLearner pipeline of ETIA.
- The k-fold protocol of ETIA stops on NOTEARS and SAM (two return values, not three), and NOTEARS ignores the rows of the fold.

## Increment 10: CRV (planned)

- The causal queries (paths, Markov boundaries), the adjustment sets, the bootstrap confidence, and a visualization without Cytoscape.
