# Defects of Tetrad 7.6.5 found by ETIA-JAX

ETIA calls Tetrad 7.6.5 (`tetradlib.jar`) for its CI tests, scores and searches. ETIA-JAX ports these parts. To get ETIA's outputs exactly, the port had to copy each behavior of Tetrad. This file lists the behaviors that do not agree with the definitions of the algorithms.

- **Source.** The Tetrad source of tag `v7.6.5` ([Tetrad 7.6.5](#ref-tetrad765)).
- **Check.** Each item has a test in ETIA-JAX. The test compares the `tetrad_compat=True` mode (or `etia_compat=True`) with outputs of ETIA and Tetrad in `reference/fixtures/`.
- **Default mode.** The default mode of ETIA-JAX does not have these defects. Its tests use oracles (d-separation, exact covariances) where possible.
- **Increment.** The column "Inc." gives the increment of ETIA-JAX that found the item. The details are in the README sections of the increments.

## Summary

| # | Area | Defect | Effect | Inc. |
|---|---|---|---|---|
| 1 | Threads | One CI test object is used from parallel threads. | Discrete searches crash or get incorrect p-values. | 4, 5, 9 |
| 2 | Threads | R4 runs its tasks in parallel. | The PAG can change from run to run. | 4, 5 |
| 3 | Threads | `IndTestFisherZ` with one processor. | Division by zero. | 9 |
| 4 | CI tests | CCI uses the wrong rows. | Incorrect p-values. | 2 |
| 5 | CI tests | DG-LRT uses a wrong mean. | Incorrect p-values. | 2 |
| 6 | Scores | The BDeu structure prior uses the number of rows. | Incorrect scores. | 5 |
| 7 | Scores | The discrete BIC structure prior has the wrong sign. | More parents give a higher score. | 5 |
| 8 | PC, CPC | The "stable" FAS is not stable. | The skeleton depends on the order of the variables. | 3 |
| 9 | PC, CPC | `colliderAllowed` has a typo. | A collider can make a cycle. | 3 |
| 10 | CPC | A side with one node is not searched. | Colliders stay ambiguous. | 3 |
| 11 | FCI | R4 reverses the discriminating path. | R4 accepts only the shortest paths. | 4 |
| 12 | FCI | R10 does not change marks. | The PAG is not complete. | 4 |
| 13 | FCI, CFCI | The possible-m-sep steps search only from x. | Fewer edges are removed. | 4 |
| 14 | FGES | The degree limit is ceil(ln n). | FGES stops early on large graphs. | 5 |
| 15 | GFCI | `SepsetsMinP` searches only adj(x), and all subsets must show independence. | Fewer separating sets. | 5 |
| 16 | GFCI | No Possible-D-SEP step. | Extra edges can stay. | 5 |
| 17 | Knowledge | Meek R4 stays off with knowledge. | Fewer orientations. | 6 |
| 18 | Knowledge | FCI-Max, RFCI and CFCI orient with empty knowledge. | Orientations against the tiers. | 6 |
| 19 | Knowledge | `getTier` sorts the names as strings. | Tiers in a different order with 10 or more variables. | 6 |
| 20 | SVAR | The similar-pair copies of a mark apply in one direction. | The lags are not symmetric. | 6 |
| 21 | Graphs | `DagToPag` is not exact. | Wrong tails and missing arrowheads. | 1 |
| 22 | Graphs | `magFromPag` can make undirected edges. | The "MAG" is not a MAG. | 9 |

## Threads

1. **One CI test object from parallel threads.** FCI, FCI-Max, RFCI, CFCI and GFCI call one CI test object from parallel streams and ForkJoin tasks. The chi-square and G-square tests keep their counts in a shared `CellTable`, so they are not thread-safe.
   - FCI crashed on 11 of 12 discrete searches, and GFCI crashed in 30 discrete searches. The errors are a `NullPointerException` in `MultiDimIntTable.getValue` or an `ArrayIndexOutOfBoundsException` in `MultiDimIntTable.getCell`.
   - In R0, `SepsetFinder.getSepsetContainingGreedy` runs the tests in a parallel stream. A JVM with one processor does not stop the race, because the common pool and the calling thread run at the same time.
   - ETIA's OCT on the dataset `disc10_s0` stopped with Java errors in all 4 tries.
   - A search that does not crash can get incorrect p-values. The references of ETIA-JAX for discrete data put a lock around each test call (`reference/make_fixtures_serial.py`).
2. **R4 in parallel.** R4 runs its tasks in parallel over a hash set of discriminating paths that has no hash codes. Thus the order of the tasks changes from run to run, and with two or more discriminating paths in one round the PAG can change. 12 FCI-family searches had more than one possible PAG, and GFCI had up to 7. ETIA-JAX uses the order in which it finds the paths. Its tests accept each PAG that a race-free Tetrad gives.
3. **FisherZ on one processor.** `CovariancesDoubleForkJoin` uses `availableProcessors()` threads, divided by 2 when there are 8 processors or fewer. With one processor this gives 0 threads, and `IndTestFisherZ` stops with "/ by zero". Thus ETIA cannot run on a machine (or a JVM) with one processor. The OCT references use a JVM with two processors.

## CI tests and scores

4. **CCI.** The kernel distance uses the rows (x, z1, ..., z(k-1)) instead of (z1, ..., zk). Thus the regression of x on Z partly uses x itself and omits the last variable of Z. The correlation of the standardized basis values also divides by n - 1 instead of n.
5. **DG-LRT.** The covariance uses a mean of sum / (n - 1) with a divisor n. Tetrad also drops the indicator of the category that appears first latest in the row order, not a fixed category.
6. **BDeu structure prior.** The prior uses v = N - 1, where N is the number of rows. It must use the number of other variables.
7. **Discrete BIC structure prior.** Tetrad adds 2 times the minus log prior, with the wrong sign. Thus more parents give a higher score.

The cache of the CI tests also gives a different answer for each direction of a pair when the test is not symmetric (CG-LRT, DG-LRT). ETIA-JAX copies this cache in `tetrad_compat` mode. It is a property of the tests, not a defect.

## PC and CPC

8. **The "stable" FAS is not stable.** At each depth, the FAS copies the adjacencies, but then it changes that copy. Later pairs thus see the removals of earlier pairs, and the result depends on the order of the variables. ETIA asks for `stable=True`. The default mode of ETIA-JAX is PC-stable as [Colombo and Maathuis (2014)](#ref-colombo2014) define it.
9. **`colliderAllowed`.** Tetrad must not orient a collider x -> y <- z when y is an ancestor of x or of z. A typo tests z two times and does not test x.
10. **CPC.** Tetrad looks for separating sets in adj(x) or adj(z) only when that set has two or more nodes. Thus a collider x -> y <- z, where x and z have only y as neighbour, stays ambiguous.

The orientation steps also visit the nodes and edges in the order of Java hash sets. The collider rules and Meek rules depend on this order. ETIA-JAX copies the hash-set order (`etia_jax/search/tetrad_graph.py`). This is an implementation choice, not a defect.

## The FCI family

11. **R4 path.** A defect in `DiscriminatingPath` puts the nodes of the path in the opposite order. Thus R4 accepts only the shortest discriminating paths.
12. **R10.** The code compares a list of edges with one edge, and this test is always false. Thus R10 never changes a mark, and the PAG is not always complete as [Zhang (2008)](#ref-zhang2008) defines it.
13. **Possible-m-sep.** The possible-m-sep step of FCI starts its search only from x. CFCI uses a different step that tests only the side of x.

These behaviors are also different from the textbook algorithm, and ETIA-JAX copies them in `tetrad_compat` mode:

- R0 of FCI finds a new separating set with a greedy search, not the set of the FAS.
- R4 runs only when R1, R2 or R3 changed a mark in the same cycle. The rules R5 to R7 (for selection bias) also run.
- CFCI keeps its own map of separating sets, which is empty at the first R0.

## FGES and GFCI

14. **Degree limit.** FGES takes the degree limit of the score. For most scores the limit is ceil(ln n), for example 7 with 1000 rows. For the discrete BIC score it is 1000. GFCI sets no limit.
15. **`SepsetsMinP`.** The search takes the subset of adj(x) with the smallest p-value. It keeps this subset only if its p-value is above alpha, so all the subsets must show independence. The second search uses adj(x) again, so adj(y) is never searched.
16. **No Possible-D-SEP step.** GFCI does not remove the edges that only a Possible-D-SEP step finds. The FGES of GFCI also assumes faithfulness, so it does only two rounds.

The GFCI collider rule also looks at a pair x, z that is adjacent in the CPDAG and in the graph. It searches for a separating set of x and z, and if it finds one, it removes the edge. ETIA-JAX copies this in `tetrad_compat` mode.

## Knowledge and time series

17. **Meek R4 with knowledge.** `MeekRules` decides `useRule4` before it gets the knowledge. Thus R4 stays off, also when knowledge is given.
18. **Empty knowledge.** FCI-Max, RFCI and CFCI give an empty knowledge to their final orientation. Only their first orientation step uses the tiers.
19. **Tier order.** `Knowledge.getTier` sorts the names as strings. Thus `X10:1` comes before `X1:1`. With 10 or more variables, the positions of one variable in two tiers can be different.
20. **SVAR similar pairs.** SVAR-FCI and SVAR-GFCI copy each edge removal and each mark to the "similar pairs" at the other lags. A mark is copied only when the first node is in the later tier. SVAR-GFCI also uses its own older FGES class (`SvarFges`), which copies each edge.

Tetrad also orders equal score changes of SVAR-GFCI by its own rounding. In 8 searches the tests of ETIA-JAX accept the other order too.

## Graph transforms

21. **`DagToPag`.** On 13 of 60 fixture graphs, the PAG is not exact. At 29 marks Tetrad gives a tail, but a MAG of the class has no tail there. At 3 marks Tetrad gives a circle, but all the MAGs of the class have an arrowhead there. The slow test `test_pag_exact_where_tetrad_differs` checks this with witness MAGs and a search of all the MAGs of the class.
22. **`magFromPag`.** `transormPagIntoRandomMag` sets each circle at random to a tail or an arrow. After each change, it runs the final orientation of FCI with an m-separation test on the graph. A circle can become a tail at both ends of an edge. Thus the "MAG" can have undirected edges. On one PAG of 10 variables, the draws had 1.6 undirected edges on average.
    - The m-separation test (`MsepTest`) keeps the descendant map of the graph at its start, but the graph changes during the orientation.
    - Java draws the random numbers without a seed that Python can set. Thus ETIA's OCT, which takes the Markov boundaries from this graph, gives different scores and winners in each run.
    - `dagFromCpdag` is random in the same way, but its result is a DAG.

## References (outside the corpus)

- []{#ref-tetrad765}Tetrad 7.6.5, Carnegie Mellon University, the source of tag `v7.6.5`: [github.com/cmu-phil/tetrad/tree/v7.6.5](https://github.com/cmu-phil/tetrad/tree/v7.6.5). The class names and the behaviors in this file come from this source and from runs of the `tetradlib.jar` in ETIA.
- []{#ref-colombo2014}Colombo, D. and Maathuis, M. H. (2014). Order-independent constraint-based causal structure learning. *Journal of Machine Learning Research* 15(116), 3921–3962. [jmlr.org/papers/v15/colombo14a](https://jmlr.org/papers/v15/colombo14a.html). Only the bibliographic record was checked here.
- []{#ref-zhang2008}Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. *Artificial Intelligence* 172(16–17), 1873–1896. [doi:10.1016/j.artint.2008.08.001](https://doi.org/10.1016/j.artint.2008.08.001). The paper defines the rules R1 to R10. The statement of each rule was not checked here.
