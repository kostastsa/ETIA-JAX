## scores

This table compares the local scores of ETIA-JAX with the local scores of Tetrad 7.6.5. Each row is one score. ETIA uses the same five scores with the same settings.

For each dataset, the Tetrad fixture has random queries. A query is one node and a set of up to four parents. The column "max_rel_error" gives the largest relative difference between the two scores. A value near 1e-12 shows that the numbers are the same. Only the last digits are different.

The column "same_nan" shows that the two programs give no score for the same queries. The column "same_max_degree" shows that the two programs give the same degree limit for FGES.

## fges_agree

This table compares the CPDAGs of FGES in ETIA-JAX with the CPDAGs of ETIA. Each row is one score and one group of datasets. ETIA-JAX uses the Tetrad mode.

The column "same_as_etia" gives the number of searches where the two CPDAGs are the same. When this value is equal to "searches", all the searches in the row agree. The column "tetrad_deterministic" gives the number of searches where three Tetrad searches gave the same CPDAG.

## gfci_agree

This table compares the PAGs of GFCI in ETIA-JAX with the PAGs of Tetrad. Each row is one CI test and one group of datasets. Each row has all the scores and both values of alpha.

The column "tetrad_crashes" gives the number of searches where Tetrad stopped with an error one time or more. This occurs with the chi-square and G-square tests, because Tetrad uses them from parallel threads. Thus ETIA-JAX is compared with a changed Tetrad that sends the test calls through a lock.

In the last step, Tetrad starts its R4 tasks in parallel. Thus some searches can give a different PAG each time. The column "race_free_deterministic" gives the number of searches where ten searches of the changed Tetrad gave one PAG (some searches had more runs).

The column "agree" gives the number of searches where two conditions apply. First, the graph before R4 is the same as in Tetrad. Second, the last PAG of ETIA-JAX is one of the PAGs of Tetrad. Two searches with the same graph before R4 share their Tetrad PAGs, because R4 uses only this graph, the test and alpha.

## time_chart

This chart shows the median time of one search in each group of datasets. Select an algorithm above the chart. Each bar is ETIA or ETIA-JAX in one mode on one device.

The y axis has a logarithmic scale. A lower bar shows a shorter time. All the times of ETIA-JAX are after the compilation. Each search of ETIA-JAX makes a new score object, thus the time includes all the local scores.

## time_table

This table gives the values of the time chart. Each row is one algorithm and one group of datasets. All the times are medians in seconds. The column "p" gives the number of observed variables.

The column "ETIA data copy" gives the time that ETIA uses to copy the data into Tetrad. Compare the columns of ETIA-JAX with the column "ETIA". A value that is smaller than the value of ETIA shows that ETIA-JAX is faster.

## timing

This table gives the times of all the searches. Each row is one search: one algorithm, one dataset, one score, and for GFCI one CI test and one value of alpha. All the times are in seconds. Type in the filter box to show only some rows.

The column "speed-up default" divides the time of ETIA by the shorter time of the default mode. The column "speed-up compat" does the same for the Tetrad mode. A value above 1 shows that ETIA-JAX is faster. An empty value shows a search where Tetrad stopped with an error.

## quality_chart

This chart shows how near the results are to the correct graphs. Select an algorithm above the chart. Each group of bars is one group of datasets. Each color is one mode of ETIA-JAX.

The y axis is the mean structural Hamming distance (SHD). For FGES, the correct graph is the CPDAG of the DAG that made the data. For GFCI, the correct graph is the PAG of that DAG with the latent variables. A lower bar shows a better result. All the GFCI searches use alpha 0.05.

## quality_table

This table gives the values of the quality chart and also the adjacency F1 score. Each row is one algorithm, one group of datasets and one mode. The column "runs" gives the number of searches in the row.

An F1 score of 1 shows that the skeleton is the same as the skeleton of the correct graph. Sometimes the SHD of the two modes is different, but the F1 is almost the same. Then the two modes find the same edges, but put different marks on them.

## defects

This table shows what Tetrad 7.6.5 does in the scores, FGES and GFCI that the definitions of the algorithms do not tell. Each row is one behavior. The last column tells how ETIA-JAX does it. ETIA uses Tetrad 7.6.5, thus ETIA has these behaviors.

## tests

This table shows the tests of the test suite after increment 5. Each row is one test function. The column result shows "pass" or "fail". The column time_s gives the time of each test in seconds. Start the slow tests with the command "pytest -m slow".
