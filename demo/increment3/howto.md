## agree

This table compares the graphs of ETIA-JAX with the graphs of ETIA. Each row is one group of datasets, one CI test and one algorithm. ETIA-JAX uses the Tetrad mode here.

The column "searches" gives the number of searches in the group: each dataset with two values of alpha. The column "same_graph" gives the number of searches where all the edge marks are the same. A value of 100 in the last column shows that all the graphs in the group are the same.

## size_chart

This chart shows the median time of one search for a number of variables. Select PC or CPC above the chart. Each line is ETIA or ETIA-JAX in one mode. The data are continuous, and the CI test is FisherZ.

The x axis is the number of variables. The y axis has a logarithmic scale. A lower line shows a shorter time. The lines "compiled" show the time after JAX compiled the code. The lines "first call" show the time of the first search of this size in the benchmark. This time includes the compilation of the new shapes.

## size_table

This table gives the values of the chart and more columns. Each row is one number of variables and one algorithm. All the times are medians in seconds.

The column "ETIA data copy" gives the time that ETIA uses to copy the data into Tetrad before each search. The column "tests" gives the number of CI queries that the search uses. The column "computed" gives the number of queries that JAX calculates in the batched calls. This number is larger, because a batched call also calculates queries that the search does not use.

## timing

This table gives the times of all the 116 searches. Each row is one search: one dataset, one CI test, one value of alpha and one algorithm. All the times are in seconds. Type in the filter box to show only some rows.

The column "speed-up" divides the time of the Tetrad search by the shorter compiled time of ETIA-JAX. A value above 1 shows that ETIA-JAX is faster. The column "same_as_tetrad" must be "True" in all the rows.

## quality_chart

This chart shows how near the graphs are to the correct graph. Select PC or CPC above the chart. Each group of bars is one group of datasets. Each color is one mode of ETIA-JAX.

The y axis is the mean structural Hamming distance (SHD) to the correct CPDAG. The SHD is the number of node pairs with an incorrect edge. A lower bar shows a better graph. Compare the two bars of each group. On the larger graphs, the PC-stable mode has a lower SHD than the Tetrad mode.

## quality_table

This table gives the values of the quality chart and also the adjacency F1 score. Each row is one group of datasets, one algorithm and one mode. The column "runs" gives the number of datasets in the group. All the searches use alpha 0.05.

An F1 score of 1 shows that the skeleton is the same as the skeleton of the correct graph. The two modes find the same edges on the small graphs. They are different only in the sequence of the tests.

## defects

This table shows what Tetrad 7.6.5 does in PC and CPC that the definitions of the algorithms do not tell. Each row is one behavior. The last column tells how ETIA-JAX does it. ETIA uses Tetrad 7.6.5, thus ETIA has these behaviors.

## tests

This table shows the tests of the test suite after increment 3. Each row is one test function. The column result shows "pass" or "fail". The column time_s gives the time of each test in seconds. Start the slow tests with the command "pytest -m slow".
