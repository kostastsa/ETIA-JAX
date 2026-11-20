## agree

This table compares the PAGs of ETIA-JAX in the Tetrad mode with the PAGs of ETIA. Each row is one algorithm and one group of datasets. Each group has datasets with latent variables, and each dataset has one or two CI tests and two values of alpha.

The column "tetrad_crashes" gives the number of searches where Tetrad stopped with an error. The column "same_as_etia" gives the number of searches where the PAG of ETIA-JAX and the PAG of ETIA are the same. For discrete data, ETIA gives incorrect results. Thus the last two columns compare with Tetrad after a change that makes the CI tests safe.

The column "race_free_deterministic" gives the number of searches where this changed Tetrad gives the same PAG each time. The column "among_race_free" gives the number of searches where the PAG of ETIA-JAX is one of the PAGs of this changed Tetrad. When this value is equal to "searches", all the searches in the row agree.

## time_chart

This chart shows the median time of one search in each group of datasets. Select an algorithm above the chart. Each bar is ETIA or ETIA-JAX in one mode on one device.

The y axis has a logarithmic scale. A lower bar shows a shorter time. All the times of ETIA-JAX are after the compilation. A group without an ETIA bar is a group where Tetrad stopped with an error.

In the group "cont50", the default mode of FCI, FCI-Max and CFCI is slow. In one dataset, the Possible-D-SEP sets have up to 21 nodes. Thus the default mode tests millions of subsets. Tetrad makes a smaller set because of a defect, and it tests fewer subsets. The GPU does these tests much faster than the CPU.

## time_table

This table gives the values of the time chart. Each row is one algorithm and one group of datasets. All the times are medians in seconds. The column "p" gives the number of observed variables.

The column "ETIA data copy" gives the time that ETIA uses to copy the data into Tetrad. Compare the columns of ETIA-JAX with the column "ETIA". A value that is smaller than the value of ETIA shows that ETIA-JAX is faster.

## timing

This table gives the times of all the searches. Each row is one search: one dataset, one CI test, one value of alpha and one algorithm. All the times are in seconds. Type in the filter box to show only some rows.

The column "speed-up default" divides the time of ETIA by the shorter time of the default mode. The column "speed-up compat" does the same for the Tetrad mode. A value above 1 shows that ETIA-JAX is faster. An empty value shows a search where Tetrad stopped with an error.

## quality_chart

This chart shows how near the PAGs are to the correct PAG. Select an algorithm above the chart. Each group of bars is one group of datasets. Each color is one mode of ETIA-JAX.

The y axis is the mean structural Hamming distance (SHD) to the correct PAG. The correct PAG comes from the DAG with the latent variables. A lower bar shows a better PAG. All the searches use alpha 0.05.

## quality_table

This table gives the values of the quality chart and also the adjacency F1 score. Each row is one algorithm, one group of datasets and one mode. The column "runs" gives the number of searches in the row.

An F1 score of 1 shows that the skeleton is the same as the skeleton of the correct PAG. Sometimes the SHD of the two modes is very different, but the F1 is almost the same. Then the two modes find the same edges, but put different marks on them.

## defects

This table shows what Tetrad 7.6.5 does in the FCI family that the definitions of the algorithms do not tell. Each row is one behavior. The last column tells how ETIA-JAX does it. ETIA uses Tetrad 7.6.5, thus ETIA has these behaviors.

## tests

This table shows the tests of the test suite after increment 4. Each row is one test function. The column result shows "pass" or "fail". The column time_s gives the time of each test in seconds. Start the slow tests with the command "pytest -m slow".
