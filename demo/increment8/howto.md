## fs_agree

This table compares FBED and SES of ETIA-JAX with MXM 1.5.8 in R, which ETIA calls. Each row is one FS algorithm on one group of datasets. Each call has one dataset, one target, and one FS configuration of the ETIA grid.

The column "same" gives the number of calls where ETIA-JAX gives the same selected columns as MXM. For FBED, the sequence of the columns must also be the same. The column "R errors" gives the number of calls where the R code of ETIA stops with an error. There are two errors: the Rfast error when FBED finds one column, and a missing column "sel" when FBED finds no column. In these calls, ETIA-JAX in the ETIA mode gives the same error.

The column "speed-up" divides the median time of ETIA by the median time of ETIA-JAX. The time of ETIA includes the start of R and the CSV files. ETIA does these steps for each call.

## afs_runs

This table compares complete AFS runs of ETIA and ETIA-JAX on the same data and target. Each row is one dataset, one target and one grid. The grid "linear" has the 8 configurations with linear regression. The grid "full" has all 24 configurations of ETIA.

The winner columns give the configuration with the largest mean score. The "columns" give the selected features. In the full grid, ETIA-JAX uses an MLP, not the random forest. Thus its winner and its columns can be different.

The bootstrap columns give the mean score of the bootstrap. The bootstrap is random. Thus a small difference between ETIA and ETIA-JAX is usual. The column "default BBC-CV" gives the score of the default mode. BBC-CV corrects the bias that comes from the selection of the winner. The default mode also uses only 2 folds. Thus this value can be higher or lower than the value of ETIA.

## time_chart

This chart shows the median time of one call for each kind of call. Each bar is ETIA or ETIA-JAX. The y axis has a logarithmic scale. A lower bar shows a shorter time.

The kinds "fbed" and "ses" are single FS calls. The kinds "afs_linear" and "afs_full" are complete AFS runs with the linear grid and the full grid.

## timing

This table gives the times of all the calls. Each row is one FS call or one AFS run. All the times are in seconds. The column "same" gives 1 when the output is the same as in ETIA. Type in the filter box to show only some rows.

The column "speed-up" divides the time of ETIA by the time of ETIA-JAX. A value above 1 shows that ETIA-JAX is faster.

## findings

This table shows what ETIA, MXM and Rfast do that the definitions of the algorithms do not tell. Each row is one behavior. The last column tells how ETIA-JAX does it.

## tests

This table shows the tests of the test suite after increment 8. Each row is one test function. The column result shows "pass" or "fail". The column time_s gives the time of each test in seconds. Start the slow tests with the command "pytest -m slow".
