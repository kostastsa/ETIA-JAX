## tetrad_agree

This table compares the Tetrad algorithms of ETIA-JAX with Tetrad 7.6.5 on time series. Each row is one algorithm. Each search uses the lagged table of ETIA and the tier knowledge of ETIA.

The column "agree" gives the number of searches where ETIA-JAX gives one of the outputs of Tetrad. When this value is equal to "searches", all the searches of the row agree.

The column "nondeterministic_in_tetrad" gives the number of searches where Tetrad gave more than one output in its runs. The column "rounding_ties" gives the number of searches where two score changes are equal except for rounding. Tetrad orders these by its own rounding, so the test also accepts the other order.

## tig_agree

This table compares PCMCI, PCMCI+ and LPCMCI of ETIA-JAX with tigramite 5.2. Each row is one algorithm and one CI test. Each search uses the raw time series, as ETIA does.

The column "same_graph" gives the number of searches where the graph and the mark matrix of ETIA are the same. The column "random_p_values" shows the tests that get their p-values from random draws in tigramite. For these tests, a decision near alpha can change, because ETIA-JAX cannot make the same random draws.

## oracle

This table shows the check of the default mode of ETIA-JAX. Each row is one algorithm. Each case is a random time-series model with the tier knowledge. The CI tests and the score are d-separation oracles.

The column "correct" gives the number of cases where three conditions apply. First, the skeleton is correct. Second, each oriented mark is true in the correct graph. Third, the output is at least as informative as the PAG or the CPDAG.

## time_chart

This chart shows the median time of one search for each algorithm. Select a family of algorithms above the chart. Each bar is ETIA or ETIA-JAX in one mode on one device.

The y axis has a logarithmic scale. A lower bar shows a shorter time. All the times of ETIA-JAX are after the compilation.

## time_table

This table gives the values of the time chart. Each row is one algorithm. All the times are medians in seconds. The column "speed-up" divides the time of ETIA by the shorter time of ETIA-JAX in the mode that gives the same output as ETIA. A value above 1 shows that ETIA-JAX is faster.

## timing

This table gives the times of all the searches. Each row is one search: one algorithm, one dataset, one CI test or score, and one value of alpha. All the times are in seconds. Type in the filter box to show only some rows.

The column "speed-up" divides the time of ETIA by the shorter time of ETIA-JAX in the mode that gives the same output as ETIA. A value above 1 shows that ETIA-JAX is faster.

## findings

This table shows what ETIA, Tetrad 7.6.5 and tigramite 5.2 do with time series that the definitions of the algorithms do not tell. Each row is one behavior. The last column tells how ETIA-JAX does it.

## tests

This table shows the tests of the test suite after increment 6. Each row is one test function. The column result shows "pass" or "fail". The column time_s gives the time of each test in seconds. Start the slow tests with the command "pytest -m slow".
