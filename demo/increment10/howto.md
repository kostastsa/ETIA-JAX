## agreement

This table compares each part of the CRV of ETIA-JAX in the "etia_compat" mode with ETIA. Each row is one part. The column "reference" tells what gives the expected output. The column "same" gives the number of outputs that are the same, and the column "total" gives the number of compared outputs.

ETIA calls dagitty in R for the adjustment sets. The expected output comes from dagitty with ETIA's steps. The bootstrap graphs come from Tetrad's FCI in ETIA on the same bootstrap rows.

## adjust_oracle

This table checks the adjustment sets against the truth. Each row is one graph type with 60 random linear models. Two variables of each MAG and PAG are latent. A set is correct when the regression on the exposure and the set gives the true total effect, with exact covariances.

Compare the column "default_correct" with "default_sets", and "etia_correct" with "etia_sets". For a CPDAG, ETIA drops the undirected edges before it calls dagitty. Thus some of its sets give a wrong effect.

## query_diff

This table compares the path queries of ETIA with the default mode. Each row is one kind of path, and each kind has 1200 queries on 400 graphs. The "found" columns give the number of queries with a path. The "mean_edges" columns give the mean number of edges of the paths that the two modes find.

ETIA does not follow the edges "--o" and "---" in a potentially directed path. Thus it finds fewer paths. The default mode gives a shortest path. Thus its paths can be shorter.

## confidence

This table gives the bootstrap confidence of each edge of the FCI graph of the dataset cont10_s0, from 50 bootstrap samples. Each row is one edge. The column "edge" gives the marks at the two ends, for example "Tail-Arrow" for source -> target.

The column "discovery" gives the fraction of bootstrap graphs with the same edge. The consistency columns give the fraction with a consistent edge, by ETIA's table and by the default rule. The default rule accepts each circle against a tail or an arrow. Thus its value can be higher.

## figure

This figure shows the FCI graph of the dataset cont10_s0. Each ellipse is one variable, and each line is one edge. An arrowhead shows an arrow mark, a small circle shows a circle mark, and no symbol shows a tail. A wider and darker line shows a higher default consistency.

## time_chart

This chart shows the time of each kind of call. Each bar is ETIA, ETIA-JAX on the GPU, or ETIA-JAX on the CPU. The y axis has a logarithmic scale. A lower bar shows a shorter time.

An adjustment set in ETIA starts R for each call. A bootstrap runs FCI on 50 bootstrap samples, and the path metrics score 30 to 40 paths.

## timing

This table gives the times of all the kinds of calls. Each row is one kind. All the times are in seconds, and a query time is the time of one call. The column "same" gives the number of outputs that are the same as in ETIA.

The column "speed-up" divides the time of ETIA by the time of ETIA-JAX on the GPU.

## findings

This table shows what ETIA does that the definitions do not tell. Each row is one behavior. The last column tells how ETIA-JAX does it.

## tests

This table shows the tests of the test suite after increment 10. Each row is one test function. The column result shows "pass" or "fail". The column time_s gives the time of each test in seconds. Start the slow tests with the command "pytest -m slow".
