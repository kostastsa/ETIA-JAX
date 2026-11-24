## steps

This table compares each step of the OCT of ETIA-JAX with the same step of ETIA. Each row is one step. Each step gets the inputs that ETIA's step had in the stored ETIA runs. Thus the random graphs of Tetrad do not change the result of this table.

The column "same" gives the number of outputs that are the same as in ETIA. The column "total" gives the number of compared outputs. A prediction or a score is the same when the difference is below 1e-8. The fold PAGs come from the searches of ETIA-JAX on the training rows of each fold.

## score_chart

This chart shows the mean score of each configuration. Each point is one configuration in ETIA or in one mode of ETIA-JAX. The x axis gives the configuration. The y axis gives the mean mutual information of the nodes, in nats. Use the selector to show the other dataset.

The two ETIA runs use the same data and the same seed, but Tetrad draws other random MAGs. Thus their points can be far apart. The bar on the "etia_compat" point goes from the lowest to the highest value of 10 seeds. The default mode takes the Markov boundary from the class of the graph. Thus its points stay the same in each run.

## time_chart

This chart shows the median time of one complete OCT run. Each bar is ETIA or ETIA-JAX. The y axis has a logarithmic scale. A lower bar shows a shorter time.

The run "lin" uses linear regression, and the run "rf" uses the random forest of ETIA. The digit gives the number of joblib workers of ETIA: "1" runs without workers, and "8" uses eight workers. ETIA-JAX runs the same settings in the "etia_compat" mode.

## timing

This table gives the time of each stored ETIA run and of the same run in ETIA-JAX. Each row is one dataset and one run setting. All the times are in seconds. The column "speed-up" divides the time of ETIA by the time of ETIA-JAX on the GPU. The column "default_s" gives the time of the default mode.

The column "pags_same" gives the fraction of the fold PAGs that are the same as in ETIA. The two "chosen" columns give the chosen configuration. They can be different, because the random MAGs of ETIA and ETIA-JAX are not the same.

## afs

This table compares complete AFS runs with the full grid of ETIA. Each row is one dataset and one target. From this increment, the random forest of ETIA-JAX replaces the MLP of increment 8.

The winner columns give the configuration with the largest mean score. The column "same_selected" shows if the selected columns are the same as in ETIA. The times are in seconds, and the time of ETIA-JAX is the time of a second run.

## findings

This table shows what ETIA and Tetrad do that the definition of the OCT does not tell. Each row is one behavior. The last column tells how ETIA-JAX does it.

## tests

This table shows the tests of the test suite after increment 9. Each row is one test function. The column result shows "pass" or "fail". The column time_s gives the time of each test in seconds. Start the slow tests with the command "pytest -m slow".
