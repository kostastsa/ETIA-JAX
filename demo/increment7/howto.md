## nt_agree

This table compares NOTEARS of ETIA-JAX in the ETIA mode with NOTEARS of causalnex, which ETIA uses. Each row is one dataset of the search fixtures. causalnex has no random step. Thus the two programs find the solution of the same problem.

The column "max |w diff|" gives the largest difference between two weights. The column "score rel. diff" gives the relative difference of the squared-error score of the two weight matrices. A small value shows that the two solvers found equally good solutions.

The last three columns compare the graph of ETIA with the graph of ETIA-JAX. Each column is one threshold of the ETIA grid. The value "same" shows equal mark matrices. An empty cell shows that the fixture has no ETIA run at that threshold. From 20 variables, one causalnex run takes minutes.

## nt_weights

This chart compares each weight of causalnex with the same weight of ETIA-JAX. Each point is one entry W[i, j] of one dataset. The x axis gives the weight in causalnex, and the y axis gives the weight in ETIA-JAX.

When the two solvers agree, the points are on the diagonal line. A point far from the diagonal shows a weight that the two solvers find differently. Only a weight that crosses a threshold can change the graph.

## nt_acc

This table gives the accuracy of NOTEARS at the default threshold of ETIA (0.8). Each row is one dataset and one mode of ETIA-JAX. The SHD is the structural Hamming distance to the correct DAG: the number of node pairs with a different edge. A lower SHD shows a better graph.

The column "same as ETIA" gives 1 when the ETIA mode gives the same mark matrix as ETIA. The default mode centers the data. Thus its graph can be different.

## sam_acc

This table compares the accuracy of SAM in ETIA and in ETIA-JAX. Each row is one dataset, one setting and one mode of ETIA-JAX. SAM is random, and ETIA-JAX cannot make the random draws of torch. Thus the table compares the accuracy, not the output.

The AUROC uses the weights of all the node pairs as scores for the correct edges. The value 1 shows a perfect sequence, and the value 0.5 shows a random sequence. A higher AUROC shows a better result. The SHD uses the DAG of the weights above 0.5 (after the division by the number of batches). A lower SHD shows a better graph.

## sam_seeds

This table compares the range of SAM results in ETIA-JAX with the one ETIA run of each dataset. Each row is one dataset. ETIA-JAX ran SAM with 10 seeds, and ETIA ran SAM one time, because one ETIA run takes about 13 minutes.

The columns "min", "ETIA-JAX mean" and "max" give the AUROC of the 10 seeds. The column "ETIA in range" gives the value "true" when the AUROC of ETIA is between the minimum and the maximum. This value shows that the difference between ETIA and ETIA-JAX is not larger than the random changes of SAM.

## time_chart

This chart shows the time of one run for each dataset. Select the algorithm above the chart. Each bar is ETIA or ETIA-JAX on one device.

The y axis has a logarithmic scale. A lower bar shows a shorter time. All the times of ETIA-JAX are after the compilation. For SAM, one run has the 8 runs of cdt. The label "(defaults)" shows the run with the default settings of ETIA (3000 training epochs and 1000 test epochs).

## timing

This table gives the times and the accuracy of all the runs. Each row is one algorithm, one dataset, one setting, one device and one mode. All the times are in seconds. The column "first run s" includes the compilation. Type in the filter box to show only some rows.

The column "speed-up" divides the time of ETIA by the time of ETIA-JAX. A value above 1 shows that ETIA-JAX is faster.

## findings

This table shows what ETIA, causalnex and cdt do that the definitions of the algorithms do not tell. Each row is one behavior. The last column tells how ETIA-JAX does it.

## tests

This table shows the tests of the test suite after increment 7. Each row is one test function. The column result shows "pass" or "fail". The column time_s gives the time of each test in seconds. Start the slow tests with the command "pytest -m slow".
