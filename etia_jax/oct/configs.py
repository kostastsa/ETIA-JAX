"""The configurations of ETIA's CausalLearner (increment 9).

A copy of ETIA's tables (``CausalLearning/configurations/``): the causal
discovery algorithms (``causal_configs.py``), the CI tests
(``ci_tests.py``), the scores (``scores.py``), the default settings
(``default_parameters.py``), the configurator
(``class_causal_configurator.py``) and scikit-learn's ``ParameterGrid``
order. These tables are data, so both modes use them as they are. Two of
their values have effects that are not clear from the names:

* ``pc`` and ``cpc`` have ``assume_faithfulness: [False]`` and ``fges`` has
  ``causal_sufficiency: [False]``. Thus ETIA's default (no causal
  sufficiency, faithfulness) has only FCI and FCI-Max for continuous data.
* the OCT of ETIA makes a MAG with Tetrad's ``magFromPag`` when the
  configuration has ``causal_sufficiency=False``, also for FGES, which
  gives a CPDAG.
"""
import itertools

import numpy as np

_ALL = ["continuous", "mixed", "categorical"]
_TETRAD_CI = ["FisherZ", "cg_lrt", "dg_lrt", "chisquare", "gsquare"]
_TETRAD_SCORES = ["sem_bic_score", "bdeu", "discrete_bic", "cg_bic", "dg_bic"]
_TIG_CI = ["ParCor", "RobustParCor", "GPDC", "CMIknn", "ParCorrWLS", "Gsquared", "CMIsymb", "RegressionCI"]


def _alg(data_type, cs, af, ts, tl, params):
    return dict(data_type=data_type, causal_sufficiency=cs, assume_faithfulness=af, time_series=ts,
                time_lagged=tl, parameters=params)


CAUSAL_CONFIGS = {
    "pc": _alg(_ALL, [True], [False], [False], [False, True], {"ci_test": _TETRAD_CI, "stable": [True]}),
    "cpc": _alg(_ALL, [True], [False], [False], [False, True], {"ci_test": _TETRAD_CI, "stable": [True]}),
    "fges": _alg(_ALL, [False], [False], [False], [False, True], {"score": _TETRAD_SCORES}),
    "fci": _alg(_ALL, [False], [True], [False], [False, True], {"ci_test": _TETRAD_CI, "stable": [True]}),
    "fcimax": _alg(_ALL, [False], [True], [False], [False, True], {"ci_test": _TETRAD_CI, "stable": [True]}),
    "rfci": _alg(_ALL, [False], [True], [False], [False, True], {"ci_test": _TETRAD_CI, "stable": [True]}),
    "gfci": _alg(_ALL, [False], [True], [False], [False, True],
                 {"ci_test": _TETRAD_CI, "stable": [True], "score": _TETRAD_SCORES}),
    "cfci": _alg(_ALL, [False], [True], [False], [False, True], {"ci_test": _TETRAD_CI, "stable": [True]}),
    "svarfci": _alg(_ALL, [False], [True], [True], [False, True], {"ci_test": _TETRAD_CI, "stable": [True]}),
    "svargfci": _alg(_ALL, [False], [True], [True], [False, True],
                     {"ci_test": _TETRAD_CI, "stable": [True], "score": _TETRAD_SCORES}),
    "PCMCI": _alg(_ALL, [True], [False], [True], [False], {"ci_test": _TIG_CI}),
    "PCMCI+": _alg(_ALL, [True], [False], [True], [False], {"ci_test": _TIG_CI}),
    "LPCMCI": _alg(_ALL, [False], [False], [True], [False], {"ci_test": _TIG_CI}),
    "sam": _alg(["continuous", "mixed"], [True], [True], [False], [False, True],
                {"lr": [0.001, 0.01, 0.1], "dlr": [0.0001, 0.001, 0.01], "lambda1": [1, 10, 100],
                 "lambda2": [0.0001, 0.001, 0.01], "nh": [10, 20, 50], "dnh": [100, 200, 300],
                 "train_epochs": [1000, 3000, 5000], "test_epochs": [500, 1000, 1500],
                 "batch_size": [50, 100, 200], "losstype": ["fgan", "gan", "mse"]}),
    "notears": _alg(_ALL, [True], [True], [False], [False],
                    {"max_iter": [100, 500, 1000], "h_tol": [1e-07, 1e-05, 0.001], "threshold": [0.0, 0.5, 0.8]}),
}

CI_TESTS = {
    "FisherZ": (["continuous"], [0.01, 0.05, 0.1]),
    "cci": (["continuous"], [0.01, 0.05]),
    "cg_lrt": (["mixed"], [0.01, 0.05]),
    "dg_lrt": (["mixed"], [0.01, 0.05]),
    "chisquare": (["categorical"], [0.01, 0.05]),
    "gsquare": (["categorical"], [0.01, 0.05]),
    "ParCor": (["continuous"], [0.01, 0.05]),
    "RobustParCor": (["continuous"], [0.01, 0.05]),
    "ParCorrWLS": (["continuous"], [0.01, 0.05]),
    "GPDC": (["continuous"], [0.01, 0.05]),
    "CMIknn": (["continuous"], [0.01, 0.05]),
    "Gsquared": (["categorical"], [0.01, 0.05]),
    "CMIsymb": (["categorical"], [0.01, 0.05]),
    "RegressionCI": (["mixed"], [0.01, 0.05]),
}

SCORES = {
    "sem_bic_score": (["continuous"], {"penalty_discount": [1], "structure_prior": [1]}),
    "bdeu": (["categorical"], {"penalty_discount": [1], "structure_prior": [1]}),
    "cg_bic": (["mixed"], {"penalty_discount": [1], "structure_prior": [1]}),
    "dg_bic": (["mixed"], {"penalty_discount": [1], "structure_prior": [1]}),
}

DEFAULT_CONF = {
    "causal_sufficiency": False,
    "assume_faithfulness": True,
    "alpha": 0.01,
    "n_permutations": 50,
    "folds": 10,
    "folds_to_run": 3,
    "regressor": ("RandomForestRegressor", {"n_trees": 100, "min_samples_leaf": 0.01, "max_depth": 10}),
    "exclude_algs": ["gfci", "rfci", "cfci"],
}


def create_causal_configs(data_type, causal_sufficiency=False, assume_faithfulness=True, time_series=False,
                          time_lagged=False, include_algs=None, exclude_algs=None):
    """CausalDiscoveryConfigurator.create_causal_configs: {algorithm: {param:
    [values]}} for the algorithms that fit the data and the assumptions."""
    ci_use = {k: v for k, v in CI_TESTS.items() if data_type in v[0]}
    sc_use = {k: v for k, v in SCORES.items() if data_type in v[0]}
    out = {}
    for name, a in CAUSAL_CONFIGS.items():
        if not (data_type in a["data_type"] and causal_sufficiency in a["causal_sufficiency"]
                and assume_faithfulness in a["assume_faithfulness"] and time_series in a["time_series"]
                and time_lagged in a["time_lagged"]):
            continue
        if include_algs and name not in include_algs:
            continue
        if exclude_algs and name in exclude_algs:
            continue
        cfg = {}
        for param, values in a["parameters"].items():
            if param == "ci_test":
                names = [t for t in values if t in ci_use]
                if names:
                    cfg["ci_test"] = names
                    for t in names:
                        cfg["significance_level"] = list(ci_use[t][1])
            elif param == "score":
                names = [s for s in values if s in sc_use]
                if names:
                    cfg["score"] = names
                    for s in names:
                        cfg.update({k: list(v) for k, v in sc_use[s][1].items()})
            else:
                cfg[param] = list(values)
        cfg["causal_sufficiency"] = list(a["causal_sufficiency"])
        cfg["assume_faithfulness"] = list(a["assume_faithfulness"])
        out[name] = cfg
    return out


def parameter_grid(grid):
    """sklearn ParameterGrid: sorted keys, the last key changes fastest."""
    items = sorted(grid.items())
    keys = [k for k, _ in items]
    return [dict(zip(keys, v)) for v in itertools.product(*[list(v) for _, v in items])]


def expand(algorithms):
    """All the configurations, in ETIA's order: the algorithms in their
    order, each with its ParameterGrid. Each configuration gets ``name``."""
    out = []
    for name, grid in algorithms.items():
        for params in parameter_grid(grid):
            params["name"] = name
            out.append(params)
    return out


def var_types(data, unique_val_thr=5):
    """ETIA's var_types_and_categorical_encoding: a column with fewer than 5
    distinct values is categorical (ordinal codes 0..k-1). Returns the
    encoded data, the kinds ("continuous" / "discrete") and the data type."""
    X = np.array(data, dtype=np.float64)
    kinds = []
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        if len(u) < unique_val_thr:
            X[:, j] = np.searchsorted(u, X[:, j])
            kinds.append("discrete")
        else:
            kinds.append("continuous")
    if all(k == "continuous" for k in kinds):
        dt = "continuous"
    elif all(k == "discrete" for k in kinds):
        dt = "categorical"
    else:
        dt = "mixed"
    return X, kinds, dt
