"""ETIA's CausalLearner on top of the OCT (increment 9).

``CausalLearner(data).learn_model()`` makes the configurations of ETIA
(``configs.create_causal_configs`` with ETIA's default settings: no causal
sufficiency, faithfulness, without GFCI, RFCI and CFCI), runs the OCT and
returns ETIA's result dict. ``etia_compat`` and ``etia_n_jobs`` go to
``run_oct``.

Two methods of ETIA stop with an error; ``etia_compat=True`` raises the
same errors, the default mode does the intended work:

* ``get_best_model_between_algorithms``: ETIA unpacks the three values of
  ``model.run`` into two names (``ValueError``).
* ``get_best_model_between_family``: ETIA compares a list (for example
  ``[False]``) with a bool, so a given ``causal_sufficiency`` or
  ``assume_faithfulness`` removes all the algorithms (``RuntimeError``),
  and ``accepts_missing_values`` reads a key that the table does not have
  (``KeyError``).
"""
import time

import numpy as np

from .configs import CAUSAL_CONFIGS, DEFAULT_CONF, create_causal_configs, expand, var_types
from .oct import run_oct, oct_select
from .algorithms import run_config


class CausalLearner:
    def __init__(self, data, algorithms=None, kinds=None, alpha=None, n_permutations=None, folds=None,
                 folds_to_run=None, regressor=None, include_algs=None, exclude_algs=None, seed=0,
                 etia_compat=False, etia_n_jobs=None):
        """``data``: an array or a DataFrame. ``algorithms``: {name: {param:
        [values]}} as ETIA's ``CausalDiscoveryAlgorithms``; None gives ETIA's
        default configurations for the data type. ``kinds``: "continuous" /
        "discrete" per column; None uses ETIA's rule (fewer than 5 distinct
        values: categorical)."""
        self.columns = list(getattr(data, "columns", [])) or None
        X = np.asarray(data, dtype=np.float64)
        Xe, k, dt = var_types(X)
        if kinds is None:
            X, kinds = Xe, k
        else:
            dt = ("continuous" if all(v == "continuous" for v in kinds) else
                  "categorical" if all(v == "discrete" for v in kinds) else "mixed")
        self.X, self.kinds, self.data_type = X, list(kinds), dt
        if algorithms is None:
            exclude = DEFAULT_CONF["exclude_algs"] if exclude_algs is None else exclude_algs
            algorithms = create_causal_configs(dt, DEFAULT_CONF["causal_sufficiency"],
                                               DEFAULT_CONF["assume_faithfulness"], include_algs=include_algs,
                                               exclude_algs=exclude)
        self.algorithms = algorithms
        self.configs = expand(algorithms)
        self.params = dict(alpha=DEFAULT_CONF["alpha"] if alpha is None else alpha,
                           n_permutations=DEFAULT_CONF["n_permutations"] if n_permutations is None else n_permutations,
                           folds=DEFAULT_CONF["folds"] if folds is None else folds,
                           folds_to_run=DEFAULT_CONF["folds_to_run"] if folds_to_run is None else folds_to_run,
                           regressor=DEFAULT_CONF["regressor"] if regressor is None else regressor)
        self.seed, self.etia_compat, self.etia_n_jobs = seed, etia_compat, etia_n_jobs
        self.result = None

    def learn_model(self):
        """Run the OCT. Returns ETIA's dict: ``optimal_conf``,
        ``matrix_mec_graph``, ``matrix_graph``, ``run_time`` (a string, as
        ETIA) and ``library_results`` (None)."""
        t0 = time.time()
        self.result = run_oct(self.X, self.configs, self.kinds, seed=self.seed, etia_compat=self.etia_compat,
                              etia_n_jobs=self.etia_n_jobs, **self.params)
        r = self.result
        return {"optimal_conf": r["config"], "matrix_mec_graph": r["mec_graph"], "matrix_graph": r["graph"],
                "run_time": str(time.time() - t0), "library_results": None}

    def _best_among(self, indices):
        r = self.result
        if r is None:
            raise RuntimeError("Run learn_model() first.")
        if not indices:
            raise RuntimeError("Cannot find best configuration among algorithms that have not been run yet.")
        mu = r["mean_mu"]
        best = indices[int(np.argmax(mu[indices]))]
        chosen = best
        for c in indices:
            if r["is_equal"][c] and r["mean_mb"][c] < r["mean_mb"][chosen]:
                chosen = c
        return chosen

    def get_best_model_between_algorithms(self, algorithms):
        """The best configuration of the given algorithms, learned on all the
        rows: (config, class graph, DAG or MAG)."""
        indices = [i for i, c in enumerate(self.configs) if c["name"] in algorithms]
        chosen = self._best_among(indices)
        if self.etia_compat:
            raise ValueError("too many values to unpack (expected 2)")
        mec, g = run_config(self.configs[chosen], self.X, self.kinds, False, seed=self.seed)
        return self.configs[chosen], mec, g

    def get_best_model_between_family(self, causal_sufficiency=None, assume_faithfulness=None, is_output_mec=None,
                                      accepts_missing_values=None):
        algs = []
        for name, a in CAUSAL_CONFIGS.items():
            if self.etia_compat:
                if causal_sufficiency is not None and a["causal_sufficiency"] != causal_sufficiency:
                    continue
                if assume_faithfulness is not None and a["assume_faithfulness"] != assume_faithfulness:
                    continue
                if accepts_missing_values is not None:
                    raise KeyError("missing_values")
            else:
                if causal_sufficiency is not None and causal_sufficiency not in a["causal_sufficiency"]:
                    continue
                if assume_faithfulness is not None and assume_faithfulness not in a["assume_faithfulness"]:
                    continue
                if accepts_missing_values:
                    continue                    # no algorithm of ETIA-JAX accepts missing values
            algs.append(name)
        return self.get_best_model_between_algorithms(algs)
