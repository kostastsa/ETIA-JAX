"""Run one causal configuration of ETIA on some rows of the data (increment 9).

``run_config`` is the ``algorithm.run(data, parameters)`` step of ETIA's
k-fold protocol and of the last step of the OCT. It returns the graph of
the class (CPDAG, PAG or DAG) and one DAG or MAG of the class:

* ``etia_compat=True``: the searches in their Tetrad modes, and the random
  graph of ETIA's Tetrad wrapper: ``dagFromCpdag`` when
  ``causal_sufficiency`` is true, else ``magFromPag`` (also for FGES).
  ETIA's wrappers of NOTEARS, SAM and tigramite return two values, not
  three, so the k-fold protocol of ETIA stops with a ``ValueError``; this
  mode raises the same error.
* default: the textbook searches, and the fixed member of
  ``graphs.markov.member``. NOTEARS and SAM run on the rows of the fold.
"""
import inspect

import numpy as np

from ..ci import CI_TESTS
from ..scores import SCORES, ETIA_SETTINGS
from ..graphs.markov import member
from ..search.pcalg import pc, cpc
from ..search.fges import fges
from ..search.fci import fci, fcimax, rfci, cfci
from ..search.gfci import gfci
from ..search.notears import notears
from ..search.sam import sam
from ..search.tetrad_members import tetrad_dag_from_cpdag, tetrad_mag_from_pag

TETRAD = ("pc", "cpc", "fges", "fci", "fcimax", "rfci", "gfci", "cfci")
TWO_VALUES = ("notears", "sam", "PCMCI", "PCMCI+", "LPCMCI")
TIME_SERIES = ("svarfci", "svargfci", "PCMCI", "PCMCI+", "LPCMCI")


def _make(cls, X, kinds, compat):
    kw = {}
    if "tetrad_compat" in inspect.signature(cls.__init__).parameters:
        kw["tetrad_compat"] = compat
    return cls(X, kinds, **kw)


def _score(params, X, kinds, compat):
    name = params["score"]
    st = dict(ETIA_SETTINGS[name])
    if "penalty_discount" in params and "penalty_discount" in st:
        st["penalty_discount"] = float(params["penalty_discount"])
    if name == "bdeu" and "structure_prior" in params:
        st["structure_prior"] = float(params["structure_prior"])
    return SCORES[name](X, kinds, tetrad_compat=compat, **st)


def search(params, X, kinds=None, etia_compat=False, seed=0):
    """The graph of the class (mark matrix) of one configuration on X."""
    name = params["name"]
    X = np.asarray(X, dtype=np.float64)
    if name in TIME_SERIES:
        raise NotImplementedError(f"{name}: the OCT of ETIA-JAX has no time-series algorithms yet")
    if name in TWO_VALUES and etia_compat:
        raise ValueError("not enough values to unpack (expected 3, got 2)")
    if name in TETRAD:
        ci = _make(CI_TESTS[params["ci_test"]], X, kinds, etia_compat) if "ci_test" in params else None
        a = params.get("significance_level", 0.05)
        if name == "pc":
            return pc(ci, a, stable=params.get("stable", True), tetrad_compat=etia_compat).graph
        if name == "cpc":
            return cpc(ci, a, stable=params.get("stable", True), tetrad_compat=etia_compat).graph
        if name == "fges":
            return fges(_score(params, X, kinds, etia_compat), tetrad_compat=etia_compat).graph
        if name == "gfci":
            return gfci(ci, _score(params, X, kinds, etia_compat), a, tetrad_compat=etia_compat).graph
        f = {"fci": fci, "fcimax": fcimax, "rfci": rfci, "cfci": cfci}[name]
        return f(ci, a, tetrad_compat=etia_compat).graph
    if name == "notears":
        return notears(X, threshold=params.get("threshold", 0.8), max_iter=params.get("max_iter", 100),
                       h_tol=params.get("h_tol", 1e-8)).G
    if name == "sam":
        keys = ("lr", "dlr", "lambda1", "lambda2", "nh", "dnh", "train_epochs", "test_epochs", "batch_size",
                "losstype")
        kw = {k: params[k] for k in keys if k in params}
        mixed = kinds is not None and any(k == "discrete" for k in kinds)
        return sam(X, mixed_data=mixed, seed=seed, **kw).G
    raise ValueError(f"unknown algorithm {name}")


def run_config(params, X, kinds=None, etia_compat=False, rng=None, seed=0):
    """(class graph, one DAG or MAG of it) of one configuration on X."""
    mec = np.asarray(search(params, X, kinds, etia_compat, seed), dtype=np.int8)
    if etia_compat:
        rng = rng if rng is not None else np.random.default_rng(seed)
        if params.get("causal_sufficiency", False):
            return mec, tetrad_dag_from_cpdag(mec, rng)
        return mec, tetrad_mag_from_pag(mec, rng)
    return mec, member(mec)
