"""Conditional independence tests. Names follow the ETIA configuration
(``ETIA/CausalLearning/configurations/ci_tests.py``)."""
from .base import CITest
from .gaussian import FisherZ, ParCorr, RobustParCorr, ParCorrWLS, DGLrt
from .discrete import ChiSquare, GSquare, Gsquared, CMIsymb
from .mixed import RegressionCI, CGLrt
from .nonparametric import CMIknn, GPDC, CCI
from ..graphs.separation import DSepOracle

CI_TESTS = {
    # Tetrad tests in ETIA
    "FisherZ": FisherZ,
    "cci": CCI,
    "cg_lrt": CGLrt,
    "dg_lrt": DGLrt,
    "chisquare": ChiSquare,
    "gsquare": GSquare,
    # tigramite tests in ETIA
    "ParCor": ParCorr,
    "RobustParCor": RobustParCorr,
    "ParCorrWLS": ParCorrWLS,
    "GPDC": GPDC,
    "CMIknn": CMIknn,
    "Gsquared": Gsquared,
    "CMIsymb": CMIsymb,
    "RegressionCI": RegressionCI,
}


def make_ci_test(name, data, kinds=None, **params):
    return CI_TESTS[name](data, kinds, **params)
