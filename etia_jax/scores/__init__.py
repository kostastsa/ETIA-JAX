"""The scores of ETIA (increment 5): SEM BIC, BDeu, discrete BIC,
conditional Gaussian BIC and degenerate Gaussian BIC."""
from .base import Score
from .gaussian import SemBicScore, DegenerateGaussianScore
from .discrete import BdeuScore, DiscreteBicScore
from .mixed import ConditionalGaussianScore

SCORES = {
    "sem_bic_score": SemBicScore,
    "bdeu": BdeuScore,
    "discrete_bic": DiscreteBicScore,
    "cg_bic": ConditionalGaussianScore,
    "dg_bic": DegenerateGaussianScore,
}

# ETIA's settings (configurations/scores.py and TetradAlgorithm._score)
ETIA_SETTINGS = {
    "sem_bic_score": {"penalty_discount": 1.0},
    "bdeu": {"structure_prior": 1.0},
    "discrete_bic": {"penalty_discount": 1.0, "structure_prior": 1.0},
    "cg_bic": {"penalty_discount": 1.0},
    "dg_bic": {"penalty_discount": 1.0},
}
