import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pathlib import Path
import numpy as np
import pytest

FIX = Path(__file__).parent.parent / "reference" / "fixtures"


@pytest.fixture(scope="session")
def graph_fixtures():
    f = np.load(FIX / "graphs.npz")
    n = int(f["n_graphs"])
    return [{k: f[f"{k}_{g}"] for k in ("A", "latent", "cpdag", "mag", "pag", "dsep")}
            for g in range(n)]
