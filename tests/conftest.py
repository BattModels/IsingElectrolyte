"""Shared fixtures. Session-scoped so the JAX solver compiles once per test run."""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # tests never open plot windows

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
DATA = EXAMPLES / "data" / "lhce_md"


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def paper_model():
    """(params, model_kwargs) of cross-validation fold 0 of the paper."""
    from IsingElectrolyte.pretrained import load_paper_model

    return load_paper_model(0)


@pytest.fixture(scope="session")
def lhce():
    """1.0 m LiTFSI in DME:TTE = 1:2 (mol:mol), the README example: (solvents, anions)."""
    solvents = {
        "DME": {"dn": 20.0, "an": 10.2, "x": 0.2434, "volume": 135.53},
        "TTE": {"dn": 1.9, "an": 20.0, "x": 0.4868, "volume": 187.76},
    }
    anions = {"TFSI": {"dn": 11.2, "x": 0.1349, "volume": 209.76}}
    return solvents, anions


@pytest.fixture(scope="session")
def physical_monotonicity():
    return {
        "sol_params_dn": "decrease",
        "salt_params_dn": "decrease",
        "params_sol_salt_an": "decrease",
        "params_sol_sol": "decrease",
        "params_anion_anion": "increase",
        "conc_factor_sol": "increase",
    }
