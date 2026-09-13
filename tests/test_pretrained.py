"""Tests that the shipped paper models load and reproduce the published predictions."""

import os

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from IsingLHCE.model import find_root, find_root_old
from IsingLHCE.pretrained import N_FOLDS, PAPER_Z, load_paper_ensemble, load_paper_model
from IsingLHCE.train import initialize_params

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "lhce_md")


def _species(row):
    solvents = {
        "solvent": {"dn": row["Solvent DN"], "an": row["Solvent AN"],
                    "x": row["solvent molar ratio"], "volume": row["solvent volume"]},
        "diluent": {"dn": row["Diluent DN"], "an": row["Diluent AN"],
                    "x": row["diluent molar ratio"], "volume": row["diluent volume"]},
    }
    anions = {"anion": {"dn": row["Anion DN"], "x": row["anion molar ratio"],
                        "volume": row["anion volume"]}}
    return solvents, anions


@pytest.mark.parametrize("fold", range(N_FOLDS))
def test_generic_interface_matches_legacy_solver(fold):
    """load_paper_model + find_root reproduces the legacy fit_model.py physics."""
    params, model_kwargs = load_paper_model(fold)
    test = pd.read_csv(os.path.join(DATA_DIR, f"fold_{fold}", "test.csv")).head(4)
    for _, row in test.iterrows():
        solvents, anions = _species(row)
        occ, ok = find_root(params, solvents, anions, PAPER_Z, **model_kwargs)
        constants = [row["Solvent DN"], row["Diluent DN"], row["Anion DN"],
                     row["solvent molar ratio"], row["diluent molar ratio"],
                     row["Solvent AN"], row["Diluent AN"], row["anion molar ratio"],
                     row["solvent volume"], row["diluent volume"], row["anion volume"], PAPER_Z]
        occ_old, ok_old = find_root_old(params, constants)
        assert bool(ok) and bool(ok_old)
        np.testing.assert_allclose(np.asarray(occ), np.asarray(occ_old), atol=1e-7)


def test_ensemble_reproduces_paper_dme_tte_litfsi():
    """1.0 m LiTFSI in DME:TTE 1:2 — ensemble mean published with the paper."""
    params_list, model_kwargs = load_paper_ensemble()
    solvents = {
        "DME": {"dn": 20.0, "an": 10.2, "x": 0.2434, "volume": 135.5337},
        "TTE": {"dn": 1.9, "an": 20.0, "x": 0.4868, "volume": 187.7604},
    }
    anions = {"TFSI": {"dn": 11.2, "x": 0.1349, "volume": 209.7631}}
    occ = np.mean([np.asarray(find_root(p, solvents, anions, PAPER_Z, **model_kwargs)[0])
                   for p in params_list], axis=0)
    np.testing.assert_allclose(occ, [0.38359, 0.00025, 0.61615], atol=5e-4)


def test_load_paper_model_rejects_bad_fold():
    with pytest.raises(ValueError):
        load_paper_model(N_FOLDS)


def test_initialize_params_defaults_to_from_scratch():
    params = initialize_params()
    assert set(params) == {"sol_params_dn", "salt_params_dn", "params_sol_salt_an",
                           "params_sol_sol", "params_anion_anion", "conc_factor_sol"}
    assert all(jnp.all(jnp.isfinite(v)) for v in params.values())


def test_analysis_imports_without_author_filesystem():
    import IsingLHCE.analysis as analysis

    assert "DME" in analysis.solvent_map_dict
