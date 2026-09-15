"""Does the model produce physically valid, differentiable predictions?"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from IsingElectrolyte.model import find_root, li_free_energy
from IsingElectrolyte.pretrained import PAPER_Z
from IsingElectrolyte.train import initialize_params


def test_prediction_is_a_valid_occupation(paper_model, lhce):
    params, kw = paper_model
    solvents, anions = lhce
    occ, ok = find_root(params, solvents, anions, PAPER_Z, **kw)
    occ = np.asarray(occ)
    assert bool(ok)
    assert occ.shape == (3,)
    assert np.all((occ >= 0) & (occ <= 1))
    assert occ.sum() == pytest.approx(1.0, abs=1e-6)


def test_prediction_matches_readme_example(paper_model, lhce):
    """README quick start: DME/TTE/LiTFSI shell is anion-rich with the diluent excluded."""
    params, kw = paper_model
    occ, _ = find_root(params, *lhce, PAPER_Z, **kw)
    np.testing.assert_allclose(np.asarray(occ), [0.3887, 0.0002, 0.6110], atol=1e-3)


def test_output_order_is_solvents_then_anions(paper_model, lhce):
    """Swapping solvent roles changes the answer, and outputs follow the dict order."""
    params, kw = paper_model
    solvents, anions = lhce
    occ, _ = find_root(params, solvents, anions, PAPER_Z, **kw)
    swapped = {"TTE": solvents["TTE"], "DME": solvents["DME"]}
    occ_swapped, ok = find_root(params, swapped, anions, PAPER_Z, groups=(1, 0, 2), **kw)
    assert bool(ok)
    # same roles (DME ranked first via groups), listed in the opposite order
    np.testing.assert_allclose(np.asarray(occ_swapped)[[1, 0, 2]], np.asarray(occ), atol=1e-7)


def test_failed_solve_is_reported(paper_model, lhce):
    """An unphysical energy (h = -1000 eV) has no solution: NaN occupations and found_valid=False."""
    params, kw = paper_model
    unphysical = lambda dn_eff, x, p: -1.0e3 + 0.0 * dn_eff
    occ, ok = find_root(params, *lhce, PAPER_Z, h_sol_func=unphysical, **kw)
    assert not bool(ok)
    assert np.all(np.isnan(np.asarray(occ)))


def test_li_free_energy_is_finite_and_negative(paper_model, lhce):
    params, kw = paper_model
    G = float(li_free_energy(params, *lhce, PAPER_Z, **kw))
    assert np.isfinite(G) and G < 0


def test_occupations_are_differentiable(paper_model, lhce):
    """Gradients flow through the root solver to every trained parameter group."""
    params, kw = paper_model
    anion_occupation = lambda p: find_root(p, *lhce, PAPER_Z, **kw)[0][2]
    grads = jax.grad(anion_occupation)(params)
    for key, g in grads.items():
        assert bool(jnp.all(jnp.isfinite(g))), key
    trained = set(params) - {"params_salt"}  # params_salt is an alias read via params_anion_anion
    assert all(float(jnp.abs(grads[k]).max()) > 0 for k in trained)


def test_three_solvents_two_anions(physical_monotonicity):
    """The generic interface handles N solvents + M anions (default interaction terms)."""
    params = initialize_params(random_seed=42)
    solvents = {
        "DME": {"dn": 20.0, "an": 10.2, "x": 0.2434, "volume": 135.53},
        "TTE": {"dn": 1.9, "an": 20.0, "x": 0.4868, "volume": 187.76},
        "DMC": {"dn": 17.1, "an": 20.0, "x": 0.10, "volume": 113.92},
    }
    anions = {
        "TFSI": {"dn": 11.2, "x": 0.08, "volume": 209.76},
        "FSI": {"dn": 10.0, "x": 0.05, "volume": 141.09},
    }
    occ, ok = find_root(params, solvents, anions, PAPER_Z, groups=(0, 1, 0, 2, 2),
                        monotonicity_dict=physical_monotonicity)
    occ = np.asarray(occ)
    assert bool(ok)
    assert occ.shape == (5,)
    assert np.all((occ >= 0) & (occ <= 1))
    assert occ.sum() == pytest.approx(1.0, abs=1e-6)
    G = li_free_energy(params, solvents, anions, PAPER_Z, groups=(0, 1, 0, 2, 2),
                       monotonicity_dict=physical_monotonicity)
    assert np.isfinite(float(G))
