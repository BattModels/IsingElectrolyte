"""Regression tests for the group-aware J symmetrization (branch `asymmetry-fix`).

See ``docs/asymmetry-fix.md`` for the full rationale. In short: ``sol_sol_func``
is asymmetric in its two inputs, so the generic ``default_J_sol_sol`` produced a
non-symmetric J block. These tests pin down the three properties the fix must
have:

1. with all-distinct groups it reproduces the legacy ``energetics_old`` exactly,
2. with a shared group it is invariant to how the species are listed,
3. J is symmetric in every case.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from IsingLHCE.model import energetics, energetics_old, find_root, symmetrize_pair_block
from IsingLHCE.interactions import _apply_softplus, expfunc, logfunc

# --- the trial-25 parameter layout uses the legacy `params_salt` key ----------
# Its anion-anion term has a different functional form from the generic
# `params_anion_anion`. With a single anion it only appears as the self term
# j22, so injecting the old form reproduces it exactly.

MONO = {
    "sol_params_dn": "decrease",
    "salt_params_dn": "decrease",
    "params_sol_salt_an": "decrease",
    "params_sol_sol": "decrease",
    "params_anion_anion": "none",
    "conc_factor_sol": "increase",
}


def old_J_an_an(dn_i, x_i, dn_j, x_j, p):
    """Legacy j22 = expfunc(dn_anion, p[:4]) + logfunc(x_anion, p[4])."""
    return expfunc(dn_i, p[:4]) + logfunc(x_i, p[4])


def old_J_an_an_rescale(p, monotonicity):
    return _apply_softplus(p, jnp.array([0, +1, +1, +1, -1]))


@pytest.fixture(scope="module")
def params():
    """Hand-built parameters with the legacy key layout.

    Values are arbitrary but structurally faithful (correct shapes, and a1 != a2
    in the DN-DN / AN-AN blocks so the asymmetry is actually exercised).
    """
    rng = np.random.default_rng(0)
    p = {
        "sol_params_dn": jnp.array(rng.normal(size=5)),
        "salt_params_dn": jnp.array(rng.normal(size=5)),
        "params_sol_salt_an": jnp.array(rng.normal(size=6)),
        "params_sol_sol": jnp.array(rng.normal(size=16)),
        "params_salt": jnp.array(rng.normal(size=5)),
        "conc_factor_sol": jnp.array(rng.normal(size=8)),
    }
    p["params_anion_anion"] = p["params_salt"]
    return p


def _energetics_generic(params, sol, anion, groups, pair_symmetry="group"):
    """sol: list of (dn, an, x, v); anion: (dn, x, v)."""
    dn, an, x, v = (jnp.array([s[k] for s in sol]) for k in range(4))
    return energetics(
        jnp.ones(len(sol) + 1) / (len(sol) + 1), dict(params),
        dn, an, x, v,
        jnp.array([anion[0]]), jnp.array([anion[1]]), jnp.array([anion[2]]), 1.86,
        J_an_an_func=old_J_an_an, rescale_J_an_an=old_J_an_an_rescale,
        monotonicity_dict=frozenset(MONO.items()),
        groups=groups, pair_symmetry=pair_symmetry,
    )


G4 = (16.6, 10.0, 0.2333, 310.8796)
FB = (3.0, 10.0, 0.4667, 123.5967)
TFSI = (11.2, 0.15, 209.763)


def test_all_distinct_groups_reproduce_legacy(params):
    """groups=(0,1,2) must match energetics_old bit-for-bit.

    This is the contract that lets existing LHCE results stand unchanged.
    """
    ic = jnp.array([G4[0], FB[0], TFSI[0], G4[2], FB[2], G4[1], FB[1],
                    TFSI[1], G4[3], FB[3], TFSI[2], 1.86])
    _, J_old, *_ = energetics_old(jnp.array([0.3, 0.3, 0.4]), dict(params), ic)
    _, J_new, _ = _energetics_generic(params, [G4, FB], TFSI, groups=(0, 1, 2))
    assert np.abs(np.array(J_old) - np.array(J_new)).max() < 1e-12


HEE_SOLVENTS = {
    "DME": (20.0, 10.2, 0.1735, 135.53),
    "EA": (17.1, 9.3, 0.1735, 126.54),
    "MA": (16.5, 10.7, 0.1735, 103.6),
    "PP": (17.0, 12.5, 0.1735, 89.3),
    "THF": (20.0, 8.0, 0.1735, 108.85),
}

# `energetics` re-traces on every call, so a sample of orderings is used rather
# than all 120 permutations — enough to catch an ordering leak without making
# the suite crawl.
_ALL_PERMS = list(itertools.permutations(HEE_SOLVENTS))
SAMPLED_PERMS = [_ALL_PERMS[i] for i in (0, 17, 41, 66, 93, 119)]


def _worst_coupling_shift(params, groups):
    """Largest change in any physical pair coupling across sampled orderings."""
    names = list(HEE_SOLVENTS)
    _, J_ref, _ = _energetics_generic(
        params, [HEE_SOLVENTS[k] for k in names], TFSI, groups=groups)
    J_ref = np.array(J_ref)
    worst = 0.0
    for perm in SAMPLED_PERMS:
        _, J_p, _ = _energetics_generic(
            params, [HEE_SOLVENTS[k] for k in perm], TFSI, groups=groups)
        J_p = np.array(J_p)
        for a, ka in enumerate(names):
            for b, kb in enumerate(names):
                got = J_p[perm.index(ka), perm.index(kb)]
                worst = max(worst, abs(got - J_ref[a, b]))
    return worst


def test_shared_group_is_listing_order_invariant(params):
    """Five interchangeable solvents (the HEE case): the physical couplings must
    not depend on the order they happen to be listed in."""
    assert _worst_coupling_shift(params, groups=(0, 0, 0, 0, 0, 1)) < 1e-12


def test_all_distinct_groups_are_order_dependent_by_design(params):
    """The complement of the test above, pinned down so the trade-off is explicit.

    All-distinct groups reproduce the legacy convention, in which list position
    encodes a role (index 0 = solvent, 1 = diluent). That makes the result
    order-dependent -- correct when the order is meaningful, wrong when the
    species are interchangeable. This is exactly why HEE must pass a shared group.
    """
    assert _worst_coupling_shift(params, groups=(0, 1, 2, 3, 4, 5)) > 1e-12


@pytest.mark.parametrize("groups", [(0, 1, 2), (0, 0, 1), (0, 1, 1)])
def test_j_is_symmetric(params, groups):
    _, J, _ = _energetics_generic(params, [G4, FB], TFSI, groups=groups)
    J = np.array(J)
    assert np.abs(J - J.T).max() < 1e-12


def test_pair_symmetry_none_is_asymmetric(params):
    """Guard: the pre-fix behaviour is still reachable, and is genuinely
    asymmetric — otherwise these tests would be vacuous."""
    _, J, _ = _energetics_generic(params, [G4, FB], TFSI, groups=None,
                                  pair_symmetry="none")
    J = np.array(J)
    assert np.abs(J - J.T).max() > 1e-9


def test_single_solvent_unaffected_by_grouping(params):
    """N=1 (HCE): there are no off-diagonal solvent pairs, so every grouping and
    both symmetry modes must agree exactly."""
    ref = None
    for groups, mode in [((0, 1), "group"), ((0, 0), "group"), (None, "none")]:
        _, J, _ = _energetics_generic(params, [G4], TFSI, groups=groups,
                                      pair_symmetry=mode)
        J = np.array(J)
        if ref is None:
            ref = J
        else:
            assert np.abs(J - ref).max() < 1e-12


def test_symmetrize_pair_block_diagonal_is_preserved():
    """Averaging must not touch the diagonal."""
    J = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = np.array(symmetrize_pair_block(J, (0, 0)))
    assert out[0, 0] == 1.0 and out[1, 1] == 4.0
    assert out[0, 1] == out[1, 0] == pytest.approx(2.5)


def test_symmetrize_pair_block_cross_group_takes_lower_rank():
    """Cross-group: the lower-ranked species' ordering wins in both cells."""
    J = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = np.array(symmetrize_pair_block(J, (0, 1)))
    assert out[0, 1] == out[1, 0] == 2.0     # J[0,1], the lower-rank-first value


def test_groups_length_is_validated(params):
    with pytest.raises(ValueError, match="length"):
        find_root(
            dict(params),
            {"G4": {"dn": 16.6, "an": 10.0, "x": 0.2333, "volume": 310.8796}},
            {"TFSI": {"dn": 11.2, "x": 0.15, "volume": 209.763}},
            z=1.86, groups=(0, 1, 2),
        )
