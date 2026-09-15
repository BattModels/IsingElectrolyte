import jax
import jax.numpy as jnp
import jax.nn as jnn
from jaxopt import Broyden
from functools import partial

from .interactions import (
    expfunc, logfunc, sol_sol_func,
    default_h_sol, default_h_an,
    default_J_sol_sol, default_J_sol_an, default_J_an_an,
    default_h_sol_rescale, default_h_an_rescale,
    default_J_sol_sol_rescale, default_J_sol_an_rescale, default_J_an_an_rescale,
    default_conc_factor_rescale,
)
from .conc_vol_correction import conc_factor_sep_x_v_sigmoid

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)

# ---------------------------------------------------------------------------
# Monotonicity configuration
# ---------------------------------------------------------------------------

#: Default monotonicity constraints for each parameter group.
#: Pass a modified copy of this dict as ``monotonicity_dict`` to any top-level
#: function (``find_root``, ``li_free_energy``, ``get_root_error``, …) to
#: experiment with different physical assumptions without touching the code.
#:
#: Allowed values per key:
#:   "decrease" — the interaction strength decreases as the descriptor increases
#:   "increase" — the interaction strength increases as the descriptor increases
#:   "none"     — no monotonicity constraint; parameters are left unconstrained
# Example:
# DEFAULT_MONOTONICITY = {
#     "sol_params_dn":      "decrease",   # h(Li-sol) decreases with DN
#     "salt_params_dn":     "decrease",   # h(Li-anion) decreases with DN
#     "params_sol_salt_an": "decrease",   # J(sol-anion) decreases with DN/AN
#     "params_sol_sol":     "decrease",   # J(sol-sol) decreases with DN/AN cross
#     "params_anion_anion": "increase",   # J(anion-anion) increases with DN
#     "conc_factor_sol":    "increase",   # concentration factor increases with x, V
# }
DEFAULT_MONOTONICITY = {
    "sol_params_dn":      "none",       # no constraint
    "salt_params_dn":     "none",       # no constraint
    "params_sol_salt_an": "none",       # no constraint
    "params_sol_sol":     "none",       # no constraint
    "params_anion_anion": "none",       # no constraint
    "conc_factor_sol":    "none",       # no constraint
}

def _freeze_mono(d):
    """Convert a monotonicity dict to a hashable frozenset for JAX static args.

    JAX requires static arguments to be hashable.  Plain Python dicts are not,
    so we convert to ``frozenset(d.items())`` at every JIT boundary.  The
    inverse (``dict(frozen)``) is used inside ``rescale_input_params`` for the
    string lookups.  Users always interact with plain dicts.
    """
    if isinstance(d, frozenset):
        return d
    return frozenset(d.items())


# ---------------------------------------------------------------------------
# Pair-block symmetrization
# ---------------------------------------------------------------------------

#: Default symmetrization mode for the J_ss / J_aa blocks.  See
#: ``symmetrize_pair_block`` and ``docs/asymmetry-fix.md`` for the rationale.
DEFAULT_PAIR_SYMMETRY = "group"


def symmetrize_pair_block(J, groups):
    """Make a same-kind J block symmetric, respecting species group structure.

    ``sol_sol_func`` is **not** symmetric in its two inputs — it computes
    ``L0 + H0/(1+exp(a0 + a1*dn + a2*an))`` and fitted ``a1 != a2``.  The
    generic ``default_J_sol_sol`` feeds ``(dn_i, dn_j)`` into the DN-DN block
    and ``(an_i, an_j)`` into the AN-AN block, so ``J[i,j] != J[j,i]``.  That is
    invalid for a pair coupling: the energy of an (i,j) pair cannot depend on
    which partner is named first.

    In those two blocks *both* slots receive the same kind of descriptor (two
    donor numbers, or two acceptor numbers), so ``a1 != a2`` carries no physical
    meaning — it only records which partner was written first.  This function
    removes that artifact while preserving the one ordering that *is* physical:
    the role ordering between different species groups.

    Rules:
      * **same group** (interchangeable partners, e.g. the five HEE solvents) →
        average the two orderings.  Symmetric and invariant to how the species
        are listed.
      * **different groups** (distinct roles, e.g. solvent vs diluent) → the
        species from the lower-ranked group takes the first slot.  Symmetric and
        independent of list position, because the choice follows group rank.

    The diagonal is untouched (``0.5*(J_ii + J_ii) == J_ii``), so single-species
    blocks are bit-for-bit unchanged.

    With ``groups`` all-distinct this reproduces the legacy ``fit_model.py``
    convention exactly (verified to 2.2e-16 on the 2-solvent LHCE case), because
    the old code hardcoded the same argument order into both ``j01`` and ``j10``.

    Args:
        J:      square block, shape (n, n)
        groups: integer group rank per species, shape (n,)

    Returns:
        Symmetric block of shape (n, n).
    """
    grp  = jnp.asarray(groups)
    same = grp[:, None] == grp[None, :]
    lo   = grp[:, None] <  grp[None, :]
    return jnp.where(same, 0.5 * (J + J.T),      # in-group: interchangeable
                     jnp.where(lo, J, J.T))      # cross-group: lower rank first


def _resolve_groups(groups, n_species):
    """Normalize ``groups`` to a hashable tuple of length ``n_species``.

    ``None`` means all-distinct (``0, 1, ..., n-1``), which reproduces the
    legacy per-species ordering and keeps existing results unchanged.
    """
    if groups is None:
        return tuple(range(n_species))
    groups = tuple(int(g) for g in groups)
    if len(groups) != n_species:
        raise ValueError(
            f"groups has length {len(groups)} but there are {n_species} species "
            f"(solvents first, then anions)."
        )
    return groups


# ---------------------------------------------------------------------------
# Legacy fixed-species functions (2 solvents + 1 anion)
# ---------------------------------------------------------------------------

@jax.jit
def rescale_input_params_old(input_params):
    """
    Rescale input parameters to ensure monotonicity
    """
    sol_params_dn_tmp = input_params["sol_params_dn"]
    salt_params_dn_tmp = input_params["salt_params_dn"]
    params_sol_salt_an_tmp = input_params["params_sol_salt_an"]
    params_sol_sol_tmp = input_params["params_sol_sol"]
    params_salt_tmp = input_params["params_salt"]
    conc_factor_sol = input_params["conc_factor_sol"]
    # sol_params_dn_tmp need to ensure Li-sol interaction strictly decreasing with DN, so a1a2a3<0, we set a1,a2 positive, a3 negative
    sol_params_dn_tmp = sol_params_dn_tmp.at[1].set(jnn.softplus(sol_params_dn_tmp[1]))
    sol_params_dn_tmp = sol_params_dn_tmp.at[2].set(jnn.softplus(sol_params_dn_tmp[2]))
    sol_params_dn_tmp = sol_params_dn_tmp.at[3].set(-jnn.softplus(sol_params_dn_tmp[3]))
    sol_params_dn_tmp = sol_params_dn_tmp.at[4].set(-jnn.softplus(sol_params_dn_tmp[4]))
    # salt_params_dn_tmp need to ensure Li-anion interaction strictly decreasing with DN, so a1 should be negative, a2, a3 positive
    salt_params_dn_tmp = salt_params_dn_tmp.at[1].set(jnn.softplus(salt_params_dn_tmp[1]))
    salt_params_dn_tmp = salt_params_dn_tmp.at[2].set(jnn.softplus(salt_params_dn_tmp[2]))
    salt_params_dn_tmp = salt_params_dn_tmp.at[3].set(-jnn.softplus(salt_params_dn_tmp[3]))
    salt_params_dn_tmp = salt_params_dn_tmp.at[4].set(-jnn.softplus(salt_params_dn_tmp[4]))
    # params_sol_salt_an_tmp need to ensure solvent-anion interaction strictly decreasing with DN and AN, so L, a1, a2 should be positive
    params_sol_salt_an_tmp = params_sol_salt_an_tmp.at[1].set(jnn.softplus(params_sol_salt_an_tmp[1]))
    params_sol_salt_an_tmp = params_sol_salt_an_tmp.at[3].set(jnn.softplus(params_sol_salt_an_tmp[3]))
    params_sol_salt_an_tmp = params_sol_salt_an_tmp.at[4].set(jnn.softplus(params_sol_salt_an_tmp[4]))
    params_sol_salt_an_tmp = params_sol_salt_an_tmp.at[5].set(-jnn.softplus(params_sol_salt_an_tmp[5]))
    # params_sol_sol_tmp need to ensure solvent-solvent interaction DN-AN part strictly decreasing with DN and AN, so H0, a1, a2 positive
    # DN-DN part strictly increasing with DN, so a1, a2 negative, H0 positive
    # AN-AN part strictly increasing with AN, so a1, a2 negative, H0 positive
    params_sol_sol_tmp = params_sol_sol_tmp.at[1].set(jnn.softplus(params_sol_sol_tmp[1]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[3].set(jnn.softplus(params_sol_sol_tmp[3]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[4].set(jnn.softplus(params_sol_sol_tmp[4]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[6].set(jnn.softplus(params_sol_sol_tmp[6]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[8].set(-jnn.softplus(params_sol_sol_tmp[8]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[9].set(-jnn.softplus(params_sol_sol_tmp[9]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[11].set(jnn.softplus(params_sol_sol_tmp[11]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[13].set(-jnn.softplus(params_sol_sol_tmp[13]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[14].set(-jnn.softplus(params_sol_sol_tmp[14]))
    params_sol_sol_tmp = params_sol_sol_tmp.at[15].set(-jnn.softplus(params_sol_sol_tmp[15]))
    # params_salt_tmp need to ensure anion-anion interaction strictly increasing with DN, so a1, a2, a3 positive
    params_salt_tmp = params_salt_tmp.at[1].set(jnn.softplus(params_salt_tmp[1]))
    params_salt_tmp = params_salt_tmp.at[2].set(jnn.softplus(params_salt_tmp[2]))
    params_salt_tmp = params_salt_tmp.at[3].set(jnn.softplus(params_salt_tmp[3]))
    params_salt_tmp = params_salt_tmp.at[4].set(-jnn.softplus(params_salt_tmp[4]))
    # conc_factor_sol need to ensure increasing with x and V, so a1, a3, negative, a2, b2, b1, b3 positive
    conc_factor_sol = conc_factor_sol.at[1].set(-jnn.softplus(conc_factor_sol[1]))
    conc_factor_sol = conc_factor_sol.at[2].set(jnn.softplus(conc_factor_sol[2]))
    conc_factor_sol = conc_factor_sol.at[3].set(-jnn.softplus(conc_factor_sol[3]))
    conc_factor_sol = conc_factor_sol.at[5].set(jnn.softplus(conc_factor_sol[5]))
    conc_factor_sol = conc_factor_sol.at[6].set(jnn.softplus(conc_factor_sol[6]))
    conc_factor_sol = conc_factor_sol.at[7].set(jnn.softplus(conc_factor_sol[7]))

    # rewrite back to input_params
    input_params["sol_params_dn"] = sol_params_dn_tmp
    input_params["salt_params_dn"] = salt_params_dn_tmp
    input_params["params_sol_salt_an"] = params_sol_salt_an_tmp
    input_params["params_sol_sol"] = params_sol_sol_tmp
    input_params["params_salt"] = params_salt_tmp
    input_params["conc_factor_sol"] = conc_factor_sol
    return input_params

def energetics_old(vars, input_params, input_constants, kT = 0.0257):
    # Rescale input parameters to ensure monotonicity
    input_params = rescale_input_params_old(input_params)
    # Unpack input parameters.
    sol_params_dn_tmp = input_params["sol_params_dn"]
    salt_params_dn_tmp = input_params["salt_params_dn"]
    params_sol_salt_an_tmp = input_params["params_sol_salt_an"]
    params_sol_sol_tmp = input_params["params_sol_sol"]
    params_salt_tmp = input_params["params_salt"]
    conc_factor_sol = input_params["conc_factor_sol"]

    (
        dn0,
        dn1,
        dn_anion,
        x0,
        x1,
        an0,
        an1,
        x_anion,
        solvent_volume,
        diluent_volume,
        anion_volume,
        z,
    ) = input_constants

    v_avg = x0 * solvent_volume + x1 * diluent_volume + x_anion * anion_volume
    v0 = solvent_volume
    v1 = diluent_volume
    v_anion = anion_volume
    phi_anion = 1.0

    dn0 = dn0 * conc_factor_sep_x_v_sigmoid(x0, x_anion, v0, v_anion, conc_factor_sol)
    an0 = an0 * conc_factor_sep_x_v_sigmoid(x0, x_anion, v0, v_anion, conc_factor_sol)
    dn1 = dn1 * conc_factor_sep_x_v_sigmoid(x1, x_anion, v1, v_anion, conc_factor_sol)
    an1 = an1 * conc_factor_sep_x_v_sigmoid(x1, x_anion, v1, v_anion, conc_factor_sol)
    dn_anion = dn_anion * phi_anion
    anion_charge_density = -1 / anion_volume

    avg_m, avg_n, avg_l = vars
    h0 = expfunc(dn0, sol_params_dn_tmp[:4]) + logfunc(x0, sol_params_dn_tmp[4])
    h1 = expfunc(dn1, sol_params_dn_tmp[:4]) + logfunc(x1, sol_params_dn_tmp[4])
    h2 = expfunc(dn_anion, salt_params_dn_tmp[:4]) + logfunc(
        x_anion, salt_params_dn_tmp[4]
    )
    h = jnp.array([h0, h1, h2]).flatten()
    j00 = (
        sol_sol_func(jnp.array([dn0, an0]), params_sol_sol_tmp[:5])
        + sol_sol_func(jnp.array([dn0, an0]), params_sol_sol_tmp[:5])
        + sol_sol_func(jnp.array([dn0, dn0]), params_sol_sol_tmp[5:10])
        + sol_sol_func(jnp.array([an0, an0]), params_sol_sol_tmp[10:15])
        + 2.0 * logfunc(x0, params_sol_sol_tmp[15])
    )
    j01 = (
        sol_sol_func(jnp.array([dn0, an1]), params_sol_sol_tmp[:5])
        + sol_sol_func(jnp.array([dn1, an0]), params_sol_sol_tmp[:5])
        + sol_sol_func(jnp.array([dn0, dn1]), params_sol_sol_tmp[5:10])
        + sol_sol_func(jnp.array([an0, an1]), params_sol_sol_tmp[10:15])
        + logfunc(x0, params_sol_sol_tmp[15])
        + logfunc(x1, params_sol_sol_tmp[15])
    )
    j10 = (
        sol_sol_func(jnp.array([dn1, an0]), params_sol_sol_tmp[:5])
        + sol_sol_func(jnp.array([dn0, an1]), params_sol_sol_tmp[:5])
        + sol_sol_func(jnp.array([dn0, dn1]), params_sol_sol_tmp[5:10])
        + sol_sol_func(jnp.array([an0, an1]), params_sol_sol_tmp[10:15])
        + logfunc(x0, params_sol_sol_tmp[15])
        + logfunc(x1, params_sol_sol_tmp[15])
    )
    j11 = (
        sol_sol_func(jnp.array([dn1, an1]), params_sol_sol_tmp[:5])
        + sol_sol_func(jnp.array([dn1, an1]), params_sol_sol_tmp[:5])
        + sol_sol_func(jnp.array([dn1, dn1]), params_sol_sol_tmp[5:10])
        + sol_sol_func(jnp.array([an1, an1]), params_sol_sol_tmp[10:15])
        + 2.0 * logfunc(x1, params_sol_sol_tmp[15])
    )
    j02 = (
        sol_sol_func(jnp.array([dn_anion,an0,]), params_sol_salt_an_tmp[:5])
        + logfunc(x0, params_sol_salt_an_tmp[5])
        + logfunc(x_anion, params_sol_salt_an_tmp[5])
    )
    j20 = (
        sol_sol_func(jnp.array([dn_anion,an0,]), params_sol_salt_an_tmp[:5])
        + logfunc(x0, params_sol_salt_an_tmp[5])
        + logfunc(x_anion, params_sol_salt_an_tmp[5])
    )
    j12 = (
        sol_sol_func(jnp.array([dn_anion, an1,]), params_sol_salt_an_tmp[:5])
        + logfunc(x1, params_sol_salt_an_tmp[5])
        + logfunc(x_anion, params_sol_salt_an_tmp[5])
    )
    j21 = (
        sol_sol_func(jnp.array([dn_anion, an1]), params_sol_salt_an_tmp[:5])
        + logfunc(x1, params_sol_salt_an_tmp[5])
        + logfunc(x_anion, params_sol_salt_an_tmp[5])
    )
    j22 = expfunc(dn_anion, params_salt_tmp[:4]) + logfunc(x_anion, params_salt_tmp[4])

    J = jnp.array([[j00, j01, j02], [j10, j11, j12], [j20, j21, j22]])

    return h, J, avg_m, avg_n, avg_l, kT, z

def equations_old(vars, input_params, input_constants):
    h, J, avg_m, avg_n, avg_l, kT, z = energetics_old(vars, input_params, input_constants)
    exp0 = jnp.exp(
        -(
            h[0]
            + J[0, 0] * z * avg_m
            + J[0, 1] * avg_n * z / 2.0
            + J[0, 2] * avg_l * z / 2.0
        )
        / kT
    )
    exp1 = jnp.exp(
        -(
            h[1]
            + J[1, 1] * z * avg_n
            + J[1, 0] * avg_m * z / 2.0
            + J[1, 2] * avg_l * z / 2.0
        )
        / kT
    )
    exp2 = jnp.exp(
        -(
            h[2]
            + J[2, 2] * z * avg_l
            + J[2, 0] * avg_m * z / 2.0
            + J[2, 1] * avg_n * z / 2.0
        )
        / kT
    )
    partition = exp0 + exp1 + exp2
    f1 = exp0 / partition - avg_m
    f3 = exp2 / partition - avg_l
    f2 = exp1 / partition - avg_n
    return jnp.array([f1, f2, f3])

BROYDEN_SOLVER_OLD = Broyden(fun=equations_old, maxiter=1000, tol=1e-8, verbose=False, jit=True)

@partial(jax.jit, static_argnames=("max_tries",))
def find_root_old(input_params, input_constants,
              initial_guess=jnp.array([0.5, 0.0, 0.5]),
              max_tries=10):

    guesses = jnp.array([
        initial_guess,
        [0.3, 0.0, 0.3],
        [0.7, 0.0, 0.7],
        [0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8],
        [0.1, 0.0, 0.9],
        [0.9, 0.0, 0.1],
        [0.4, 0.3, 0.6],
        [0.6, 0.4, 0.3],
    ])
    guesses = guesses[:max_tries]

    def try_solve(carry, guess):
        best_sol, found_valid = carry

        def skip_solve(_):
            return best_sol, found_valid

        def do_solve(_):
            sol = BROYDEN_SOLVER_OLD.run(
                guess,
                input_params=input_params,
                input_constants=input_constants,
            )
            params = sol.params
            in_bounds = (params >= 0.0) & (params <= 1.0)
            finite = jnp.all(jnp.isfinite(params))
            is_valid = jnp.all(in_bounds) & finite

            update = is_valid & (~found_valid)
            new_best = jnp.where(update, params, best_sol)
            new_found = found_valid | is_valid
            return new_best, new_found

        best_sol_out, found_valid_out = jax.lax.cond(
            found_valid,
            skip_solve,
            do_solve,
            operand=None,
        )
        return (best_sol_out, found_valid_out), None

    init_best = jnp.full((3,), jnp.nan)
    init_found = jnp.array(False)

    (final_sol, found_valid), _ = jax.lax.scan(
        try_solve,
        (init_best, init_found),
        guesses,
    )

    return final_sol, found_valid


def get_root_error_old(avg_m, avg_n, avg_l, input_params, input_constants):
    """
    Get the error of the root by comparing f(n)-n with 0
    """
    vars = jnp.array([avg_m, avg_n, avg_l])
    f1, f2, f3 = equations_old(vars, input_params, input_constants)
    return jnp.array([f1, f2, f3])


@jax.jit
def li_free_energy_old(input_params, input_constants):
    (avg_m, avg_n, avg_l), _ = find_root_old(input_params, input_constants, initial_guess=jnp.array([0.5, 0.0, 0.5]))
    h, J, _, _, _, kT, z = energetics_old(
        jnp.array([avg_m, avg_n, avg_l]), input_params, input_constants
    )
     # Free energy calculation
    G = h[0] * z * avg_m + h[1] * z * avg_n + h[2] * z * avg_l
    return G


# ---------------------------------------------------------------------------
# Generic multi-species functions (N solvents + M anions)
# ---------------------------------------------------------------------------

def _species_props_to_arrays(solvents, anions):
    """Convert species property dicts to JAX arrays for JIT-compatible use.

    Args:
        solvents: dict of {name: {"dn": ..., "an": ..., "x": ..., "volume": ...}}
        anions:   dict of {name: {"dn": ..., "x": ..., "volume": ...}}

    Returns:
        dn_sol, an_sol, x_sol, v_sol: shape (N,) arrays for solvents
        dn_an, x_an, v_an:            shape (M,) arrays for anions
    """
    sol_vals = list(solvents.values())
    an_vals  = list(anions.values())
    dn_sol = jnp.array([p["dn"]     for p in sol_vals])
    an_sol = jnp.array([p["an"]     for p in sol_vals])
    x_sol  = jnp.array([p["x"]      for p in sol_vals])
    v_sol  = jnp.array([p["volume"] for p in sol_vals])
    dn_an  = jnp.array([p["dn"]     for p in an_vals])
    x_an   = jnp.array([p["x"]      for p in an_vals])
    v_an   = jnp.array([p["volume"] for p in an_vals])
    return dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an


@partial(jax.jit, static_argnames=(
    "monotonicity_dict",
    "rescale_h_sol", "rescale_h_an",
    "rescale_J_sol_sol", "rescale_J_sol_an", "rescale_J_an_an",
    "rescale_conc_factor",
))
def rescale_input_params(
    input_params,
    monotonicity_dict=None,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Rescale parameters for the generic multi-species model.

    A generic dispatcher: each parameter group is rescaled by its own injected
    function, defaulting to the companions of the default h/J term functions.
    When you inject a custom term function with different parameter semantics,
    pass a matching custom_*_rescale alongside it.

    Args:
        input_params:      parameter dict (sol_params_dn, salt_params_dn,
                           params_sol_salt_an, params_sol_sol,
                           params_anion_anion, conc_factor_sol).
        monotonicity_dict: frozenset from _freeze_mono(), or None → DEFAULT_MONOTONICITY.
                           Parameter groups it does not mention fall back to
                           DEFAULT_MONOTONICITY (no constraint).
        rescale_h_sol:     rescaling function for sol_params_dn.
        rescale_h_an:      rescaling function for salt_params_dn.
        rescale_J_sol_sol: rescaling function for params_sol_sol.
        rescale_J_sol_an:  rescaling function for params_sol_salt_an.
        rescale_J_an_an:   rescaling function for params_anion_anion.
        rescale_conc_factor: rescaling function for conc_factor_sol.
    """
    mono = {**DEFAULT_MONOTONICITY, **dict(monotonicity_dict or {})}
    input_params["sol_params_dn"]      = rescale_h_sol(      input_params["sol_params_dn"],      mono["sol_params_dn"])
    input_params["salt_params_dn"]     = rescale_h_an(       input_params["salt_params_dn"],     mono["salt_params_dn"])
    input_params["params_sol_salt_an"] = rescale_J_sol_an(   input_params["params_sol_salt_an"], mono["params_sol_salt_an"])
    input_params["params_sol_sol"]     = rescale_J_sol_sol(  input_params["params_sol_sol"],     mono["params_sol_sol"])
    input_params["params_anion_anion"] = rescale_J_an_an(    input_params["params_anion_anion"], mono["params_anion_anion"])
    input_params["conc_factor_sol"]    = rescale_conc_factor(input_params["conc_factor_sol"],    mono["conc_factor_sol"])
    return input_params


def energetics(
    vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
    monotonicity_dict=None,
    groups=None,
    pair_symmetry=DEFAULT_PAIR_SYMMETRY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Compute h and J for a generic N-solvent + M-anion Ising model.

    Args:
        vars:        occupation fractions, shape (N+M,)
        input_params: parameter dict with keys sol_params_dn, salt_params_dn,
                      params_sol_salt_an, params_sol_sol, params_anion_anion,
                      conc_factor_sol
        dn_sol: donor numbers of solvents, shape (N,)
        an_sol: acceptor numbers of solvents, shape (N,)
        x_sol:  molar fractions of solvents, shape (N,)
        v_sol:  molar volumes of solvents, shape (N,)
        dn_an:  donor numbers of anions, shape (M,)
        x_an:   molar fractions of anions, shape (M,)
        v_an:   molar volumes of anions, shape (M,)
        z:      coordination number (scalar)
        h_sol_func:    callable(dn_eff, x, params) → scalar
        h_an_func:     callable(dn_an, x, params) → scalar
        J_sol_sol_func: callable(dn_i, an_i, x_i, dn_j, an_j, x_j, params) → scalar
        J_sol_an_func:  callable(dn_an, an_sol, x_sol, x_an, params) → scalar
        J_an_an_func:   callable(dn_i, x_i, dn_j, x_j, params) → scalar
        monotonicity_dict: frozenset from _freeze_mono(), or None for default.
        groups:        integer group rank per species, length N+M, solvents
                       first then anions.  Species sharing a rank are treated as
                       interchangeable; different ranks encode distinct roles.
                       None (default) = all-distinct, which preserves the legacy
                       per-species ordering.  See ``symmetrize_pair_block``.
        pair_symmetry: "group" (default) applies ``symmetrize_pair_block`` to the
                       J_ss and J_aa blocks; "none" leaves them as-is, which
                       reproduces the pre-fix (non-symmetric) behaviour.
        rescale_h_sol, rescale_h_an, rescale_J_sol_sol, rescale_J_sol_an,
        rescale_J_an_an, rescale_conc_factor:
                       companion rescaling functions for each parameter group.

    Returns:
        h:   single-particle energies, shape (N+M,)
        J:   pairwise interaction matrix, shape (N+M, N+M)
        kT:  thermal energy (scalar)
    """
    kT = 0.0257
    input_params = rescale_input_params(
        input_params,
        monotonicity_dict=monotonicity_dict,
        rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
    )

    sol_params_dn_tmp      = input_params["sol_params_dn"]
    salt_params_dn_tmp     = input_params["salt_params_dn"]
    params_sol_salt_an_tmp = input_params["params_sol_salt_an"]
    params_sol_sol_tmp     = input_params["params_sol_sol"]
    params_anion_anion_tmp = input_params["params_anion_anion"]
    conc_factor_sol        = input_params["conc_factor_sol"]

    # Aggregate anion properties for concentration correction
    x_an_total = jnp.sum(x_an)
    v_an_avg   = jnp.sum(x_an * v_an) / x_an_total

    # --- Item 6: vmap dn_eff / an_eff (replaces list comprehensions) ---
    # conc_factor_sep_x_v_sigmoid is called once per solvent with its own
    # (x_sol[i], v_sol[i]) and shared (x_an_total, v_an_avg, params).
    conc = jax.vmap(
        conc_factor_sep_x_v_sigmoid, in_axes=(0, None, 0, None, None)
    )(x_sol, x_an_total, v_sol, v_an_avg, conc_factor_sol)  # shape (N,)
    dn_eff = dn_sol * conc
    an_eff = an_sol * conc

    # --- Items 1 & 2: vmap h terms (replaces Python for loops) ---
    # Each call gets one species' scalar (dn/x) and the shared param array.
    h_sol_vals = jax.vmap(h_sol_func, in_axes=(0, 0, None))(
        dn_eff, x_sol, sol_params_dn_tmp
    )  # shape (N,)
    h_an_vals = jax.vmap(h_an_func, in_axes=(0, 0, None))(
        dn_an, x_an, salt_params_dn_tmp
    )  # shape (M,)
    h = jnp.concatenate([h_sol_vals, h_an_vals])  # shape (N+M,)

    # --- Items 2 & 3: vmap J blocks, then assemble with jnp.block ---
    # J_ss: (N, N) — outer vmap over row i, inner over column j.
    # Outer in_axes=(0,0,0, None,None,None, None): scalars dn_i/an_i/x_i are
    # mapped (axis 0); the full j-side arrays and params are broadcast (None).
    def _J_ss_row(dn_i, an_i, x_i):
        return jax.vmap(
            lambda dn_j, an_j, x_j: J_sol_sol_func(
                dn_i, an_i, x_i, dn_j, an_j, x_j, params_sol_sol_tmp
            )
        )(dn_eff, an_eff, x_sol)

    J_ss = jax.vmap(_J_ss_row)(dn_eff, an_eff, x_sol)  # (N, N)

    # J_sa: (N, M) — row = solvent i, col = anion j.
    # Outer maps over (an_eff[i], x_sol[i]); inner maps over anion j arrays.
    def _J_sa_row(an_i, x_i):
        return jax.vmap(
            lambda dn_j, x_j: J_sol_an_func(dn_j, an_i, x_i, x_j, params_sol_salt_an_tmp)
        )(dn_an, x_an)

    J_sa = jax.vmap(_J_sa_row)(an_eff, x_sol)  # (N, M)

    # J_aa: (M, M) — symmetric, outer maps over anion i, inner over anion j.
    def _J_aa_row(dn_i, x_i):
        return jax.vmap(
            lambda dn_j, x_j: J_an_an_func(dn_i, x_i, dn_j, x_j, params_anion_anion_tmp)
        )(dn_an, x_an)

    J_aa = jax.vmap(_J_aa_row)(dn_an, x_an)  # (M, M)

    # --- Symmetrize the same-kind blocks ---
    # J_ss and J_aa are built from a J term whose two slots receive the same kind
    # of descriptor, so the raw blocks are asymmetric whenever the underlying
    # function weights its slots differently (sol_sol_func does: a1 != a2).
    # J_sa needs no treatment: its slots hold genuinely different quantities
    # (anion DN vs solvent AN) and the block is symmetrized below by J_sa.T.
    if pair_symmetry == "group":
        n_sol = dn_sol.shape[0]
        grp = _resolve_groups(groups, n_sol + dn_an.shape[0])
        J_ss = symmetrize_pair_block(J_ss, grp[:n_sol])
        J_aa = symmetrize_pair_block(J_aa, grp[n_sol:])
    elif pair_symmetry != "none":
        raise ValueError(f"pair_symmetry must be 'group' or 'none', got {pair_symmetry!r}")

    # --- Item 3: jnp.block assembles the 2×2 block matrix in one XLA op ---
    # J_sa.T gives the anion-solvent (M, N) block; symmetry is exact by construction.
    J = jnp.block([[J_ss, J_sa],
                   [J_sa.T, J_aa]])  # (N+M, N+M)

    return h, J, kT


def equations(
    vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
    monotonicity_dict=None,
    groups=None,
    pair_symmetry=DEFAULT_PAIR_SYMMETRY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Mean-field self-consistency equations for N-solvent + M-anion Ising model.

    Returns residuals f_i = exp_i / Z - vars_i for each species i.
    All term and rescaling functions are passed through to energetics unchanged.
    """
    h, J, kT = energetics(
        vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func, monotonicity_dict=monotonicity_dict,
        groups=groups, pair_symmetry=pair_symmetry,
        rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
    )
    energies = -(h + (z / 2.0) * (J @ vars + jnp.diag(J) * vars)) / kT
    exps = jnp.exp(energies)
    return exps / jnp.sum(exps) - vars


@partial(jax.jit, static_argnames=(
    "max_tries",
    "h_sol_func", "h_an_func",
    "J_sol_sol_func", "J_sol_an_func", "J_an_an_func",
    "monotonicity_dict", "groups", "pair_symmetry",
    "rescale_h_sol", "rescale_h_an",
    "rescale_J_sol_sol", "rescale_J_sol_an", "rescale_J_an_an",
    "rescale_conc_factor",
))
def _find_root_impl(
    input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
    initial_guess, max_tries,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
    monotonicity_dict=None,
    groups=None,
    pair_symmetry=DEFAULT_PAIR_SYMMETRY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """JIT-compiled multi-start Broyden solver.

    Term functions are static arguments: JAX recompiles when they change
    (which is the desired behavior when testing a new hypothesis) and caches
    the compiled version for repeated calls with the same functions.
    """
    _eq = partial(
        equations,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func, monotonicity_dict=monotonicity_dict,
        groups=groups, pair_symmetry=pair_symmetry,
        rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
    )
    # jit=True: jaxopt uses jax.lax.while_loop internally, which is required
    # for compatibility with jax.lax.cond / jax.lax.scan (traced context).
    # Nested JIT is idempotent in JAX so this does not cause recompilation.
    solver = Broyden(fun=_eq, maxiter=1000, tol=1e-8, verbose=False, jit=True)

    n_species = dn_sol.shape[0] + dn_an.shape[0]
    ones = jnp.ones(n_species)
    guesses = jnp.stack([
        initial_guess,
        ones * 0.3,
        ones * 0.7,
        ones * 0.5,
        ones * 0.2,
        ones * 0.8,
        ones * 0.1,
        ones * 0.9,
        ones * 0.4,
        ones * 0.6,
    ])  # shape (10, n_species)
    guesses = guesses[:max_tries]

    def try_solve(carry, guess):
        best_sol, found_valid = carry

        def skip_solve(_):
            return best_sol, found_valid

        def do_solve(_):
            sol = solver.run(
                guess,
                input_params=input_params,
                dn_sol=dn_sol, an_sol=an_sol, x_sol=x_sol, v_sol=v_sol,
                dn_an=dn_an, x_an=x_an, v_an=v_an,
                z=z,
            )
            params = sol.params
            in_bounds = (params >= 0.0) & (params <= 1.0)
            finite    = jnp.all(jnp.isfinite(params))
            is_valid  = jnp.all(in_bounds) & finite
            update    = is_valid & (~found_valid)
            new_best  = jnp.where(update, params, best_sol)
            new_found = found_valid | is_valid
            return new_best, new_found

        best_sol_out, found_valid_out = jax.lax.cond(
            found_valid, skip_solve, do_solve, operand=None
        )
        return (best_sol_out, found_valid_out), None

    init_best  = jnp.full((n_species,), jnp.nan)
    init_found = jnp.array(False)

    (final_sol, found_valid), _ = jax.lax.scan(
        try_solve, (init_best, init_found), guesses
    )
    return final_sol, found_valid

def find_root(
    input_params, solvents, anions, z,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
    monotonicity_dict=DEFAULT_MONOTONICITY,
    groups=None,
    pair_symmetry=DEFAULT_PAIR_SYMMETRY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
    initial_guess=None, max_tries=10,
):
    """Multi-start root solver for the generic N-solvent + M-anion Ising model.

    Args:
        input_params:      parameter dict (must include params_anion_anion, 6 elements)
        solvents:          dict of {name: {"dn": ..., "an": ..., "x": ..., "volume": ...}}
        anions:            dict of {name: {"dn": ..., "x": ..., "volume": ...}}
        z:                 coordination number
        h_sol_func:        h term for solvents — default: expfunc + logfunc
        h_an_func:         h term for anions  — default: expfunc + logfunc
        J_sol_sol_func:    J term for solvent pairs — default: DN/AN cross + DN-DN + AN-AN
        J_sol_an_func:     J term for solvent-anion — default: sol_sol_func of (dn_an, an_sol)
        J_an_an_func:      J term for anion pairs   — default: sol_sol_func of (dn_i, dn_j)
        monotonicity_dict: dict controlling monotonicity constraints — see DEFAULT_MONOTONICITY.
                           Converted to a frozenset internally for JAX static-arg hashing.
        groups:            integer group rank per species, length N+M, solvents first then
                           anions.  Same rank = interchangeable partners (averaged); different
                           ranks = distinct roles (lower rank takes the first slot).  None
                           (default) = all-distinct, preserving legacy ordering.  Example: an
                           equimolar 5-solvent mixture with one salt is ``(0,0,0,0,0,1)``.
        pair_symmetry:     "group" (default) symmetrizes the J_ss / J_aa blocks;
                           "none" reproduces the pre-fix non-symmetric behaviour.
        rescale_h_sol, rescale_h_an, rescale_J_sol_sol, rescale_J_sol_an,
        rescale_J_an_an, rescale_conc_factor:
                           companion rescaling functions (defaults match default term functions).
        initial_guess:     shape (N+M,), defaults to uniform 1/(N+M)
        max_tries:         number of initial guesses to try (default 10)

    Returns:
        (occupations, found_valid): shape (N+M,) array and boolean flag
    """
    dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an = _species_props_to_arrays(solvents, anions)
    n_species = len(solvents) + len(anions)
    if initial_guess is None:
        initial_guess = jnp.ones(n_species) / n_species
    return _find_root_impl(
        input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        initial_guess, max_tries,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func, monotonicity_dict=_freeze_mono(monotonicity_dict),
        groups=_resolve_groups(groups, n_species), pair_symmetry=pair_symmetry,
        rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
    )


def get_root_error(
    vars, input_params, solvents, anions, z,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
    monotonicity_dict=DEFAULT_MONOTONICITY,
    groups=None,
    pair_symmetry=DEFAULT_PAIR_SYMMETRY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Diagnostic: returns mean-field equation residuals for given occupation fractions."""
    dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an = _species_props_to_arrays(solvents, anions)
    return equations(
        vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func, monotonicity_dict=_freeze_mono(monotonicity_dict),
        groups=_resolve_groups(groups, len(solvents) + len(anions)),
        pair_symmetry=pair_symmetry,
        rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
    )


def li_free_energy(
    input_params, solvents, anions, z,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
    monotonicity_dict=DEFAULT_MONOTONICITY,
    groups=None,
    pair_symmetry=DEFAULT_PAIR_SYMMETRY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Li+ solvation free energy for a generic N-solvent + M-anion system.

    Args:
        input_params:      parameter dict (must include params_anion_anion, 6 elements)
        solvents:          dict of {name: {"dn": ..., "an": ..., "x": ..., "volume": ...}}
        anions:            dict of {name: {"dn": ..., "x": ..., "volume": ...}}
        z:                 coordination number
        h_sol_func, h_an_func, J_sol_sol_func, J_sol_an_func, J_an_an_func:
                           injectable term functions (defaults = current physics)
        monotonicity_dict: dict controlling monotonicity constraints — see DEFAULT_MONOTONICITY.
        rescale_h_sol, rescale_h_an, rescale_J_sol_sol, rescale_J_sol_an,
        rescale_J_an_an, rescale_conc_factor:
                           companion rescaling functions (defaults match default term functions).

    Returns:
        G: scalar free energy
    """
    dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an = _species_props_to_arrays(solvents, anions)
    n_species = len(solvents) + len(anions)
    initial_guess = jnp.ones(n_species) / n_species
    mono_frozen = _freeze_mono(monotonicity_dict)
    grp = _resolve_groups(groups, n_species)
    vars, _ = _find_root_impl(
        input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        initial_guess, max_tries=10,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func, monotonicity_dict=mono_frozen,
        groups=grp, pair_symmetry=pair_symmetry,
        rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
    )
    h, J, kT = energetics(
        vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func, monotonicity_dict=mono_frozen,
        groups=grp, pair_symmetry=pair_symmetry,
        rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
    )
    return jnp.sum(h * z * vars)
