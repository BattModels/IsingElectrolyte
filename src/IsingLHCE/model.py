import jax
import jax.numpy as jnp
import jax.nn as jnn
from jaxopt import Broyden
from functools import partial

from .interactions import (
    expfunc, logfunc, sol_sol_func,
    default_h_sol, default_h_an,
    default_J_sol_sol, default_J_sol_an, default_J_an_an,
)
from .conc_vol_correction import conc_factor_sep_x_v_sigmoid

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)

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


# Sign masks for rescale_input_params.
# +1 → softplus(p[i])  (force positive)
# -1 → -softplus(p[i]) (force negative)
#  0 → p[i] unchanged
#
# Read alongside the physics comments in rescale_input_params_old for context.
_SIGNS_SOL_PARAMS_DN      = jnp.array([ 0, +1, +1, -1, -1])
_SIGNS_SALT_PARAMS_DN     = jnp.array([ 0, +1, +1, -1, -1])
_SIGNS_SOL_SALT_AN        = jnp.array([ 0, +1,  0, +1, +1, -1])
_SIGNS_SOL_SOL            = jnp.array([ 0, +1,  0, +1, +1,  0,
                                        +1,  0, -1, -1,  0, +1,
                                         0, -1, -1, -1])
_SIGNS_ANION_ANION        = jnp.array([ 0, +1,  0, -1, -1, -1])
_SIGNS_CONC_FACTOR        = jnp.array([ 0, -1, +1, -1,  0, +1, +1, +1])


def _apply_softplus(p, signs):
    """Vectorized signed-softplus transform.

    Replaces p[i] with signs[i]*softplus(p[i]) where signs[i] != 0,
    leaving p[i] unchanged where signs[i] == 0.  Single XLA op — no
    Python-level indexing loop.
    """
    return jnp.where(signs == 0, p, signs * jnn.softplus(p))


@jax.jit
def rescale_input_params(input_params):
    """Rescale parameters for the generic multi-species model.

    Each parameter array has a corresponding sign-mask (_SIGNS_*) that
    encodes which indices must be forced positive (+1), negative (-1),
    or left free (0).  _apply_softplus applies the transform in one
    vectorized pass instead of per-index .at[].set() calls.
    """
    input_params["sol_params_dn"]      = _apply_softplus(input_params["sol_params_dn"],      _SIGNS_SOL_PARAMS_DN)
    input_params["salt_params_dn"]     = _apply_softplus(input_params["salt_params_dn"],     _SIGNS_SALT_PARAMS_DN)
    input_params["params_sol_salt_an"] = _apply_softplus(input_params["params_sol_salt_an"], _SIGNS_SOL_SALT_AN)
    input_params["params_sol_sol"]     = _apply_softplus(input_params["params_sol_sol"],     _SIGNS_SOL_SOL)
    input_params["params_anion_anion"] = _apply_softplus(input_params["params_anion_anion"], _SIGNS_ANION_ANION)
    input_params["conc_factor_sol"]    = _apply_softplus(input_params["conc_factor_sol"],    _SIGNS_CONC_FACTOR)
    return input_params


def energetics(
    vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
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

    Returns:
        h:   single-particle energies, shape (N+M,)
        J:   pairwise interaction matrix, shape (N+M, N+M)
        kT:  thermal energy (scalar)
    """
    kT = 0.0257
    input_params = rescale_input_params(input_params)

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
):
    """Mean-field self-consistency equations for N-solvent + M-anion Ising model.

    Returns residuals f_i = exp_i / Z - vars_i for each species i.
    Term functions are passed through to energetics unchanged.
    """
    h, J, kT = energetics(
        vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func,
    )
    energies = -(h + (z / 2.0) * (J @ vars + jnp.diag(J) * vars)) / kT
    exps = jnp.exp(energies)
    return exps / jnp.sum(exps) - vars


@partial(jax.jit, static_argnames=(
    "max_tries",
    "h_sol_func", "h_an_func",
    "J_sol_sol_func", "J_sol_an_func", "J_an_an_func",
))
def _find_root_impl(
    input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
    initial_guess, max_tries,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
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
        J_an_an_func=J_an_an_func,
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
    initial_guess=None, max_tries=10,
):
    """Multi-start root solver for the generic N-solvent + M-anion Ising model.

    Args:
        input_params:  parameter dict (must include params_anion_anion, 6 elements)
        solvents:      dict of {name: {"dn": ..., "an": ..., "x": ..., "volume": ...}}
        anions:        dict of {name: {"dn": ..., "x": ..., "volume": ...}}
        z:             coordination number
        h_sol_func:    h term for solvents — default: expfunc + logfunc
        h_an_func:     h term for anions  — default: expfunc + logfunc
        J_sol_sol_func: J term for solvent pairs — default: DN/AN cross + DN-DN + AN-AN
        J_sol_an_func:  J term for solvent-anion — default: sol_sol_func of (dn_an, an_sol)
        J_an_an_func:   J term for anion pairs   — default: sol_sol_func of (dn_i, dn_j)
        initial_guess: shape (N+M,), defaults to uniform 1/(N+M)
        max_tries:     number of initial guesses to try (default 10)

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
        J_an_an_func=J_an_an_func,
    )


def get_root_error(
    vars, input_params, solvents, anions, z,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
):
    """Diagnostic: returns mean-field equation residuals for given occupation fractions."""
    dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an = _species_props_to_arrays(solvents, anions)
    return equations(
        vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func,
    )


def li_free_energy(
    input_params, solvents, anions, z,
    h_sol_func=default_h_sol,
    h_an_func=default_h_an,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_an_func=default_J_sol_an,
    J_an_an_func=default_J_an_an,
):
    """Li+ solvation free energy for a generic N-solvent + M-anion system.

    Args:
        input_params: parameter dict (must include params_anion_anion, 6 elements)
        solvents:     dict of {name: {"dn": ..., "an": ..., "x": ..., "volume": ...}}
        anions:       dict of {name: {"dn": ..., "x": ..., "volume": ...}}
        z:            coordination number
        h_sol_func, h_an_func, J_sol_sol_func, J_sol_an_func, J_an_an_func:
                      injectable term functions (defaults = current physics)

    Returns:
        G: scalar free energy
    """
    dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an = _species_props_to_arrays(solvents, anions)
    n_species = len(solvents) + len(anions)
    initial_guess = jnp.ones(n_species) / n_species
    vars, _ = _find_root_impl(
        input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        initial_guess, max_tries=10,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func,
    )
    h, J, kT = energetics(
        vars, input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_an_func=J_sol_an_func,
        J_an_an_func=J_an_an_func,
    )
    return jnp.sum(h * z * vars)
