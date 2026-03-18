import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve, root
import jax, optax, os
import jax.numpy as jnp
from jaxopt import Broyden
from jax import grad
import pickle
import copy
from jax import random
from functools import partial
import jax.nn as jnn
import time

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)  # Disable JIT compilation for debugging

from matplotlib import rcParams

fontsize = 15
plot_settings = {
    # "font.family": "times new roman",
    "axes.labelsize": fontsize,
    "axes.labelweight": "bold",
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.size": fontsize,
    "axes.linewidth": 2.0,
    "lines.dashed_pattern": [5, 2.5],
    "lines.markersize": 10,
    "lines.linewidth": 2,
    "lines.markeredgewidth": 1,
    # "lines.markeredgecolor": "k",
    "legend.fontsize": fontsize,
    "legend.frameon": False,
    "figure.figsize": [6, 6],
}

# Update rcParams with settings from JSON file
rcParams.update(plot_settings)

@jax.jit
def mlp(x, params):
    """
    A simple MLP.
    x is the input vector.
    """
    W1, b1, W2, b2, W3, b3 = params
    # W1, b1, W2, b2 = params_dn
    hidden1 = jax.nn.sigmoid(jnp.dot(x, W1) + b1)
    hidden2 = jax.nn.sigmoid(jnp.dot(hidden1, W2) + b2)
    output = jnp.dot(hidden2, W3) + b3
    return output

@jax.jit
def langmuirfunc(x, params):
    """
    A simple Langmuir isotherm function.
    x is the input vector.
    params is an array of parameters
    """
    fmax, n, K, x0, b = params
    output = fmax * (x - x0) ** n / (K**n + (x - x0) ** n) + b
    return output

@jax.jit
def linearfunc(x, params):
    """
    A simple linear function.
    x is the input vector.
    params is an array of parameters
    """
    a0, a1 = params
    output = a0 + a1 * x
    return output

@jax.jit
def logfunc_conc_factor(x, params):
    """
    A simple log function with concentration factor.
    x is the input vector.
    params is an array of parameters
    """
    a0, a1 = params
    output = a0 * jnp.log(x) + a1
    return output

@jax.jit
def expfunc(x, params):
    """
    A simple exp function.
    x is the input vector.
    """
    a0, a1, a2, a3 = params
    output = a0 + a1 / (1 + a2 * jnp.exp(-(a3 * x)))
    return output

@jax.jit
def logfunc(x, params):
    """
    A simple log function.
    x is the input vector.
    """
    a0 = params
    output = a0 * jnp.log(x)
    return output

@jax.jit
def sol_sol_func(X, params):
    """
    function computing solvent-solvent interaction.
    """
    dn, an = X
    L0, H0, a0, a1, a2 = params
    output = L0 + H0 / (1.0 + jnp.exp(a0 + a1 * dn + a2 * an))
    return output

@jax.jit
def polynomial_func(x, params, n=2):
    """
    A simple polynomial function.
    x is the input vector.
    params is an array of parameters
    """
    output = 0
    for i in range(n+1):
        output += params[i] * x**i
    return output

@jax.jit
def conc_factor(x0, x_ref, V0, V_ref, params):
    output = expfunc((x0 / x_ref), params = params[:4]) * expfunc((V_ref / V0), params = params[4:8])
    return output

@jax.jit
def dn_loss(params, x, y):
    # Vectorize the MLP evaluation over the batch.
    preds = jax.vmap(lambda x: mlp(x, params))(x)
    return jnp.sqrt(jnp.mean((preds - y) ** 2))

@jax.jit
def rescale_input_params(input_params):
    """
    Rescale input parameters to ensure monotonicity
    """
    # # check that all values inside each key are below 50, if not set them to 50
    # for key in input_params.keys():
    #     input_params[key] = jnp.where(input_params[key] > 50, 50.0, input_params[key])
    #     input_params[key] = jnp.where(input_params[key] < -50, -50.0, input_params[key])

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

def energetics(vars, input_params, input_constants):
    # Define constants.
    Mref = 1.0
    kT = 0.0257
    cref = 1.0
    # z = 2
    # Rescale input parameters to ensure monotonicity
    input_params = rescale_input_params(input_params)
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
    # if jax.config.jax_disable_jit:
    #     print('formulation ', dn0, dn1, dn_anion, x_anion)    

    dn0 = dn0 * conc_factor(x0, x_anion, v0, v_anion, conc_factor_sol)
    an0 = an0 * conc_factor(x0, x_anion, v0, v_anion, conc_factor_sol)
    dn1 = dn1 * conc_factor(x1, x_anion, v1, v_anion, conc_factor_sol)
    an1 = an1 * conc_factor(x1, x_anion, v1, v_anion, conc_factor_sol)
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

    # if jax.config.jax_disable_jit:
    #     print('J00:', j00, " J01:", j01, " J02:", j02, 
    #           " J10:", j10, " J11:", j11, " J12:", j12,
    #           " J20:", j20, " J21:", j21, " J22:", j22)
    return h, J, avg_m, avg_n, avg_l, kT, z  

def equations(vars, input_params, input_constants):
    h, J, avg_m, avg_n, avg_l, kT, z = energetics(vars, input_params, input_constants)
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
    # if jax.config.jax_disable_jit:
    #     # For debugging purposes, print the intermediate values.
    #     print("h:", h)
    #     print("J:", J)
    #     print("exp0:", exp0, "exp1:", exp1, "exp2:", exp2, "partition:", partition)
    #     print("avg_m:", avg_m, "avg_n:", avg_n, "avg_l:", avg_l)
    f1 = exp0 / partition - avg_m
    f3 = exp2 / partition - avg_l
    f2 = exp1 / partition - avg_n
    return jnp.array([f1, f2, f3])

BROYDEN_SOLVER = Broyden(fun=equations, maxiter=1000, tol=1e-8, verbose=False, jit=True)
@partial(jax.jit, static_argnames=("max_tries",))
def find_root(input_params, input_constants,
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

    def try_solve(carry, guess):
        best_sol, found_valid = carry

        def skip_solve(_):
            return best_sol, found_valid

        def do_solve(_):
            sol = BROYDEN_SOLVER.run(
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


def get_root_error(avg_m, avg_n, avg_l, input_params, input_constants):
    """
    Get the error of the root by comparing f(n)-n with 0
    """
    vars = jnp.array([avg_m, avg_n, avg_l])
    f1, f2, f3 = equations(vars, input_params, input_constants)
    return jnp.array([f1, f2, f3])


def objective_single(
    input_params,
    input_constants,
    m_target,
    n_target,
    l_target,
    init_guess,
):
    # Get the outputs from your function.
    (avg_m, avg_n, avg_l), found_valid = find_root(
        input_params=input_params, input_constants=input_constants,
        initial_guess=init_guess,
    )
    # We define a loss as the squared difference.
    loss = (avg_m - m_target) ** 2 + (avg_n - n_target) ** 2 + (avg_l - l_target) ** 2
    new_init = jnp.where(found_valid, jnp.array([avg_m, avg_n, avg_l]), init_guess)
    return loss, new_init

@jax.jit
def total_objective(params, data):
    # data is a dictionary where each key holds an array of shape (num_points,)
    input_constants = jnp.vstack(
        [
            data["dn0"],
            data["dn1"],
            data["dn_anion"],
            data["x0"],
            data["x1"],
            data["an0"],
            data["an1"],
            data["x_anion"],
            data["solvent_volume"],
            data["diluent_volume"],
            data["anion_volume"],
            data["z"],
        ]
    )
    # Use vmap to vectorize the single-point objective over the data.
    # in_axes: each data argument is mapped over axis 0, and params is shared (None).
    v_objective = jax.vmap(
        objective_single,
        in_axes=(None, 1, 0, 0, 0, 0),
    )
    losses, new_inits = v_objective(
        params,
        input_constants,
        data["m_target"],
        data["n_target"],
        data["l_target"],
        data['init_guess'],
    )
    mean_loss = jnp.sqrt(jnp.mean(losses))
    return mean_loss, new_inits


def initialize_params(
    mode="from_file",
    file_path=None,
    input_dim=1,
    hidden_dim=4,
    output_dim=1,
    random_seed = 42,
):
    if mode == "from_scratch":
        sol_params_dn = jnp.array(
            [
                -1.4226977825164795, 
                0.6868864893913269, 
                -1.29103684425354, 
                -2.2087085247039795, 
                -2.837172,
            ]
        )
        salt_params_dn = jnp.array(
            [
                -1.933838129043579, 
                0.21721802651882172, 
                -1.0301377773284912, 
                -2.815300941467285, 
                -1.930697,
            ]
        )
        params_sol_salt_an = jnp.array(
            [
                -0.28391966223716736, 
                -0.4041173458099365, 
                0.6561649441719055, 
                -2.1234798431396484, 
                -1.09547758102417, 
                -1.3783741,
            ]
        )
        params_sol_sol = jnp.array(
            [
                0.11802631616592407, 
                0.04696842283010483, 
                0.039969541132450104, 
                -3.7262208461761475, 
                -3.0505568981170654, 
                -0.10816801339387894, 
                -2.2049686908721924, 
                0.375214546918869, 
                -3.009303092956543, 
                -2.591977834701538, 
                -0.11441854387521744, 
                -2.53210186958313, 
                0.3858657479286194, 
                -2.321645736694336, 
                -2.3250560760498047, 
                -3.2535532,
            ]
        )
        params_salt = jnp.array(
            [
                -0.37612465023994446, 
                -0.13344359397888184, 
                -1.854067325592041, 
                -3.3515141010284424, 
                -3.6836671829223633
            ]
        )
        conc_factor_sol = jnp.array(
            [
                0.6028587818145752, 
                0.5641694664955139, 
                -0.0062898872420191765, 
                -1.0074571371078491, 
                -2177.35791015625, 
                2189.39013671875, 
                -6.164828777313232, 
                2.281510829925537,
            ]
        )

        # add random noise to the initial params
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=6)
        noise_scale = 0.05
        sol_params_dn = sol_params_dn * (1.0 + noise_scale * random.normal(
            keys[0], shape=sol_params_dn.shape
        ))
        salt_params_dn = salt_params_dn * (1.0 + noise_scale * random.normal(
            keys[1], shape=salt_params_dn.shape
        ))
        params_sol_salt_an = params_sol_salt_an * (1.0 + noise_scale * random.normal(
            keys[2], shape=params_sol_salt_an.shape
        ))
        params_sol_sol = params_sol_sol * (1.0 + noise_scale * random.normal(
            keys[3], shape=params_sol_sol.shape
        ))
        params_salt = params_salt * (1.0 + noise_scale * random.normal(
            keys[4], shape=params_salt.shape
        ))
        init_params = {
            "sol_params_dn": sol_params_dn,
            "salt_params_dn": salt_params_dn,
            "params_sol_salt_an": params_sol_salt_an,
            "params_sol_sol": params_sol_sol,
            "params_salt": params_salt,
            "conc_factor_sol": conc_factor_sol,
        }
        return init_params
    elif mode == "from_file":
        if file_path is None:
            raise ValueError("file_path must be provided when mode is 'from_file'")
        with open(file_path, "rb") as f:
            init_params = pickle.load(f)
        # add random noise to the initial params
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=6)
        noise_scale = 0.05
        init_params['sol_params_dn'] = init_params['sol_params_dn'] * (1.0 + noise_scale * random.normal(
            keys[0], shape=init_params['sol_params_dn'].shape
        ))
        init_params['salt_params_dn'] = init_params['salt_params_dn'] * (1.0 + noise_scale * random.normal(
            keys[1], shape=init_params['salt_params_dn'].shape
        ))
        init_params['params_sol_salt_an'] = init_params['params_sol_salt_an'] * (1.0 + noise_scale * random.normal(
            keys[2], shape=init_params['params_sol_salt_an'].shape
        ))
        init_params['params_sol_sol'] = init_params['params_sol_sol'] * (1.0 + noise_scale * random.normal(
            keys[3], shape=init_params['params_sol_sol'].shape
        ))
        init_params['params_salt'] = init_params['params_salt'] * (1.0 + noise_scale * random.normal(
            keys[4], shape=init_params['params_salt'].shape
        ))
        init_params['conc_factor_sol'] = init_params['conc_factor_sol'] * (1.0 + noise_scale * random.normal(
            keys[5], shape=init_params['conc_factor_sol'].shape
        ))
        return init_params

@jax.jit
def li_free_energy(input_params, input_constants):
    (avg_m, avg_n, avg_l), _ = find_root(input_params, input_constants, initial_guess=jnp.array([0.5, 0.0, 0.5]))
    h, J, _, _, _, kT, z = energetics(
        jnp.array([avg_m, avg_n, avg_l]), input_params, input_constants
    )
     # Free energy calculation
    G = h[0] * z * avg_m + h[1] * z * avg_n + h[2] * z * avg_l
    return G  

@jax.jit
def update(params, opt_state, data):
    (loss, new_inits), grads = jax.value_and_grad(total_objective, has_aux=True)(
        params, data
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, new_inits

def train(params, num_epochs, train_data, test_data, val_data, opt_state, random_seed=42):
    # Initialize backup variables
    prev_params = copy.deepcopy(params)  # backup from the previous step
    best_params = copy.deepcopy(params)  # backup for best parameters

    train_loss_log = jnp.zeros(num_epochs)
    val_loss_log = jnp.zeros(num_epochs)

    # Set up early stopping parameters
    best_val_loss = jnp.inf
    best_train_loss = jnp.inf
    patience = 100  # how many steps to wait for an improvement
    patience_counter = 0

    # batch size implementation
    n_train = train_data["dn0"].shape[0]
    init_guess_train = np.tile(np.array([0.5, 0.0, 0.5]), (n_train, 1))
    init_guess_train = jnp.array(init_guess_train)
    n_test = test_data["dn0"].shape[0]
    init_guess_test = np.tile(np.array([0.5, 0.0, 0.5]), (n_test, 1))
    init_guess_test = jnp.array(init_guess_test)
    n_val = val_data["dn0"].shape[0]
    init_guess_val = np.tile(np.array([0.5, 0.0, 0.5]), (n_val, 1))
    init_guess_val = jnp.array(init_guess_val)
    key = random.PRNGKey(random_seed)
    batch_size = n_train
    steps_per_ep = n_train // batch_size

    for i, epoch in enumerate(range(num_epochs)):
        prev_params = copy.deepcopy(params)  # backup from the previous epoch
        
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_train)

        epoch_loss = 0.0
        for step in range(steps_per_ep):
            batch_idx = perm[step * batch_size : (step + 1) * batch_size]
            # Create a batch of data
            batch = {k: jnp.take(v, batch_idx, axis=0) for k,v in train_data.items()}
            batch['init_guess'] = jnp.take(init_guess_train, batch_idx, axis=0)
            # Compute gradients & update parameters on this batch
            params, opt_state, loss, new_inits = update(params, opt_state, batch)
            init_guess_train = init_guess_train.at[batch_idx].set(new_inits)
            epoch_loss += loss
        
        epoch_loss /= steps_per_ep
        val_data['init_guess'] = init_guess_val
        test_data['init_guess'] = init_guess_test
        val_loss, _ = total_objective(params, val_data)
        test_loss, _ = total_objective(params, test_data)
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch}: loss = {epoch_loss:.4f}, val loss = {val_loss:.4f}, test loss = {test_loss:.4f}"
            )
        train_loss_log = train_loss_log.at[i].set(epoch_loss)
        val_loss_log = val_loss_log.at[i].set(val_loss)

        if round(val_loss,4) <= round(best_val_loss, 4):
            best_val_loss = val_loss
            best_params = copy.deepcopy(params)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if jnp.isnan(epoch_loss) or jnp.isnan(val_loss):
            params = copy.deepcopy(best_params)
            break

        # Early stopping if no improvement for 'patience' consecutive steps
        if patience_counter >= patience:
            print(
                f"Early stopping triggered at step {epoch}: No improvement in validation loss for {patience} steps."
            )
            # Revert to the best parameters encountered during training
            params = copy.deepcopy(best_params)
            break

        # Other optional stopping conditions (e.g., based on thresholds) can also be included:
        if jnp.abs(epoch_loss) < 0.04 and jnp.abs(val_loss) < 0.04:
            print("Losses are below threshold; stopping training.")
            break
        # # break when val_loss starts increasing consistently
        # if i > 100 and jnp.mean(val_loss_log[i-100:i]) < jnp.mean(val_loss_log[i-50:i]):
        #     break
        # only output the not nan values from 0 to i+1
    train_loss_log = train_loss_log.at[i+1:].set(jnp.nan)
    val_loss_log = val_loss_log.at[i+1:].set(jnp.nan)
    train_loss_log = train_loss_log[~jnp.isnan(train_loss_log)]
    val_loss_log = val_loss_log[~jnp.isnan(val_loss_log)]
    return params, train_loss_log, val_loss_log

def parity_results(data_dict, input_params):
    for key in list(data_dict.keys())[:]:
        df = pd.DataFrame({kk: data_dict[key][kk] for kk in data_dict[key].keys() if kk != "init_guess"})
        df1 = pd.read_csv(f"{key}.csv")
        # merge df1 into df where df1['Solvent DN'] == df['dn0'] and df1['Diluent DN'] == df['dn1']
        df = df.merge(
            df1,
            left_on=["dn0", "dn1", "an0", "an1", "m_target"],
            right_on=[
                "Solvent DN",
                "Diluent DN",
                "Solvent AN",
                "Diluent AN",
                "Fractional CN solvent",
            ],
            how="left",
        )
        # make sure formulation column is at the left
        df = df[["formulation"] + list(df.columns.difference(["formulation"]))]
        ising_avg_m = np.zeros(len(df))
        ising_avg_n = np.zeros(len(df))
        ising_avg_l = np.zeros(len(df))
        for i in range(len(df)):
            dn0 = df["dn0"].iloc[i]
            dn1 = df["dn1"].iloc[i]
            dn_anion = df["dn_anion"].iloc[i]
            x0 = df["x0"].iloc[i]
            x1 = df["x1"].iloc[i]
            an0 = df["an0"].iloc[i]
            an1 = df["an1"].iloc[i]
            x_anion = df["x_anion"].iloc[i]
            solvent_volume = df["solvent_volume"].iloc[i]
            diluent_volume = df["diluent_volume"].iloc[i]
            anion_volume = df["anion_volume"].iloc[i]
            z = 3.72 / 2.0 # mean value of total CN from MD
            (avg_m, avg_n, avg_l), _ = find_root(
                input_params,
                [
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
                ],
            )
            ising_avg_m[i] = avg_m
            ising_avg_n[i] = avg_n
            ising_avg_l[i] = avg_l
        loss, _ = total_objective(input_params, data_dict[key])
        print(f"Loss on the {key} data: {loss:.4f}")
        # calculate rmse
        rmse = np.sqrt(
            np.mean(
                (ising_avg_m - df["m_target"]) ** 2
                + (ising_avg_n - df["n_target"]) ** 2
                + (ising_avg_l - df["l_target"]) ** 2
            )
        )
        print(f"RMSE on the {key} data: {rmse:.4f}")

        df["ising_avg_m"] = ising_avg_m
        df["ising_avg_n"] = ising_avg_n
        df["ising_avg_l"] = ising_avg_l
        df = df.round(4)
        df.to_csv(f"ising_results_{key}.csv", index=False)
        df = pd.read_csv(f"ising_results_{key}.csv")
        # compute rmse over whole dataset
        rmse = np.sqrt(
            np.mean(
                (df["ising_avg_m"] - df["m_target"]) ** 2
                + (df["ising_avg_n"] - df["n_target"]) ** 2
                + (df["ising_avg_l"] - df["l_target"]) ** 2
            )
        )
        # compute the loss over whole dataset
        loss, _ = total_objective(input_params, data_dict["train"])
        # calculate R2
        ss_res = np.sum(
            (df["ising_avg_m"] - df["m_target"]) ** 2
            + (df["ising_avg_n"] - df["n_target"]) ** 2
            + (df["ising_avg_l"] - df["l_target"]) ** 2
        )
        ss_tot = np.sum(
            (df["m_target"] - np.mean(df["m_target"])) ** 2
            + (df["n_target"] - np.mean(df["n_target"])) ** 2
            + (df["l_target"] - np.mean(df["l_target"])) ** 2
        )
        r2 = 1 - ss_res / ss_tot
        # plot the results
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.plot(
            df["m_target"] * 100,
            df["ising_avg_m"] * 100,
            "o",
            color="b",
            label="Solvent",
        )
        plt.plot(
            df["n_target"] * 100,
            df["ising_avg_n"] * 100,
            "x",
            color="orange",
            label="Diluent",
        )
        plt.plot(
            df["l_target"] * 100,
            df["ising_avg_l"] * 100,
            "^",
            mec="g",
            mfc="none",
            label="Salt",
        )
        plt.plot([0, 100], [0, 100], "k--", label=f"RMSE: {rmse:.4f}\nR2: {r2:.4f}")
        plt.xlabel("MD Fractional CN (%)")
        plt.ylabel("Ising Fractional CN (%)")
        plt.legend()
        plt.savefig(f"ising_{key}.png", dpi=300, bbox_inches="tight")
        plt.close(fig=fig)

        # free solvent fraction prediction
        df_free_solvent = pd.read_csv(f"{key}.csv")
        # join the df_free_solvent with df on 'Solvent CID' and 'Diluent CID' and only add columns that are not in df_free_solvent
        df_to_merge = df[
            [
                key
                for key in df.keys()
                if (key not in df_free_solvent.keys())
                or (key in ["Solvent CID", "Diluent CID", "m_target"])
            ]
        ].copy()
        df_to_merge = df_to_merge.round(4)
        df_free_solvent = df_free_solvent.round(4)
        df_free_solvent = df_free_solvent.merge(
            df_to_merge,
            left_on=["Solvent CID", "Diluent CID", "Fractional CN solvent"],
            right_on=["Solvent CID", "Diluent CID", "m_target"],
            how="left",
        )
        ising_coordinated_solvent = (
            df_free_solvent["ising_avg_m"].to_numpy()
            * 3.72
            * df_free_solvent["anion molar ratio"].to_numpy()
        )
        ising_free_solvent = (
            df_free_solvent["solvent molar ratio"].to_numpy()
            - ising_coordinated_solvent
        ) / df_free_solvent["solvent molar ratio"].to_numpy()
        print(len(df_free_solvent), len(df))
        # all the negative ising_free_solvent values should be set to 0
        ising_free_solvent[ising_free_solvent < 0] = 0.0
        # calculate rmse
        target_free_solvent = df_free_solvent["Fraction Free Solvents"].to_numpy()
        ising_free_solvent_tmp = ising_free_solvent.copy()
        rmse_free_solvent = np.sqrt(
            np.mean((ising_free_solvent_tmp - target_free_solvent) ** 2)
        )
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.plot(
            target_free_solvent * 100,
            ising_free_solvent_tmp * 100,
            "o",
            mfc="r",
            mec="k",
        )
        plt.plot([0, 100], [0, 100], "k--", label=f"RMSE: {rmse_free_solvent:.4f}")
        plt.xlabel("MD Free Solvent Fraction (%)")
        plt.ylabel("Ising Free Solvent Fraction (%)")
        plt.legend()
        plt.savefig(f"free_solvent_{key}.png", dpi=300, bbox_inches="tight")
        plt.close(fig=fig)
        df["ising_free_solvent"] = ising_free_solvent
        df["free_solvent_target"] = df_free_solvent["Fraction Free Solvents"].to_numpy()
        df = df.round(4)
        # reorder the keys, such that the sequence is [formulation,Solvent CID,Diluent CID,molality,Solvent DN,Diluent DN,Solvent AN,Diluent AN,Fractional CN anion,Fractional CN solvent,Fractional CN diluent,ising_avg_m,ising_avg_n,ising_avg_l,...]
        # other keys can stay as-is
        keys = df.keys()
        key_first_half = [
            "formulation",
            "Solvent CID",
            "Diluent CID",
            "molality",
            "Solvent DN",
            "Diluent DN",
            "Solvent AN",
            "Diluent AN",
            "Fractional CN anion",
            "Fractional CN solvent",
            "Fractional CN diluent",
            "ising_avg_m",
            "ising_avg_n",
            "ising_avg_l",
            "free_solvent_target",
            "ising_free_solvent",
        ]
        key_second_half = [key for key in keys if key not in key_first_half]
        keys = key_first_half + key_second_half
        df = df[keys]
        df.to_csv(f"ising_results_{key}.csv", index=False)    

if __name__ == "__main__":
    # Initialize parameters
    model_idx = os.getcwd().split("/")[-1]

    train_test_val_df = {}
    train_test_val_df["train"] = pd.read_csv("train.csv")
    train_test_val_df["test"] = pd.read_csv("test.csv")
    train_test_val_df["val"] = pd.read_csv("val.csv")

    data_dict = {}
    for key in train_test_val_df.keys():
        data_dict[key] = {
            "dn0": jnp.array(train_test_val_df[key]["Solvent DN"].to_numpy()),
            "dn1": jnp.array(train_test_val_df[key]["Diluent DN"].to_numpy()),
            "dn_anion": jnp.array(train_test_val_df[key]["Anion DN"].to_numpy()),
            "x0": jnp.array(train_test_val_df[key]["solvent molar ratio"].to_numpy()),
            "x1": jnp.array(train_test_val_df[key]["diluent molar ratio"].to_numpy()),
            "an0": jnp.array(train_test_val_df[key]["Solvent AN"].to_numpy()),
            "an1": jnp.array(train_test_val_df[key]["Diluent AN"].to_numpy()),
            "x_anion": jnp.array(train_test_val_df[key]["anion molar ratio"].to_numpy()),
            "solvent_volume": jnp.array(
                train_test_val_df[key]["solvent volume"].to_numpy()
            ),
            "diluent_volume": jnp.array(
                train_test_val_df[key]["diluent volume"].to_numpy()
            ),
            "anion_volume": jnp.array(
                train_test_val_df[key]["anion volume"].to_numpy()
            ),
            "z": 3.72 / 2.0
            * jnp.ones_like(jnp.array(train_test_val_df[key]["Solvent DN"].to_numpy())), # mean from MD
            "m_target": jnp.array(
                train_test_val_df[key]["Fractional CN solvent"].to_numpy()
            ),
            "n_target": jnp.array(
                train_test_val_df[key]["Fractional CN diluent"].to_numpy()
            ),
            "l_target": jnp.array(
                train_test_val_df[key]["Fractional CN anion"].to_numpy()
            ),
            'init_guess': jnp.array(
                np.tile(np.array([0.5, 0.0, 0.5]), (len(train_test_val_df[key]), 1))
            ),
        }

    train_data = data_dict["train"]
    test_data = data_dict["test"]
    val_data = data_dict["val"]

    num_epochs = 1000
    # optimize parameters using optax
    schedule = optax.exponential_decay(
        init_value=0.002, transition_steps=num_epochs, decay_rate=0.95,
    )
    # Set up an optimizer with optax.
    optimizer = optax.adam(learning_rate=schedule)
    update_keys = [
        "sol_params_dn",
        "salt_params_dn",
        "params_sol_salt_an",
        "params_sol_sol",
        "params_salt",
        "conc_factor_sol",
    ]

    # multiple trainings with different starting seeds to ensure that we get a good minimum
    best_train_loss = jnp.inf
    best_val_loss = jnp.inf
    best_train_loss_log = None
    best_val_loss_log = None
    time_log = []
    for i, trial in enumerate(range(10)):
        print("-------------------------------------------------------")
        print(f"Training trial {i+1} ")
        start_time = time.time()
        lb = int(os.getcwd().split("/")[-1]) * 1000 + 10 * i
        ub = lb + 10
        random_seed = np.random.randint(lb, ub)
        init_params = initialize_params(
            mode="from_scratch",
            random_seed = random_seed,
        )
        mask = {k: (k in update_keys) for k in init_params}
        optimizer = optax.masked(optimizer, mask=mask)
        opt_state = optimizer.init(init_params)
        params = copy.deepcopy(init_params)

        params, train_loss_log, val_loss_log = train(
            params,
            num_epochs,
            train_data,
            test_data,
            val_data,
            opt_state,
            random_seed = random_seed,
        )
        valid_val_loss_idx = jnp.where(~jnp.isnan(val_loss_log))[0]
        valid_train_loss_idx = jnp.where(~jnp.isnan(train_loss_log))[0]
        valid_idx = jnp.intersect1d(valid_val_loss_idx, valid_train_loss_idx)
        val_loss_log = val_loss_log[valid_idx]
        train_loss_log = train_loss_log[valid_idx]
        if len(val_loss_log) == 0 or len(train_loss_log) == 0:
            best_val_loss_by_trial = jnp.inf
        else:
            best_val_loss_by_trial = jnp.min(val_loss_log)
        print('Best val loss so far: ', best_val_loss)
        print('Val loss in this trial: ', best_val_loss_by_trial)
        if best_val_loss_by_trial < best_val_loss:
            best_val_loss_idx = jnp.argmin(val_loss_log)
            best_val_loss = val_loss_log[best_val_loss_idx]
            best_train_loss = train_loss_log[best_val_loss_idx]
            best_params = copy.deepcopy(params)
            print('parameters updated.')
            print(best_params)
            best_train_loss_log = train_loss_log
            best_val_loss_log = val_loss_log
        # jump out if a minimum is already reached, given by both train and val loss < 0.08
        if best_train_loss < 0.08 and best_val_loss < 0.08:
            best_params = copy.deepcopy(params)
            print(f'Good minimum reached, stopping further trials. Trial: {i}, train loss: {train_loss_log[-1]:.4f}, val loss: {val_loss_log[-1]:.4f}')
            break
        end_time = time.time()
        time_log.append(end_time - start_time)

    print("Trained parameters:")
    print(best_params)

    # Save parameters to a file
    with open("trained_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

    # Save parameters to a file
    with open("trained_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    # combine train_loss_log and val_loss_log into a single dictionary and save to a file
    loss_df = pd.DataFrame({
        "train_loss": best_train_loss_log,
        "val_loss": best_val_loss_log,
    })
    loss_df.to_csv("train_val_loss_log.csv", index=False)

    with open("trained_params.pkl", "rb") as f:
        input_params = pickle.load(f)
    for key in input_params.keys():
        print(f"{key}: {input_params[key].tolist()}")
    # do prediction on the full dataset
    parity_results(data_dict, input_params)

    # plot training and validation loss
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    loss_df = pd.read_csv("train_val_loss_log.csv")
    best_train_loss_log = loss_df["train_loss"].to_numpy()
    best_val_loss_log = loss_df["val_loss"].to_numpy()
    step = len(best_train_loss_log)
    plt.semilogy(np.arange(0, step, 1), best_train_loss_log[:step], label="Train Loss")
    plt.semilogy(np.arange(0, step, 1), best_val_loss_log[:step], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("train_val_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig=fig)

    # get the total number of parameters
    n_total = sum(a.size for a in params.values())
    print(f"Total parameters: {n_total}")

    # save time log as txt file
    with open("time_log.txt", "w") as f:
        for item in time_log:
            f.write(f"{item}\n")
            
