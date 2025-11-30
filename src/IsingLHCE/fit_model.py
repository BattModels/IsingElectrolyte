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
from IsingLHCE.functions import (
    expfunc, mlp, langmuirfunc, linearfunc, logfunc, polynomial_func, sol_sol_func,
)
from IsingLHCE.conc_factor_func import *

@jax.jit
def dn_loss(params, x, y):
    # Vectorize the MLP evaluation over the batch.
    preds = jax.vmap(lambda x: mlp(x, params))(x)
    return jnp.sqrt(jnp.mean((preds - y) ** 2))

def energetics(vars, input_params, input_constants, conc_factor = conc_factor_sep_x_v_sigmoid):
    # Define constants.
    Mref = 1.0
    kT = 0.0257
    cref = 1.0
    # z = 2
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

    dn0 = dn0 * conc_factor(x0, x_anion, v0, v_anion, conc_factor_sol)
    an0 = an0 * conc_factor(x0, x_anion, v0, v_anion, conc_factor_sol)
    dn1 = dn1 * conc_factor(x1, x_anion, v1, v_anion, conc_factor_sol)
    an1 = an1 * conc_factor(x1, x_anion, v1, v_anion, conc_factor_sol)
    dn_anion = dn_anion * phi_anion
    if jax.config.jax_disable_jit:
        print(dn0, dn1, dn_anion)

    avg_m, avg_n, avg_l = vars
    h0 = expfunc(dn0, sol_params_dn_tmp[:4]) + logfunc(x0, sol_params_dn_tmp[4])
    h1 = expfunc(dn1, sol_params_dn_tmp[:4]) + logfunc(x1, sol_params_dn_tmp[4])
    h2 = expfunc(dn_anion, salt_params_dn_tmp[:4]) + logfunc(
        x_anion, salt_params_dn_tmp[4]
    )
    h = jnp.array([h0, h1, h2]).flatten()
    j00 = (
        sol_sol_func(jnp.array([dn0, an0]), params_sol_sol_tmp[:6])
        + sol_sol_func(jnp.array([dn0, an0]), params_sol_sol_tmp[:6])
        + 2.0 * logfunc(x0, params_sol_sol_tmp[6])
    )
    j01 = (
        sol_sol_func(jnp.array([dn0, an1]), params_sol_sol_tmp[:6])
        + sol_sol_func(jnp.array([dn1, an0]), params_sol_sol_tmp[:6])
        + logfunc(x0, params_sol_sol_tmp[6])
        + logfunc(x1, params_sol_sol_tmp[6])
    )
    j10 = (
        sol_sol_func(jnp.array([dn1, an0]), params_sol_sol_tmp[:6])
        + sol_sol_func(jnp.array([dn0, an1]), params_sol_sol_tmp[:6])
        + logfunc(x0, params_sol_sol_tmp[6])
        + logfunc(x1, params_sol_sol_tmp[6])
    )
    j11 = (
        sol_sol_func(jnp.array([dn1, an1]), params_sol_sol_tmp[:6])
        + sol_sol_func(jnp.array([dn1, an1]), params_sol_sol_tmp[:6])
        + 2.0 * logfunc(x1, params_sol_sol_tmp[6])
    )
    j02 = (
        expfunc(an0, params_sol_salt_an_tmp[:4])
        + logfunc(x0, params_sol_salt_an_tmp[4])
        + logfunc(x_anion, params_sol_salt_an_tmp[5])
    )
    j20 = (
        expfunc(an0, params_sol_salt_an_tmp[:4])
        + logfunc(x0, params_sol_salt_an_tmp[4])
        + logfunc(x_anion, params_sol_salt_an_tmp[5])
    )
    j12 = (
        expfunc(an1, params_sol_salt_an_tmp[:4])
        + logfunc(x1, params_sol_salt_an_tmp[4])
        + logfunc(x_anion, params_sol_salt_an_tmp[5])
    )
    j21 = (
        expfunc(an1, params_sol_salt_an_tmp[:4])
        + logfunc(x1, params_sol_salt_an_tmp[4])
        + logfunc(x_anion, params_sol_salt_an_tmp[5])
    )
    j22 = expfunc(dn_anion, params_salt_tmp[:4]) + logfunc(x_anion, params_salt_tmp[4])

    J = jnp.array([[j00, j01, j02], [j10, j11, j12], [j20, j21, j22]])
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
    if jax.config.jax_disable_jit:
        # For debugging purposes, print the intermediate values.
        print("h:", h)
        print("J:", J)
        print("exp0:", exp0, "exp1:", exp1, "exp2:", exp2, "partition:", partition)
        print("avg_m:", avg_m, "avg_n:", avg_n, "avg_l:", avg_l)
    f1 = exp0 / partition - avg_m
    f3 = exp2 / partition - avg_l
    f2 = exp1 / partition - avg_n
    return jnp.array([f1, f2, f3])

BROYDEN_SOLVER = Broyden(fun=equations, maxiter=1000, tol=1e-8, verbose=False, jit=True)
@partial(jax.jit, static_argnames=("max_tries",))
# @jax.jit
def find_root(input_params, input_constants, initial_guess = jnp.array([0.5, 0.0, 0.5]), max_tries=10):
    """
    Find the root of the equations using Broyden's method.
    """
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
            # Run solver with current guess
            sol = BROYDEN_SOLVER.run(
                guess, 
                input_params=input_params, 
                input_constants=input_constants
            )
            params = sol.params
            in_bounds = (params >= 0.0) & (params <= 1.0)
            is_valid = jnp.all(in_bounds)

            # only update best_sol on the first valid solution found
            update = is_valid & (~found_valid)
            new_best = jnp.where(update, params, best_sol)
            new_found = found_valid | is_valid
            return new_best, new_found
        best_sol_out, found_valid_out = jax.lax.cond(
            found_valid,
            skip_solve,
            do_solve,
            operand=None
        )
        return (best_sol_out, found_valid_out), None
        
    init_best = jnp.full((3,), jnp.nan)
    init_found = jnp.array(False)
    (final_sol, found_valid), _ = jax.lax.scan(
        try_solve,
        (init_best, init_found),
        guesses[:max_tries],
    )
    if jax.config.jax_disable_jit:
        print('avg_m: ', final_sol[0], 'avg_n: ', final_sol[1], 'avg_l: ', final_sol[2], 'found_any: ', found_any)
    return final_sol

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
):
    # Get the outputs from your function.
    avg_m, avg_n, avg_l = find_root(
        input_params=input_params, input_constants=input_constants, 
    )
    # We define a loss as the squared difference.
    loss = (avg_m - m_target) ** 2 + (avg_n - n_target) ** 2 + (avg_l - l_target) ** 2
    return loss

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
        in_axes=(None, 1, 0, 0, 0),
    )
    losses = v_objective(
        params,
        input_constants,
        data["m_target"],
        data["n_target"],
        data["l_target"],
    )
    return jnp.sqrt(jnp.mean(losses))

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
                -1.73,
                0.75,
                0.003,
                -0.35,
                -0.026,
            ]
        )
        salt_params_dn = jnp.array(
            [
                -1.73,
                0.75,
                0.003,
                -0.35,
                -0.026,
            ]
        )
        params_sol_salt_an = jnp.array(
            [
                -0.25,
                0.2,
                0.8,
                0.05,
                -0.026,
                -0.026,
            ]
        )
        params_sol_sol = jnp.array(
            [
                0.28,
                0.1,
                -32.7,
                2.82,
                -5.93,
                0.06,
                0.001,
            ]
        )
        params_salt = jnp.array(
            [
                -0.2,
                0.87,
                0.14,
                -0.03,
                -0.026,
            ]
        )
        conc_factor_sol = jnp.array([2.86, -3.04, 0.22, -0.31, 1.86, -2.04, 0.22, -0.31])

        # add random noise to the initial params
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=6)
        sol_params_dn = sol_params_dn + 1e-3 * random.normal(
            keys[0], shape=sol_params_dn.shape
        )
        salt_params_dn = salt_params_dn + 1e-3 * random.normal(
            keys[1], shape=salt_params_dn.shape
        )
        params_sol_salt_an = params_sol_salt_an + 1e-3 * random.normal(
            keys[2], shape=params_sol_salt_an.shape
        )
        params_sol_sol = params_sol_sol + 1e-3 * random.normal(
            keys[3], shape=params_sol_sol.shape
        )
        params_salt = params_salt + 1e-3 * random.normal(
            keys[4], shape=params_salt.shape
        )
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
        if len(init_params['conc_factor_sol']) == 4:
            init_params['conc_factor_sol'] = jnp.concatenate([init_params['conc_factor_sol'], init_params['conc_factor_sol']])
        # add random noise to the initial params
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=6)
        init_params['sol_params_dn'] = init_params['sol_params_dn'] + 1e-3 * random.normal(
            keys[0], shape=init_params['sol_params_dn'].shape
        )
        init_params['salt_params_dn'] = init_params['salt_params_dn'] + 1e-3 * random.normal(
            keys[1], shape=init_params['salt_params_dn'].shape
        )
        init_params['params_sol_salt_an'] = init_params['params_sol_salt_an'] + 1e-3 * random.normal(
            keys[2], shape=init_params['params_sol_salt_an'].shape
        )
        init_params['params_sol_sol'] = init_params['params_sol_sol'] + 1e-3 * random.normal(
            keys[3], shape=init_params['params_sol_sol'].shape
        )
        init_params['params_salt'] = init_params['params_salt'] + 1e-3 * random.normal(
            keys[4], shape=init_params['params_salt'].shape
        )
        init_params['conc_factor_sol'] = init_params['conc_factor_sol'] + 1e-3 * random.normal(
            keys[5], shape=init_params['conc_factor_sol'].shape
        )
        return init_params

@jax.jit
def li_free_energy(input_params, input_constants):
    avg_m, avg_n, avg_l = find_root(input_params, input_constants)
    h, J, _, _, _, kT, z = energetics(
        jnp.array([avg_m, avg_n, avg_l]), input_params, input_constants
    )
     # Free energy calculation
    G = h[0] * z * avg_m + h[1] * z * avg_n + h[2] * z * avg_l
    return G  

@jax.jit
def update(params, opt_state, data):
    loss, grads = jax.value_and_grad(total_objective)(params, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss    

def train(params, num_epochs, train_data, test_data, val_data, optimizer=None, random_seed=42):
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
    n_test = test_data["dn0"].shape[0]
    n_val = val_data["dn0"].shape[0]
    key = random.PRNGKey(random_seed)
    batch_size = n_train
    steps_per_ep = n_train // batch_size

    if optimizer is None:
        schedule = optax.exponential_decay(
            init_value = 0.002, transition_steps=num_epochs, decay_rate=0.9,
        )
        optimizer = optax.adam(schedule)

    update_keys = [
        "sol_params_dn",
        "salt_params_dn",
        "params_sol_salt_an",
        "params_sol_sol",
        "params_salt",
        "conc_factor_sol",
    ]
    mask = {k: (k in update_keys) for k in params.keys()}
    optimizer = optax.masked(optimizer, mask)
    opt_state = optimizer.init(params)

    for i, epoch in enumerate(range(num_epochs)):
        prev_params = copy.deepcopy(params)  # backup from the previous epoch
        
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_train)

        epoch_loss = 0.0
        for step in range(steps_per_ep):
            batch_idx = perm[step * batch_size : (step + 1) * batch_size]
            # Create a batch of data
            batch = {k: jnp.take(v, batch_idx, axis=0) for k,v in train_data.items()}
            # Compute gradients & update parameters on this batch
            params, opt_state, loss = update(params, opt_state, batch)
            epoch_loss += loss
        
        epoch_loss /= steps_per_ep
        val_loss = total_objective(params, val_data)
        test_loss = total_objective(params, test_data)
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch}: loss = {epoch_loss:.4f}, val loss = {val_loss:.4f}, test loss = {test_loss:.4f}"
            )
        train_loss_log = train_loss_log.at[i].set(epoch_loss)
        val_loss_log = val_loss_log.at[i].set(val_loss)

        if round(val_loss,3) <= round(best_val_loss, 3):
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
        df = pd.DataFrame(data_dict[key])
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
            z = df['z'].iloc[i]
            avg_m, avg_n, avg_l = find_root(
                input_params = input_params,
                input_constants = [
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
        loss = total_objective(input_params, data_dict[key])
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
        loss = total_objective(input_params, data_dict["train"])
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
            * df_free_solvent['z'].to_numpy()
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
    return None