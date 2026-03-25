import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import os
import pickle
import copy
from jax import random

from .model import find_root

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)

# Module-level optimizer — must be set by the user before calling update() or train()
optimizer = None


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
