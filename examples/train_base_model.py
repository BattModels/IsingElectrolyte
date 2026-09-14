"""
Replication of the original fit_model.py using the IsingElectrolyte package.

Physics backend: IsingElectrolyte.model.*_old functions, which preserve the original
2-solvent + 1-anion formulation with params_salt (5 elem, expfunc-based j22)
and hard-coded monotonicity constraints — identical to the legacy fit_model.py.

Usage (run from a directory containing train.csv, test.csv, val.csv):
    python examples/train_base_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import os
import pickle
import copy
import time
from jax import random
from functools import partial

from IsingElectrolyte.model import (
    find_root_old,
    li_free_energy_old,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)

from matplotlib import rcParams

fontsize = 15
rcParams.update({
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
    "legend.fontsize": fontsize,
    "legend.frameon": False,
    "figure.figsize": [6, 6],
})

# Module-level optimizer — set before calling update_old() or train_old()
optimizer = None


# ---------------------------------------------------------------------------
# Parameter initialisation (old interface: params_salt, 5 elem)
# ---------------------------------------------------------------------------

def initialize_params_old(mode="from_scratch", file_path=None, random_seed=42):
    """Initialise the parameter dict for the legacy 2-sol + 1-anion model.

    Param keys: sol_params_dn(5), salt_params_dn(5), params_sol_salt_an(6),
                params_sol_sol(16), params_salt(5), conc_factor_sol(8).
    """
    _HAND_TUNED = {
        "sol_params_dn": jnp.array([
            -1.4226977825164795, 0.6868864893913269, -1.29103684425354,
            -2.2087085247039795, -2.837172,
        ]),
        "salt_params_dn": jnp.array([
            -1.933838129043579, 0.21721802651882172, -1.0301377773284912,
            -2.815300941467285, -1.930697,
        ]),
        "params_sol_salt_an": jnp.array([
            -0.28391966223716736, -0.4041173458099365, 0.6561649441719055,
            -2.1234798431396484, -1.09547758102417, -1.3783741,
        ]),
        "params_sol_sol": jnp.array([
            0.11802631616592407, 0.04696842283010483, 0.039969541132450104,
            -3.7262208461761475, -3.0505568981170654, -0.10816801339387894,
            -2.2049686908721924, 0.375214546918869, -3.009303092956543,
            -2.591977834701538, -0.11441854387521744, -2.53210186958313,
            0.3858657479286194, -2.321645736694336, -2.3250560760498047,
            -3.2535532,
        ]),
        "params_salt": jnp.array([
            -0.37612465023994446, -0.13344359397888184, -1.854067325592041,
            -3.3515141010284424, -3.6836671829223633,
        ]),
        "conc_factor_sol": jnp.array([
            0.6028587818145752, 0.5641694664955139, -0.0062898872420191765,
            -1.0074571371078491, -2177.35791015625, 2189.39013671875,
            -6.164828777313232, 2.281510829925537,
        ]),
    }
    _KEY_ORDER = list(_HAND_TUNED.keys())
    noise_scale = 0.05

    if mode == "from_scratch":
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=len(_KEY_ORDER))
        init_params = {}
        for i, name in enumerate(_KEY_ORDER):
            base = _HAND_TUNED[name]
            init_params[name] = base * (1.0 + noise_scale * random.normal(
                keys[i], shape=base.shape
            ))
        return init_params

    elif mode == "from_file":
        if file_path is None:
            raise ValueError("file_path must be provided when mode is 'from_file'")
        with open(file_path, "rb") as f:
            init_params = pickle.load(f)
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=len(_KEY_ORDER))
        for i, name in enumerate(_KEY_ORDER):
            arr = init_params[name]
            init_params[name] = arr * (1.0 + noise_scale * random.normal(
                keys[i], shape=arr.shape
            ))
        return init_params

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'from_scratch' or 'from_file'.")


# ---------------------------------------------------------------------------
# Objective and update (old interface)
# ---------------------------------------------------------------------------

def objective_single_old(input_params, input_constants, m_target, n_target, l_target, init_guess):
    (avg_m, avg_n, avg_l), found_valid = find_root_old(
        input_params=input_params,
        input_constants=input_constants,
        initial_guess=init_guess,
    )
    loss = (avg_m - m_target) ** 2 + (avg_n - n_target) ** 2 + (avg_l - l_target) ** 2
    new_init = jnp.where(found_valid, jnp.array([avg_m, avg_n, avg_l]), init_guess)
    return loss, new_init


@jax.jit
def total_objective_old(params, data):
    input_constants = jnp.vstack([
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
    ])
    v_objective = jax.vmap(
        objective_single_old,
        in_axes=(None, 1, 0, 0, 0, 0),
    )
    losses, new_inits = v_objective(
        params,
        input_constants,
        data["m_target"],
        data["n_target"],
        data["l_target"],
        data["init_guess"],
    )
    mean_loss = jnp.sqrt(jnp.mean(losses))
    return mean_loss, new_inits


@jax.jit
def update_old(params, opt_state, data):
    (loss, new_inits), grads = jax.value_and_grad(total_objective_old, has_aux=True)(
        params, data
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, new_inits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_old(params, num_epochs, train_data, test_data, val_data, opt_state, random_seed=42):
    prev_params = copy.deepcopy(params)
    best_params = copy.deepcopy(params)

    train_loss_log = jnp.zeros(num_epochs)
    val_loss_log = jnp.zeros(num_epochs)

    best_val_loss = jnp.inf
    best_train_loss = jnp.inf
    patience = 100
    patience_counter = 0

    n_train = train_data["dn0"].shape[0]
    init_guess_train = jnp.tile(jnp.array([0.5, 0.0, 0.5]), (n_train, 1))
    n_test = test_data["dn0"].shape[0]
    init_guess_test = jnp.tile(jnp.array([0.5, 0.0, 0.5]), (n_test, 1))
    n_val = val_data["dn0"].shape[0]
    init_guess_val = jnp.tile(jnp.array([0.5, 0.0, 0.5]), (n_val, 1))

    key = random.PRNGKey(random_seed)
    batch_size = n_train
    steps_per_ep = n_train // batch_size

    for i, epoch in enumerate(range(num_epochs)):
        prev_params = copy.deepcopy(params)

        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_train)

        epoch_loss = 0.0
        for step in range(steps_per_ep):
            batch_idx = perm[step * batch_size : (step + 1) * batch_size]
            batch = {k: jnp.take(v, batch_idx, axis=0) for k, v in train_data.items()}
            batch["init_guess"] = jnp.take(init_guess_train, batch_idx, axis=0)
            params, opt_state, loss, new_inits = update_old(params, opt_state, batch)
            init_guess_train = init_guess_train.at[batch_idx].set(new_inits)
            epoch_loss += loss

        epoch_loss /= steps_per_ep
        val_data["init_guess"] = init_guess_val
        test_data["init_guess"] = init_guess_test
        val_loss, _ = total_objective_old(params, val_data)
        test_loss, _ = total_objective_old(params, test_data)

        print(f"Epoch {epoch}: loss = {epoch_loss:.4f}, val loss = {val_loss:.4f}, test loss = {test_loss:.4f}")

        train_loss_log = train_loss_log.at[i].set(epoch_loss)
        val_loss_log = val_loss_log.at[i].set(val_loss)

        if round(val_loss, 4) <= round(best_val_loss, 4):
            best_val_loss = val_loss
            best_params = copy.deepcopy(params)
            patience_counter = 0
        else:
            patience_counter += 1

        if jnp.isnan(epoch_loss) or jnp.isnan(val_loss):
            params = copy.deepcopy(best_params)
            break

        if patience_counter >= patience:
            print(f"Early stopping triggered at step {epoch}: No improvement for {patience} steps.")
            params = copy.deepcopy(best_params)
            break

        if jnp.abs(epoch_loss) < 0.04 and jnp.abs(val_loss) < 0.04:
            print("Losses are below threshold; stopping training.")
            break

    train_loss_log = train_loss_log.at[i + 1:].set(jnp.nan)
    val_loss_log = val_loss_log.at[i + 1:].set(jnp.nan)
    train_loss_log = train_loss_log[~jnp.isnan(train_loss_log)]
    val_loss_log = val_loss_log[~jnp.isnan(val_loss_log)]
    return params, train_loss_log, val_loss_log


# ---------------------------------------------------------------------------
# Parity results
# ---------------------------------------------------------------------------

def parity_results_old(data_dict, input_params):
    for key in list(data_dict.keys()):
        df = pd.DataFrame({kk: data_dict[key][kk] for kk in data_dict[key] if kk != "init_guess"})
        df1 = pd.read_csv(f"{key}.csv")
        df = df.merge(
            df1,
            left_on=["dn0", "dn1", "an0", "an1", "m_target"],
            right_on=["Solvent DN", "Diluent DN", "Solvent AN", "Diluent AN", "Fractional CN solvent"],
            how="left",
        )
        df = df[["formulation"] + list(df.columns.difference(["formulation"]))]

        ising_avg_m = np.zeros(len(df))
        ising_avg_n = np.zeros(len(df))
        ising_avg_l = np.zeros(len(df))
        for i in range(len(df)):
            input_constants = [
                df["dn0"].iloc[i],
                df["dn1"].iloc[i],
                df["dn_anion"].iloc[i],
                df["x0"].iloc[i],
                df["x1"].iloc[i],
                df["an0"].iloc[i],
                df["an1"].iloc[i],
                df["x_anion"].iloc[i],
                df["solvent_volume"].iloc[i],
                df["diluent_volume"].iloc[i],
                df["anion_volume"].iloc[i],
                3.72 / 2.0,
            ]
            (avg_m, avg_n, avg_l), _ = find_root_old(input_params, input_constants)
            ising_avg_m[i] = avg_m
            ising_avg_n[i] = avg_n
            ising_avg_l[i] = avg_l

        loss, _ = total_objective_old(input_params, data_dict[key])
        print(f"Loss on the {key} data: {loss:.4f}")
        rmse = np.sqrt(np.mean(
            (ising_avg_m - df["m_target"]) ** 2
            + (ising_avg_n - df["n_target"]) ** 2
            + (ising_avg_l - df["l_target"]) ** 2
        ))
        print(f"RMSE on the {key} data: {rmse:.4f}")

        df["ising_avg_m"] = ising_avg_m
        df["ising_avg_n"] = ising_avg_n
        df["ising_avg_l"] = ising_avg_l
        df = df.round(4)
        df.to_csv(f"ising_results_{key}.csv", index=False)
        df = pd.read_csv(f"ising_results_{key}.csv")

        rmse = np.sqrt(np.mean(
            (df["ising_avg_m"] - df["m_target"]) ** 2
            + (df["ising_avg_n"] - df["n_target"]) ** 2
            + (df["ising_avg_l"] - df["l_target"]) ** 2
        ))
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

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.plot(df["m_target"] * 100, df["ising_avg_m"] * 100, "o", color="b", label="Solvent")
        plt.plot(df["n_target"] * 100, df["ising_avg_n"] * 100, "x", color="orange", label="Diluent")
        plt.plot(df["l_target"] * 100, df["ising_avg_l"] * 100, "^", mec="g", mfc="none", label="Salt")
        plt.plot([0, 100], [0, 100], "k--", label=f"RMSE: {rmse:.4f}\nR2: {r2:.4f}")
        plt.xlabel("MD Fractional CN (%)")
        plt.ylabel("Ising Fractional CN (%)")
        plt.legend()
        plt.savefig(f"ising_{key}.png", dpi=300, bbox_inches="tight")
        plt.close(fig=fig)

        df_free_solvent = pd.read_csv(f"{key}.csv")
        df_to_merge = df[
            [k for k in df.keys() if (k not in df_free_solvent.keys()) or (k in ["Solvent CID", "Diluent CID", "m_target"])]
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
            df_free_solvent["solvent molar ratio"].to_numpy() - ising_coordinated_solvent
        ) / df_free_solvent["solvent molar ratio"].to_numpy()
        ising_free_solvent[ising_free_solvent < 0] = 0.0
        target_free_solvent = df_free_solvent["Fraction Free Solvents"].to_numpy()
        rmse_free_solvent = np.sqrt(np.mean((ising_free_solvent - target_free_solvent) ** 2))

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.plot(target_free_solvent * 100, ising_free_solvent * 100, "o", mfc="r", mec="k")
        plt.plot([0, 100], [0, 100], "k--", label=f"RMSE: {rmse_free_solvent:.4f}")
        plt.xlabel("MD Free Solvent Fraction (%)")
        plt.ylabel("Ising Free Solvent Fraction (%)")
        plt.legend()
        plt.savefig(f"free_solvent_{key}.png", dpi=300, bbox_inches="tight")
        plt.close(fig=fig)

        df["ising_free_solvent"] = ising_free_solvent
        df["free_solvent_target"] = df_free_solvent["Fraction Free Solvents"].to_numpy()
        df = df.round(4)
        key_first_half = [
            "formulation", "Solvent CID", "Diluent CID", "molality",
            "Solvent DN", "Diluent DN", "Solvent AN", "Diluent AN",
            "Fractional CN anion", "Fractional CN solvent", "Fractional CN diluent",
            "ising_avg_m", "ising_avg_n", "ising_avg_l",
            "free_solvent_target", "ising_free_solvent",
        ]
        key_second_half = [k for k in df.keys() if k not in key_first_half]
        df = df[key_first_half + key_second_half]
        df.to_csv(f"ising_results_{key}.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_idx = os.getcwd().split("/")[-1]

    train_test_val_df = {
        "train": pd.read_csv("train.csv"),
        "test":  pd.read_csv("test.csv"),
        "val":   pd.read_csv("val.csv"),
    }

    data_dict = {}
    for split, df in train_test_val_df.items():
        n = len(df)
        data_dict[split] = {
            "dn0":            jnp.array(df["Solvent DN"].to_numpy()),
            "dn1":            jnp.array(df["Diluent DN"].to_numpy()),
            "dn_anion":       jnp.array(df["Anion DN"].to_numpy()),
            "x0":             jnp.array(df["solvent molar ratio"].to_numpy()),
            "x1":             jnp.array(df["diluent molar ratio"].to_numpy()),
            "an0":            jnp.array(df["Solvent AN"].to_numpy()),
            "an1":            jnp.array(df["Diluent AN"].to_numpy()),
            "x_anion":        jnp.array(df["anion molar ratio"].to_numpy()),
            "solvent_volume": jnp.array(df["solvent volume"].to_numpy()),
            "diluent_volume": jnp.array(df["diluent volume"].to_numpy()),
            "anion_volume":   jnp.array(df["anion volume"].to_numpy()),
            "z":              3.72 / 2.0 * jnp.ones(n),
            "m_target":       jnp.array(df["Fractional CN solvent"].to_numpy()),
            "n_target":       jnp.array(df["Fractional CN diluent"].to_numpy()),
            "l_target":       jnp.array(df["Fractional CN anion"].to_numpy()),
            "init_guess":     jnp.tile(jnp.array([0.5, 0.0, 0.5]), (n, 1)),
        }

    train_data = data_dict["train"]
    test_data  = data_dict["test"]
    val_data   = data_dict["val"]

    num_epochs = 1000
    schedule = optax.exponential_decay(
        init_value=0.002, transition_steps=num_epochs, decay_rate=0.95,
    )
    base_optimizer = optax.adam(learning_rate=schedule)

    update_keys = [
        "sol_params_dn", "salt_params_dn", "params_sol_salt_an",
        "params_sol_sol", "params_salt", "conc_factor_sol",
    ]

    best_val_loss = jnp.inf
    best_train_loss = jnp.inf
    best_params = None
    best_train_loss_log = None
    best_val_loss_log = None
    time_log = []

    for i in range(10):
        print("-------------------------------------------------------")
        print(f"Training trial {i + 1}")
        start_time = time.time()

        lb = int(model_idx) * 1000 + 10 * i if model_idx.isdigit() else 10 * i
        ub = lb + 10
        random_seed = np.random.randint(lb, ub)

        init_params = initialize_params_old(mode="from_scratch", random_seed=random_seed)
        mask = {k: (k in update_keys) for k in init_params}
        optimizer = optax.masked(base_optimizer, mask=mask)
        opt_state = optimizer.init(init_params)
        params = copy.deepcopy(init_params)

        params, train_loss_log, val_loss_log = train_old(
            params, num_epochs, train_data, test_data, val_data, opt_state,
            random_seed=random_seed,
        )

        valid_val_idx   = jnp.where(~jnp.isnan(val_loss_log))[0]
        valid_train_idx = jnp.where(~jnp.isnan(train_loss_log))[0]
        valid_idx = jnp.intersect1d(valid_val_idx, valid_train_idx)
        val_loss_log   = val_loss_log[valid_idx]
        train_loss_log = train_loss_log[valid_idx]

        if len(val_loss_log) == 0 or len(train_loss_log) == 0:
            best_val_loss_by_trial = jnp.inf
        else:
            best_val_loss_by_trial = jnp.min(val_loss_log)

        print(f"Best val loss so far: {best_val_loss}")
        print(f"Val loss in this trial: {best_val_loss_by_trial}")

        if best_val_loss_by_trial < best_val_loss:
            best_val_loss_idx = jnp.argmin(val_loss_log)
            best_val_loss   = val_loss_log[best_val_loss_idx]
            best_train_loss = train_loss_log[best_val_loss_idx]
            best_params = copy.deepcopy(params)
            print("Parameters updated.")
            print(best_params)
            best_train_loss_log = train_loss_log
            best_val_loss_log   = val_loss_log

        if best_train_loss < 0.08 and best_val_loss < 0.08:
            print(f"Good minimum reached, stopping. Trial: {i}, "
                  f"train: {train_loss_log[-1]:.4f}, val: {val_loss_log[-1]:.4f}")
            break

        time_log.append(time.time() - start_time)

    print("Trained parameters:")
    print(best_params)

    with open("trained_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

    loss_df = pd.DataFrame({
        "train_loss": best_train_loss_log,
        "val_loss":   best_val_loss_log,
    })
    loss_df.to_csv("train_val_loss_log.csv", index=False)

    with open("trained_params.pkl", "rb") as f:
        input_params = pickle.load(f)
    for k, v in input_params.items():
        print(f"{k}: {v.tolist()}")

    parity_results_old(data_dict, input_params)

    loss_df = pd.read_csv("train_val_loss_log.csv")
    step = len(loss_df)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.semilogy(np.arange(step), loss_df["train_loss"].to_numpy(), label="Train Loss")
    plt.semilogy(np.arange(step), loss_df["val_loss"].to_numpy(),   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("train_val_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig=fig)

    n_total = sum(a.size for a in input_params.values())
    print(f"Total parameters: {n_total}")

    with open("time_log.txt", "w") as f:
        for t in time_log:
            f.write(f"{t}\n")
