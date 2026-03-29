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
from functools import partial

from .model import _find_root_impl, DEFAULT_MONOTONICITY, _freeze_mono
from .interactions import (
    default_h_sol_rescale, default_h_an_rescale,
    default_J_sol_sol_rescale, default_J_sol_an_rescale, default_J_an_an_rescale,
    default_conc_factor_rescale,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)

# Module-level optimizer — must be set by the user before calling update() or train()
optimizer = None


def objective_single(
    input_params,
    dn_sol, an_sol, x_sol, v_sol,
    dn_an, x_an, v_an,
    z,
    targets,
    init_guess,
    monotonicity_dict=None,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Compute the squared loss for a single data point.

    Args:
        input_params:           parameter dict (must include params_anion_anion)
        dn_sol, an_sol, x_sol, v_sol: shape (N,) arrays for solvents
        dn_an, x_an, v_an:            shape (M,) arrays for anions
        z:                      coordination number (scalar)
        targets:                target occupation fractions, shape (N+M,)
        init_guess:             initial guess for the root solver, shape (N+M,)
        monotonicity_dict:      frozenset from _freeze_mono(), or None for default.
        rescale_h_sol, rescale_h_an, rescale_J_sol_sol, rescale_J_sol_an,
        rescale_J_an_an, rescale_conc_factor:
                                companion rescaling functions.

    Returns:
        (loss, new_init): squared loss (scalar), updated initial guess shape (N+M,)
    """
    occupations, found_valid = _find_root_impl(
        input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an, z,
        init_guess, max_tries=10, monotonicity_dict=monotonicity_dict,
        rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
    )
    loss = jnp.sum((occupations - targets) ** 2)
    new_init = jnp.where(found_valid, occupations, init_guess)
    return loss, new_init


@partial(jax.jit, static_argnames=(
    "monotonicity_dict",
    "rescale_h_sol", "rescale_h_an",
    "rescale_J_sol_sol", "rescale_J_sol_an", "rescale_J_an_an",
    "rescale_conc_factor",
))
def total_objective(
    params, data, monotonicity_dict=None,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Compute RMSE loss over all data points.

    data must contain:
        dn_sol, an_sol, x_sol, v_sol: shape (n, N) arrays for solvents
        dn_an, x_an, v_an:            shape (n, M) arrays for anions
        z:                            shape (n,) coordination numbers
        targets:                      shape (n, N+M) target occupation fractions
        init_guess:                   shape (n, N+M)
    monotonicity_dict: frozenset from _freeze_mono(), or None for default.
    """
    v_objective = jax.vmap(
        partial(
            objective_single,
            monotonicity_dict=monotonicity_dict,
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        ),
        in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    losses, new_inits = v_objective(
        params,
        data["dn_sol"], data["an_sol"], data["x_sol"], data["v_sol"],
        data["dn_an"], data["x_an"], data["v_an"],
        data["z"],
        data["targets"],
        data["init_guess"],
    )
    mean_loss = jnp.sqrt(jnp.mean(losses))
    return mean_loss, new_inits


def initialize_params(
    mode="from_file",
    file_path=None,
    input_dim=1,
    hidden_dim=4,
    output_dim=1,
    random_seed=42,
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
        # 6-element anion-anion coupling (sol_sol_func form); initialised near zero
        params_anion_anion = jnp.zeros(6)
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
        keys = random.split(key, num=7)
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
        # params_anion_anion starts at zero: use additive noise
        params_anion_anion = params_anion_anion + noise_scale * random.normal(
            keys[4], shape=params_anion_anion.shape
        )
        conc_factor_sol = conc_factor_sol * (1.0 + noise_scale * random.normal(
            keys[5], shape=conc_factor_sol.shape
        ))
        init_params = {
            "sol_params_dn": sol_params_dn,
            "salt_params_dn": salt_params_dn,
            "params_sol_salt_an": params_sol_salt_an,
            "params_sol_sol": params_sol_sol,
            "params_anion_anion": params_anion_anion,
            "conc_factor_sol": conc_factor_sol,
        }
        return init_params

    elif mode == "from_file":
        if file_path is None:
            raise ValueError("file_path must be provided when mode is 'from_file'")
        with open(file_path, "rb") as f:
            init_params = pickle.load(f)
        if "params_anion_anion" not in init_params:
            raise ValueError(
                "The loaded checkpoint does not contain 'params_anion_anion'. "
                "It was likely saved with the legacy interface which used 'params_salt' (5 elements). "
                "Please re-initialize from scratch with initialize_params(mode='from_scratch')."
            )
        # add random noise to the initial params
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=7)
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
        init_params['params_anion_anion'] = init_params['params_anion_anion'] * (1.0 + noise_scale * random.normal(
            keys[4], shape=init_params['params_anion_anion'].shape
        ))
        init_params['conc_factor_sol'] = init_params['conc_factor_sol'] * (1.0 + noise_scale * random.normal(
            keys[5], shape=init_params['conc_factor_sol'].shape
        ))
        return init_params


@partial(jax.jit, static_argnames=(
    "monotonicity_dict",
    "rescale_h_sol", "rescale_h_an",
    "rescale_J_sol_sol", "rescale_J_sol_an", "rescale_J_an_an",
    "rescale_conc_factor",
))
def update(
    params, opt_state, data, monotonicity_dict=None,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    (loss, new_inits), grads = jax.value_and_grad(
        partial(
            total_objective,
            monotonicity_dict=monotonicity_dict,
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        ), has_aux=True
    )(params, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, new_inits


def train(params, num_epochs, train_data, test_data, val_data, opt_state,
          monotonicity_dict=DEFAULT_MONOTONICITY,
          rescale_h_sol=default_h_sol_rescale,
          rescale_h_an=default_h_an_rescale,
          rescale_J_sol_sol=default_J_sol_sol_rescale,
          rescale_J_sol_an=default_J_sol_an_rescale,
          rescale_J_an_an=default_J_an_an_rescale,
          rescale_conc_factor=default_conc_factor_rescale,
          random_seed=42):
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

    # Infer dataset sizes and species count from data
    n_train = train_data["targets"].shape[0]
    n_species = train_data["dn_sol"].shape[1] + train_data["dn_an"].shape[1]
    init_guess_train = jnp.full((n_train, n_species), 1.0 / n_species)

    n_test = test_data["targets"].shape[0]
    init_guess_test = jnp.full((n_test, n_species), 1.0 / n_species)

    n_val = val_data["targets"].shape[0]
    init_guess_val = jnp.full((n_val, n_species), 1.0 / n_species)

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
            batch = {k: jnp.take(v, batch_idx, axis=0) for k, v in train_data.items()}
            batch['init_guess'] = jnp.take(init_guess_train, batch_idx, axis=0)
            # Compute gradients & update parameters on this batch
            params, opt_state, loss, new_inits = update(
                params, opt_state, batch,
                monotonicity_dict=_freeze_mono(monotonicity_dict),
                rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
                rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
                rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
            )
            init_guess_train = init_guess_train.at[batch_idx].set(new_inits)
            epoch_loss += loss

        epoch_loss /= steps_per_ep
        val_data['init_guess'] = init_guess_val
        test_data['init_guess'] = init_guess_test
        val_loss, _ = total_objective(
            params, val_data,
            monotonicity_dict=_freeze_mono(monotonicity_dict),
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        test_loss, _ = total_objective(
            params, test_data,
            monotonicity_dict=_freeze_mono(monotonicity_dict),
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch}: loss = {epoch_loss:.4f}, val loss = {val_loss:.4f}, test loss = {test_loss:.4f}"
            )
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

    train_loss_log = train_loss_log.at[i+1:].set(jnp.nan)
    val_loss_log = val_loss_log.at[i+1:].set(jnp.nan)
    train_loss_log = train_loss_log[~jnp.isnan(train_loss_log)]
    val_loss_log = val_loss_log[~jnp.isnan(val_loss_log)]
    return params, train_loss_log, val_loss_log


def parity_results(
    data_dict, input_params, monotonicity_dict=DEFAULT_MONOTONICITY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Evaluate trained parameters against each dataset split and produce parity plots.

    data_dict maps split names (e.g. "train", "val", "test") to data dicts with keys:
        dn_sol, an_sol, x_sol, v_sol: shape (n, N)
        dn_an, x_an, v_an:            shape (n, M)
        z:                            shape (n,)
        targets:                      shape (n, N+M)
    The function stores per-species predictions as "pred_{i}" columns and computes
    RMSE and R² across all species. The CSV merging block below is left as a
    placeholder — update column names to match your specific data files.
    """
    for split_name in list(data_dict.keys()):
        data = data_dict[split_name]
        n = data["targets"].shape[0]
        n_sol = data["dn_sol"].shape[1]
        n_an = data["dn_an"].shape[1]
        n_species = n_sol + n_an

        # Per-row prediction via root solver
        preds = np.zeros((n, n_species))
        init_guess = jnp.ones(n_species) / n_species
        for i in range(n):
            occupations, _ = _find_root_impl(
                input_params,
                data["dn_sol"][i], data["an_sol"][i],
                data["x_sol"][i], data["v_sol"][i],
                data["dn_an"][i], data["x_an"][i], data["v_an"][i],
                data["z"][i],
                init_guess, max_tries=10,
                monotonicity_dict=_freeze_mono(monotonicity_dict),
                rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
                rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
                rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
            )
            preds[i] = np.array(occupations)

        targets = np.array(data["targets"])

        # Build a flat dataframe with per-species columns
        df_dict = {}
        for s in range(n_sol):
            df_dict[f"dn_sol_{s}"] = np.array(data["dn_sol"][:, s])
            df_dict[f"an_sol_{s}"] = np.array(data["an_sol"][:, s])
            df_dict[f"x_sol_{s}"]  = np.array(data["x_sol"][:, s])
        for a in range(n_an):
            df_dict[f"dn_an_{a}"]  = np.array(data["dn_an"][:, a])
            df_dict[f"x_an_{a}"]   = np.array(data["x_an"][:, a])
        for k in range(n_species):
            df_dict[f"target_{k}"] = targets[:, k]
            df_dict[f"pred_{k}"]   = preds[:, k]
        df = pd.DataFrame(df_dict)

        # RMSE and R²
        residuals = preds - targets
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot

        loss, _ = total_objective(
            input_params, data_dict[split_name],
            monotonicity_dict=_freeze_mono(monotonicity_dict),
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        print(f"Loss on the {split_name} data: {loss:.4f}")
        print(f"RMSE on the {split_name} data: {rmse:.4f}")
        print(f"R² on the {split_name} data:   {r2:.4f}")

        df = df.round(4)
        df.to_csv(f"ising_results_{split_name}.csv", index=False)

        # Parity plot — one panel per species
        labels = [f"Solvent {s}" for s in range(n_sol)] + [f"Anion {a}" for a in range(n_an)]
        markers = ["o", "x", "^", "s", "D", "v"]
        colors  = ["b", "orange", "g", "r", "purple", "brown"]

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        for k in range(n_species):
            ax.plot(
                targets[:, k] * 100,
                preds[:, k] * 100,
                markers[k % len(markers)],
                color=colors[k % len(colors)],
                label=labels[k],
            )
        ax.plot([0, 100], [0, 100], "k--", label=f"RMSE: {rmse:.4f}\nR²: {r2:.4f}")
        ax.set_xlabel("MD Fractional CN (%)")
        ax.set_ylabel("Ising Fractional CN (%)")
        ax.legend()
        plt.savefig(f"ising_{split_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
