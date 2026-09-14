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
    default_h_sol, default_h_anion,
    default_J_sol_sol, default_J_sol_anion, default_J_anion_anion,
    default_h_sol_rescale, default_h_anion_rescale,
    default_J_sol_sol_rescale, default_J_sol_anion_rescale, default_J_anion_anion_rescale,
    default_conc_factor_rescale,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)

# Module-level optimizer — must be set by the user before calling update() or train()
optimizer = None


def objective_single(
    input_params,
    sol_props, anion_props,
    z,
    targets,
    init_guess,
    monotonicity_dict=None,
    h_sol_func=default_h_sol,
    h_anion_func=default_h_anion,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_anion_func=default_J_sol_anion,
    J_anion_anion_func=default_J_anion_anion,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_anion=default_h_anion_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_anion=default_J_sol_anion_rescale,
    rescale_J_anion_anion=default_J_anion_anion_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Compute the squared loss for a single data point.

    Args:
        input_params:  parameter dict (must include params_anion_anion)
        sol_props:     dict with keys "dn","an","x","v" — shape (N,) arrays
        anion_props:   dict with keys "dn","x","v" — shape (M,) arrays
        z:             coordination number (scalar)
        targets:       target occupation fractions, shape (N+M,)
        init_guess:    initial guess for the root solver, shape (N+M,)
        monotonicity_dict: frozenset from _freeze_mono(), or None for default.
        h_sol_func, h_anion_func, J_sol_sol_func, J_sol_anion_func, J_anion_anion_func:
                       injectable term functions (defaults = current physics).
        rescale_h_sol, rescale_h_anion, rescale_J_sol_sol, rescale_J_sol_anion,
        rescale_J_anion_anion, rescale_conc_factor:
                       companion rescaling functions.

    Returns:
        (loss, new_init): squared loss (scalar), updated initial guess shape (N+M,)
    """
    occupations, found_valid = _find_root_impl(
        input_params, sol_props, anion_props, z,
        init_guess, max_tries=10, monotonicity_dict=monotonicity_dict,
        h_sol_func=h_sol_func, h_anion_func=h_anion_func,
        J_sol_sol_func=J_sol_sol_func, J_sol_anion_func=J_sol_anion_func,
        J_anion_anion_func=J_anion_anion_func,
        rescale_h_sol=rescale_h_sol, rescale_h_anion=rescale_h_anion,
        rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_anion=rescale_J_sol_anion,
        rescale_J_anion_anion=rescale_J_anion_anion, rescale_conc_factor=rescale_conc_factor,
    )
    loss = jnp.sum((occupations - targets) ** 2)
    new_init = jnp.where(found_valid, occupations, init_guess)
    return loss, new_init


@partial(jax.jit, static_argnames=(
    "monotonicity_dict",
    "h_sol_func", "h_anion_func",
    "J_sol_sol_func", "J_sol_anion_func", "J_anion_anion_func",
    "rescale_h_sol", "rescale_h_anion",
    "rescale_J_sol_sol", "rescale_J_sol_anion", "rescale_J_anion_anion",
    "rescale_conc_factor",
))
def total_objective(
    params, data, monotonicity_dict=None,
    h_sol_func=default_h_sol,
    h_anion_func=default_h_anion,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_anion_func=default_J_sol_anion,
    J_anion_anion_func=default_J_anion_anion,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_anion=default_h_anion_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_anion=default_J_sol_anion_rescale,
    rescale_J_anion_anion=default_J_anion_anion_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Compute RMSE loss over all data points.

    data must contain:
        sol_props:   dict with keys "dn","an","x","v" — shape (n, N) arrays per key
        anion_props: dict with keys "dn","x","v" — shape (n, M) arrays per key
        z:           shape (n,) coordination numbers
        targets:     shape (n, N+M) target occupation fractions
        init_guess:  shape (n, N+M)
    monotonicity_dict: frozenset from _freeze_mono(), or None for default.
    h_sol_func, h_anion_func, J_sol_sol_func, J_sol_anion_func, J_anion_anion_func:
                       injectable term functions (defaults = current physics).
    """
    v_objective = jax.vmap(
        partial(
            objective_single,
            monotonicity_dict=monotonicity_dict,
            h_sol_func=h_sol_func, h_anion_func=h_anion_func,
            J_sol_sol_func=J_sol_sol_func, J_sol_anion_func=J_sol_anion_func,
            J_anion_anion_func=J_anion_anion_func,
            rescale_h_sol=rescale_h_sol, rescale_h_anion=rescale_h_anion,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_anion=rescale_J_sol_anion,
            rescale_J_anion_anion=rescale_J_anion_anion, rescale_conc_factor=rescale_conc_factor,
        ),
        in_axes=(None, 0, 0, 0, 0, 0),
    )
    losses, new_inits = v_objective(
        params,
        data["sol_props"],
        data["anion_props"],
        data["z"],
        data["targets"],
        data["init_guess"],
    )
    mean_loss = jnp.sqrt(jnp.mean(losses))
    return mean_loss, new_inits


#: Default number of elements in each trainable parameter array.
#: Used by initialize_params to decide whether to apply hand-tuned initial
#: values or zeros + noise when a custom size is requested via param_sizes=.
_DEFAULT_PARAM_SIZES = {
    "sol_params_dn":      5,
    "salt_params_dn":     5,
    "params_sol_salt_an": 6,
    "params_sol_sol":     16,
    "params_anion_anion": 6,
    "conc_factor_sol":    8,
}

# Hand-tuned starting points for the default parameter sizes.
# None signals "always zero-initialize" (additive noise only).
_HAND_TUNED_PARAMS = {
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
    "params_anion_anion": None,   # 6-element anion-anion; always zero-initialized
    "conc_factor_sol": jnp.array([
        0.6028587818145752, 0.5641694664955139, -0.0062898872420191765,
        -1.0074571371078491, -2177.35791015625, 2189.39013671875,
        -6.164828777313232, 2.281510829925537,
    ]),
}


def initialize_params(
    mode="from_file",
    file_path=None,
    input_dim=1,
    hidden_dim=4,
    output_dim=1,
    random_seed=42,
    param_sizes=None,
):
    """Initialize the parameter dict for the Ising electrolyte model.

    Args:
        mode:        "from_scratch" or "from_file".
        file_path:   Required when mode="from_file". Path to a .pkl checkpoint.
        random_seed: PRNG seed for additive/multiplicative initialization noise.
        param_sizes: Optional dict mapping parameter names to custom array lengths.
                     Keys absent from param_sizes use their default size.
                     Arrays with a custom (non-default) size are initialized as
                     zeros + noise because the hand-tuned values are only valid
                     for the default parameter semantics.
                     Example: {"params_sol_sol": 9, "sol_params_dn": 8}
                     Default sizes: sol_params_dn=5, salt_params_dn=5,
                     params_sol_salt_an=6, params_sol_sol=16,
                     params_anion_anion=6, conc_factor_sol=8.
                     Ignored when mode="from_file" (sizes come from checkpoint).

    Returns:
        dict with keys: sol_params_dn, salt_params_dn, params_sol_salt_an,
        params_sol_sol, params_anion_anion, conc_factor_sol.
    """
    if mode == "from_scratch":
        # Resolve effective sizes — default unless overridden by param_sizes.
        _sizes = dict(_DEFAULT_PARAM_SIZES)
        if param_sizes:
            for k, v in param_sizes.items():
                if k in _sizes:
                    _sizes[k] = int(v)

        # PRNG keys: split order matches _DEFAULT_PARAM_SIZES insertion order
        # (sol_params_dn, salt_params_dn, params_sol_salt_an, params_sol_sol,
        #  params_anion_anion, conc_factor_sol) so that keys[i] assignments are
        # deterministic and identical to the original hard-coded behavior for
        # default sizes. num=7 preserves the original PRNG sequence exactly.
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=len(_DEFAULT_PARAM_SIZES) + 1)
        noise_scale = 0.05

        init_params = {}
        for i, name in enumerate(_DEFAULT_PARAM_SIZES):
            size = _sizes[name]
            base = _HAND_TUNED_PARAMS[name]
            if base is not None and size == _DEFAULT_PARAM_SIZES[name]:
                # Default size with hand-tuned starting point: multiplicative noise
                arr = base * (1.0 + noise_scale * random.normal(keys[i], shape=base.shape))
            else:
                # Custom size or inherently zero-initialized: additive noise from zeros
                arr = jnp.zeros(size) + noise_scale * random.normal(keys[i], shape=(size,))
            init_params[name] = arr

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
        # add random noise to the loaded params (sizes come from the checkpoint)
        key = random.PRNGKey(random_seed)
        keys = random.split(key, num=len(_DEFAULT_PARAM_SIZES) + 1)
        noise_scale = 0.05
        for i, name in enumerate(_DEFAULT_PARAM_SIZES):
            arr = init_params[name]
            init_params[name] = arr * (1.0 + noise_scale * random.normal(
                keys[i], shape=arr.shape
            ))
        return init_params


@partial(jax.jit, static_argnames=(
    "monotonicity_dict",
    "h_sol_func", "h_anion_func",
    "J_sol_sol_func", "J_sol_anion_func", "J_anion_anion_func",
    "rescale_h_sol", "rescale_h_anion",
    "rescale_J_sol_sol", "rescale_J_sol_anion", "rescale_J_anion_anion",
    "rescale_conc_factor",
))
def update(
    params, opt_state, data, monotonicity_dict=None,
    h_sol_func=default_h_sol,
    h_anion_func=default_h_anion,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_anion_func=default_J_sol_anion,
    J_anion_anion_func=default_J_anion_anion,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_anion=default_h_anion_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_anion=default_J_sol_anion_rescale,
    rescale_J_anion_anion=default_J_anion_anion_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    (loss, new_inits), grads = jax.value_and_grad(
        partial(
            total_objective,
            monotonicity_dict=monotonicity_dict,
            h_sol_func=h_sol_func, h_anion_func=h_anion_func,
            J_sol_sol_func=J_sol_sol_func, J_sol_anion_func=J_sol_anion_func,
            J_anion_anion_func=J_anion_anion_func,
            rescale_h_sol=rescale_h_sol, rescale_h_anion=rescale_h_anion,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_anion=rescale_J_sol_anion,
            rescale_J_anion_anion=rescale_J_anion_anion, rescale_conc_factor=rescale_conc_factor,
        ), has_aux=True
    )(params, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, new_inits


def _take_batch(data, idx):
    """Slice a data dict (possibly with nested dicts) along axis 0."""
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = {kk: jnp.take(vv, idx, axis=0) for kk, vv in v.items()}
        else:
            result[k] = jnp.take(v, idx, axis=0)
    return result


def train(params, num_epochs, train_data, test_data, val_data, opt_state,
          monotonicity_dict=DEFAULT_MONOTONICITY,
          h_sol_func=default_h_sol,
          h_anion_func=default_h_anion,
          J_sol_sol_func=default_J_sol_sol,
          J_sol_anion_func=default_J_sol_anion,
          J_anion_anion_func=default_J_anion_anion,
          rescale_h_sol=default_h_sol_rescale,
          rescale_h_anion=default_h_anion_rescale,
          rescale_J_sol_sol=default_J_sol_sol_rescale,
          rescale_J_sol_anion=default_J_sol_anion_rescale,
          rescale_J_anion_anion=default_J_anion_anion_rescale,
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
    n_species = (train_data["sol_props"]["dn"].shape[1]
                 + train_data["anion_props"]["dn"].shape[1])
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
            # Create a batch of data (handles nested dicts for sol_props/anion_props)
            batch = _take_batch(train_data, batch_idx)
            batch['init_guess'] = jnp.take(init_guess_train, batch_idx, axis=0)
            # Compute gradients & update parameters on this batch
            params, opt_state, loss, new_inits = update(
                params, opt_state, batch,
                monotonicity_dict=_freeze_mono(monotonicity_dict),
                h_sol_func=h_sol_func, h_anion_func=h_anion_func,
                J_sol_sol_func=J_sol_sol_func, J_sol_anion_func=J_sol_anion_func,
                J_anion_anion_func=J_anion_anion_func,
                rescale_h_sol=rescale_h_sol, rescale_h_anion=rescale_h_anion,
                rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_anion=rescale_J_sol_anion,
                rescale_J_anion_anion=rescale_J_anion_anion, rescale_conc_factor=rescale_conc_factor,
            )
            init_guess_train = init_guess_train.at[batch_idx].set(new_inits)
            epoch_loss += loss

        epoch_loss /= steps_per_ep
        val_data['init_guess'] = init_guess_val
        test_data['init_guess'] = init_guess_test
        val_loss, _ = total_objective(
            params, val_data,
            monotonicity_dict=_freeze_mono(monotonicity_dict),
            h_sol_func=h_sol_func, h_anion_func=h_anion_func,
            J_sol_sol_func=J_sol_sol_func, J_sol_anion_func=J_sol_anion_func,
            J_anion_anion_func=J_anion_anion_func,
            rescale_h_sol=rescale_h_sol, rescale_h_anion=rescale_h_anion,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_anion=rescale_J_sol_anion,
            rescale_J_anion_anion=rescale_J_anion_anion, rescale_conc_factor=rescale_conc_factor,
        )
        test_loss, _ = total_objective(
            params, test_data,
            monotonicity_dict=_freeze_mono(monotonicity_dict),
            h_sol_func=h_sol_func, h_anion_func=h_anion_func,
            J_sol_sol_func=J_sol_sol_func, J_sol_anion_func=J_sol_anion_func,
            J_anion_anion_func=J_anion_anion_func,
            rescale_h_sol=rescale_h_sol, rescale_h_anion=rescale_h_anion,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_anion=rescale_J_sol_anion,
            rescale_J_anion_anion=rescale_J_anion_anion, rescale_conc_factor=rescale_conc_factor,
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
    h_sol_func=default_h_sol,
    h_anion_func=default_h_anion,
    J_sol_sol_func=default_J_sol_sol,
    J_sol_anion_func=default_J_sol_anion,
    J_anion_anion_func=default_J_anion_anion,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_anion=default_h_anion_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_anion=default_J_sol_anion_rescale,
    rescale_J_anion_anion=default_J_anion_anion_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Evaluate trained parameters against each dataset split and produce parity plots.

    data_dict maps split names (e.g. "train", "val", "test") to data dicts with keys:
        sol_props:   dict with keys "dn","an","x","v" — shape (n, N) per key
        anion_props: dict with keys "dn","x","v" — shape (n, M) per key
        z:           shape (n,)
        targets:     shape (n, N+M)
    The function stores per-species predictions as "pred_{i}" columns and computes
    RMSE and R² across all species. The CSV merging block below is left as a
    placeholder — update column names to match your specific data files.
    """
    for split_name in list(data_dict.keys()):
        data = data_dict[split_name]
        n = data["targets"].shape[0]
        sol_props_data   = data["sol_props"]
        anion_props_data = data["anion_props"]
        n_sol     = sol_props_data["dn"].shape[1]
        n_anion   = anion_props_data["dn"].shape[1]
        n_species = n_sol + n_anion

        # Per-row prediction via root solver
        preds = np.zeros((n, n_species))
        init_guess = jnp.ones(n_species) / n_species
        for i in range(n):
            # Slice each leaf of the nested dicts for row i
            sol_props_i   = {k: v[i] for k, v in sol_props_data.items()}
            anion_props_i = {k: v[i] for k, v in anion_props_data.items()}
            occupations, _ = _find_root_impl(
                input_params,
                sol_props_i,
                anion_props_i,
                data["z"][i],
                init_guess, max_tries=10,
                monotonicity_dict=_freeze_mono(monotonicity_dict),
                h_sol_func=h_sol_func, h_anion_func=h_anion_func,
                J_sol_sol_func=J_sol_sol_func, J_sol_anion_func=J_sol_anion_func,
                J_anion_anion_func=J_anion_anion_func,
                rescale_h_sol=rescale_h_sol, rescale_h_anion=rescale_h_anion,
                rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_anion=rescale_J_sol_anion,
                rescale_J_anion_anion=rescale_J_anion_anion, rescale_conc_factor=rescale_conc_factor,
            )
            preds[i] = np.array(occupations)

        targets = np.array(data["targets"])

        # Build a flat dataframe with per-species columns
        df_dict = {}
        for s in range(n_sol):
            df_dict[f"dn_sol_{s}"] = np.array(sol_props_data["dn"][:, s])
            df_dict[f"an_sol_{s}"] = np.array(sol_props_data["an"][:, s])
            df_dict[f"x_sol_{s}"]  = np.array(sol_props_data["x"][:, s])
        for a in range(n_anion):
            df_dict[f"dn_an_{a}"]  = np.array(anion_props_data["dn"][:, a])
            df_dict[f"x_an_{a}"]   = np.array(anion_props_data["x"][:, a])
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
            h_sol_func=h_sol_func, h_anion_func=h_anion_func,
            J_sol_sol_func=J_sol_sol_func, J_sol_anion_func=J_sol_anion_func,
            J_anion_anion_func=J_anion_anion_func,
            rescale_h_sol=rescale_h_sol, rescale_h_anion=rescale_h_anion,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_anion=rescale_J_sol_anion,
            rescale_J_anion_anion=rescale_J_anion_anion, rescale_conc_factor=rescale_conc_factor,
        )
        print(f"Loss on the {split_name} data: {loss:.4f}")
        print(f"RMSE on the {split_name} data: {rmse:.4f}")
        print(f"R² on the {split_name} data:   {r2:.4f}")

        df = df.round(4)
        df.to_csv(f"ising_results_{split_name}.csv", index=False)

        # Parity plot — one panel per species
        labels = [f"Solvent {s}" for s in range(n_sol)] + [f"Anion {a}" for a in range(n_anion)]
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
