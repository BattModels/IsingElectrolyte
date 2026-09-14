"""Train the Ising electrolyte model with dielectric-constant-dependent interaction terms.

This script demonstrates how to plug in the new ``h_anion_conc_step`` and
``J_anion_anion_conc_step`` functions (from ``IsingElectrolyte.interactions``) that
account for the dielectric constant of the anion's local environment.

Key differences from ``train_lhce.py``
---------------------------------------
1. The effective dielectric constant (epsilon) is loaded from the CSV columns
   "Solvent Epsilon" and "Diluent Epsilon", averaged by molar fraction, and
   stored as an extra column in ``anion_props``.
2. ``h_anion_func``  is replaced with ``h_anion_conc_step``   (11 params).
3. ``J_anion_anion_func`` is replaced with ``J_anion_anion_conc_step`` (13 params).
4. Non-standard parameter sizes are declared via ``initialize_params_kwargs``.

Expected CSV columns (in addition to the standard ones)
---------------------------------------------------------
    Solvent Epsilon    — dielectric constant of the solvating solvent
    Diluent Epsilon    — dielectric constant of the diluent solvent

Usage
-----
    python examples/train_dielectric.py                    # uses config_dielectric.yaml
    python examples/train_dielectric.py --train data/train.csv --val data/val.csv --test data/test.csv
    python examples/train_dielectric.py config_dielectric.yaml --epochs 500
"""

import copy
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

import IsingElectrolyte.train as ising_train
from IsingElectrolyte.train import initialize_params, train, parity_results
from IsingElectrolyte.model import DEFAULT_MONOTONICITY
from IsingElectrolyte.interactions import (
    default_h_sol,
    h_anion_conc_step,
    h_anion_conc_step_rescale,
    default_J_sol_sol,
    default_J_sol_anion,
    J_anion_anion_conc_step,
    J_anion_anion_conc_step_rescale,
    default_h_sol_rescale,
    default_J_sol_sol_rescale,
    default_J_sol_anion_rescale,
    default_conc_factor_rescale,
)

# ---------------------------------------------------------------------------
# CUSTOMIZATION — term functions
# ---------------------------------------------------------------------------
# h_anion_conc_step uses 11 params (vs. default 5).
# J_anion_anion_conc_step uses 13 params (vs. default 6).
# All other terms keep their defaults.

H_SOL_FUNC           = default_h_sol
H_ANION_FUNC         = h_anion_conc_step
J_SOL_SOL_FUNC       = default_J_sol_sol
J_SOL_ANION_FUNC     = default_J_sol_anion
J_ANION_ANION_FUNC   = J_anion_anion_conc_step

# ---------------------------------------------------------------------------
# CUSTOMIZATION — rescale functions (match the term functions above)
# ---------------------------------------------------------------------------
RESCALE_H_SOL          = default_h_sol_rescale
RESCALE_H_ANION        = h_anion_conc_step_rescale
RESCALE_J_SOL_SOL      = default_J_sol_sol_rescale
RESCALE_J_SOL_ANION    = default_J_sol_anion_rescale
RESCALE_J_ANION_ANION  = J_anion_anion_conc_step_rescale
RESCALE_CONC_FACTOR    = default_conc_factor_rescale

# ---------------------------------------------------------------------------
# CUSTOMIZATION — monotonicity (None → package DEFAULT_MONOTONICITY)
# ---------------------------------------------------------------------------
MONOTONICITY_DICT = None

# ---------------------------------------------------------------------------
# CUSTOMIZATION — parameter initialization
# ---------------------------------------------------------------------------
# Passed directly to initialize_params(). random_seed is always set per-trial.
# param_sizes must match the term functions above:
#   salt_params_dn     = 11  for h_anion_conc_step      (default: 5)
#   params_anion_anion = 13  for J_anion_anion_conc_step (default: 6)
INITIALIZE_PARAMS_KWARGS = {
    "mode": "from_scratch",
    "param_sizes": {
        "salt_params_dn":     11,
        "params_anion_anion": 13,
    },
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRAIN_PATH    = "data/train.csv"
VAL_PATH      = "data/val.csv"
TEST_PATH     = "data/test.csv"
EPOCHS        = 1000
TRIALS        = 10
BASE_SEED     = 42
LR            = 0.01
LR_DECAY_STEPS = 100
LR_DECAY_RATE  = 0.99
CHECKPOINT    = "trained_params_dielectric.pkl"


def _compute_anion_epsilon(df):
    """Compute per-sample effective dielectric constant for the anion environment.

    Uses the mole-fraction-weighted average of the solvent and diluent
    dielectric constants, normalized to the total solvent mole fraction
    (excluding the anion).

      ε_eff = (x_sol * ε_sol + x_dil * ε_dil) / (x_sol + x_dil)

    This approximates the dielectric constant of the local solvation shell
    seen by the anion.

    Returns a 1-D numpy array of shape (n,).
    """
    x_sol = df["solvent molar ratio"].values
    x_dil = df["diluent molar ratio"].values
    eps_sol = df["Solvent Epsilon"].values
    eps_dil = df["Diluent Epsilon"].values
    x_total = x_sol + x_dil
    return (x_sol * eps_sol + x_dil * eps_dil) / x_total


def load_split(path):
    """Load a CSV split and build the nested sol_props / anion_props data dict.

    Identical to the standard load_split in train_lhce.py, except:
    - Computes the effective anion dielectric constant from solvent columns.
    - Adds "epsilon" as an extra key in anion_props.
    """
    import jax.numpy as jnp

    df = pd.read_csv(path)

    # --- Standard solvent properties ---
    sol_props = {
        "dn": jnp.array(np.stack([df["Solvent DN"].values,
                                   df["Diluent DN"].values], axis=1), dtype=float),
        "an": jnp.array(np.stack([df["Solvent AN"].values,
                                   df["Diluent AN"].values], axis=1), dtype=float),
        "x":  jnp.array(np.stack([df["solvent molar ratio"].values,
                                   df["diluent molar ratio"].values], axis=1), dtype=float),
        "v":  jnp.array(np.stack([df["solvent volume"].values,
                                   df["diluent volume"].values], axis=1), dtype=float),
    }

    # --- Standard anion properties + effective dielectric ---
    epsilon_eff = _compute_anion_epsilon(df)   # shape (n,)
    anion_props = {
        "dn":      jnp.array(df["Anion DN"].values[:, None],           dtype=float),
        "x":       jnp.array(df["anion molar ratio"].values[:, None],   dtype=float),
        "v":       jnp.array(df["anion volume"].values[:, None],         dtype=float),
        "epsilon": jnp.array(epsilon_eff[:, None],                      dtype=float),
    }

    targets     = jnp.array(np.stack([
        df["Fractional CN solvent"].values,
        df["Fractional CN diluent"].values,
        df["Fractional CN anion"].values,
    ], axis=1), dtype=float)
    z           = jnp.array(df["z"].values if "z" in df.columns
                            else np.full(len(df), 3.72 / 2.0), dtype=float)
    init_guess  = jnp.ones_like(targets) / targets.shape[1]

    return {
        "sol_props":   sol_props,
        "anion_props": anion_props,
        "targets":     targets,
        "z":           z,
        "init_guess":  init_guess,
    }


def main():
    # ------------------------------------------------------------------
    # Monotonicity
    # ------------------------------------------------------------------
    monotonicity_dict = MONOTONICITY_DICT if MONOTONICITY_DICT is not None else DEFAULT_MONOTONICITY

    print("=== IsingElectrolyte Training — Dielectric Experiment ===")
    print(f"  h_anion_func:          {H_ANION_FUNC.__name__}")
    print(f"  J_anion_anion_func:    {J_ANION_ANION_FUNC.__name__}")
    print(f"  rescale_h_anion:       {RESCALE_H_ANION.__name__}")
    print(f"  rescale_J_anion_anion: {RESCALE_J_ANION_ANION.__name__}")
    print(f"  monotonicity_dict:     {monotonicity_dict}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    train_data = load_split(TRAIN_PATH)
    val_data   = load_split(VAL_PATH)
    test_data  = load_split(TEST_PATH)

    print(f"Loaded {train_data['targets'].shape[0]} train / "
          f"{val_data['targets'].shape[0]} val / "
          f"{test_data['targets'].shape[0]} test samples.")

    # ------------------------------------------------------------------
    # Multi-trial training
    # ------------------------------------------------------------------
    best_params    = None
    best_val_loss  = float("inf")
    best_train_log = None
    best_val_log   = None
    time_log       = []

    for trial in range(TRIALS):
        trial_seed = BASE_SEED + trial * 10
        print(f"\n--- Trial {trial + 1}/{TRIALS}  (seed={trial_seed}) ---")

        ip_kwargs = dict(INITIALIZE_PARAMS_KWARGS)
        ip_kwargs["random_seed"] = trial_seed
        params = initialize_params(**ip_kwargs)

        schedule  = optax.exponential_decay(LR, LR_DECAY_STEPS, LR_DECAY_RATE)
        optimizer = optax.adam(schedule)
        ising_train.optimizer = optimizer
        opt_state = optimizer.init(params)

        t0 = time.time()
        params, train_log, val_log = train(
            params=params,
            num_epochs=EPOCHS,
            train_data=train_data,
            test_data=test_data,
            val_data=val_data,
            opt_state=opt_state,
            monotonicity_dict=monotonicity_dict,
            h_sol_func=H_SOL_FUNC,
            h_anion_func=H_ANION_FUNC,
            J_sol_sol_func=J_SOL_SOL_FUNC,
            J_sol_anion_func=J_SOL_ANION_FUNC,
            J_anion_anion_func=J_ANION_ANION_FUNC,
            rescale_h_sol=RESCALE_H_SOL,
            rescale_h_anion=RESCALE_H_ANION,
            rescale_J_sol_sol=RESCALE_J_SOL_SOL,
            rescale_J_sol_anion=RESCALE_J_SOL_ANION,
            rescale_J_anion_anion=RESCALE_J_ANION_ANION,
            rescale_conc_factor=RESCALE_CONC_FACTOR,
            random_seed=trial_seed,
        )
        elapsed = time.time() - t0
        time_log.append(elapsed)

        final_val_loss = float(val_log[-1]) if len(val_log) > 0 else float("inf")
        print(f"  Trial {trial + 1} finished in {elapsed:.1f}s — val_loss={final_val_loss:.4f}")

        if final_val_loss < best_val_loss:
            best_val_loss  = final_val_loss
            best_params    = copy.deepcopy(params)
            best_train_log = train_log
            best_val_log   = val_log

        final_train_loss = float(train_log[-1]) if len(train_log) > 0 else float("inf")
        if final_train_loss < 0.08 and final_val_loss < 0.08:
            print("  Losses below threshold — stopping early.")
            break

    # ------------------------------------------------------------------
    # Save best parameters
    # ------------------------------------------------------------------
    with open(CHECKPOINT, "wb") as f:
        pickle.dump(best_params, f)
    print(f"\nBest params saved to {CHECKPOINT}  (val_loss={best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # Save loss log
    # ------------------------------------------------------------------
    max_len   = max(len(best_train_log), len(best_val_log))
    train_pad = np.full(max_len, np.nan)
    val_pad   = np.full(max_len, np.nan)
    train_pad[:len(best_train_log)] = np.array(best_train_log)
    val_pad[:len(best_val_log)]     = np.array(best_val_log)

    pd.DataFrame({"train_loss": train_pad, "val_loss": val_pad}).to_csv(
        "train_val_loss_log_dielectric.csv", index=False
    )
    print("Loss log saved to train_val_loss_log_dielectric.csv")

    with open("time_log_dielectric.txt", "w") as f:
        for i, t in enumerate(time_log):
            f.write(f"Trial {i + 1}: {t:.1f}s\n")

    # ------------------------------------------------------------------
    # Parity evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating best parameters on all splits...")
    parity_results(
        {"train": train_data, "val": val_data, "test": test_data},
        best_params,
        monotonicity_dict=monotonicity_dict,
        h_sol_func=H_SOL_FUNC,
        h_anion_func=H_ANION_FUNC,
        J_sol_sol_func=J_SOL_SOL_FUNC,
        J_sol_anion_func=J_SOL_ANION_FUNC,
        J_anion_anion_func=J_ANION_ANION_FUNC,
        rescale_h_sol=RESCALE_H_SOL,
        rescale_h_anion=RESCALE_H_ANION,
        rescale_J_sol_sol=RESCALE_J_SOL_SOL,
        rescale_J_sol_anion=RESCALE_J_SOL_ANION,
        rescale_J_anion_anion=RESCALE_J_ANION_ANION,
        rescale_conc_factor=RESCALE_CONC_FACTOR,
    )

    # ------------------------------------------------------------------
    # Loss curve plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs_arr = np.arange(1, len(train_pad) + 1)
    ax.plot(epochs_arr, train_pad, label="Train")
    ax.plot(epochs_arr, val_pad,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE Loss")
    ax.legend()
    ax.set_title("Training Loss — Dielectric Experiment")
    fig.tight_layout()
    fig.savefig("loss_curve_dielectric.png", dpi=150)
    print("Loss curve saved to loss_curve_dielectric.png")


if __name__ == "__main__":
    main()
