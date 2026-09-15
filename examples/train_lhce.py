"""Train the Ising electrolyte model on a 2-solvent + 1-anion dataset.

Usage
-----
# Via YAML config (recommended), from the repository root:
    python examples/train_lhce.py examples/config.yaml

# Via CLI flags:
    python examples/train_lhce.py --train examples/data/lhce_md/fold_0/train.csv \
        --val examples/data/lhce_md/fold_0/val.csv --test examples/data/lhce_md/fold_0/test.csv

# CLI flags override YAML when both are supplied:
    python examples/train_lhce.py examples/config.yaml --epochs 500 --trials 3

Config fields (YAML keys = long CLI flag names)
-----------------------------------------------
    train           Path to training CSV
    val             Path to validation CSV
    test            Path to test CSV
    epochs          Max training epochs per trial  (default: 1000)
    trials          Number of random-seed trials   (default: 10)
    seed            Base random seed               (default: 42)
    checkpoint      Output path for best params    (default: trained_params.pkl)
    learning_rate   Initial Adam LR                (default: 0.002)
    lr_decay_steps  Exponential decay steps        (default: 1000)
    lr_decay_rate   Exponential decay rate         (default: 0.95)

    # Optional — term function overrides (string names from IsingElectrolyte.interactions)
    h_sol_func      Name of h(Li-sol) function     (default: package default)
    h_an_func       Name of h(Li-anion) function   (default: package default)
    j_sol_sol_func  Name of J(sol-sol) function    (default: package default)
    j_sol_an_func   Name of J(sol-anion) function  (default: package default)
    j_an_an_func    Name of J(anion-anion) function(default: package default)

    # Optional - rescale parameter function overrides (string names from IsingElectrolyte.interactions)
    rescale_h_sol          Name of h(Li-sol) rescale function       (default: package default)
    rescale_h_an           Name of h(Li-anion) rescale function     (default: package default)
    rescale_J_sol_sol      Name of J(sol-sol) rescale function      (default: package default)
    rescale_J_sol_an       Name of J(sol-anion) rescale function    (default: package default)
    rescale_J_an_an        Name of J(anion-anion) rescale function  (default: package default)
    rescale_conc_factor    Name of concentration rescale function   (default: package default)

    # Optional - monotonicity constraint dict
    monotonicity_dict      Dict mapping param keys to "increase"/"decrease"/"none".
                           null/omitted → PHYSICAL_MONOTONICITY below (the same
                           constraints as examples/config.yaml; training without
                           constraints usually diverges).
                           {} → disable all constraints; keys not listed → "none".

Expected CSV columns
--------------------
    Solvent DN, Diluent DN, Anion DN
    solvent molar ratio, diluent molar ratio, anion molar ratio
    Solvent AN, Diluent AN
    solvent volume, diluent volume, anion volume
    Fractional CN solvent, Fractional CN diluent, Fractional CN anion
"""

import argparse
import copy
import pickle
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import yaml

import IsingElectrolyte.interactions as _interactions
import IsingElectrolyte.train as ising_train
from IsingElectrolyte.interactions import (
    default_h_sol, default_h_an,
    default_J_sol_sol, default_J_sol_an, default_J_an_an,
    default_h_sol_rescale, default_h_an_rescale,
    default_J_sol_sol_rescale, default_J_sol_an_rescale, default_J_an_an_rescale,
    default_conc_factor_rescale,
)
from IsingElectrolyte.model import DEFAULT_MONOTONICITY
from IsingElectrolyte.train import initialize_params, train, parity_results


# ---------------------------------------------------------------------------
# Term-function resolver
# ---------------------------------------------------------------------------

def _resolve_func(spec, default):
    """Resolve a term function from a callable, string name, or None.

    Args:
        spec:    None    → return ``default`` (package default)
                 callable → return as-is (function defined anywhere)
                 str      → look up by name from ``IsingElectrolyte.interactions``
                            (add your new function there; no import needed here)
        default: fallback callable when spec is None

    Returns:
        A JAX-traceable callable matching the relevant signature contract.

    Raises:
        ValueError: if ``spec`` is a string not found in IsingElectrolyte.interactions
        TypeError:  if ``spec`` is not None, callable, or str
    """
    if spec is None:
        return default
    if callable(spec):
        return spec
    if isinstance(spec, str):
        fn = getattr(_interactions, spec, None)
        if fn is None:
            raise ValueError(
                f"No function named {spec!r} found in IsingElectrolyte.interactions. "
                "Define it there and reinstall the package (pip install -e .)."
            )
        return fn
    raise TypeError(f"Expected None, callable, or str; got {type(spec).__name__!r}")


# ---------------------------------------------------------------------------
# CUSTOMIZATION — swap in your own term functions here
# ---------------------------------------------------------------------------
# Each function must be JAX-traceable (use jax.numpy, not numpy).
#
# Signature contracts:
#   h_sol_func     (dn_eff: scalar, x: scalar, params: array) -> scalar
#   h_an_func      (dn_an:  scalar, x: scalar, params: array) -> scalar
#   J_sol_sol_func (dn_i, an_i, x_i, dn_j, an_j, x_j, params) -> scalar
#   J_sol_an_func  (dn_an, an_sol, x_sol, x_an, params) -> scalar
#   J_an_an_func   (dn_i, x_i, dn_j, x_j, params) -> scalar
#
# Three ways to specify each slot:
#   None       — use the package default (no change needed)
#   callable   — a function defined below or imported above
#   str        — name of a function in IsingElectrolyte.interactions
#                (add it there, no import needed here; same name works in config.yaml)
#
# Examples:
#   H_AN_FUNC = "my_new_h_an"            # define my_new_h_an in interactions.py
#
#   from IsingElectrolyte.interactions import langmuirfunc
#   def my_h_sol(dn_eff, x, params): return langmuirfunc(dn_eff, params)
#   H_SOL_FUNC = my_h_sol
#
# NOTE: functions with different parameter array lengths than the defaults
# require matching changes to initialize_params() or a compatible checkpoint.
# Default parameter counts: h_sol/h_an=5, J_sol_sol=16, J_sol_an/J_an_an=6.

H_SOL_FUNC     = None   # default: expfunc(dn_eff) + logfunc(x),  5 params
H_AN_FUNC      = None   # default: expfunc(dn_an)  + logfunc(x),  5 params
J_SOL_SOL_FUNC = None   # default: DN/AN cross + DN-DN + AN-AN,  16 params
J_SOL_AN_FUNC  = None   # default: sol_sol_func(dn_an, an_sol),   6 params
J_AN_AN_FUNC   = None   # default: sol_sol_func(dn_i, dn_j),      6 params

# ---------------------------------------------------------------------------
# CUSTOMIZATION — swap in your own rescale functions here (optional)
# ---------------------------------------------------------------------------
RESCALE_H_SOL     = None   # default: decrease with DN
RESCALE_H_AN      = None   # default: decrease with DN
RESCALE_J_SOL_SOL = None   # default: decrease with DN
RESCALE_J_SOL_AN  = None   # default: decrease with DN
RESCALE_J_AN_AN   = None   # default: increase with DN
RESCALE_CONC_FACTOR = None # default: increase with concentration and volume

# ---------------------------------------------------------------------------
# CUSTOMIZATION — override the monotonicity constraint dict (optional)
# ---------------------------------------------------------------------------
# A dict mapping parameter groups to "increase", "decrease" or "none".
# None → PHYSICAL_MONOTONICITY (below), also used when config.yaml omits the key.
# {} (empty dict) → disable all monotonicity constraints.
# Groups not listed in a dict are unconstrained ("none").
#
# Example — constrain Li-solvent and Li-anion binding only:
#   MONOTONICITY_DICT = {"sol_params_dn": "decrease", "salt_params_dn": "decrease"}
#
# Can also be set from config.yaml under the 'monotonicity_dict' key
# (script-level value takes priority).
MONOTONICITY_DICT = None

# Physically motivated constraints from the paper; the default of this script.
# Without constraints the root solver usually goes NaN within the first epoch.
PHYSICAL_MONOTONICITY = {
    "sol_params_dn":      "decrease",   # h(Li-solvent) decreases with DN
    "salt_params_dn":     "decrease",   # h(Li-anion) decreases with anion DN
    "params_sol_salt_an": "decrease",   # J(solvent-anion)
    "params_sol_sol":     "decrease",   # J(solvent-solvent)
    "params_anion_anion": "increase",   # J(anion-anion)
    "conc_factor_sol":    "increase",   # concentration-volume correction
}

# ---------------------------------------------------------------------------
# Coordination number — fixed for the standard DME/TTE/LiTFSI MD dataset
# ---------------------------------------------------------------------------
Z_FIXED = 3.72 / 2.0


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULTS = {
    "train":           None,
    "val":             None,
    "test":            None,
    "epochs":          1000,
    "trials":          10,
    "seed":            42,
    "checkpoint":      "trained_params.pkl",
    "learning_rate":   2e-3,
    "lr_decay_steps":  1000,
    "lr_decay_rate":   0.95,
    # Optional term-function overrides (string names from IsingElectrolyte.interactions)
    "h_sol_func":      None,
    "h_an_func":       None,
    "j_sol_sol_func":  None,
    "j_sol_an_func":   None,
    "j_an_an_func":    None,
    # Optional rescale function overrides (string names from IsingElectrolyte.interactions)
    "rescale_h_sol":      None,
    "rescale_h_an":       None,
    "rescale_J_sol_sol":  None,
    "rescale_J_sol_an":   None,
    "rescale_J_an_an":    None,
    "rescale_conc_factor": None,
    # Optional monotonicity constraint dict.
    # None → PHYSICAL_MONOTONICITY; {} → disable all constraints.
    "monotonicity_dict": None,
    # Optional kwargs forwarded verbatim to initialize_params() each trial.
    # 'random_seed' is always computed per-trial and is silently ignored here.
    "initialize_params_kwargs": {},
}


def _check_monotonicity(mono):
    """Validate a monotonicity dict and fill unlisted parameter groups with "none"."""
    if not isinstance(mono, dict):
        raise SystemExit(f"monotonicity_dict must be a mapping, got {mono!r}")
    unknown = sorted(set(mono) - set(DEFAULT_MONOTONICITY))
    if unknown:
        raise SystemExit(f"monotonicity_dict: unknown parameter group(s) {unknown}; "
                         f"valid groups are {sorted(DEFAULT_MONOTONICITY)}")
    bad = {k: v for k, v in mono.items() if v not in ("increase", "decrease", "none")}
    if bad:
        raise SystemExit(f"monotonicity_dict: values must be 'increase', 'decrease' or 'none', got {bad}")
    return {**DEFAULT_MONOTONICITY, **mono}


def _load_config(argv):
    """Parse config from an optional YAML file and CLI flags.

    Priority (highest to lowest): CLI flags > YAML file > defaults.
    """
    parser = argparse.ArgumentParser(
        description="Train Ising electrolyte model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("yaml_config", nargs="?", default=None,
                        help="Optional YAML config file")
    parser.add_argument("--train",          default=None)
    parser.add_argument("--val",            default=None)
    parser.add_argument("--test",           default=None)
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--trials",         type=int,   default=None)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--checkpoint",     default=None)
    parser.add_argument("--learning_rate",  type=float, default=None)
    parser.add_argument("--lr_decay_steps", type=int,   default=None)
    parser.add_argument("--lr_decay_rate",  type=float, default=None)
    # Term-function overrides: string names looked up in IsingElectrolyte.interactions
    parser.add_argument("--h_sol_func",     default=None,
                        help="Name of h(Li-sol) function in IsingElectrolyte.interactions")
    parser.add_argument("--h_an_func",      default=None,
                        help="Name of h(Li-anion) function in IsingElectrolyte.interactions")
    parser.add_argument("--j_sol_sol_func", default=None,
                        help="Name of J(sol-sol) function in IsingElectrolyte.interactions")
    parser.add_argument("--j_sol_an_func",  default=None,
                        help="Name of J(sol-anion) function in IsingElectrolyte.interactions")
    parser.add_argument("--j_an_an_func",   default=None,
                        help="Name of J(anion-anion) function in IsingElectrolyte.interactions")
    # Rescale function overrides: string names looked up in IsingElectrolyte.interactions
    parser.add_argument("--rescale_h_sol",      default=None,
                        help="Name of h(Li-sol) rescale function in IsingElectrolyte.interactions")
    parser.add_argument("--rescale_h_an",       default=None,
                        help="Name of h(Li-anion) rescale function in IsingElectrolyte.interactions")
    parser.add_argument("--rescale_J_sol_sol",  default=None,
                        help="Name of J(sol-sol) rescale function in IsingElectrolyte.interactions")
    parser.add_argument("--rescale_J_sol_an",   default=None,
                        help="Name of J(sol-anion) rescale function in IsingElectrolyte.interactions")
    parser.add_argument("--rescale_J_an_an",    default=None,
                        help="Name of J(anion-anion) rescale function in IsingElectrolyte.interactions")
    parser.add_argument("--rescale_conc_factor", default=None,
                        help="Name of concentration factor rescale function in IsingElectrolyte.interactions")

    args = parser.parse_args(argv)

    # Start from defaults, overlay YAML, then overlay explicit CLI flags
    cfg = dict(DEFAULTS)

    if args.yaml_config is not None:
        with open(args.yaml_config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg.update(yaml_cfg)

    for key in DEFAULTS:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val

    for required in ("train", "val", "test"):
        if cfg[required] is None:
            parser.error(f"--{required} is required (or set '{required}' in the YAML config)")

    # Normalise initialize_params_kwargs: YAML may produce None for an empty mapping
    if not cfg.get("initialize_params_kwargs"):
        cfg["initialize_params_kwargs"] = {}
    # monotonicity_dict: keep None as None (→ PHYSICAL_MONOTONICITY at call site).
    # An explicit {} in YAML stays {} (meaning: disable all constraints).
    if "monotonicity_dict" not in cfg:
        cfg["monotonicity_dict"] = None

    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(path):
    """Load a CSV split and return a data dict for train.py.

    The new API expects species properties as 2D arrays (n_samples × n_species).
    For the standard 2-solvent + 1-anion system:
        dn_sol, an_sol, x_sol, v_sol : shape (n, 2)
        dn_an, x_an, v_an            : shape (n, 1)
        targets                      : shape (n, 3)   [solvent, diluent, anion]
        z                            : shape (n,)      fixed scalar broadcast
        init_guess                   : shape (n, 3)    uniform 1/3
    """
    df = pd.read_csv(path)
    n = len(df)

    def col(name):
        return jnp.array(df[name].values, dtype=jnp.float64)

    dn_sol = jnp.stack([col("Solvent DN"),            col("Diluent DN")],           axis=1)
    an_sol = jnp.stack([col("Solvent AN"),            col("Diluent AN")],           axis=1)
    x_sol  = jnp.stack([col("solvent molar ratio"),   col("diluent molar ratio")],  axis=1)
    v_sol  = jnp.stack([col("solvent volume"),        col("diluent volume")],       axis=1)
    dn_an  = col("Anion DN")[:, None]
    x_an   = col("anion molar ratio")[:, None]
    v_an   = col("anion volume")[:, None]

    targets = jnp.stack([
        col("Fractional CN solvent"),
        col("Fractional CN diluent"),
        col("Fractional CN anion"),
    ], axis=1)

    z          = jnp.full((n,), Z_FIXED, dtype=jnp.float64)
    init_guess = jnp.full((n, 3), 1.0 / 3.0, dtype=jnp.float64)

    return {
        "dn_sol":     dn_sol,
        "an_sol":     an_sol,
        "x_sol":      x_sol,
        "v_sol":      v_sol,
        "dn_an":      dn_an,
        "x_an":       x_an,
        "v_an":       v_an,
        "z":          z,
        "targets":    targets,
        "init_guess": init_guess,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    cfg = _load_config(sys.argv[1:] if argv is None else argv)

    # ------------------------------------------------------------------
    # Resolve term functions
    # Script-level CUSTOMIZATION values take precedence over YAML/CLI strings.
    # ------------------------------------------------------------------
    h_sol_func     = _resolve_func(H_SOL_FUNC     or cfg.get("h_sol_func"),     default_h_sol)
    h_an_func      = _resolve_func(H_AN_FUNC      or cfg.get("h_an_func"),      default_h_an)
    j_sol_sol_func = _resolve_func(J_SOL_SOL_FUNC or cfg.get("j_sol_sol_func"), default_J_sol_sol)
    j_sol_an_func  = _resolve_func(J_SOL_AN_FUNC  or cfg.get("j_sol_an_func"),  default_J_sol_an)
    j_an_an_func   = _resolve_func(J_AN_AN_FUNC   or cfg.get("j_an_an_func"),   default_J_an_an)

    # -------------------------------------------------------------------
    # Resolve rescale functions (optional)
    # Script-level CUSTOMIZATION values take precedence over YAML/CLI strings.
    # -------------------------------------------------------------------
    rescale_h_sol      = _resolve_func(RESCALE_H_SOL      or cfg.get("rescale_h_sol"),      default_h_sol_rescale)
    rescale_h_an       = _resolve_func(RESCALE_H_AN       or cfg.get("rescale_h_an"),       default_h_an_rescale)
    rescale_J_sol_sol     = _resolve_func(RESCALE_J_SOL_SOL  or cfg.get("rescale_J_sol_sol"),  default_J_sol_sol_rescale)
    rescale_J_sol_an      = _resolve_func(RESCALE_J_SOL_AN   or cfg.get("rescale_J_sol_an"),   default_J_sol_an_rescale)
    rescale_J_an_an       = _resolve_func(RESCALE_J_AN_AN    or cfg.get("rescale_J_an_an"),    default_J_an_an_rescale)
    rescale_conc_factor   = _resolve_func(RESCALE_CONC_FACTOR or cfg.get("rescale_conc_factor"), default_conc_factor_rescale)

    # -------------------------------------------------------------------
    # Resolve monotonicity dict
    # Priority: script-level MONOTONICITY_DICT > YAML/CLI > PHYSICAL_MONOTONICITY
    # None at any level → fall through to the next.
    # {} means "disable all constraints"; unlisted groups are "none".
    # -------------------------------------------------------------------
    _mono_raw = MONOTONICITY_DICT if MONOTONICITY_DICT is not None else cfg["monotonicity_dict"]
    monotonicity_dict = _check_monotonicity(
        PHYSICAL_MONOTONICITY if _mono_raw is None else _mono_raw
    )

    print("=== IsingElectrolyte Training ===")
    print(f"  train:    {cfg['train']}")
    print(f"  val:      {cfg['val']}")
    print(f"  test:     {cfg['test']}")
    print(f"  epochs:   {cfg['epochs']}  trials: {cfg['trials']}  seed: {cfg['seed']}")
    print(f"  lr:       {cfg['learning_rate']}  decay_steps: {cfg['lr_decay_steps']}  decay_rate: {cfg['lr_decay_rate']}")
    print(f"  checkpoint: {cfg['checkpoint']}")
    print(f"  h_sol_func:     {h_sol_func.__name__}")
    print(f"  h_an_func:      {h_an_func.__name__}")
    print(f"  j_sol_sol_func: {j_sol_sol_func.__name__}")
    print(f"  j_sol_an_func:  {j_sol_an_func.__name__}")
    print(f"  j_an_an_func:   {j_an_an_func.__name__}")
    print(f"  rescale_h_sol:      {rescale_h_sol.__name__ if rescale_h_sol else None}")
    print(f"  rescale_h_an:       {rescale_h_an.__name__ if rescale_h_an else None}")
    print(f"  rescale_J_sol_sol:  {rescale_J_sol_sol.__name__ if rescale_J_sol_sol else None}")
    print(f"  rescale_J_sol_an:   {rescale_J_sol_an.__name__ if rescale_J_sol_an else None}")
    print(f"  rescale_J_an_an:    {rescale_J_an_an.__name__ if rescale_J_an_an else None}")
    print(f"  rescale_conc_factor: {rescale_conc_factor.__name__ if rescale_conc_factor else None}")
    _ip_kwargs_display = {k: v for k, v in cfg["initialize_params_kwargs"].items()
                          if k != "random_seed"}
    print(f"  initialize_params_kwargs: {_ip_kwargs_display}  (random_seed per trial)")
    print(f"  monotonicity_dict: {monotonicity_dict}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    train_data = load_split(cfg["train"])
    val_data   = load_split(cfg["val"])
    test_data  = load_split(cfg["test"])

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

    for trial in range(cfg["trials"]):
        trial_seed = cfg["seed"] + trial * 10
        print(f"\n--- Trial {trial + 1}/{cfg['trials']}  (seed={trial_seed}) ---")

        ip_kwargs = dict(cfg["initialize_params_kwargs"])
        ip_kwargs.pop("random_seed", None)   # always per-trial, never user-overridable
        ip_kwargs["random_seed"] = trial_seed
        params = initialize_params(**ip_kwargs)

        schedule  = optax.exponential_decay(
            cfg["learning_rate"], cfg["lr_decay_steps"], cfg["lr_decay_rate"]
        )
        optimizer = optax.adam(schedule)
        ising_train.optimizer = optimizer
        opt_state = optimizer.init(params)

        t0 = time.time()
        params, train_log, val_log = train(
            params=params,
            num_epochs=cfg["epochs"],
            train_data=train_data,
            test_data=test_data,
            val_data=val_data,
            opt_state=opt_state,
            monotonicity_dict=monotonicity_dict,
            h_sol_func=h_sol_func, h_an_func=h_an_func,
            J_sol_sol_func=j_sol_sol_func, J_sol_an_func=j_sol_an_func,
            J_an_an_func=j_an_an_func,
            rescale_h_sol = rescale_h_sol,
            rescale_h_an = rescale_h_an,
            rescale_J_sol_sol = rescale_J_sol_sol,
            rescale_J_sol_an = rescale_J_sol_an,
            rescale_J_an_an = rescale_J_an_an,
            rescale_conc_factor = rescale_conc_factor,
            random_seed=trial_seed,
        )
        elapsed = time.time() - t0
        time_log.append(elapsed)

        final_val_loss = float(val_log[-1]) if len(val_log) > 0 else float("inf")
        if np.isfinite(final_val_loss):
            print(f"  Trial {trial + 1} finished in {elapsed:.1f}s — val_loss={final_val_loss:.4f}")
        else:
            print(f"  Trial {trial + 1} diverged after {elapsed:.1f}s (loss became NaN); discarded")

        if final_val_loss < best_val_loss:
            best_val_loss  = final_val_loss
            best_params    = copy.deepcopy(params)
            best_train_log = train_log
            best_val_log   = val_log

        # Early exit if both losses are already very small
        final_train_loss = float(train_log[-1]) if len(train_log) > 0 else float("inf")
        if final_train_loss < 0.08 and final_val_loss < 0.08:
            print("  Losses below threshold — stopping early.")
            break

    if best_params is None:
        sys.exit(
            f"\nERROR: all {len(time_log)} trial(s) diverged (the loss became NaN), so there are no "
            "trained parameters and nothing was saved.\n"
            f"  monotonicity_dict: {monotonicity_dict}\n"
            "  Try the physically motivated constraints (the default when monotonicity_dict is not set),\n"
            "  a lower --learning_rate, or more --trials / a different --seed."
        )

    # ------------------------------------------------------------------
    # Save best parameters
    # ------------------------------------------------------------------
    with open(cfg["checkpoint"], "wb") as f:
        pickle.dump(best_params, f)
    print(f"\nBest params saved to {cfg['checkpoint']}  (val_loss={best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # Save loss logs
    # ------------------------------------------------------------------
    max_len = max(len(best_train_log), len(best_val_log))
    train_pad = np.full(max_len, np.nan)
    val_pad   = np.full(max_len, np.nan)
    train_pad[:len(best_train_log)] = np.array(best_train_log)
    val_pad[:len(best_val_log)]     = np.array(best_val_log)

    loss_df = pd.DataFrame({"train_loss": train_pad, "val_loss": val_pad})
    loss_df.to_csv("train_val_loss_log.csv", index=False)
    print("Loss log saved to train_val_loss_log.csv")

    # ------------------------------------------------------------------
    # Save per-trial timing
    # ------------------------------------------------------------------
    with open("time_log.txt", "w") as f:
        for i, t in enumerate(time_log):
            f.write(f"Trial {i + 1}: {t:.1f}s\n")

    # ------------------------------------------------------------------
    # Parity evaluation on all splits
    # ------------------------------------------------------------------
    print("\nEvaluating best parameters on all splits...")
    parity_results(
        {"train": train_data, "val": val_data, "test": test_data},
        best_params,
        monotonicity_dict=monotonicity_dict,
        h_sol_func=h_sol_func, h_an_func=h_an_func,
        J_sol_sol_func=j_sol_sol_func, J_sol_an_func=j_sol_an_func,
        J_an_an_func=j_an_an_func,
        rescale_h_sol=rescale_h_sol,
        rescale_h_an=rescale_h_an,
        rescale_J_sol_sol=rescale_J_sol_sol,
        rescale_J_sol_an=rescale_J_sol_an,
        rescale_J_an_an=rescale_J_an_an,
        rescale_conc_factor=rescale_conc_factor,
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
    ax.set_title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig("train_val_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Loss curve saved to train_val_loss.png")


if __name__ == "__main__":
    main()
