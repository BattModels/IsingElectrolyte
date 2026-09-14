import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import jax
import jax.numpy as jnp

from functools import partial
from ..model import _find_root_impl, energetics, DEFAULT_MONOTONICITY, _freeze_mono
from ..interactions import (
    default_h_sol_rescale, default_h_an_rescale,
    default_J_sol_sol_rescale, default_J_sol_an_rescale, default_J_an_an_rescale,
    default_conc_factor_rescale,
)
from . import solvent_map_dict


def frac_occupation(
    input_params, monotonicity_dict=DEFAULT_MONOTONICITY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Fractional occupation of each species as a function of solvent 1 and solvent 2 DN.

    Sweeps DN of both solvents over a 2D grid, with all other properties fixed to
    the DME/TTE/LiTFSI reference system (molar ratio 3:6:1, AN=10.2).
    """
    npoints = 50
    dn1 = jnp.linspace(0, 40, npoints)
    dn2 = jnp.linspace(0, 40, npoints)
    dn1, dn2 = jnp.meshgrid(dn1, dn2)
    dn1_flat = dn1.flatten()
    dn2_flat = dn2.flatten()

    # Fixed species properties for the reference system
    an0          = 10.2
    an1          = 10.2
    x0           = 0.24
    x1           = 0.48
    x_anion      = 0.14
    dn_anion     = 11.2
    v_sol0       = solvent_map_dict["DME"]['volume A3']
    v_sol1       = solvent_map_dict["TTE"]['volume A3']
    v_an         = solvent_map_dict["TFSI"]['volume A3']
    z            = 3.72 / 2.0
    init_guess   = jnp.array([1/3, 1/3, 1/3])

    def predict(d1, d2):
        dn_sol = jnp.array([d1,      d2])
        an_sol = jnp.array([an0,     an1])
        x_sol  = jnp.array([x0,      x1])
        v_sol  = jnp.array([v_sol0,  v_sol1])
        dn_an  = jnp.array([dn_anion])
        x_an   = jnp.array([x_anion])
        v_an_  = jnp.array([v_an])
        roots, _ = _find_root_impl(
            input_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an_, z,
            init_guess, max_tries=10,
            monotonicity_dict=monotonicity_dict,
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        return roots

    batched_predict = jax.vmap(predict)
    roots_flat = batched_predict(dn1_flat, dn2_flat)
    roots = roots_flat.reshape(npoints, npoints, 3)

    def update_array(arr):
        # Replace out-of-range values with the average of their 4-neighbors
        up    = jnp.roll(arr, shift=1,  axis=0)
        down  = jnp.roll(arr, shift=-1, axis=0)
        left  = jnp.roll(arr, shift=1,  axis=1)
        right = jnp.roll(arr, shift=-1, axis=1)
        neighbor_avg = (up + down + left + right) / 4.0
        arr = jnp.where(arr > 1, neighbor_avg, arr)
        arr = jnp.where(arr < 0, neighbor_avg, arr)
        return arr

    roots = jnp.where(roots < 0, 0, roots)
    roots = update_array(roots)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=True)
    vmin = np.min(roots)
    vmax = np.max(roots)
    axes[0].set_title("Solvent 1 Occupation")
    axes[0].contourf(
        dn1.reshape(npoints, npoints), dn2.reshape(npoints, npoints),
        roots[:, :, 0], levels=100, vmin=vmin, vmax=vmax, cmap="plasma",
    )
    axes[1].set_title("Solvent 2 Occupation")
    axes[1].contourf(
        dn1.reshape(npoints, npoints), dn2.reshape(npoints, npoints),
        roots[:, :, 1], levels=100, vmin=vmin, vmax=vmax, cmap="plasma",
    )
    axes[2].set_title("Anion Occupation")
    axes[2].contourf(
        dn1.reshape(npoints, npoints), dn2.reshape(npoints, npoints),
        roots[:, :, 2], levels=100, vmin=vmin, vmax=vmax, cmap="plasma",
    )
    for ax in axes:
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.set_yticks(np.arange(0, 31, 5))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
    axes[0].set_xlabel("Solvent 1 DN")
    axes[1].set_xlabel("Solvent 1 DN")
    axes[2].set_xlabel("Solvent 1 DN")
    axes[0].set_ylabel("Solvent 2 DN")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical")
    cbar.set_label("Fractional Occupation")
    plt.savefig("occupations.jpg", dpi=300, bbox_inches="tight")
    roots = np.array(roots)
    np.save("roots.npy", roots)


def occupations_dn2_an2_csv(
    trained_params, monotonicity_dict=DEFAULT_MONOTONICITY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Occupation as a function of solvent 2 DN and AN; solvent 1 fixed to DME properties."""
    dn0      = 20.0
    an0      = 10.2
    dn1      = jnp.linspace(1.0, 40, 100)
    an1      = jnp.linspace(1.0, 40, 100)
    dn1, an1 = jnp.meshgrid(dn1, an1)
    dn1_flat = dn1.flatten()
    an1_flat = an1.flatten()

    dn_anion    = 11.2
    x0, x1     = 0.24, 0.48
    x_anion     = 0.14
    v_sol0      = solvent_map_dict["DME"]['volume A3']
    v_sol1      = solvent_map_dict["TTE"]['volume A3']
    v_an        = solvent_map_dict["TFSI"]['volume A3']
    z           = 3.72 / 2.0
    init_guess  = jnp.array([1/3, 1/3, 1/3])

    def predict(d1, a1):
        dn_sol = jnp.array([dn0, d1])
        an_sol = jnp.array([an0, a1])
        x_sol  = jnp.array([x0,  x1])
        v_sol  = jnp.array([v_sol0, v_sol1])
        dn_an  = jnp.array([dn_anion])
        x_an   = jnp.array([x_anion])
        v_an_  = jnp.array([v_an])
        roots, _ = _find_root_impl(
            trained_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an_, z,
            init_guess, max_tries=10,
            monotonicity_dict=monotonicity_dict,
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        return roots

    batched_predict = jax.vmap(predict)
    y_pred_flat = batched_predict(dn1_flat, an1_flat)
    y_pred = y_pred_flat.reshape(100, 100, 3)

    fig = plt.figure(figsize=(18, 5))
    axes = fig.subplots(1, 3, sharex=False, sharey=True)
    axes[0].set_title("Solvent 1 Occupation")
    axes[0].contourf(dn1.reshape(100, 100), an1.reshape(100, 100),
                     y_pred[:, :, 0].reshape(100, 100), levels=100, cmap="plasma")
    axes[1].set_title("Solvent 2 Occupation")
    axes[1].contourf(dn1.reshape(100, 100), an1.reshape(100, 100),
                     y_pred[:, :, 1].reshape(100, 100), levels=100, cmap="plasma")
    axes[2].set_title("Anion Occupation")
    axes[2].contourf(dn1.reshape(100, 100), an1.reshape(100, 100),
                     y_pred[:, :, 2].reshape(100, 100), levels=100, cmap="plasma")
    for ax in axes:
        ax.set_xlabel("Solvent 2 DN")
        ax.set_ylabel("Solvent 2 AN")
    cbar = fig.colorbar(axes[0].collections[0], ax=axes, orientation="vertical")
    cbar.set_label("Fractional Occupation")
    plt.savefig("occpupations_dn2_an2_contour.png", dpi=300, bbox_inches="tight")
    plt.close(fig=fig)

    dn1_flat = np.array(dn1_flat)
    an1_flat = np.array(an1_flat)
    df = pd.DataFrame({
        "solvent2_dn":         dn1_flat,
        "solvent2_an":         an1_flat,
        "solvent1_occupation": np.array(y_pred[:, :, 0]).flatten(),
        "solvent2_occupation": np.array(y_pred[:, :, 1]).flatten(),
        "anion_occupation":    np.array(y_pred[:, :, 2]).flatten(),
    })
    df.to_csv("occupations_dn2_an2_contour.csv", index=False)


def occupations_dn2_an2_contour(
    trained_params, monotonicity_dict=DEFAULT_MONOTONICITY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Line plots of occupation vs. solvent 2 DN for several fixed AN values."""
    dn0, an0 = 20.0, 10.2
    dn1      = jnp.linspace(1.0, 40, 100)
    an1_vals = jnp.array([5, 10, 15, 20])

    dn_anion   = 11.2
    x0, x1    = 0.24, 0.48
    x_anion    = 0.14
    v_sol0     = solvent_map_dict["DME"]['volume A3']
    v_sol1     = solvent_map_dict["TTE"]['volume A3']
    v_an       = solvent_map_dict["TFSI"]['volume A3']
    z          = 3.72 / 2.0
    init_guess = jnp.array([1/3, 1/3, 1/3])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)
    for a1 in an1_vals:
        dn1_flat = dn1.flatten()
        an1_flat = a1 * jnp.ones_like(dn1_flat)

        def predict(d1, a1_):
            dn_sol = jnp.array([dn0, d1])
            an_sol = jnp.array([an0, a1_])
            x_sol  = jnp.array([x0,  x1])
            v_sol  = jnp.array([v_sol0, v_sol1])
            dn_an  = jnp.array([dn_anion])
            x_an   = jnp.array([x_anion])
            v_an_  = jnp.array([v_an])
            roots, _ = _find_root_impl(
                trained_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an_, z,
                init_guess, max_tries=10,
                monotonicity_dict=monotonicity_dict,
                rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
                rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
                rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
            )
            return roots

        batched_predict = jax.vmap(predict)
        y_pred_flat = batched_predict(dn1_flat, an1_flat)
        y_pred = y_pred_flat.reshape(100, 3)
        axs[0].plot(dn1, y_pred[:, 0], label=f"AN={a1}")
        axs[1].plot(dn1, y_pred[:, 1], label=f"AN={a1}")
        axs[2].plot(dn1, y_pred[:, 2], label=f"AN={a1}")

    axs[0].set_ylabel("Solvent 1 Occupation")
    axs[1].set_ylabel("Solvent 2 Occupation")
    axs[2].set_ylabel("Anion Occupation")
    for ax in axs:
        ax.set_xlabel("Solvent 2 DN")
        ax.legend()
    plt.savefig("occpupations_dn2_an2.png", bbox_inches="tight")


def visualize_li_free_energy_contour(
    trained_params, monotonicity_dict=DEFAULT_MONOTONICITY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Li solvation free energy as a 2D contour over solvent 1 and solvent 2 DN."""
    dn1      = jnp.linspace(1.0, 40, 100)
    dn2      = jnp.linspace(1.0, 40, 100)
    dn1, dn2 = jnp.meshgrid(dn1, dn2)
    dn1_flat = dn1.flatten()
    dn2_flat = dn2.flatten()

    an0        = 10.2
    an1        = 10.2
    x0, x1    = 0.24, 0.48
    x_anion   = 0.14
    dn_anion  = 11.2
    v_sol0    = solvent_map_dict["DME"]['volume A3']
    v_sol1    = solvent_map_dict["TTE"]['volume A3']
    v_an      = solvent_map_dict["TFSI"]['volume A3']
    z         = 3.72 / 2.0
    init_guess = jnp.array([1/3, 1/3, 1/3])

    def predict(d1, d2):
        dn_sol = jnp.array([d1,     d2])
        an_sol = jnp.array([an0,    an1])
        x_sol  = jnp.array([x0,     x1])
        v_sol  = jnp.array([v_sol0, v_sol1])
        dn_an  = jnp.array([dn_anion])
        x_an   = jnp.array([x_anion])
        v_an_  = jnp.array([v_an])
        vars, _ = _find_root_impl(
            trained_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an_, z,
            init_guess, max_tries=10,
            monotonicity_dict=monotonicity_dict,
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        h, J, kT = energetics(
            vars, trained_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an_, z,
            monotonicity_dict=monotonicity_dict,
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        return jnp.sum(h * z * vars)

    batched_predict = jax.vmap(predict)
    y_pred_flat = batched_predict(dn1_flat, dn2_flat)
    y_pred = y_pred_flat.reshape(100, 100)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        dn1.reshape(100, 100), dn2.reshape(100, 100),
        y_pred.reshape(100, 100), levels=100, cmap="plasma",
    )
    plt.xlabel("Solvent 1 DN")
    plt.ylabel("Solvent 2 DN")
    plt.colorbar(label="Energy (eV)")
    plt.savefig("li_free_energy_contour.png", dpi=300, bbox_inches="tight")


def visualize_li_free_energy_dn_an_contour(
    trained_params, monotonicity_dict=DEFAULT_MONOTONICITY,
    rescale_h_sol=default_h_sol_rescale,
    rescale_h_an=default_h_an_rescale,
    rescale_J_sol_sol=default_J_sol_sol_rescale,
    rescale_J_sol_an=default_J_sol_an_rescale,
    rescale_J_an_an=default_J_an_an_rescale,
    rescale_conc_factor=default_conc_factor_rescale,
):
    """Li solvation free energy as a 2D contour over solvent 2 DN and AN."""
    dn0      = 20.0
    an0      = 10.2
    dn1      = jnp.linspace(5.0, 40, 100)
    an1      = jnp.linspace(1.0, 40, 100)
    dn1, an1 = jnp.meshgrid(dn1, an1)
    dn1_flat = dn1.flatten()
    an1_flat = an1.flatten()

    x0, x1  = 0.24, 0.48
    x_anion = 0.14
    dn_anion = 11.2
    v_sol0  = solvent_map_dict["DME"]['volume A3']
    v_sol1  = solvent_map_dict["TTE"]['volume A3']
    v_an    = solvent_map_dict["TFSI"]['volume A3']
    z       = 3.72 / 2.0
    init_guess = jnp.array([1/3, 1/3, 1/3])

    def predict(d1, a1):
        dn_sol = jnp.array([dn0,    d1])
        an_sol = jnp.array([an0,    a1])
        x_sol  = jnp.array([x0,     x1])
        v_sol  = jnp.array([v_sol0, v_sol1])
        dn_an  = jnp.array([dn_anion])
        x_an   = jnp.array([x_anion])
        v_an_  = jnp.array([v_an])
        vars, _ = _find_root_impl(
            trained_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an_, z,
            init_guess, max_tries=10,
            monotonicity_dict=monotonicity_dict,
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        h, J, kT = energetics(
            vars, trained_params, dn_sol, an_sol, x_sol, v_sol, dn_an, x_an, v_an_, z,
            monotonicity_dict=monotonicity_dict,
            rescale_h_sol=rescale_h_sol, rescale_h_an=rescale_h_an,
            rescale_J_sol_sol=rescale_J_sol_sol, rescale_J_sol_an=rescale_J_sol_an,
            rescale_J_an_an=rescale_J_an_an, rescale_conc_factor=rescale_conc_factor,
        )
        return jnp.sum(h * z * vars)

    batched_predict = jax.vmap(predict)
    y_pred_flat = batched_predict(dn1_flat, an1_flat)
    y_pred = y_pred_flat.reshape(100, 100)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.contourf(
        dn1.reshape(100, 100), an1.reshape(100, 100),
        y_pred.reshape(100, 100), levels=100, cmap="plasma",
    )
    plt.xlabel("Solvent 2 DN")
    plt.ylabel("Solvent 2 AN")
    plt.colorbar(label="Energy (eV)")
    plt.savefig("li_free_energy_dn_an_contour.png", dpi=300, bbox_inches="tight")
